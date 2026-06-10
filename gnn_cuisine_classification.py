"""
gnn_cuisine_classification.py
==============================
Вспомогательный модуль для задач:
  - Задача 1: предсказание кухни (nb04) — GraphSAGE baseline + HeteroGAT
  - Задача 3: real vs fake recipe (nb03) — HeteroGAT на полном I + feat_df

Архитектура:
  GraphSAGE (homo)  — однородный граф (рецепт ↔ ингредиент слиты в один тип узлов)
  HeteroSAGE        — тот же гетерограф, что у HeteroGAT, но SAGEConv (ablation)
  HeteroGAT         — гетерогенный граф с двумя типами узлов: 'recipe' и 'ingredient'
                ребро: ('recipe', 'contains', 'ingredient') + обратное

Входные данные:
  • G_nx: nx.Graph — двудольный граф рецепт ↔ ингредиент (из nb01)
  • I_nx: nx.Graph — граф ингредиент-ингредиент (из nb01)
  • feat_df: pd.DataFrame — структурные признаки ингредиентов (из nb02)
  • labels: list[str] — метки кухни (или 0/1 для задачи 3) для каждого рецепта
  • recipe_ids: list[str] — идентификаторы рецептных узлов в G_nx (вида 'recipe::N')

Возвращаемое значение каждой функции train_* — SimpleNamespace с полями:
  .accuracy  .f1  .auc  .report  (строка classification_report)
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.nn import GATConv as PyGGATConv


# ─────────────────────────────────────────────────────────────────────────────
# 1. КОНВЕРТАЦИЯ ГРАФА В PyTorch Geometric
# ─────────────────────────────────────────────────────────────────────────────

def _ingredient_features(I_nx: nx.Graph, feat_df) -> tuple[dict, np.ndarray]:
    """
    Возвращает:
      ing2idx  — словарь {имя_ингредиента: индекс}
      X_ing    — нормализованная матрица признаков (N_ing × F)
    """
    COLS = ['degree', 'degree_w', 'pagerank', 'clustering', 'kcore', 'betweenness', 'log_freq']
    available = [c for c in COLS if c in feat_df.columns]

    ing_nodes = sorted(I_nx.nodes())
    ing2idx = {n: i for i, n in enumerate(ing_nodes)}

    if available:
        X_raw = feat_df.reindex(ing_nodes)[available].fillna(0).values.astype(float)
    else:
        # fallback: только degree
        deg = dict(I_nx.degree())
        X_raw = np.array([[deg.get(n, 0)] for n in ing_nodes], dtype=float)

    scaler = StandardScaler()
    X_ing = scaler.fit_transform(X_raw).astype(np.float32)
    return ing2idx, X_ing


def build_homogeneous_data(
    G_nx: nx.Graph,
    recipe_ids: list[str],
    I_nx: nx.Graph,
    feat_df,
    labels_encoded: np.ndarray,
) -> Data:
    """
    Строит однородный граф для GraphSAGE.

    Узлы:  [recipe_0, recipe_1, ..., ing_0, ing_1, ...]
    Рёбра: рецепт ↔ ингредиент из G_nx (двунаправленные)
    Признаки рецепта: среднее признаков его ингредиентов (простой prior)
    """
    ing2idx, X_ing = _ingredient_features(I_nx, feat_df)
    F_dim = X_ing.shape[1]

    n_recipes = len(recipe_ids)
    rec2idx = {r: i for i, r in enumerate(recipe_ids)}

    # признаки рецептов = среднее признаков ингредиентов
    X_rec = np.zeros((n_recipes, F_dim), dtype=np.float32)
    for rec_id in recipe_ids:
        r_i = rec2idx[rec_id]
        if rec_id not in G_nx:
            continue
        nbrs = [n for n in G_nx.neighbors(rec_id) if n in ing2idx]
        if nbrs:
            X_rec[r_i] = X_ing[[ing2idx[n] for n in nbrs]].mean(axis=0)

    # глобальный индекс: 0..n_recipes-1 — рецепты, n_recipes.. — ингредиенты
    offset = n_recipes
    src_list, dst_list = [], []
    for rec_id in recipe_ids:
        if rec_id not in G_nx:
            continue
        ri = rec2idx[rec_id]
        for ing_name in G_nx.neighbors(rec_id):
            if ing_name not in ing2idx:
                continue
            ii = ing2idx[ing_name] + offset
            src_list += [ri, ii]
            dst_list += [ii, ri]

    X_all = np.vstack([X_rec, X_ing])
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    x = torch.tensor(X_all, dtype=torch.float)

    # метки только для рецептных узлов
    y = torch.full((len(X_all),), -1, dtype=torch.long)
    y[:n_recipes] = torch.tensor(labels_encoded, dtype=torch.long)

    # маска: только рецепты имеют метку
    recipe_mask = torch.zeros(len(X_all), dtype=torch.bool)
    recipe_mask[:n_recipes] = True

    data = Data(x=x, edge_index=edge_index, y=y)
    data.recipe_mask = recipe_mask
    data.n_recipes = n_recipes
    return data


def build_hetero_data(
    G_nx: nx.Graph,
    recipe_ids: list[str],
    I_nx: nx.Graph,
    feat_df,
    labels_encoded: np.ndarray,
) -> HeteroData:
    """
    Строит гетерогенный граф для HeteroGAT.

    Типы узлов:   'recipe'     (признаки = среднее признаков ингредиентов)
                  'ingredient' (признаки = структурные из feat_df)
    Типы рёбер:   ('recipe', 'contains', 'ingredient')
                  ('ingredient', 'in', 'recipe')   — обратное
    """
    ing2idx, X_ing = _ingredient_features(I_nx, feat_df)
    F_dim = X_ing.shape[1]

    n_recipes = len(recipe_ids)
    rec2idx = {r: i for i, r in enumerate(recipe_ids)}

    X_rec = np.zeros((n_recipes, F_dim), dtype=np.float32)
    for rec_id in recipe_ids:
        ri = rec2idx[rec_id]
        if rec_id not in G_nx:
            continue
        nbrs = [n for n in G_nx.neighbors(rec_id) if n in ing2idx]
        if nbrs:
            X_rec[ri] = X_ing[[ing2idx[n] for n in nbrs]].mean(axis=0)

    rec_src, ing_dst = [], []
    for rec_id in recipe_ids:
        if rec_id not in G_nx:
            continue
        ri = rec2idx[rec_id]
        for ing_name in G_nx.neighbors(rec_id):
            if ing_name not in ing2idx:
                continue
            ii = ing2idx[ing_name]
            rec_src.append(ri)
            ing_dst.append(ii)

    data = HeteroData()
    data['recipe'].x = torch.tensor(X_rec, dtype=torch.float)
    data['recipe'].y = torch.tensor(labels_encoded, dtype=torch.long)
    data['ingredient'].x = torch.tensor(X_ing, dtype=torch.float)

    if rec_src:
        ei_fwd = torch.tensor([rec_src, ing_dst], dtype=torch.long)
        ei_rev = torch.tensor([ing_dst, rec_src], dtype=torch.long)
    else:
        ei_fwd = torch.zeros((2, 0), dtype=torch.long)
        ei_rev = torch.zeros((2, 0), dtype=torch.long)

    data['recipe', 'contains', 'ingredient'].edge_index = ei_fwd
    data['ingredient', 'in', 'recipe'].edge_index = ei_rev
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 2. МОДЕЛИ
# ─────────────────────────────────────────────────────────────────────────────

class GraphSAGECuisine(nn.Module):
    """2-слойный GraphSAGE для классификации рецептов."""

    def __init__(self, in_channels: int, hidden: int, num_classes: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.clf = nn.Linear(hidden, num_classes)

    def embed(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return x

    def forward(self, x, edge_index):
        return self.clf(self.embed(x, edge_index))


class HeteroGATCuisine(nn.Module):
    """
    2-слойный HeteroGAT для классификации рецептов.

    Каждый слой — HeteroConv с GATConv по каждому типу ребра.
    После двух слоёв берём эмбеддинг рецептного узла → линейный классификатор.
    """

    def __init__(self, in_channels: int, hidden: int, num_classes: int, heads: int = 4):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        # Слой 1: recipe→ing и ing→recipe
        self.conv1 = HeteroConv({
            ('recipe', 'contains', 'ingredient'): PyGGATConv(
                in_channels, hidden // heads, heads=heads, dropout=0.3, add_self_loops=False,
            ),
            ('ingredient', 'in', 'recipe'): PyGGATConv(
                in_channels, hidden // heads, heads=heads, dropout=0.3, add_self_loops=False,
            ),
        }, aggr='sum')

        # Слой 2
        self.conv2 = HeteroConv({
            ('recipe', 'contains', 'ingredient'): PyGGATConv(
                hidden, hidden // heads, heads=heads, dropout=0.3, add_self_loops=False,
            ),
            ('ingredient', 'in', 'recipe'): PyGGATConv(
                hidden, hidden // heads, heads=heads, dropout=0.3, add_self_loops=False,
            ),
        }, aggr='sum')

        self.clf = nn.Linear(hidden, num_classes)

    def embed(self, x_dict, edge_index_dict):
        """Эмбеддинги всех узлов после двух слоёв (до классификатора)."""
        x = self.conv1(x_dict, edge_index_dict)
        x = {k: v.relu() for k, v in x.items()}
        x = self.conv2(x, edge_index_dict)
        x = {k: v.relu() for k, v in x.items()}
        return x

    def forward(self, x_dict, edge_index_dict):
        return self.clf(self.embed(x_dict, edge_index_dict)['recipe'])

    @torch.no_grad()
    def recipe_attention(self, x_dict, edge_index_dict):
        """
        Настоящие attention-веса первого слоя GAT по рёбрам
        ('ingredient', 'in', 'recipe'): насколько модель «слушает»
        каждый ингредиент при формировании эмбеддинга рецепта.

        Возвращает (edge_index [2, E], alpha [E]) — alpha усреднён по головам.
        edge_index[0] = индекс ингредиента, edge_index[1] = индекс рецепта.
        """
        self.eval()
        conv = self.conv1.convs[('ingredient', 'in', 'recipe')]
        ei = edge_index_dict[('ingredient', 'in', 'recipe')]
        _, (att_ei, alpha) = conv(
            (x_dict['ingredient'], x_dict['recipe']),
            ei,
            return_attention_weights=True,
        )
        return att_ei, alpha.mean(dim=1)


class HeteroSAGECuisine(nn.Module):
    """2-слойный HeteroSAGE на том же гетерографе, что HeteroGAT (ablation: SAGE vs attention)."""

    def __init__(self, in_channels: int, hidden: int, num_classes: int):
        super().__init__()
        self.conv1 = HeteroConv({
            ('recipe', 'contains', 'ingredient'): SAGEConv((-1, -1), hidden),
            ('ingredient', 'in', 'recipe'): SAGEConv((-1, -1), hidden),
        }, aggr='sum')
        self.conv2 = HeteroConv({
            ('recipe', 'contains', 'ingredient'): SAGEConv((hidden, hidden), hidden),
            ('ingredient', 'in', 'recipe'): SAGEConv((hidden, hidden), hidden),
        }, aggr='sum')
        self.clf = nn.Linear(hidden, num_classes)

    def embed(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = {k: v.relu() for k, v in x.items()}
        x = self.conv2(x, edge_index_dict)
        x = {k: v.relu() for k, v in x.items()}
        return x

    def forward(self, x_dict, edge_index_dict):
        return self.clf(self.embed(x_dict, edge_index_dict)['recipe'])


@torch.no_grad()
def top_attention_ingredients(model, data, recipe_idx, ing_names, top_k=5):
    """
    Для рецепта `recipe_idx` возвращает его ингредиенты, отсортированные
    по attention-весу HeteroGAT (наиболее «влиятельные» — первыми).

    model      : обученный HeteroGATCuisine
    data       : HeteroData (из build_hetero_data)
    recipe_idx : индекс рецепта в data['recipe']
    ing_names  : список имён ингредиентов (sorted(I_nx.nodes()))

    Веса нормированы внутри рецепта; сравнивать абсолютные значения между рецептами
    разной длины нельзя. Attention — не causal importance.
    """
    att_ei, alpha = model.recipe_attention(data.x_dict, data.edge_index_dict)
    mask = att_ei[1] == recipe_idx
    ings = att_ei[0][mask].tolist()
    scores = alpha[mask].tolist()
    pairs = sorted(zip(ings, scores), key=lambda t: -t[1])[:top_k]
    return [(ing_names[i], float(s)) for i, s in pairs]


# ─────────────────────────────────────────────────────────────────────────────
# 3. ОБУЧЕНИЕ
# ─────────────────────────────────────────────────────────────────────────────

def stratified_recipe_split(
    n_rec: int,
    y_enc: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
    train_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Стратифицированный train/test по меткам кухни (одинаковый для всех моделей)."""
    if train_idx is not None and test_idx is not None:
        return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)
    idx = np.arange(n_rec)
    return train_test_split(
        idx, test_size=test_ratio, random_state=seed, stratify=y_enc,
    )


def _monitor_val_indices(
    tr_idx: np.ndarray,
    y_enc: np.ndarray,
    seed: int,
    val_ratio: float = 0.15,
) -> np.ndarray:
    """Stratified val из train — только для логов во время обучения, не для финальной оценки."""
    y_tr = y_enc[tr_idx]
    if len(tr_idx) < 4 or len(np.unique(y_tr)) < 2:
        return np.asarray(tr_idx[: max(1, len(tr_idx) // 5)], dtype=int)
    _, val_idx = train_test_split(
        tr_idx, test_size=val_ratio, random_state=seed + 1, stratify=y_tr,
    )
    return np.asarray(val_idx, dtype=int)


def _train_loop(model, data, optimizer, mask, is_hetero: bool):
    model.train()
    optimizer.zero_grad()
    if is_hetero:
        out = model(data.x_dict, data.edge_index_dict)   # (n_recipes, C)
        loss = F.cross_entropy(out, data['recipe'].y)
    else:
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def _eval(model, data, train_mask, test_mask, le, is_hetero: bool):
    model.eval()
    if is_hetero:
        out = model(data.x_dict, data.edge_index_dict)
        y_true = data['recipe'].y.cpu().numpy()
        proba = torch.softmax(out, dim=1).cpu().numpy()
        pred = proba.argmax(axis=1)
        y_tr, pred_tr = y_true[train_mask], pred[train_mask]
        y_te, pred_te = y_true[test_mask], pred[test_mask]
        proba_te = proba[test_mask]
    else:
        out = model(data.x, data.edge_index)
        y_true = data.y.cpu().numpy()
        proba = torch.softmax(out, dim=1).cpu().numpy()
        pred = proba.argmax(axis=1)
        y_tr, pred_tr = y_true[train_mask], pred[train_mask]
        y_te, pred_te = y_true[test_mask], pred[test_mask]
        proba_te = proba[test_mask]
    return y_tr, pred_tr, y_te, pred_te, proba_te


def _make_result(y_te, pred_te, proba_te, le):
    acc = accuracy_score(y_te, pred_te)
    f1 = f1_score(y_te, pred_te, average='macro', zero_division=0)
    n_classes = proba_te.shape[1]
    if n_classes == 2:
        auc = roc_auc_score(y_te, proba_te[:, 1])
    else:
        try:
            auc = roc_auc_score(y_te, proba_te, multi_class='ovr', average='macro')
        except Exception:
            auc = float('nan')
    # target_names только если все классы присутствуют в y_te
    unique_in_te = np.unique(y_te)
    tnames = (
        le.classes_[unique_in_te]
        if hasattr(le, 'classes_') and len(unique_in_te) == len(le.classes_)
        else None
    )
    report = classification_report(
        y_te, pred_te,
        target_names=tnames,
        zero_division=0,
    )
    return SimpleNamespace(accuracy=acc, f1=f1, auc=auc, report=report,
                           y_true=y_te, y_pred=pred_te, proba=proba_te)


def train_graphsage_cuisine(
    G_nx: nx.Graph,
    recipe_ids: list[str],
    labels: list[str],
    I_nx: nx.Graph,
    feat_df,
    hidden: int = 128,
    epochs: int = 80,
    lr: float = 3e-3,
    test_ratio: float = 0.2,
    train_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> SimpleNamespace:
    """
    GraphSAGE baseline (однородный граф) для классификации кухни.

    Transductive: train и test в одном графе, различаются только маской.
    Для сравнения с HeteroGAT передайте одинаковые ``train_idx`` / ``test_idx``.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    n_classes = len(le.classes_)

    data = build_homogeneous_data(G_nx, recipe_ids, I_nx, feat_df, y_enc)

    n_rec = data.n_recipes
    tr_idx, te_idx = stratified_recipe_split(
        n_rec, y_enc, test_ratio, seed, train_idx, test_idx,
    )
    val_idx = _monitor_val_indices(tr_idx, y_enc, seed)

    train_mask = torch.zeros(len(data.x), dtype=torch.bool)
    train_mask[tr_idx] = True

    model = GraphSAGECuisine(data.x.shape[1], hidden, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for ep in range(1, epochs + 1):
        loss = _train_loop(model, data, opt, train_mask, is_hetero=False)
        if verbose and ep % 20 == 0:
            _, _, y_val, pred_val, _ = _eval(
                model, data, tr_idx, val_idx, le, is_hetero=False)
            acc = accuracy_score(y_val, pred_val)
            print(f'  [GraphSAGE] epoch {ep:3d}  loss={loss:.4f}  val_acc={acc:.3f}')

    _, _, y_te, pred_te, proba_te = _eval(
        model, data, tr_idx, te_idx, le, is_hetero=False)

    model.eval()
    with torch.no_grad():
        recipe_emb = model.embed(data.x, data.edge_index)[:data.n_recipes].cpu().numpy()

    res = _make_result(y_te, pred_te, proba_te, le)
    res.model = model
    res.data = data
    res.le = le
    res.train_idx = tr_idx
    res.test_idx = te_idx
    res.recipe_emb = recipe_emb
    res.y_all = data.y[:data.n_recipes].cpu().numpy()
    return res


def train_hetgat_cuisine(
    G_nx: nx.Graph,
    recipe_ids: list[str],
    labels: list[str],
    I_nx: nx.Graph,
    feat_df,
    hidden: int = 128,
    heads: int = 4,
    epochs: int = 80,
    lr: float = 3e-3,
    test_ratio: float = 0.2,
    train_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> SimpleNamespace:
    """
    HeteroGAT для классификации кухни (или задачи real vs fake при binary labels).

    Transductive: test-рецепты в общем графе участвуют в message passing.
    Для nb04 / §3.8.1 передайте одинаковые ``train_idx`` и ``test_idx``.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    le = LabelEncoder()
    y_enc = le.fit_transform([str(l) for l in labels])
    n_classes = len(le.classes_)

    data = build_hetero_data(G_nx, recipe_ids, I_nx, feat_df, y_enc)

    n_rec = len(recipe_ids)
    tr_idx, te_idx = stratified_recipe_split(
        n_rec, y_enc, test_ratio, seed, train_idx, test_idx,
    )
    val_idx = _monitor_val_indices(tr_idx, y_enc, seed)

    in_ch = data['recipe'].x.shape[1]
    model = HeteroGATCuisine(in_ch, hidden, n_classes, heads=heads)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out[tr_idx], data['recipe'].y[tr_idx])
        loss.backward()
        opt.step()

        if verbose and ep % 20 == 0:
            model.eval()
            with torch.no_grad():
                out_ev = model(data.x_dict, data.edge_index_dict)
            pred_val = out_ev[val_idx].argmax(dim=1).cpu().numpy()
            y_val = data['recipe'].y[val_idx].cpu().numpy()
            acc = accuracy_score(y_val, pred_val)
            print(f'  [HeteroGAT] epoch {ep:3d}  loss={loss:.4f}  val_acc={acc:.3f}')

    model.eval()
    with torch.no_grad():
        out_final = model(data.x_dict, data.edge_index_dict)
        recipe_emb = model.embed(data.x_dict, data.edge_index_dict)['recipe'].cpu().numpy()
    proba_te = torch.softmax(out_final[te_idx], dim=1).cpu().numpy()
    pred_te  = proba_te.argmax(axis=1)
    y_te     = data['recipe'].y[te_idx].cpu().numpy()

    res = _make_result(y_te, pred_te, proba_te, le)
    res.model = model
    res.data = data
    res.le = le
    res.ing_names = sorted(I_nx.nodes())
    res.recipe_ids = recipe_ids
    res.train_idx = tr_idx
    res.test_idx = te_idx
    res.recipe_emb = recipe_emb
    res.y_all = data['recipe'].y.cpu().numpy()
    return res


def train_hetero_sage_cuisine(
    G_nx: nx.Graph,
    recipe_ids: list[str],
    labels: list[str],
    I_nx: nx.Graph,
    feat_df,
    hidden: int = 128,
    epochs: int = 80,
    lr: float = 3e-3,
    test_ratio: float = 0.2,
    train_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> SimpleNamespace:
    """
    HeteroSAGE на том же гетерографе, что HeteroGAT — ablation «гетерограф vs attention».

    Transductive; передайте те же ``train_idx`` / ``test_idx``,
    что и для homo-GraphSAGE / HeteroGAT.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    le = LabelEncoder()
    y_enc = le.fit_transform([str(l) for l in labels])
    n_classes = len(le.classes_)

    data = build_hetero_data(G_nx, recipe_ids, I_nx, feat_df, y_enc)
    n_rec = len(recipe_ids)
    tr_idx, te_idx = stratified_recipe_split(
        n_rec, y_enc, test_ratio, seed, train_idx, test_idx,
    )
    val_idx = _monitor_val_indices(tr_idx, y_enc, seed)

    in_ch = data['recipe'].x.shape[1]
    model = HeteroSAGECuisine(in_ch, hidden, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out[tr_idx], data['recipe'].y[tr_idx])
        loss.backward()
        opt.step()

        if verbose and ep % 20 == 0:
            model.eval()
            with torch.no_grad():
                out_ev = model(data.x_dict, data.edge_index_dict)
            pred_val = out_ev[val_idx].argmax(dim=1).cpu().numpy()
            y_val = data['recipe'].y[val_idx].cpu().numpy()
            acc = accuracy_score(y_val, pred_val)
            print(f'  [HeteroSAGE] epoch {ep:3d}  loss={loss:.4f}  val_acc={acc:.3f}')

    model.eval()
    with torch.no_grad():
        out_final = model(data.x_dict, data.edge_index_dict)
        recipe_emb = model.embed(data.x_dict, data.edge_index_dict)['recipe'].cpu().numpy()
    proba_te = torch.softmax(out_final[te_idx], dim=1).cpu().numpy()
    pred_te = proba_te.argmax(axis=1)
    y_te = data['recipe'].y[te_idx].cpu().numpy()

    res = _make_result(y_te, pred_te, proba_te, le)
    res.model = model
    res.data = data
    res.le = le
    res.ing_names = sorted(I_nx.nodes())
    res.recipe_ids = recipe_ids
    res.train_idx = tr_idx
    res.test_idx = te_idx
    res.recipe_emb = recipe_emb
    res.y_all = data['recipe'].y.cpu().numpy()
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ЗАДАЧИ 3 (real vs fake через HeteroGAT)
# ─────────────────────────────────────────────────────────────────────────────

def build_bipartite_from_recipes(
    recipes: list[list[str]],
    fake: list[list[str]],
    I_nx: nx.Graph,
) -> tuple[nx.Graph, list[str], list[str]]:
    """
    Строит двудольный граф G из списков рецептов + фейков.
    Возвращает: G_nx, recipe_ids, labels ('real'/'fake')
    """
    G = nx.Graph()
    recipe_ids = []
    labels = []

    for i, ings in enumerate(recipes):
        rid = f'recipe::{i}'
        G.add_node(rid, ntype='recipe')
        for ing in ings:
            if ing in I_nx:
                G.add_node(ing, ntype='ingredient')
                G.add_edge(rid, ing)
        recipe_ids.append(rid)
        labels.append('real')

    offset = len(recipes)
    for i, ings in enumerate(fake):
        rid = f'recipe::{i + offset}'
        G.add_node(rid, ntype='recipe')
        for ing in ings:
            if ing in I_nx:
                G.add_node(ing, ntype='ingredient')
                G.add_edge(rid, ing)
        recipe_ids.append(rid)
        labels.append('fake')

    return G, recipe_ids, labels
