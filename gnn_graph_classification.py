"""GraphSAGE graph classification: реальное vs нереальное блюдо."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from torch_geometric.nn import global_mean_pool

from gnn_link_prediction import HeteroRecipeSAGE


@dataclass
class ValidityClfResult:
    accuracy: float
    f1: float
    auc: float
    model: nn.Module
    data: HeteroData
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_proba: np.ndarray


class RecipeValiditySAGE(nn.Module):
    """Чтение эмбеддинга узла-рецепта (readout='node')."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.encoder = HeteroRecipeSAGE(hidden=hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_dict, edge_index_dict):
        z = self.encoder.encode(x_dict, edge_index_dict)
        return self.head(z['recipe']).view(-1)


class RecipeValidityGlobalPool(nn.Module):
    """
    Graph classification через настоящий `global_mean_pool`:
    кодируем ингредиенты GraphSAGE-энкодером, затем усредняем эмбеддинги
    ингредиентов КАЖДОГО рецепта (readout по всему «подграфу» рецепта)
    и классифицируем полученный graph-level вектор.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.encoder = HeteroRecipeSAGE(hidden=hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_dict, edge_index_dict):
        z = self.encoder.encode(x_dict, edge_index_dict)
        ei = edge_index_dict[('recipe', 'contains', 'ingredient')]
        recipe_batch = ei[0]                 # к какому рецепту относится ингредиент-вхождение
        ing_nodes = ei[1]
        ing_emb = z['ingredient'][ing_nodes]
        num_recipes = z['recipe'].size(0)
        pooled = global_mean_pool(ing_emb, recipe_batch, size=num_recipes)
        return self.head(pooled).view(-1)


def _edge_density(recipe: list[str], I: nx.Graph | None) -> float:
    if not I:
        return 0.0
    pairs = edges = 0
    for i in range(len(recipe)):
        for j in range(i + 1, len(recipe)):
            pairs += 1
            if I.has_edge(recipe[i], recipe[j]):
                edges += 1
    return edges / pairs if pairs else 0.0


def recipes_to_hetero(
    recipes: list[list[str]],
    labels: np.ndarray | None = None,
    I: nx.Graph | None = None,
) -> HeteroData:
    all_ings = sorted({ing for r in recipes for ing in r})
    ing_idx = {ing: i for i, ing in enumerate(all_ings)}

    edges = []
    for ri, ings in enumerate(recipes):
        for ing in ings:
            edges.append((ri, ing_idx[ing]))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    if I is not None:
        deg = dict(I.degree())
        ing_x = torch.tensor(
            [[float(np.log1p(deg.get(ing, 0)))] for ing in all_ings],
            dtype=torch.float,
        )
        recipe_x = torch.tensor(
            [[float(len(r))] for r in recipes],
            dtype=torch.float,
        )
    else:
        ing_x = torch.ones(len(all_ings), 1)
        recipe_x = torch.ones(len(recipes), 1)

    data = HeteroData()
    data['recipe'].num_nodes = len(recipes)
    data['ingredient'].num_nodes = len(all_ings)
    data['recipe'].x = recipe_x
    data['ingredient'].x = ing_x
    data['recipe', 'contains', 'ingredient'].edge_index = edge_index
    data['ingredient', 'rev_contains', 'recipe'].edge_index = edge_index.flip(0)
    if labels is not None:
        data['recipe'].y = torch.tensor(labels, dtype=torch.float)
    data.ingredient_id = all_ings
    return data


def make_fake_recipes(
    recipes: list[list[str]],
    ingredient_pool: list[str] | None = None,
    mix_ratio: float = 0.5,
    seed: int = 42,
) -> list[list[str]]:
    """Фейковый рецепт: часть ингредиентов заменена случайными из пула."""
    rng = np.random.default_rng(seed)
    pool = ingredient_pool or sorted({ing for r in recipes for ing in r})
    fake = []
    for r in recipes:
        n = len(r)
        n_replace = max(1, int(n * mix_ratio))
        keep_n = n - n_replace
        kept = list(rng.choice(r, size=keep_n, replace=False)) if keep_n else []
        candidates = [x for x in pool if x not in r]
        if len(candidates) >= n_replace:
            replaced = list(rng.choice(candidates, size=n_replace, replace=False))
        else:
            replaced = candidates
        fake.append([str(x) for x in kept + replaced])
    return fake


@torch.no_grad()
def counterfactual_culprits(
    model: nn.Module,
    recipe: list[str],
    I: nx.Graph,
    device: str | None = None,
):
    """
    Counterfactual-интерпретация: по очереди убираем каждый ингредиент
    и смотрим, как меняется вероятность «фейк» у обученной модели.

    Возвращает (base_fake_prob, [(ингредиент, delta), ...]) — список отсортирован
    по убыванию delta. delta > 0 означает, что удаление ингредиента СНИЖАЕТ
    fake-вероятность, т.е. этот ингредиент «виновен» в несочетаемости.

    Замечание: вариант рецепта прогоняется как маленький отдельный граф, поэтому
    эмбеддинги слегка отличаются от обучающего контекста (индуктивность GraphSAGE),
    но относительные delta между ингредиентами интерпретируемы.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ings = [x for x in recipe if x in I]
    if len(ings) < 2:
        return None, []

    variants = [ings] + [
        [x for j, x in enumerate(ings) if j != k] for k in range(len(ings))
    ]
    data = recipes_to_hetero(variants, labels=None, I=I).to(device)
    model.eval()
    logits = model(data.x_dict, data.edge_index_dict)
    fake_prob = (1.0 - torch.sigmoid(logits)).cpu().numpy()   # P(fake) = 1 - P(real)

    base = float(fake_prob[0])
    out = [(ing, base - float(fake_prob[k + 1])) for k, ing in enumerate(ings)]
    out.sort(key=lambda t: -t[1])
    return base, out


def incompatible_pairs(recipe: list[str], I: nx.Graph) -> list[tuple[str, str]]:
    """Пары ингредиентов без ребра в графе совместимости."""
    bad = []
    for i in range(len(recipe)):
        for j in range(i + 1, len(recipe)):
            a, b = recipe[i], recipe[j]
            if not I.has_edge(a, b):
                bad.append((a, b))
    return bad


def train_graphsage_validity(
    recipes: list[list[str]],
    labels: np.ndarray,
    I: nx.Graph | None = None,
    hidden: int = 64,
    epochs: int = 40,
    lr: float = 1e-3,
    test_size: float = 0.2,
    seed: int = 42,
    device: str | None = None,
    readout: str = 'node',
) -> ValidityClfResult:
    """readout='node' — эмбеддинг узла-рецепта; readout='mean' — global_mean_pool."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    idx = np.arange(len(recipes))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels,
    )

    data = recipes_to_hetero(recipes, labels, I=I)
    if readout == 'mean':
        model = RecipeValidityGlobalPool(hidden=hidden).to(device)
    else:
        model = RecipeValiditySAGE(hidden=hidden).to(device)
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_mask = torch.zeros(len(recipes), dtype=torch.bool, device=device)
    train_mask[torch.tensor(train_idx, device=device)] = True

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(data.x_dict, data.edge_index_dict)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], data['recipe'].y[train_mask])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        proba = torch.sigmoid(logits).cpu().numpy()

    y_test = labels[test_idx]
    pred = (proba[test_idx] >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred))
    auc = float(roc_auc_score(y_test, proba[test_idx]))

    return ValidityClfResult(
        accuracy=acc,
        f1=f1,
        auc=auc,
        model=model,
        data=data,
        train_idx=train_idx,
        test_idx=test_idx,
        test_proba=proba[test_idx],
    )


def eval_graphsage_inductive_split(
    rec_tr: list[list[str]],
    rec_te: list[list[str]],
    y_tr: np.ndarray,
    y_te: np.ndarray,
    feat_graph: nx.Graph | None = None,
    hidden: int = 64,
    epochs: int = 40,
    lr: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
    readout: str = 'node',
) -> ValidityClfResult:
    """Inductive GraphSAGE на заранее заданных train/test рецептах."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    y_tr = np.asarray(y_tr)
    y_te = np.asarray(y_te)

    torch.manual_seed(seed)
    np.random.seed(seed)

    if feat_graph is None:
        from recipe_negatives import build_cooc_graph
        feat_graph = build_cooc_graph(rec_tr, min_freq=1, min_cooc=1)

    data_tr = recipes_to_hetero(rec_tr, y_tr, I=feat_graph).to(device)
    data_te = recipes_to_hetero(rec_te, y_te, I=feat_graph).to(device)

    model_cls = RecipeValidityGlobalPool if readout == 'mean' else RecipeValiditySAGE
    model = model_cls(hidden=hidden).to(device)
    model(data_tr.x_dict, data_tr.edge_index_dict)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(data_tr.x_dict, data_tr.edge_index_dict)
        loss = F.binary_cross_entropy_with_logits(logits, data_tr['recipe'].y)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(data_te.x_dict, data_te.edge_index_dict)).cpu().numpy()

    pred = (proba >= 0.5).astype(int)
    return ValidityClfResult(
        accuracy=float(accuracy_score(y_te, pred)),
        f1=float(f1_score(y_te, pred)),
        auc=float(roc_auc_score(y_te, proba)),
        model=model,
        data=data_te,
        train_idx=np.arange(len(rec_tr)),
        test_idx=np.arange(len(rec_te)),
        test_proba=proba,
    )


def train_graphsage_validity_inductive(
    recipes: list[list[str]],
    labels: np.ndarray,
    feat_graph: nx.Graph | None = None,
    hidden: int = 64,
    epochs: int = 40,
    lr: float = 1e-3,
    test_size: float = 0.2,
    seed: int = 42,
    device: str | None = None,
    readout: str = 'node',
) -> ValidityClfResult:
    """Inductive-оценка real vs fake.

    train и test — разные графы: тестовые рецепты не участвуют в обучении
    (message passing только по train-графу). Признаки ингредиентов из `feat_graph`
    (обычно граф из обучающих рецептов). Модель применяется к test-графу как есть.

    Отличие от `train_graphsage_validity` (transductive): там единый граф из всех
    рецептов и маски train/test; тестовые узлы видны при свёртке.
    """
    labels = np.asarray(labels)
    idx = np.arange(len(recipes))
    tr_idx, te_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels,
    )
    rec_tr = [recipes[i] for i in tr_idx]
    rec_te = [recipes[i] for i in te_idx]
    return eval_graphsage_inductive_split(
        rec_tr, rec_te, labels[tr_idx], labels[te_idx],
        feat_graph=feat_graph, hidden=hidden, epochs=epochs, lr=lr,
        seed=seed, device=device, readout=readout,
    )
