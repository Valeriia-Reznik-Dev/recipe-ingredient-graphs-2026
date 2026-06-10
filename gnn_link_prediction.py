"""GraphSAGE link prediction на двудольном графе рецепт–ингредиент."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


@dataclass
class LinkPredResult:
    auc: float
    hits_at_k: dict[int, float]
    model: nn.Module
    data: HeteroData
    train_data: HeteroData
    examples: list[dict]
    hits_at_k_hard: dict[int, float] | None = None
    val_ei: torch.Tensor | None = None
    train_ei: torch.Tensor | None = None
    true_pairs: set[tuple[int, int]] | None = None
    seed: int = 42


class HeteroRecipeSAGE(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.conv1 = HeteroConv({
            ('recipe', 'contains', 'ingredient'): SAGEConv((-1, -1), hidden),
            ('ingredient', 'rev_contains', 'recipe'): SAGEConv((-1, -1), hidden),
        }, aggr='sum')
        self.conv2 = HeteroConv({
            ('recipe', 'contains', 'ingredient'): SAGEConv((hidden, hidden), hidden),
            ('ingredient', 'rev_contains', 'recipe'): SAGEConv((hidden, hidden), hidden),
        }, aggr='sum')
        self.link_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return x_dict

    def decode(self, z_dict, src, dst):
        return self.link_mlp(torch.cat([z_dict['recipe'][src], z_dict['ingredient'][dst]], dim=-1)).view(-1)


_FEAT_COLS = ['degree', 'degree_w', 'pagerank', 'clustering', 'kcore', 'betweenness', 'log_freq']


def _parse_bipartite_nx(G: nx.Graph) -> tuple[list, list, list[tuple[int, int]]]:
    """Узлы и рёбра двудольного графа как (recipe_idx, ingredient_idx)."""
    recipes, ingredients = [], []
    for n, d in G.nodes(data=True):
        if d.get('ntype') == 'recipe' or str(n).startswith('recipe::'):
            recipes.append(n)
        else:
            ingredients.append(n)

    r_idx = {n: i for i, n in enumerate(recipes)}
    i_idx = {n: i for i, n in enumerate(ingredients)}

    edges: list[tuple[int, int]] = []
    for u, v in G.edges():
        if u in r_idx and v in i_idx:
            edges.append((r_idx[u], i_idx[v]))
        elif v in r_idx and u in i_idx:
            edges.append((r_idx[v], i_idx[u]))
    return recipes, ingredients, edges


def _link_node_features(
    recipes: list,
    ingredients: list,
    edges: list[tuple[int, int]],
    feat_df,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Признаки узлов вместо all-ones (иначе эмбеддинги схлопываются → AUC ≈ 0.5):
      ingredient = log-степень в train-подграфе [+ структурные признаки из feat_df]
      recipe     = среднее признаков видимых (train) ингредиентов

    ``edges`` — только train-рёбра, чтобы спрятанные val-связи не утекали в x.
    """
    ing_deg = np.zeros(len(ingredients), dtype=np.float32)
    for _, i in edges:
        ing_deg[i] += 1.0
    deg_feat = np.array([[np.log1p(d)] for d in ing_deg], dtype=np.float32)

    if feat_df is not None:
        cols = [c for c in _FEAT_COLS if c in feat_df.columns]
        raw = feat_df.reindex(ingredients)[cols].fillna(0.0).values.astype(np.float32)
        mu, sd = raw.mean(0), raw.std(0) + 1e-9
        ing_feat = np.hstack([deg_feat, (raw - mu) / sd]).astype(np.float32)
    else:
        ing_feat = deg_feat

    n_feat = ing_feat.shape[1]
    rec_feat = np.zeros((len(recipes), n_feat), dtype=np.float32)
    counts = np.zeros(len(recipes), dtype=np.float32)
    for r, i in edges:
        rec_feat[r] += ing_feat[i]
        counts[r] += 1.0
    nz = counts > 0
    rec_feat[nz] /= counts[nz, None]
    return torch.tensor(ing_feat), torch.tensor(rec_feat)


def bipartite_nx_to_hetero(
    G: nx.Graph,
    feat_df=None,
    *,
    feature_edges: list[tuple[int, int]] | None = None,
) -> HeteroData:
    """HeteroData с полным edge_index; признаки — только по ``feature_edges`` (по умолчанию train=all)."""
    recipes, ingredients, edges = _parse_bipartite_nx(G)
    feat_src = feature_edges if feature_edges is not None else edges

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    rev_edge_index = edge_index.flip(0)

    ing_x, rec_x = _link_node_features(recipes, ingredients, feat_src, feat_df)

    data = HeteroData()
    data['recipe'].num_nodes = len(recipes)
    data['ingredient'].num_nodes = len(ingredients)
    data['recipe'].x = rec_x
    data['ingredient'].x = ing_x
    data['recipe', 'contains', 'ingredient'].edge_index = edge_index
    data['ingredient', 'rev_contains', 'recipe'].edge_index = rev_edge_index
    data.recipe_id = recipes
    data.ingredient_id = ingredients
    return data


def sample_bipartite_subgraph(G: nx.Graph, max_recipes: int = 3000, seed: int = 42) -> nx.Graph:
    """Подграф для быстрого обучения GraphSAGE."""
    rng = np.random.default_rng(seed)
    recipes = [n for n, d in G.nodes(data=True) if d.get('ntype') == 'recipe' or str(n).startswith('recipe::')]
    if len(recipes) <= max_recipes:
        return G
    pick = set(rng.choice(recipes, size=max_recipes, replace=False))
    nodes = set(pick)
    for r in pick:
        nodes.update(G.neighbors(r))
    return G.subgraph(nodes).copy()


def _split_edges(edge_index: torch.Tensor, val_ratio: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = edge_index.size(1)
    perm = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return edge_index[:, train_idx], edge_index[:, val_idx]


def _edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return set(map(tuple, edge_index.t().cpu().tolist()))


def _sample_ranking_negatives(
    r: int,
    i_pos: int,
    num_ings: int,
    true_pairs: set[tuple[int, int]],
    rng: np.random.Generator,
    num_neg: int,
) -> list[int]:
    """Filtered negatives: исключаем все истинные рёбра рецепта r."""
    negs: list[int] = []
    while len(negs) < num_neg:
        j = int(rng.integers(num_ings))
        if j != i_pos and (r, j) not in true_pairs:
            negs.append(j)
    return negs


def _hits_at_k_per_edge(
    model: HeteroRecipeSAGE,
    z_dict: dict,
    val_ei: torch.Tensor,
    true_pairs: set[tuple[int, int]],
    num_ings: int,
    device: str,
    rng: np.random.Generator,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    num_neg: int = 50,
) -> dict[int, float]:
    """Hit@K: для каждого val-ребра ранжируем true среди num_neg случайных негативов (filtered)."""
    tallies = {k: [] for k in ks}
    val_pairs = val_ei.t().cpu().tolist()

    for r, i_pos in val_pairs:
        r, i_pos = int(r), int(i_pos)
        negs = _sample_ranking_negatives(r, i_pos, num_ings, true_pairs, rng, num_neg)
        cand = [i_pos] + negs
        r_t = torch.full((len(cand),), r, dtype=torch.long, device=device)
        i_t = torch.tensor(cand, dtype=torch.long, device=device)
        with torch.no_grad():
            scores = torch.sigmoid(model.decode(z_dict, r_t, i_t)).cpu().numpy()
        perm = rng.permutation(len(cand))
        cand_p = np.array(cand)[perm]
        order = np.argsort(-scores[perm])
        rank = int(np.where(cand_p[order] == i_pos)[0][0]) + 1
        for k in ks:
            tallies[k].append(float(rank <= k))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in tallies.items()}


def _hits_at_k_hard_per_edge(
    model: HeteroRecipeSAGE,
    z_dict: dict,
    val_ei: torch.Tensor,
    train_ei: torch.Tensor,
    true_pairs: set[tuple[int, int]],
    data: HeteroData,
    hard_neg_fn,
    device: str,
    rng: np.random.Generator,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    num_neg: int = 50,
) -> dict[int, float]:
    """Hit@K с hard-негативами из PPR-окрестности контекста рецепта."""
    ing_names = list(data.ingredient_id)
    name_to_idx = {n: k for k, n in enumerate(ing_names)}
    ing_set = set(ing_names)
    train_pairs = _edge_set(train_ei)
    tallies = {k: [] for k in ks}

    for r, i_pos in val_ei.t().cpu().tolist():
        r, i_pos = int(r), int(i_pos)
        context = [ing_names[i] for rr, i in train_pairs if int(rr) == r]
        hard_names = hard_neg_fn(context, ing_set, num_neg * 3)
        negs: list[int] = []
        for name in hard_names:
            if name not in name_to_idx:
                continue
            j = name_to_idx[name]
            if j != i_pos and (r, j) not in true_pairs:
                negs.append(j)
            if len(negs) >= num_neg:
                break
        while len(negs) < num_neg:
            j = int(rng.integers(len(ing_names)))
            if j != i_pos and (r, j) not in true_pairs and j not in negs:
                negs.append(j)

        cand = [i_pos] + negs[:num_neg]
        r_t = torch.full((len(cand),), r, dtype=torch.long, device=device)
        i_t = torch.tensor(cand, dtype=torch.long, device=device)
        with torch.no_grad():
            scores = torch.sigmoid(model.decode(z_dict, r_t, i_t)).cpu().numpy()
        perm = rng.permutation(len(cand))
        cand_p = np.array(cand)[perm]
        order = np.argsort(-scores[perm])
        rank = int(np.where(cand_p[order] == i_pos)[0][0]) + 1
        for k in ks:
            tallies[k].append(float(rank <= k))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in tallies.items()}


def _cn_score_one(cand: str, context: list[str], neighbors: dict[str, set]) -> float:
    ctx = set(context)
    if cand in ctx:
        return 0.0
    return float(len(neighbors.get(cand, set()) & ctx))


def _ppr_scores_for_candidates(
    context: list[str],
    candidates: list[str],
    graph: nx.Graph,
) -> dict[str, float]:
    ctx = [x for x in context if x in graph]
    if not ctx:
        return {c: 0.0 for c in candidates}
    w = 1.0 / len(ctx)
    personalization = {c: w for c in ctx}
    pr = nx.pagerank(graph, alpha=0.85, personalization=personalization, weight='weight')
    return {c: pr.get(c, 0.0) for c in candidates}


def eval_heuristic_hits_at_k(
    result: LinkPredResult,
    I_nx: nx.Graph,
    score_mode: str = 'cn',
    ks: tuple[int, ...] = (1, 3, 5, 10),
    num_neg: int = 50,
) -> dict[int, float]:
    """Hit@K для CN/PPR: val-рёбра, 1 true + num_neg random (filtered)."""
    if result.val_ei is None or result.train_ei is None or result.true_pairs is None:
        raise ValueError('LinkPredResult must include val_ei, train_ei, true_pairs')

    rng = np.random.default_rng(result.seed)
    ing_names = list(result.data.ingredient_id)
    ing_set = set(ing_names)
    idx_to_name = {k: n for k, n in enumerate(ing_names)}
    neighbors = {n: set(I_nx.neighbors(n)) for n in ing_set if n in I_nx}
    train_pairs = _edge_set(result.train_ei)
    true_pairs = result.true_pairs
    tallies = {k: [] for k in ks}

    for r, i_pos in result.val_ei.t().cpu().tolist():
        r, i_pos = int(r), int(i_pos)
        context = [idx_to_name[i] for rr, i in train_pairs if int(rr) == r]
        negs = _sample_ranking_negatives(r, i_pos, len(ing_names), true_pairs, rng, num_neg)
        cand_idxs = [i_pos] + negs
        cand_names = [idx_to_name[j] for j in cand_idxs]

        if score_mode == 'cn':
            scores = np.array([_cn_score_one(n, context, neighbors) for n in cand_names])
        elif score_mode == 'ppr':
            pr_map = _ppr_scores_for_candidates(context, cand_names, I_nx)
            scores = np.array([pr_map[n] for n in cand_names])
        else:
            raise ValueError(f'unknown score_mode: {score_mode!r}')

        perm = rng.permutation(len(cand_idxs))
        cand_p = np.array(cand_idxs)[perm]
        order = np.argsort(-scores[perm])
        rank = int(np.where(cand_p[order] == i_pos)[0][0]) + 1
        for k in ks:
            tallies[k].append(float(rank <= k))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in tallies.items()}


def _build_substitution_examples(
    model: HeteroRecipeSAGE,
    z_dict: dict,
    val_ei: torch.Tensor,
    train_ei: torch.Tensor,
    true_pairs: set[tuple[int, int]],
    ingredient_names: list,
    recipe_names: list,
    device: str,
    rng: np.random.Generator,
    n: int = 8,
    num_neg: int = 50,
) -> list[dict]:
    """Примеры: скрытый ингредиент vs топ-5 GraphSAGE (per-recipe ranking)."""
    train_pairs = _edge_set(train_ei)
    num_ings = len(ingredient_names)
    val_pairs = val_ei.t().cpu().tolist()
    pick = rng.permutation(len(val_pairs))[: min(n, len(val_pairs))]

    examples = []
    for idx in pick:
        r, i_pos = val_pairs[int(idx)]
        r, i_pos = int(r), int(i_pos)
        context = sorted({
            int(i)
            for rr, i in train_pairs
            if int(rr) == r and int(i) != i_pos
        })

        negs = _sample_ranking_negatives(r, i_pos, num_ings, true_pairs, rng, num_neg)
        cand = [i_pos] + negs
        r_t = torch.full((len(cand),), r, dtype=torch.long, device=device)
        i_t = torch.tensor(cand, dtype=torch.long, device=device)
        with torch.no_grad():
            scores = torch.sigmoid(model.decode(z_dict, r_t, i_t)).cpu().numpy()
        perm = rng.permutation(len(cand))
        cand_p = np.array(cand)[perm]
        scores_p = scores[perm]
        order = np.argsort(-scores_p)
        ranked = [(int(cand_p[i]), float(scores_p[i])) for i in order[:5]]
        rank_hidden = int(np.where(cand_p[order] == i_pos)[0][0]) + 1

        examples.append({
            'recipe': recipe_names[r],
            'hidden': ingredient_names[i_pos],
            'context': [ingredient_names[i] for i in context[:8]],
            'top5': [(ingredient_names[i], sc) for i, sc in ranked],
            'rank_hidden': rank_hidden,
        })
    return examples


def _sample_negatives(
    num_recipes: int,
    num_ings: int,
    n: int,
    forbidden: set[tuple[int, int]],
    rng: np.random.Generator,
) -> torch.Tensor:
    max_possible = num_recipes * num_ings - len(forbidden)
    n = min(n, max_possible)
    negs = []
    attempts = 0
    limit = max(n * 200, 1)
    while len(negs) < n and attempts < limit:
        r = int(rng.integers(num_recipes))
        i = int(rng.integers(num_ings))
        attempts += 1
        if (r, i) not in forbidden:
            negs.append([r, i])
            forbidden.add((r, i))
    if not negs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(negs, dtype=torch.long).t()


def _build_hard_negatives(
    train_ei: torch.Tensor,
    data: HeteroData,
    hard_neg_fn,
    ratio: float,
    forbidden: set[tuple[int, int]],
    rng: np.random.Generator,
    max_hard_recipes: int,
) -> torch.Tensor:
    """
    Строит негативы, выровненные по позитивным рёбрам: для доли `ratio`
    позитивов (recipe r, ing i_pos) берём hard-негатив из hard_neg_fn(context_r),
    остальные — случайные. hard_neg_fn(context, ing_set, n) -> список имён.
    """
    ing_names = list(data.ingredient_id)
    name_to_idx = {n: k for k, n in enumerate(ing_names)}
    ing_set = set(ing_names)
    num_ings = len(ing_names)

    pairs = train_ei.t().tolist()
    ctx: dict[int, list[str]] = defaultdict(list)
    for r, i in pairs:
        ctx[r].append(ing_names[i])

    hard_by_recipe: dict[int, list[int]] = {}
    for r in list(ctx)[:max_hard_recipes]:
        names = hard_neg_fn(ctx[r], ing_set, 10)
        idxs = [
            name_to_idx[n] for n in names
            if n in name_to_idx and (r, name_to_idx[n]) not in forbidden
        ]
        if idxs:
            hard_by_recipe[r] = idxs

    neg = []
    for r, i_pos in pairs:
        if r in hard_by_recipe and rng.random() < ratio:
            j = int(rng.choice(hard_by_recipe[r]))
        else:
            while True:
                j = int(rng.integers(num_ings))
                if j != i_pos and (r, j) not in forbidden:
                    break
        neg.append([r, j])
        forbidden.add((r, j))
    return torch.tensor(neg, dtype=torch.long).t()


def train_graphsage_link_prediction(
    G: nx.Graph,
    hidden: int = 64,
    epochs: int = 25,
    lr: float = 1e-3,
    val_ratio: float = 0.15,
    max_recipes: int = 3000,
    seed: int = 42,
    device: str | None = None,
    hard_neg_fn=None,
    hard_neg_ratio: float = 0.5,
    max_hard_recipes: int = 300,
    feat_df=None,
) -> LinkPredResult:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    G = sample_bipartite_subgraph(G, max_recipes=max_recipes, seed=seed)
    recipes, ingredients, all_edges = _parse_bipartite_nx(G)
    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    train_ei, val_ei = _split_edges(edge_index, val_ratio=val_ratio, seed=seed)

    train_edge_list = train_ei.t().cpu().tolist()
    data = bipartite_nx_to_hetero(G, feat_df=feat_df, feature_edges=train_edge_list)

    true_pairs = _edge_set(edge_index)
    val_neg = _sample_negatives(
        data['recipe'].num_nodes, data['ingredient'].num_nodes,
        val_ei.size(1), true_pairs.copy(), rng,
    )
    forbidden_train = true_pairs | _edge_set(val_neg)

    if hard_neg_fn is not None:
        train_neg = _build_hard_negatives(
            train_ei, data, hard_neg_fn, hard_neg_ratio,
            forbidden_train.copy(), rng, max_hard_recipes,
        )
    else:
        train_neg = _sample_negatives(
            data['recipe'].num_nodes, data['ingredient'].num_nodes,
            train_ei.size(1), forbidden_train.copy(), rng,
        )

    train_data = HeteroData()
    train_data['recipe'].num_nodes = data['recipe'].num_nodes
    train_data['ingredient'].num_nodes = data['ingredient'].num_nodes
    train_data['recipe'].x = data['recipe'].x
    train_data['ingredient'].x = data['ingredient'].x
    train_data['recipe', 'contains', 'ingredient'].edge_index = train_ei
    train_data['ingredient', 'rev_contains', 'recipe'].edge_index = train_ei.flip(0)

    model = HeteroRecipeSAGE(hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_data = train_data.to(device)

    def batch_loss(pos_ei, neg_ei):
        z = model.encode(train_data.x_dict, train_data.edge_index_dict)
        pos = model.decode(z, pos_ei[0], pos_ei[1])
        neg = model.decode(z, neg_ei[0], neg_ei[1])
        y = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        logits = torch.cat([pos, neg])
        return F.binary_cross_entropy_with_logits(logits, y)

    pos_t = train_ei.to(device)
    neg_t = train_neg.to(device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = batch_loss(pos_t, neg_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(train_data.x_dict, train_data.edge_index_dict)
        pos_scores = torch.sigmoid(model.decode(z, val_ei[0].to(device), val_ei[1].to(device))).cpu().numpy()
        neg_scores = torch.sigmoid(model.decode(z, val_neg[0].to(device), val_neg[1].to(device))).cpu().numpy()

    y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    y_score = np.concatenate([pos_scores, neg_scores])
    auc = float(roc_auc_score(y_true, y_score))

    hits = _hits_at_k_per_edge(
        model, z, val_ei, true_pairs, data['ingredient'].num_nodes,
        device, rng,
    )
    hits_hard = None
    if hard_neg_fn is not None:
        hits_hard = _hits_at_k_hard_per_edge(
            model, z, val_ei, train_ei, true_pairs, data, hard_neg_fn,
            device, rng,
        )

    examples = _build_substitution_examples(
        model, z, val_ei, train_ei, true_pairs,
        list(data.ingredient_id), list(data.recipe_id),
        device, rng, n=8,
    )

    return LinkPredResult(
        auc=auc,
        hits_at_k=hits,
        model=model,
        data=data,
        train_data=train_data.cpu(),
        examples=examples,
        hits_at_k_hard=hits_hard,
        val_ei=val_ei.cpu(),
        train_ei=train_ei.cpu(),
        true_pairs=true_pairs,
        seed=seed,
    )
