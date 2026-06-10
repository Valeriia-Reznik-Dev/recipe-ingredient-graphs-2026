"""
gnn_multitask.py
================
Мультитаск: validity (real vs fake) + link (пропущенный ингредиент).

Общий HeteroRecipeSAGE-энкодер (как §3.8.1 / nb05):
  * validity — BCE по меткам рецептов;
  * link     — BCE по рёбрам рецепт↔ингредиент (decode-MLP).

Validity: train_auc/train_acc на train-рецептах; test_AUC — если переданы test_recipes/test_labels.
Link Hit@k — per-recipe ranking на val-рёбрах (50 негативов, nb05).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn_graph_classification import recipes_to_hetero
from gnn_link_prediction import (
    HeteroRecipeSAGE,
    _build_hard_negatives,
    _edge_set,
    _hits_at_k_per_edge,
    _hits_at_k_hard_per_edge,
    _sample_negatives,
    _split_edges,
)


class MultiTaskHeteroGNN(nn.Module):
    """HeteroRecipeSAGE + validity head; link через encoder.decode."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.encoder = HeteroRecipeSAGE(hidden=hidden)
        self.val_head = nn.Linear(hidden, 1)
        self.hidden = hidden

    def encode(self, x_dict, edge_index_dict):
        return self.encoder.encode(x_dict, edge_index_dict)

    def validity_logits(self, x_dict, edge_index_dict):
        z = self.encode(x_dict, edge_index_dict)
        return self.val_head(z["recipe"]).view(-1)

    def decode_link(self, z_dict, r, i):
        return self.encoder.decode(z_dict, r, i)


def _eval_validity_on_recipes(model, recipes, labels, I_train, device):
    from sklearn.metrics import accuracy_score, roc_auc_score

    if not recipes or len(np.unique(labels)) < 2:
        return {}
    data = recipes_to_hetero(recipes, labels, I_train).to(device)
    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(
            model.validity_logits(data.x_dict, data.edge_index_dict)
        ).cpu().numpy()
    y = np.asarray(labels)
    return {
        "test_auc": float(roc_auc_score(y, proba)),
        "test_acc": float(accuracy_score(y, (proba >= 0.5).astype(int))),
        "n_test_recipes": int(len(y)),
    }


def train_multitask(
    real_recipes,
    ingredients,
    *,
    fake_recipes=None,
    hard_neg_fn=None,
    ing_set=None,
    I_train=None,
    hidden=64,
    epochs=60,
    lr=1e-3,
    lambda_link=1.0,
    lambda_val=1.0,
    task="both",
    hide_frac=0.15,
    val_frac=0.2,
    link_eval_all=True,
    test_real_recipes=None,
    test_fake_recipes=None,
    inductive=True,
    n_pos_per_epoch=2048,
    max_ppr_recipes=1500,
    k_eval=(1, 3, 5),
    seed=42,
    device=None,
    verbose=False,
):
    """Обучает мультитаск и возвращает метрики.

    I_train — train-only граф совместимости (обязателен).
    test_AUC — на test_real_recipes / test_fake_recipes (протокол §3.8.1).
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    if I_train is None:
        raise ValueError("Передайте I_train — граф из train-рецептов (§3.3)")

    fake_recipes = fake_recipes or []
    rec_all = list(real_recipes) + list(fake_recipes)
    y_all = np.r_[np.ones(len(real_recipes)), np.zeros(len(fake_recipes))]
    n_real = len(real_recipes)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    data = recipes_to_hetero(rec_all, y_all, I_train)
    edge_index = data["recipe", "contains", "ingredient"].edge_index

    real_edge_mask = edge_index[0] < n_real
    real_ei = edge_index[:, real_edge_mask]
    if real_ei.size(1) < 4 and task in ("both", "link"):
        raise ValueError("Мало рёбер real-рецептов для link-задачи")

    link_val_frac = max(val_frac, 0.15) if link_eval_all else hide_frac
    train_ei, val_ei = _split_edges(real_ei, val_ratio=link_val_frac, seed=seed)

    model = MultiTaskHeteroGNN(hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    true_pairs = _edge_set(edge_index)
    n_neg = min(max(train_ei.size(1), 64), n_pos_per_epoch)
    forbidden_train = true_pairs.copy()
    if hard_neg_fn is not None and task in ("both", "link"):
        train_neg = _build_hard_negatives(
            train_ei, data, hard_neg_fn, 0.5, forbidden_train, rng,
            min(max_ppr_recipes, n_real),
        )
    else:
        train_neg = _sample_negatives(
            data["recipe"].num_nodes, data["ingredient"].num_nodes,
            n_neg, forbidden_train, rng,
        )

    fake_ei = edge_index[:, ~real_edge_mask]
    if inductive:
        if fake_ei.size(1) > 0:
            fake_train_ei, _ = _split_edges(fake_ei, val_ratio=link_val_frac, seed=seed + 1)
            mp_ei = torch.cat([train_ei, fake_train_ei], dim=1)
        else:
            mp_ei = train_ei
    else:
        mp_ei = edge_index

    train_data = data.clone()
    train_data["recipe", "contains", "ingredient"].edge_index = mp_ei
    train_data["ingredient", "rev_contains", "recipe"].edge_index = mp_ei.flip(0)
    train_data = train_data.to(device)

    do_link = task in ("both", "link") and train_ei.size(1) > 0
    do_val = task in ("both", "validity")

    pos_t = train_ei.to(device)
    neg_t = train_neg.to(device)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        z = model.encode(train_data.x_dict, train_data.edge_index_dict)
        loss = z["recipe"].new_zeros(())

        if do_link:
            pos = model.decode_link(z, pos_t[0], pos_t[1])
            neg = model.decode_link(z, neg_t[0], neg_t[1])
            link_loss = F.binary_cross_entropy_with_logits(
                torch.cat([pos, neg]),
                torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]),
            )
            loss = loss + lambda_link * link_loss

        if do_val:
            logit = model.val_head(z["recipe"]).view(-1)
            loss = loss + lambda_val * F.binary_cross_entropy_with_logits(
                logit, train_data["recipe"].y,
            )

        loss.backward()
        opt.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:>3}  loss={loss.item():.4f}")

    out = {
        "task": task,
        "inductive": bool(inductive),
        "hidden": hidden,
    }

    model.eval()
    with torch.no_grad():
        z = model.encode(train_data.x_dict, train_data.edge_index_dict)
        if do_val:
            proba = torch.sigmoid(model.val_head(z["recipe"]).view(-1)).cpu().numpy()
            if len(np.unique(y_all)) > 1:
                out["train_auc"] = float(roc_auc_score(y_all, proba))
            out["train_acc"] = float(accuracy_score(y_all, (proba >= 0.5).astype(int)))
            out["n_train_recipes"] = int(len(y_all))

        if task in ("both", "link") and val_ei.size(1) > 0:
            hits = _hits_at_k_per_edge(
                model.encoder, z, val_ei, true_pairs,
                data["ingredient"].num_nodes, device, rng, ks=k_eval, num_neg=50,
            )
            out.update({f"link_hit@{k}": v for k, v in hits.items()})
            if hard_neg_fn is not None:
                hits_hard = _hits_at_k_hard_per_edge(
                    model.encoder, z, val_ei, train_ei, true_pairs, data,
                    hard_neg_fn, device, rng, ks=k_eval, num_neg=50,
                )
                out.update({f"link_hit_hard@{k}": v for k, v in hits_hard.items()})
            out["n_held"] = int(val_ei.size(1))

    if test_real_recipes is not None and test_fake_recipes is not None and do_val:
        rec_te = list(test_real_recipes) + list(test_fake_recipes)
        y_te = np.r_[np.ones(len(test_real_recipes)), np.zeros(len(test_fake_recipes))]
        out.update(_eval_validity_on_recipes(model, rec_te, y_te, I_train, device))

    return out
