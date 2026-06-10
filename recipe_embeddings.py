"""
recipe_embeddings.py
====================
Блок E — сравнение представлений рецепта для классификации кухонь.

Представления рецепта и их сравнение:
  * graph   — то, что уже есть (вектор сообществ Leiden / структурные признаки);
  * nlp     — TF-IDF + LSA по тексту (ингредиенты и/или directions);
  * nlp_bge — плотный эмбеддинг BGE (sentence-transformers) для сравнения с LSA;
  * process — признаки процесса из `directions` (варка/жарка/сырое, время, шаги);
  * joint   — конкатенация доступных представлений.

extract_process_features / process_matrix полностью реализованы и протестированы.
Колонка `directions` загружается в `data_utils.load_recipes_stratified`.
"""

from __future__ import annotations

import re

import numpy as np

# --------------------------------------------------------------------------- #
# Процессные признаки из directions
# --------------------------------------------------------------------------- #
METHOD_KEYWORDS = {
    "bake": ["bake", "baked", "baking", "oven"],
    "boil": ["boil", "boiled", "boiling"],
    "fry": ["fry", "fried", "frying", "deep-fry", "deep fry"],
    "saute": ["saute", "sauté", "sauteed", "pan-fry", "pan fry"],
    "roast": ["roast", "roasted", "roasting"],
    "grill": ["grill", "grilled", "grilling", "broil", "broiled"],
    "simmer": ["simmer", "simmered", "simmering"],
    "steam": ["steam", "steamed", "steaming"],
    "stir": ["stir", "stirred", "stir-fry", "stir fry"],
    "mix": ["mix", "whisk", "blend", "beat", "fold", "combine"],
    "chill": ["chill", "refrigerate", "freeze", "fridge", "cool"],
    "marinate": ["marinate", "marinated", "soak"],
}
_HEAT_METHODS = ["bake", "boil", "fry", "saute", "roast", "grill", "simmer", "steam"]
_TIME_RE = re.compile(r"(\d+)\s*(hour|hours|hr|hrs|minute|minutes|min|mins)")


def extract_process_features(directions) -> dict:
    """Признаки кулинарного процесса из текста инструкций.

    directions : str | list[str]  (шаги рецепта; RecipeNLG: список строк)
    """
    if isinstance(directions, (list, tuple)):
        steps = [str(s) for s in directions]
        text = " . ".join(steps)
    else:
        text = str(directions or "")
    low = text.lower()

    feats = {f"m_{m}": float(sum(low.count(k) for k in kws)) for m, kws in METHOD_KEYWORDS.items()}

    minutes = 0
    for num, unit in _TIME_RE.findall(low):
        n = int(num)
        minutes += n * 60 if unit.startswith(("hour", "hr")) else n
    feats["total_minutes"] = float(minutes)

    n_steps = len(directions) if isinstance(directions, (list, tuple)) else (low.count(".") + 1)
    feats["n_steps"] = float(max(1, n_steps))
    feats["text_len"] = float(len(low))

    heat = sum(feats[f"m_{m}"] for m in _HEAT_METHODS)
    feats["has_heat"] = 1.0 if heat > 0 else 0.0
    feats["no_cook"] = 1.0 - feats["has_heat"]
    return feats


def process_matrix(directions_series):
    """Матрица процессных признаков [n_recipes × F]. Возвращает (X, cols)."""
    rows = [extract_process_features(d) for d in directions_series]
    cols = sorted(rows[0].keys()) if rows else []
    X = np.array([[r[c] for c in cols] for r in rows], dtype=float) if rows else np.zeros((0, 0))
    return X, cols


# --------------------------------------------------------------------------- #
# NLP-эмбеддинг: TF-IDF + LSA
# --------------------------------------------------------------------------- #
def lsa_embedding(texts, dim=64, ngram=(1, 2), max_features=4000, seed=42):
    """Латентно-семантическое представление текста рецепта (ингредиенты/directions)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram)
    T = vec.fit_transform(texts)
    dim = max(1, min(dim, T.shape[1] - 1)) if T.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    return svd.fit_transform(T).astype(np.float32)


def recipes_to_text(ingredient_lists, ru_fn=None):
    """Список рецептов (списки ингредиентов) -> строки для TF-IDF."""
    name = ru_fn or (lambda x: x)
    return [" ".join(name(x).replace(" ", "_") for x in ings) for ings in ingredient_lists]


# --------------------------------------------------------------------------- #
# NLP-эмбеддинг: BGE (sentence-transformers) — альтернатива LSA для сравнения
# --------------------------------------------------------------------------- #
def bge_embedding(texts, model_name="BAAI/bge-small-en-v1.5", batch_size=64,
                  device=None, verbose=True):
    """Плотный эмбеддинг текстов предобученной BGE-моделью.

    Для сравнения TF-IDF+LSA с плотным эмбеддером BGE / BGE-M3 (MTEB retrieval).

    model_name:
        "BAAI/bge-small-en-v1.5" — лёгкая английская (тексты RecipeNLG английские);
        "BAAI/bge-m3"            — мультиязычная тяжёлая (~2.3 GB), если тексты
                                   переведены на русский (ru_fn=ru).

    Возвращает np.ndarray [n, d] float32 либо None, если sentence-transformers
    не установлен (сравнение в ноутбуке пропускается с пометкой).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        if verbose:
            print("bge_embedding пропущен: пакет sentence-transformers не установлен "
                  "(pip install sentence-transformers).")
        return None

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        list(texts), batch_size=batch_size, show_progress_bar=verbose,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Сравнение представлений: classification + clustering
# --------------------------------------------------------------------------- #
def evaluate_representations(reps: dict, y, seed=42, cv=4):
    """Для каждого представления: CV-accuracy/F1 классификатора + silhouette, NMI, ARI."""
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    le = LabelEncoder()
    yy = le.fit_transform([str(v) for v in y])
    k = len(le.classes_)

    out = []
    for name, X in reps.items():
        X = np.asarray(X, dtype=float)
        Xs = StandardScaler().fit_transform(X)
        acc = float(cross_val_score(
            LogisticRegression(max_iter=2000), Xs, yy, cv=cv).mean())
        f1 = float(cross_val_score(
            LogisticRegression(max_iter=2000), Xs, yy, cv=cv, scoring="f1_macro").mean())
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xs)
        out.append(dict(
            representation=name, dim=X.shape[1], acc_cv=acc, f1_cv=f1,
            silhouette=float(silhouette_score(Xs, yy)),
            NMI=float(normalized_mutual_info_score(yy, km)),
            ARI=float(adjusted_rand_score(yy, km)),
        ))
    return out


def pair_separability(X, y, a, b, seed=42, cv=4):
    """CV-accuracy бинарного классификатора «кухня a vs кухня b» в представлении X.
    Низкая accuracy => кухни «смешаны» в этом представлении (как индийская/французская)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    y = np.asarray([str(v) for v in y])
    mask = np.isin(y, [str(a), str(b)])
    if mask.sum() < 2 * cv:
        return None
    yy = (y[mask] == str(a)).astype(int)
    if len(np.unique(yy)) < 2:
        return None
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=float)[mask])
    return float(cross_val_score(LogisticRegression(max_iter=2000), Xs, yy, cv=cv).mean())


def build_joint(*reps):
    """Конкатенация представлений (пропускает None/пустые)."""
    from sklearn.preprocessing import StandardScaler

    parts = [StandardScaler().fit_transform(np.asarray(r, dtype=float))
             for r in reps if r is not None and np.asarray(r).size]
    return np.hstack(parts) if parts else None


def all_pair_separabilities(reps: dict, y, seed=42, cv=4):
    """CV-accuracy бинарного «a vs b» для каждой пары кухонь и каждого представления."""
    from itertools import combinations

    cuis = sorted({str(v) for v in y})
    rows = []
    for a, b in combinations(cuis, 2):
        row = {'a': a, 'b': b, 'pair': f'{a} / {b}'}
        for name, X in reps.items():
            row[name] = pair_separability(X, y, a, b, seed=seed, cv=cv)
        rows.append(row)
    return rows
