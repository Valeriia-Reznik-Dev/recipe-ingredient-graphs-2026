"""Загрузка рецептов RecipeNLG с proxy-метками кухни по домену URL."""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

DOMAIN_TO_CUISINE = {
    'cookbooks.com': 'american',
    'food.com': 'american',
    'allrecipes.com': 'american',
    'recipeland.com': 'american',
    'geniuskitchen.com': 'american',
    'tasteofhome.com': 'american',
    'bbcgoodfood.com': 'british',
    'bbc.co.uk': 'british',
    'jamieoliver.com': 'british',
    '750g.com': 'french',
    'marmiton.org': 'french',
    'recettes.net': 'french',
    'giallozafferano.it': 'italian',
    'ricette.giallozafferano.it': 'italian',
    'chefkoch.de': 'german',
    'japanesecooking101.com': 'japanese',
    'justonecookbook.com': 'japanese',
    'chinasichuanfood.com': 'chinese',
    'redhousespice.com': 'chinese',
    'mexicanfoodjournal.com': 'mexican',
    'indianhealthyrecipes.com': 'indian',
}


# Характерные ингредиенты (proxy, если URL не даёт разнообразия)
CUISINE_SIGNATURES: dict[str, set[str]] = {
    'italian': {
        'pasta', 'basil', 'oregano', 'parmesan', 'mozzarella', 'ricotta',
        'prosciutto', 'marinara', 'balsamic vinegar', 'olive oil',
    },
    'mexican': {
        'tortilla', 'cumin', 'cilantro', 'jalapeno', 'black beans',
        'salsa', 'monterey jack', 'corn tortillas', 'chili powder', 'lime',
    },
    'chinese': {
        'soy sauce', 'sesame oil', 'rice vinegar', 'ginger', 'bok choy',
        'hoisin sauce', 'oyster sauce', 'green onions', 'sesame seeds',
    },
    'indian': {
        'curry', 'turmeric', 'garam masala', 'cumin', 'coriander',
        'ghee', 'chickpeas', 'lentils', 'cardamom', 'yogurt',
    },
    'french': {
        'dijon mustard', 'herbes de provence', 'gruyere', 'cognac',
        'shallots', 'creme fraiche', 'tarragon', 'wine',
    },
    'japanese': {
        'miso', 'mirin', 'dashi', 'nori', 'wasabi', 'sake', 'tofu',
        'soy sauce', 'rice vinegar', 'sesame seeds',
    },
    'american': {
        'peanut butter', 'brown sugar', 'corn syrup', 'cheddar cheese',
        'cream cheese', 'ground beef', 'bacon', 'barbecue sauce',
    },
}


def cuisine_from_url(url: str | None) -> str | None:
    if not isinstance(url, str):
        return None
    url_lower = url.lower()
    for domain, cuisine in DOMAIN_TO_CUISINE.items():
        if domain in url_lower:
            return cuisine
    return None


def cuisine_from_ingredients(ingredients: list[str], min_hits: int = 2) -> str | None:
    """Proxy-метка кухни по пересечению с набором характерных ингредиентов."""
    ing_set = set(ingredients)
    scores = {
        cuisine: len(ing_set & sigs)
        for cuisine, sigs in CUISINE_SIGNATURES.items()
    }
    best_c, best_s = max(scores.items(), key=lambda x: x[1])
    if best_s >= min_hits:
        return best_c
    return None


def label_cuisine(url: str | None, ingredients: list[str]) -> str | None:
    """Сначала URL; если american-dominated subset — эвристика по ингредиентам."""
    by_url = cuisine_from_url(url)
    by_ing = cuisine_from_ingredients(ingredients)
    if by_ing and by_ing != 'american':
        return by_ing
    return by_url or by_ing


def load_recipes_stratified(
    csv_path: Path | str,
    per_cuisine: int = 1500,
    max_chunks: int | None = None,
    min_ingredients: int = 3,
) -> pd.DataFrame:
    """Читает CSV чанками и набирает до per_cuisine рецептов на каждую кухню."""
    csv_path = Path(csv_path)
    cuisines = sorted(CUISINE_SIGNATURES.keys())
    done = {c: 0 for c in cuisines}
    rows: list[dict] = []

    chunk_iter = pd.read_csv(
        csv_path,
        usecols=['NER', 'link', 'title'],
        chunksize=50_000,
    )
    for chunk_idx, chunk in enumerate(chunk_iter):
        if max_chunks is not None and chunk_idx >= max_chunks:
            break
        if all(done[c] >= per_cuisine for c in cuisines):
            break

        for _, row in chunk.iterrows():
            try:
                ings = [
                    x.strip().lower()
                    for x in ast.literal_eval(row['NER'])
                    if isinstance(x, str) and x.strip()
                ]
            except Exception:
                continue
            if len(ings) < min_ingredients:
                continue

            cuisine = label_cuisine(row.get('link'), ings)
            if cuisine is None or done[cuisine] >= per_cuisine:
                continue

            rows.append({
                'title': row.get('title', ''),
                'link': row['link'],
                'cuisine': cuisine,
                'ingredients': ings,
            })
            done[cuisine] += 1

    df = pd.DataFrame(rows)
    if not df.empty:
        df['cuisine'] = pd.Categorical(df['cuisine'], categories=cuisines)
    return df


def recipe_community_vector(ingredients: list[str], comm: dict[str, int], n_comm: int) -> list[float]:
    """Доля ингредиентов рецепта в каждом сообществе Leiden."""
    vec = [0.0] * n_comm
    for ing in ingredients:
        c = comm.get(ing)
        if c is not None:
            vec[c] += 1.0
    total = sum(vec)
    if total > 0:
        vec = [v / total for v in vec]
    return vec


def cuisine_counts(df: pd.DataFrame) -> Counter:
    return Counter(df['cuisine'].astype(str))
