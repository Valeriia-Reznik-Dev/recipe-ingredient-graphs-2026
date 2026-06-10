"""
recipe_negatives.py
===================
Помощники для nb03 — валидность рецепта.

Блок A — граф ко-вхождений из переданных рецептов:
    build_cooc_graph(recipes, ...)   -> граф только из указанного списка рецептов
    edge_density(ings, graph)        -> плотность связей в переданном графе

Блок B — усиление негативов:
    personalized_pagerank_candidates(...) -> кандидаты по PPR (как в nb05)
    make_hard_fake_recipes(...)           -> hard negatives через PPR-замены
    verify_negatives_llm(...)             -> верификация «можно ли приготовить»
                                            через DeepSeek/OpenAI-совместимый API

Блок C — корпус негативов со шкалой уверенности и конфликт-майнинг:
    score_negatives_llm(...)    -> шкала уверенности LLM 0..100 (а не bool)
    build_negative_corpus(...)  -> наращивание корпуса до target_n негативов:
                                   генерация без повторов + кэш на диске + resume
    split_corpus(...)           -> разбиение корпуса на hard / soft / rejected
    difficulty_score(...)       -> формальная метрика «сложности» примера:
                                   конфликт «модель говорит МОЖНО» vs «LLM говорит НЕЛЬЗЯ»

Токен LLM НИКОГДА не хардкодится: передаётся аргументом api_key или
читается из переменной окружения вызывающим кодом.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import networkx as nx

__all__ = [
    "build_cooc_graph",
    "edge_density",
    "personalized_pagerank_candidates",
    "make_hard_fake_recipes",
    "verify_negatives_llm",
    "score_negatives_llm",
    "build_negative_corpus",
    "split_corpus",
    "difficulty_score",
    "load_corpus",
]


# --------------------------------------------------------------------------- #
# Блок A — граф ко-вхождений
# --------------------------------------------------------------------------- #
def build_cooc_graph(recipes, min_freq=50, min_cooc=10):
    """Строит граф ко-вхождений ингредиентов ТОЛЬКО из переданного списка
    рецептов. Логика и пороги повторяют nb01, но граф фитится на train-выборке,
    поэтому тестовые рецепты не участвуют в статистике, если граф строится только из train.

    Parameters
    ----------
    recipes : list[list[str]]   рецепты (списки ингредиентов)
    min_freq : int              мин. число рецептов, где встречается ингредиент
    min_cooc : int              мин. число совместных вхождений для ребра

    Returns
    -------
    networkx.Graph  с атрибутами рёбер weight (ко-вхождения) и pmi.
    """
    ing_freq = Counter()
    cooc = Counter()
    for ings in recipes:
        uniq = sorted(set(ings))
        for a in uniq:
            ing_freq[a] += 1
        for a, b in combinations(uniq, 2):
            cooc[(a, b)] += 1

    valid = {ing for ing, c in ing_freq.items() if c >= min_freq}
    total_pairs = sum(cooc.values()) or 1
    total_ings = float(sum(ing_freq.values())) or 1.0

    G = nx.Graph()
    G.add_nodes_from(valid)
    for (a, b), w in cooc.items():
        if a in valid and b in valid and w >= min_cooc:
            pab = w / total_pairs
            pa = ing_freq[a] / total_ings
            pb = ing_freq[b] / total_ings
            pmi = math.log(pab / (pa * pb + 1e-9))
            G.add_edge(a, b, weight=w, pmi=pmi)
    nx.set_node_attributes(G, dict(ing_freq), name="freq")
    return G


def edge_density(ings, graph):
    """Доля пар ингредиентов рецепта, между которыми есть ребро в `graph`.

    Граф передаётся явно — можно сравнить плотность на графе из обучения
    и на полном графе из этапа 1.
    """
    pairs = edges = 0
    n = len(ings)
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            if graph.has_edge(ings[i], ings[j]):
                edges += 1
    return edges / pairs if pairs else 0.0


# --------------------------------------------------------------------------- #
# Блок B — hard negatives через Personalized PageRank
# --------------------------------------------------------------------------- #
def personalized_pagerank_candidates(context, graph, ing_set, top_k=20, alpha=0.85):
    """top_k ингредиентов по Personalized PageRank относительно context,
    исключая сам context. (Та же логика, что в nb05 §5.5 — вынесена сюда,
    чтобы nb03 не зависел от рантайма nb05.)"""
    ctx = [x for x in context if x in graph]
    if not ctx:
        return []
    w = 1.0 / len(ctx)
    personalization = {c: w for c in ctx}  # nx дополнит остальные узлы нулём
    pr = nx.pagerank(graph, alpha=alpha, personalization=personalization, weight="weight")
    ctx_set = set(ctx)
    cands = [(n, s) for n, s in pr.items() if n not in ctx_set and n in ing_set]
    cands.sort(key=lambda x: -x[1])
    return cands[:top_k]


def make_hard_fake_recipes(
    recipes,
    graph,
    ing_set,
    mix_ratio=0.4,
    seed=42,
    ppr_top_k=40,
    exclude_top=5,
):
    """Генерация HARD негативов для задачи валидности.

    Для каждого рецепта заменяем долю ингредиентов на PPR-СОСЕДНИЕ
    (по отдельности правдоподобные), но пропускаем самые верхние PPR-хиты —
    они как раз сочетаемы. Берём середину ранжирования: ингредиенты «рядом по
    графу, но в эту комбинацию не просятся» -> near-miss рецепт, который трудно
    отличить от настоящего. Если PPR пуст — откат на случайную замену.

    Это лишь КАНДИДАТЫ в негативы; финально их отбирает LLM-верификация
    (verify_negatives_llm), оставляя подтверждённые «невозможные».
    """
    from ingredient_ru import filter_ingredients

    rng = np.random.default_rng(seed)
    pool = sorted(ing_set)
    fakes = []
    for ings in recipes:
        ings = list(ings)
        if not ings:
            fakes.append(ings)
            continue
        n_swap = max(1, int(round(len(ings) * mix_ratio)))
        swap_idx = rng.choice(len(ings), size=min(n_swap, len(ings)), replace=False)
        cands = personalized_pagerank_candidates(ings, graph, ing_set, top_k=ppr_top_k)
        cand_names = [c for c, _ in cands[exclude_top:]] or pool
        new = ings[:]
        for k, idx in enumerate(swap_idx):
            new[int(idx)] = cand_names[k % len(cand_names)]
        fakes.append(filter_ingredients(new))
    return fakes


# --------------------------------------------------------------------------- #
# Блок B — LLM-верификация негативов (DeepSeek / OpenAI-совместимый API)
# --------------------------------------------------------------------------- #
def verify_negatives_llm(
    recipes_ings,
    ru_fn=None,
    api_key=None,
    base_url="https://api.deepseek.com/v1/chat/completions",
    model="deepseek-chat",
    batch=8,
    timeout=60,
    sleep=0.3,
    verbose=True,
):
    """Спрашивает у LLM, можно ли из набора ингредиентов приготовить осмысленное
    реальное блюдо. Возвращает per-рецепт вердикты и долю подтверждённых
    «невозможных» (это и есть качественные hard negatives).

    Parameters
    ----------
    recipes_ings : list[list[str]]  кандидаты в негативы
    ru_fn : callable | None         перевод имени ингредиента (например ingredient_ru.ru)
    api_key : str | None            токен; если None — функция вернёт None (пропуск)
    base_url, model                 эндпоинт и модель (DeepSeek по умолчанию,
                                    совместимо с OpenAI chat/completions)

    Returns
    -------
    dict | None
        {"verdicts": list[bool],         # True = невозможно/неправдоподобно
         "frac_impossible": float,       # доля True
         "impossible_idx": list[int]}    # индексы подтверждённых негативов
        либо None, если api_key не задан.
    """
    if not api_key:
        if verbose:
            print("LLM-верификация пропущена: api_key не задан. "
                  "Задайте переменную окружения с токеном профессора и передайте сюда.")
        return None

    import json as _json
    import time
    import requests

    from ingredient_ru import filter_ingredients

    name = ru_fn or (lambda x: x)
    sys_prompt = (
        "Ты кулинарный эксперт. Для каждого списка ингредиентов реши, можно ли из них "
        "приготовить осмысленное РЕАЛЬНОЕ блюдо. Ответь строго JSON-массивом булевых "
        "значений: true = блюдо невозможно/неправдоподобно, false = вполне возможно. "
        "Никаких пояснений, только JSON-массив."
    )

    verdicts = []
    for start in range(0, len(recipes_ings), batch):
        chunk = recipes_ings[start:start + batch]
        listing = "\n".join(
            f"{i + 1}. " + ", ".join(name(x) for x in filter_ingredients(r))
            for i, r in enumerate(chunk)
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user",
                 "content": f"Списки:\n{listing}\n\nОтветь JSON-массивом из {len(chunk)} булевых."},
            ],
            "temperature": 0.0,
        }
        try:
            resp = requests.post(
                base_url, timeout=timeout, json=payload,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            txt = resp.json()["choices"][0]["message"]["content"].strip()
            txt = txt.strip("`")
            if txt.lower().startswith("json"):
                txt = txt[4:].strip()
            arr = [bool(x) for x in _json.loads(txt)][:len(chunk)]
            arr += [True] * (len(chunk) - len(arr))   # консервативно дополняем
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  батч {start}: ошибка LLM ({e}) -> помечаю как 'не подтверждено'")
            arr = [False] * len(chunk)
        verdicts.extend(arr)
        if verbose:
            print(f"  обработано {min(start + batch, len(recipes_ings))}/{len(recipes_ings)}")
        time.sleep(sleep)

    frac = float(np.mean(verdicts)) if verdicts else 0.0
    impossible_idx = [i for i, v in enumerate(verdicts) if v]
    return {"verdicts": verdicts, "frac_impossible": frac, "impossible_idx": impossible_idx}


# --------------------------------------------------------------------------- #
# Блок C — шкала уверенности LLM, корпус негативов, конфликт-майнинг
# --------------------------------------------------------------------------- #
def _llm_chat(messages, api_key, base_url, model, timeout=60):
    """Один запрос к OpenAI-совместимому chat/completions. Возвращает текст ответа."""
    import requests

    resp = requests.post(
        base_url, timeout=timeout,
        json={"model": model, "messages": messages, "temperature": 0.0},
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _parse_json_array(txt):
    """Достаёт JSON-массив из ответа LLM (срезает ```json ... ``` и прочий мусор)."""
    txt = txt.strip().strip("`")
    if txt.lower().startswith("json"):
        txt = txt[4:].strip()
    lo, hi = txt.find("["), txt.rfind("]")
    if lo >= 0 and hi > lo:
        txt = txt[lo:hi + 1]
    return json.loads(txt)


def score_negatives_llm(
    recipes_ings,
    ru_fn=None,
    api_key=None,
    base_url="https://api.deepseek.com/v1/chat/completions",
    model="deepseek-chat",
    batch=8,
    timeout=60,
    sleep=0.3,
    verbose=True,
):
    """Шкала уверенности 0..100 вместо бинарного вердикта LLM.
    Для каждого набора ингредиентов LLM даёт score 0..100:

        0   = из этого точно можно приготовить осмысленное реальное блюдо;
        100 = точно НЕЛЬЗЯ (комбинация невозможна/неправдоподобна).

    Возвращает list[float] длины len(recipes_ings); при ошибке батча — NaN
    (не подтверждено, пример не попадает ни в hard, ни в soft).
    """
    if not api_key:
        if verbose:
            print("score_negatives_llm пропущена: api_key не задан.")
        return None

    import time

    from ingredient_ru import filter_ingredients

    name = ru_fn or (lambda x: x)
    sys_prompt = (
        "Ты кулинарный эксперт. Для каждого списка ингредиентов оцени по шкале 0..100, "
        "насколько НЕВОЗМОЖНО приготовить из них осмысленное реальное блюдо: "
        "0 = точно возможно (обычное блюдо), 50 = сомнительно, "
        "100 = точно невозможно (несочетаемая комбинация). "
        "Ответь строго JSON-массивом чисел, без пояснений."
    )

    scores: list[float] = []
    for start in range(0, len(recipes_ings), batch):
        chunk = recipes_ings[start:start + batch]
        listing = "\n".join(
            f"{i + 1}. " + ", ".join(name(x) for x in filter_ingredients(r))
            for i, r in enumerate(chunk)
        )
        try:
            txt = _llm_chat(
                [{"role": "system", "content": sys_prompt},
                 {"role": "user",
                  "content": f"Списки:\n{listing}\n\nОтветь JSON-массивом из {len(chunk)} чисел 0..100."}],
                api_key, base_url, model, timeout=timeout,
            )
            arr = [float(x) for x in _parse_json_array(txt)][:len(chunk)]
            arr += [float("nan")] * (len(chunk) - len(arr))
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  батч {start}: ошибка LLM ({e}) -> NaN")
            arr = [float("nan")] * len(chunk)
        scores.extend(arr)
        if verbose:
            print(f"  оценено {min(start + batch, len(recipes_ings))}/{len(recipes_ings)}")
        time.sleep(sleep)
    return scores


def difficulty_score(model_p_real, llm_score):
    """Метрика «сложности» негатива: конфликт модели и LLM.

    difficulty = P_model(реальный) * P_LLM(невозможный)
               = model_p_real * (llm_score / 100)

    Максимум — когда наша модель уверена, что рецепт настоящий, а LLM уверена,
    что приготовить нельзя. Это и есть конфликтный (по-настоящему hard) пример.
    """
    if model_p_real is None or llm_score is None or np.isnan(llm_score):
        return float("nan")
    return float(model_p_real) * float(llm_score) / 100.0


def load_corpus(cache_path):
    """Читает кэш корпуса негативов (JSON-список записей) или возвращает []."""
    p = Path(cache_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []


def _save_corpus(corpus, cache_path):
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8")


def split_corpus(corpus, hard_thr=70.0, soft_thr=40.0):
    """Разбивает корпус по шкале LLM:

        hard     : llm_score >= hard_thr   (LLM уверена — приготовить нельзя)
        soft     : soft_thr <= llm_score < hard_thr  (сомнительно)
        rejected : llm_score < soft_thr    (съедобно — негативом не считаем)

    Возвращает dict {"hard": [...], "soft": [...], "rejected": [...]} записей корпуса.
    """
    out = {"hard": [], "soft": [], "rejected": []}
    for rec in corpus:
        s = rec.get("llm_score")
        if s is None or (isinstance(s, float) and np.isnan(s)):
            continue
        if s >= hard_thr:
            out["hard"].append(rec)
        elif s >= soft_thr:
            out["soft"].append(rec)
        else:
            out["rejected"].append(rec)
    return out


def build_negative_corpus(
    recipes,
    graph,
    ing_set,
    target_n=500,
    api_key=None,
    cache_path="output_graphs/negative_corpus.json",
    model_score_fn=None,
    ru_fn=None,
    hard_thr=70.0,
    soft_thr=40.0,
    mix_ratio=0.4,
    batch=8,
    gen_chunk=120,
    max_llm_calls=2000,
    seed=42,
    verbose=True,
    **llm_kwargs,
):
    """Наращивает корпус негативов до `target_n` подтверждённых (hard+soft) примеров.

    Цикл «генерация -> LLM-оценка» до target_n; дедупликация по множеству ингредиентов;
    кэш на диске с resume. Если передан `model_score_fn(list_of_recipes) -> P(real)`,
        кандидаты оцениваются моделью и в LLM уходят В ПЕРВУЮ ОЧЕРЕДЬ те, кого
        модель считает «настоящими» — именно из них получаются hard-конфликты.

    Каждая запись корпуса:
        {"ingredients": [...], "llm_score": 0..100, "model_p_real": float|None,
         "difficulty": float|None, "gen_seed": int}

    Кэш пишется на диск после каждого LLM-батча: прерывание ничего не теряет.
    Возвращает (corpus, split) — полный корпус и разбиение split_corpus.
    """
    from ingredient_ru import filter_ingredients

    corpus = load_corpus(cache_path)
    seen = {frozenset(rec["ingredients"]) for rec in corpus}

    def n_confirmed():
        sp = split_corpus(corpus, hard_thr, soft_thr)
        return len(sp["hard"]) + len(sp["soft"])

    if verbose:
        print(f"Кэш: {len(corpus)} записей, подтверждённых негативов: {n_confirmed()} "
              f"(цель {target_n})")

    if api_key is None:
        if verbose:
            print("api_key не задан — возвращаю текущее состояние кэша без генерации.")
        return corpus, split_corpus(corpus, hard_thr, soft_thr)

    rng = np.random.default_rng(seed)
    llm_calls = 0
    gen_round = 0

    while n_confirmed() < target_n and llm_calls < max_llm_calls:
        gen_round += 1
        gen_seed = int(rng.integers(0, 2**31 - 1))
        base_idx = rng.choice(len(recipes), size=min(gen_chunk, len(recipes)), replace=False)
        base = [recipes[i] for i in base_idx]
        cands = make_hard_fake_recipes(
            base, graph, ing_set, mix_ratio=mix_ratio, seed=gen_seed,
        )
        # дедупликация против кэша и внутри раунда
        fresh = []
        for c in cands:
            c = filter_ingredients(c)
            key = frozenset(c)
            if len(c) >= 3 and key not in seen:
                seen.add(key)
                fresh.append(c)
        if not fresh:
            continue

        # конфликт-майнинг: сперва те, кого модель считает «настоящими»
        p_real = None
        if model_score_fn is not None:
            p_real = np.asarray(model_score_fn(fresh), dtype=float)
            order = np.argsort(-p_real)
            fresh = [fresh[i] for i in order]
            p_real = p_real[order]

        need = target_n - n_confirmed()
        # запас x2: часть кандидатов LLM отклонит как съедобные
        take = min(len(fresh), max(batch, 2 * need))
        chunk_recipes = fresh[:take]
        scores = score_negatives_llm(
            chunk_recipes, ru_fn=ru_fn, api_key=api_key, batch=batch,
            verbose=False, **llm_kwargs,
        )
        llm_calls += len(chunk_recipes)

        for j, (r, s) in enumerate(zip(chunk_recipes, scores)):
            pr = float(p_real[j]) if p_real is not None else None
            corpus.append({
                "ingredients": list(r),
                "llm_score": None if np.isnan(s) else float(s),
                "model_p_real": pr,
                "difficulty": None if (pr is None or np.isnan(s)) else difficulty_score(pr, s),
                "gen_seed": gen_seed,
            })
        _save_corpus(corpus, cache_path)
        if verbose:
            sp = split_corpus(corpus, hard_thr, soft_thr)
            print(f"  раунд {gen_round}: +{len(chunk_recipes)} оценено | "
                  f"hard={len(sp['hard'])} soft={len(sp['soft'])} "
                  f"rejected={len(sp['rejected'])} | LLM-вызовов: {llm_calls}")

    if verbose:
        sp = split_corpus(corpus, hard_thr, soft_thr)
        total = len(sp["hard"]) + len(sp["soft"])
        print(f"Готово: подтверждённых негативов {total}/{target_n} "
              f"(hard={len(sp['hard'])}, soft={len(sp['soft'])}). Кэш: {cache_path}")
    return corpus, split_corpus(corpus, hard_thr, soft_thr)
