# Recipe–Ingredient Graphs

Анализ корпуса рецептов **RecipeNLG** через графовые модели: построение сети ингредиентов, извлечение структурных признаков и три задачи предсказания на графах.

Учебный проект курса **SNA Magolego**.

**Автор:** [Valeriia Reznik](https://github.com/Valeriia-Reznik-Dev)

## О проекте

Из рецептов строятся два графа:

- **Граф I** — ингредиенты, связанные совместным появлением в блюдах (вес ребра = число общих рецептов).
- **Граф G** — двудольный граф «рецепт ↔ ингредиент».

На этих графах решаются три задачи. Для каждой сравниваются классические методы (логистическая регрессия, эвристики на графе) и графовые нейросети: **GraphSAGE** и **HeteroGAT**.

Признаки узлов — только **структурные**: степень, PageRank, clustering, k-core, betweenness, log-частота. Названия ингредиентов в выводе переводятся на русский (`ingredient_ru.py`).

## Задачи

| Задача | Ноутбук | Суть |
|--------|---------|------|
| Предсказание кухни | [`04_cuisine_prediction.ipynb`](04_cuisine_prediction.ipynb) | Сообщества ингредиентов (Leiden) → классификация кухни рецепта |
| Подбор замены ингредиента | [`05_ingredient_prediction.ipynb`](05_ingredient_prediction.ipynb) | Link prediction на графе «рецепт — ингредиент» |
| Реалистичность рецепта | [`03_dish_validity_prediction.ipynb`](03_dish_validity_prediction.ipynb) | Отличие настоящего рецепта от искусственного (graph classification) |

## Структура репозитория

```
.
├── 01_graph_construction.ipynb      # построение графов I и G
├── 02_feature_engineering.ipynb     # структурные признаки узлов
├── 03_dish_validity_prediction.ipynb
├── 04_cuisine_prediction.ipynb
├── 05_ingredient_prediction.ipynb
├── data_utils.py                    # загрузка рецептов, метки кухни
├── gnn_cuisine_classification.py    # HeteroGAT для кухни
├── gnn_graph_classification.py      # GraphSAGE / HeteroGAT для real vs fake
├── gnn_link_prediction.py           # GraphSAGE для link prediction
├── ingredient_ru.py                 # словарь перевода ингредиентов
├── data_sample.csv                  # небольшой сэмпл для быстрого прогона
├── output_graphs/                   # локальные артефакты (не в git)
├── requirements.txt
└── setup_env.sh
```

## Требования

- Python 3.10+
- Jupyter Notebook или JupyterLab

Основные библиотеки: NetworkX, scikit-learn, python-igraph, leidenalg, PyTorch, PyTorch Geometric.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Либо через conda-скрипт (если используется окружение `pydata-book`):

```bash
bash setup_env.sh
```

## Данные

| Файл | Описание |
|------|----------|
| `data_sample.csv` | Уже в репозитории; подходит для проверки пайплайна |
| `full_dataset.csv` | Полный RecipeNLG (~2.2 ГБ); скачивается отдельно и кладётся в корень проекта |

Ноутбуки автоматически переключаются на `data_sample.csv`, если полный датасет не найден.

## Запуск

1. Установите зависимости (см. выше).
2. Положите данные в корень проекта.
3. Запустите ноутбуки **по порядку** с 01 по 05:

```bash
jupyter notebook
```

**Порядок выполнения**

| Шаг | Ноутбук | Результат |
|-----|---------|-----------|
| 1 | `01_graph_construction.ipynb` | `ingredient_graph.graphml`, `bipartite_sample.graphml` |
| 2 | `02_feature_engineering.ipynb` | `node_features_structural.csv`, `node_features.npz` |
| 3–5 | `03` / `04` / `05` | Модели, метрики, визуализации |

Графики и промежуточные файлы сохраняются локально в `output_graphs/` при запуске ноутбуков.

## Модули Python

| Модуль | Назначение |
|--------|------------|
| `data_utils.py` | Загрузка RecipeNLG, proxy-метки кухни, векторы по сообществам |
| `gnn_cuisine_classification.py` | Обучение HeteroGAT для задачи кухни |
| `gnn_graph_classification.py` | Graph classification (real vs fake), counterfactual-анализ |
| `gnn_link_prediction.py` | GraphSAGE для восстановления рёбер |
| `ingredient_ru.py` | Перевод названий ингредиентов на русский |
