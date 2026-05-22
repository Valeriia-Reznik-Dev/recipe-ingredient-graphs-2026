# Recipe–Ingredient Graphs: Compatibility, Substitution and the Food-Pairing Hypothesis

Course project for **SNA Magolego**.
The project applies social-network analysis to the cooking domain:
recipes and ingredients are organised as a network, and we study
its structure, learn node embeddings, predict missing links,
recommend ingredient substitutions, and test the *food-pairing hypothesis*.

## Pipeline overview

The project is split into six self-contained Jupyter notebooks that
are intended to be executed in order. Each notebook reads artifacts
produced by the previous one and writes its own outputs into
`output_graphs/`.

| Notebook | Stage | What it does |
|---|---|---|
| `01_graph_construction.ipynb` | Graph construction | Builds the ingredient–ingredient graph `I` and a bipartite recipe–ingredient sample `G` from RecipeNLG. |
| `02_feature_engineering.ipynb` | Feature engineering | Adds 7 structural features (degree, PageRank, clustering, k-core, betweenness, log_freq) and 512-dim chemical Morgan fingerprints from FlavorDB. |
| `03_cuisine_clustering.ipynb` | Cuisine clustering | Trains GraphSAGE (unsupervised) on the bipartite graph, projects with UMAP, clusters with HDBSCAN, evaluates against proxy cuisine labels (URL domain). |
| `04_link_prediction.ipynb` | Link prediction | Compares heuristics (Common Neighbors, Jaccard, Adamic-Adar, Resource Allocation), Node2Vec + LR, and a Graph Autoencoder (GAE). |
| `05_substitution_recommender.ipynb` | Substitution recommender | Combines GAE embeddings and chemical fingerprints into a substitution score; evaluates with Recall@K and Precision@K. |
| `06_food_pairing_hypothesis.ipynb` | Food-pairing hypothesis | Tests the Ahn et al. (2011) hypothesis globally and per cuisine using Spearman ρ, point-biserial, Mann-Whitney U, and a permutation test. |

## Mapping to the course material

| Lecture topic | Where it is used |
|---|---|
| Graph construction, bipartite graphs, projections | notebook 01 |
| Degree distribution, density, LCC, ego-graphs | notebook 01 |
| Centralities (degree, PageRank, betweenness), k-core, clustering | notebook 02 |
| EDA (heavy-tail degree distribution, filtering, PMI, log-weights) | notebooks 01–02 |
| Unsupervised embeddings (matrix decomposition, random walks, Node2Vec) | notebook 04 |
| Message-passing GNNs (GraphSAGE, GCN, GAE) | notebooks 03–04 |
| Link prediction (Adamic-Adar / Jaccard / CN / RA, GAE) | notebook 04 |
| Recommendation on graphs | notebook 05 |
| Statistical hypothesis testing (Spearman ρ, permutation) | notebook 06 |

## Datasets

* **RecipeNLG** — 2.23 M recipes with normalised ingredient lists.
  The full `full_dataset.csv` is ~2.2 GB and **cannot** be hosted on
  GitHub directly. See the three options below.
* **FlavorDB** — public flavor-molecule database
  (<https://cosylab.iiitd.edu.in/flavordb/>). Notebook 02 downloads
  ~1 000 entities once and caches them to
  `output_graphs/flavordb_cache.json`. No manual setup needed.

### How to get the recipe data

| Option | What you get | How |
|---|---|---|
| **1. Smoke test (already in repo)** | A 5 000-recipe `data_sample.csv` (~3 MB) that ships with the repository. All notebooks automatically fall back to it if `full_dataset.csv` is not present. | Already there — just run the notebooks. |
| **2. Full dataset from the official source** | 2.23 M recipes. | Register at <https://recipenlg.cs.put.poznan.pl/>, download `full_dataset.csv` and place it in the repository root next to the notebooks. |
| **3. Full dataset from this repo's GitHub Releases** | Same as option 2. | Download `full_dataset.csv.zip` from the [latest Release](https://github.com/Valeriia-Reznik-Dev/Recipe-Ingredient-Graphs/releases), unzip into the repository root. |

> The notebooks contain a small fallback at the top of each stage:
> ```python
> DATA_PATH = Path('full_dataset.csv')
> if not DATA_PATH.exists():
>     DATA_PATH = Path('data_sample.csv')
> ```
> so the same code runs in both "smoke" and "full" mode.

## How to run

```bash
# 1. Create an isolated environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Place full_dataset.csv (RecipeNLG) in the repository root
#    for the full pipeline. Otherwise the included data_sample.csv is used.

# 4. Launch Jupyter and run the notebooks in order 01 -> 06.
jupyter notebook
```

Approximate runtimes on a modern laptop (CPU only):

| Notebook | Time |
|---|---|
| 01 | 3–6 min |
| 02 | 4–8 min (mostly betweenness + FlavorDB download on first run) |
| 03 | 5–10 min (GraphSAGE + UMAP + HDBSCAN on 220k nodes) |
| 04 | 4–7 min |
| 05 | 1–2 min |
| 06 | 1–2 min |

## Repository layout

```
.
|-- 01_graph_construction.ipynb
|-- 02_feature_engineering.ipynb
|-- 03_cuisine_clustering.ipynb
|-- 04_link_prediction.ipynb
|-- 05_substitution_recommender.ipynb
|-- 06_food_pairing_hypothesis.ipynb
|-- data_sample.csv                # 5 000-recipe smoke-test sample (3 MB)
|-- output_graphs/                 # plots and small result files (regenerated by notebooks)
|-- recipe_graph_project_architecture.svg
|-- requirements.txt
|-- README.md
`-- .gitignore
```

Large generated artefacts (the full bipartite graphml, embeddings,
co-occurrence pickle, recipe embeddings) are **not** committed: they
are recreated when the notebooks are run. Only summary plots, small
CSVs and JSON results stay in `output_graphs/` for quick inspection.

### What is and is not in git

The repository keeps everything needed to **read** the project:
all six notebooks include their plots and tables inline, every figure
under `output_graphs/*.png` and every summary file
(`*_summary.json`, `*_scores.csv`, `*_results.csv`,
`flavordb_cache.json`) is committed.

What is intentionally **not** committed:

| File | Size | Reason |
|---|---|---|
| `recipe_embeddings.npz` | 276 MB | exceeds the 100 MB GitHub file limit |
| `cooc.pkl` | 125 MB | exceeds the 100 MB GitHub file limit |
| `bipartite_sample.graphml` | 103 MB | exceeds the 100 MB GitHub file limit |
| `ingredient_graph.graphml` | 76 MB | large; regenerated by notebook 01 |
| `chem_embs.pkl`, `ing_freq.pkl`, `*.npz`, `*.pt`, `ingredient_freq.csv`, `node_features_structural.csv` | 100 KB – 14 MB each | regenerated by notebooks 01–04 |

To reproduce these artefacts, run notebooks 01 → 04 in order with
either the bundled `data_sample.csv` (smoke mode) or the full
RecipeNLG dataset.

## Key results

* **Graph I** (after filtering): ~7 200 ingredients,
  hundreds of thousands of edges with `weight` and `pmi` attributes,
  heavy-tailed degree distribution.
* **Link prediction**: GAE with chemical features outperforms the
  Adamic-Adar / Jaccard / Common-Neighbors baselines on AUC and AP;
  see `output_graphs/link_prediction_scores.csv`.
* **Substitution recommender**: best mix of GAE and chemical signal at
  `alpha ≈ 0.4–0.6`; see `output_graphs/substitution_summary.json`.
* **Food-pairing hypothesis**: tested globally and per cuisine; full
  numbers in `output_graphs/food_pairing_results.csv` and
  `output_graphs/food_pairing_summary.json`.

## References

1. Marin et al., *RecipeNLG: A Cooking Recipes Dataset for
   Semi-Structured Text Generation*, 2020.
2. Garg et al., *FlavorDB: a database of flavor molecules*, NAR 2017.
3. Ahn et al., *Flavor network and the principles of food pairing*,
   Scientific Reports 1, 196 (2011).
4. Hamilton, Ying, Leskovec, *Inductive Representation Learning on
   Large Graphs* (GraphSAGE), NeurIPS 2017.
5. Kipf and Welling, *Variational Graph Auto-Encoders*, 2016.
