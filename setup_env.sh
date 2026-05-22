#!/usr/bin/env bash
# Install project dependencies into conda env pydata-book (Jupyter kernel).
set -euo pipefail

ENV="${CONDA_ENV:-pydata-book}"
PYTHON="/opt/miniconda3/envs/${ENV}/bin/python"
PIP="/opt/miniconda3/envs/${ENV}/bin/pip"
CONDA="/opt/miniconda3/bin/conda"

if [[ ! -x "$PYTHON" ]]; then
  echo "Create env first: conda create -n pydata-book python=3.12 -y"
  exit 1
fi

echo "==> pip: core + GNN + clustering"
"$PIP" install -r requirements.txt

echo "==> conda: rdkit (prefer conda on macOS)"
"$CONDA" install -n "$ENV" -c conda-forge rdkit -y

echo "==> numpy 2.x (node2vec works; avoid 1.x/2.x mix in Jupyter)"
"$PIP" install "numpy>=2.0" --force-reinstall

echo "==> verify"
KMP_DUPLICATE_LIB_OK=TRUE "$PYTHON" -c "
import numpy, torch, networkx, rdkit, umap, hdbscan, node2vec
print('numpy', numpy.__version__)
print('torch', torch.__version__)
print('rdkit', rdkit.__version__)
print('OK')
"

echo ""
echo "Done. In Jupyter: kernel 'Python (pydata-book)', then Kernel -> Restart."
