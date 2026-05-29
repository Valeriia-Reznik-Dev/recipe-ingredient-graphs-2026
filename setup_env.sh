#!/usr/bin/env bash
# Install project dependencies into conda env pydata-book (Jupyter kernel).
set -euo pipefail

ENV="${CONDA_ENV:-pydata-book}"
PYTHON="/opt/miniconda3/envs/${ENV}/bin/python"
PIP="/opt/miniconda3/envs/${ENV}/bin/pip"

if [[ ! -x "$PYTHON" ]]; then
  echo "Create env first: conda create -n pydata-book python=3.12 -y"
  exit 1
fi

echo "==> pip: core + GNN stack"
"$PIP" install -r requirements.txt

echo "==> verify"
KMP_DUPLICATE_LIB_OK=TRUE "$PYTHON" -c "
import numpy, torch, networkx, sklearn
print('numpy', numpy.__version__)
print('torch', torch.__version__)
print('OK')
"

echo ""
echo "Done. In Jupyter: kernel 'Python (pydata-book)', then Kernel -> Restart."
