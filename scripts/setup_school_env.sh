#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/anaconda3}"
ENV_NAME="${ENV_NAME:-finance_gfm39}"
REPO_ROOT="${REPO_ROOT:-$HOME/finance-gfm}"
QLIB_FORK_PATH_DEFAULT="$HOME/refs/qlib_sjtu"

source "$CONDA_ROOT/etc/profile.d/conda.sh"

if ! "$CONDA_ROOT/bin/conda" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_ROOT/bin/conda" create -y -n "$ENV_NAME" python=3.9 pip
fi

"$CONDA_ROOT/bin/conda" install -y -n "$ENV_NAME" pytorch==2.6.0 pytorch-cuda=12.4 -c pytorch -c nvidia

"$CONDA_ROOT/bin/conda" run -n "$ENV_NAME" pip install --upgrade pip setuptools wheel
"$CONDA_ROOT/bin/conda" run -n "$ENV_NAME" pip install --prefer-binary \
  numpy==1.24.4 \
  pandas==2.2.3 \
  scipy==1.13.1 \
  scikit-learn==1.0.2 \
  pyqlib==0.9.7 \
  einops==0.8.0 \
  pyyaml==6.0 \
  tqdm==4.64.1 \
  lightgbm==4.6.0 \
  xgboost==2.1.4 \
  mlflow==3.1.4 \
  dask==2022.7.0 \
  joblib==1.4.2 \
  matplotlib==3.5.2 \
  seaborn==0.11.2 \
  fire==0.7.1 \
  loguru==0.7.3 \
  tables==3.6.1

cat <<EOF
Environment ready.

Use:
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  export PYTHONPATH="$REPO_ROOT"
  export QLIB_FORK_PATH="${QLIB_FORK_PATH:-$QLIB_FORK_PATH_DEFAULT}"
EOF
