# School Server Migration Notes (2026-04-23)

This note records the company-side runtime we used for the reproducible `FDG/MLP/LSTM/MASTER/HIST` experiments, plus the exact steps to rebuild a matching environment on the ZJU server.

## Company Runtime

- Python: `3.9.13`
- PyTorch: `2.6.0+cu124`
- NumPy: `1.24.4`
- pandas: `2.2.3`
- pyqlib: `0.9.7`
- einops: `0.8.0`
- PyYAML: `6.0`
- scikit-learn: `1.0.2`
- scipy: `1.13.1`
- tqdm: `4.64.1`
- lightgbm: `4.6.0`
- xgboost: `2.1.4`
- mlflow: `3.1.4`
- dask: `2022.7.0`
- joblib: `1.4.2`
- matplotlib: `3.5.2`
- seaborn: `0.11.2`
- fire: `0.7.1`
- loguru: `0.7.3`
- tables: `3.6.1`

The company server also used:

- Qlib data ending at `2026-04-17`
- local Qlib fork path: `/project/user186_refs/qlib_sjtu`

## ZJU Server Findings

- The existing repo copy is `~/fdg-finace`
- Default `python` on login shell is still `Python 2.7.17`
- Existing conda envs:
  - `base`
  - `py310`
- Existing ZJU-side `.qlib` data ends at `2026-04-20`

## Recommended Portable Layout

Use a fresh project root and a local fork directory:

- repo: `~/finance-gfm`
- qlib fork root: `~/refs/qlib_sjtu`
- conda env: `finance_gfm39`

`run_qworkflow.py` now supports:

- `QLIB_FORK_PATH=/some/path`
- fallback `/project/user186_refs/qlib_sjtu`
- fallback `~/refs/qlib_sjtu`

That means the school server no longer needs the company-only `/project/...` path.

## Rebuild Commands On ZJU Server

```bash
~/anaconda3/bin/conda create -y -n finance_gfm39 python=3.9 pip
source ~/anaconda3/etc/profile.d/conda.sh
conda activate finance_gfm39

pip install --upgrade pip setuptools wheel
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install \
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
```

## Runtime Exports

Before running qworkflow models on the school server:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate finance_gfm39
export PYTHONPATH=$HOME/finance-gfm
export QLIB_FORK_PATH=$HOME/refs/qlib_sjtu
```

## First Repro Check

Use this as the first parity check:

```bash
python train/train_single.py \
  --config configs/qlib_recent_fdg.yaml \
  --override model.skip_hidden_dim=32 \
  --override train.seed=2026 \
  --override log.exp_name=fdg_csi300_qlib_recent_skip32_seed2026
```

If the school server is fully aligned, this should be the first experiment to compare against the company-side `skip32` reference.

## Important Caveat

The school server currently had a different local data snapshot (`2026-04-20` vs `2026-04-17`) and a different default env (`py310`, `torch 2.3.0+cpu`). Those differences are enough to move `skip32` noticeably, even when the code files themselves are identical.
