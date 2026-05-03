#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jiajun/finance-gfm"
PY="/home/jiajun/anaconda3/envs/finance_gfm_gpu/bin/python"
OUT_DIR="$ROOT/results/gats_rescue_20260503"
LOG_DIR="$OUT_DIR/run_logs"
EXP_NAME="gats_icir_csi300_recent_seed512"

mkdir -p "$OUT_DIR/tables" "$LOG_DIR"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export QLIB_FORK_PATH="${QLIB_FORK_PATH:-/home/jiajun/refs/qlib_sjtu}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QLIB_NUM_WORKERS="${QLIB_NUM_WORKERS:-4}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

"$PY" scripts/run_qworkflow.py \
  --config "$ROOT/configs/gats_alpha158_csi300_recent_rankicir.yaml" \
  --experiment "$EXP_NAME" \
  --summary_out "$OUT_DIR/tables/${EXP_NAME}.json" \
  --override "task.model.kwargs.seed=512" \
  --override "task.model.kwargs.GPU=0" \
  --override "task.model.kwargs.metric=icir" \
  > "$LOG_DIR/${EXP_NAME}.log" 2>&1
