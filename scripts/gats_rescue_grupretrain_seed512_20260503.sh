#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jiajun/finance-gfm"
PY="/home/jiajun/anaconda3/envs/finance_gfm_gpu/bin/python"
OUT_DIR="$ROOT/results/gats_rescue_20260503"
LOG_DIR="$OUT_DIR/run_logs"
PRETRAIN="$OUT_DIR/pretrain/gru_csi300_recent_seed512.pt"
EXP_NAME="gats_grupretrain_loss_csi300_recent_seed512"

mkdir -p "$OUT_DIR/tables" "$LOG_DIR" "$OUT_DIR/pretrain"
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

if [[ ! -f "$PRETRAIN" ]]; then
  "$PY" scripts/run_qworkflow.py \
    --config "$ROOT/configs/gru_alpha158_csi300_recent_gats_pretrain.yaml" \
    --experiment "gru_pretrain_csi300_recent_seed512" \
    --override "task.model.kwargs.seed=512" \
    --override "task.model.kwargs.GPU=0" \
    --override "task.model_save_path=${PRETRAIN}" \
    > "$LOG_DIR/gru_pretrain_csi300_recent_seed512.log" 2>&1
fi

"$PY" scripts/run_qworkflow.py \
  --config "$ROOT/configs/gats_alpha158_csi300_recent_samplerfixed.yaml" \
  --experiment "$EXP_NAME" \
  --summary_out "$OUT_DIR/tables/${EXP_NAME}.json" \
  --override "task.model.kwargs.seed=512" \
  --override "task.model.kwargs.GPU=0" \
  --override "task.model.kwargs.model_path=${PRETRAIN}" \
  > "$LOG_DIR/${EXP_NAME}.log" 2>&1
