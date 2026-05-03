#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jiajun/finance-gfm"
PY="/home/jiajun/anaconda3/envs/finance_gfm_gpu/bin/python"
OUT_DIR="$ROOT/results/gats_rescue_20260503"
LOG_DIR="$OUT_DIR/run_logs"

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

run_one() {
  local tag="$1"
  local config="$2"
  local seed="$3"
  local exp_name="gats_${tag}_csi300_recent_seed${seed}"
  local result_path="$OUT_DIR/tables/${exp_name}.json"
  local log_path="$LOG_DIR/${exp_name}.log"

  echo "[$(date '+%F %T')] start $exp_name"
  "$PY" scripts/run_qworkflow.py \
    --config "$config" \
    --experiment "$exp_name" \
    --summary_out "$result_path" \
    --override "task.model.kwargs.seed=${seed}" \
    --override "task.model.kwargs.GPU=0" \
    > "$log_path" 2>&1
  echo "[$(date '+%F %T')] done $exp_name"
}

run_one "rankicir" "$ROOT/configs/gats_alpha158_csi300_recent_rankicir.yaml" 512
