#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jiajun/finance-gfm"
PY="/home/jiajun/anaconda3/envs/finance_gfm_gpu/bin/python"
CONFIG="$ROOT/configs/gats_alpha158_csi300_recent_samplerfixed.yaml"
OUT_DIR="$ROOT/results/seed100_gpu_gats"
LOG_DIR="$OUT_DIR/run_logs"
SEEDS=(512 23 37 5 40 55 25 49 17 79)

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

for seed in "${SEEDS[@]}"; do
  exp_name="gats_samplerfixed_csi300_recent_seed${seed}"
  result_path="$OUT_DIR/tables/${exp_name}.json"
  log_path="$LOG_DIR/${exp_name}.log"
  if [[ -f "$result_path" ]]; then
    echo "[$(date '+%F %T')] skip existing $exp_name"
    continue
  fi

  echo "[$(date '+%F %T')] start $exp_name"
  "$PY" scripts/run_qworkflow.py \
    --config "$CONFIG" \
    --experiment "$exp_name" \
    --summary_out "$result_path" \
    --override "task.model.kwargs.seed=${seed}" \
    --override "task.model.kwargs.GPU=0" \
    > "$log_path" 2>&1
  echo "[$(date '+%F %T')] done $exp_name"
done

echo "[$(date '+%F %T')] all requested GATs recent top10 runs finished"
