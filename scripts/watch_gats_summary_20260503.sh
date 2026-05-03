#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jiajun/finance-gfm"
PY="/home/jiajun/anaconda3/envs/finance_gfm_gpu/bin/python"
REPORT_DIR="$ROOT/reports/gats_recent_top10_20260503/summaries"
RESULT_DIR="$ROOT/results/seed100_gpu_gats/tables"
WINDOWS_DIR="/mnt/c/Users/Jiajun/Desktop/icmlworkshop"

cd "$ROOT"
mkdir -p "$WINDOWS_DIR"

while true; do
  "$PY" scripts/summarize_gats_recent_top10_20260503.py
  cp "$REPORT_DIR"/gats_recent_top10_*_20260503.csv "$WINDOWS_DIR"/
  cp "$REPORT_DIR"/gats_recent_top10_20260503.md "$WINDOWS_DIR"/

  done_count=$(find "$RESULT_DIR" -maxdepth 1 -type f -name 'gats_samplerfixed_csi300_recent_seed*.json' 2>/dev/null | wc -l)
  echo "[$(date '+%F %T')] refreshed GATs summary: ${done_count}/10"
  if [[ "$done_count" -ge 10 ]]; then
    break
  fi
  if ! pgrep -f "scripts/gats_recent_top10_20260503.sh" >/dev/null; then
    break
  fi
  sleep 300
done

"$PY" scripts/summarize_gats_recent_top10_20260503.py
cp "$REPORT_DIR"/gats_recent_top10_*_20260503.csv "$WINDOWS_DIR"/
cp "$REPORT_DIR"/gats_recent_top10_20260503.md "$WINDOWS_DIR"/
echo "[$(date '+%F %T')] GATs summary watcher finished"
