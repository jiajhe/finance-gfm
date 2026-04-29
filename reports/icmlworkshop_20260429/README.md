# ICML Workshop Experiment Results Snapshot

Generated: 2026-04-29 11:54:09 CST

This directory is a lightweight GitHub snapshot of the experiment result tables and the latest summary reports from the local ICML workshop run. It intentionally excludes checkpoints, mlruns, and verbose logs.

## Contents

- `tables/seed100_gpu_fdg/`: FDG, random FDG, and symmetric-share FDG JSON result tables.
- `tables/seed100_gpu_master_hist_fixed/`: MASTER, HIST, and LSTM JSON result tables available at snapshot time.
- `tables/seed100_grouped_cpu/`: LGBM JSON result tables, plus earlier CPU-root LSTM artifacts if present in the source tables.
- `summaries/`: manually curated CSV/Markdown reports for the skip32 recent top-seed comparisons.
- `experiment_settings_log.md`: copied experiment-setting log for reproducibility notes.

## Completion Snapshot

| model | official tables | recent tables |
| --- | ---: | ---: |
| fdg_skip32 | 101 | 101 |
| fdg_skip32_random | 101 | 101 |
| fdg_skip32_symshare | 101 | 101 |
| master | 101 | 101 |
| hist | 101 | 101 |
| lstm | 101 | 67 |
| lgbm | 101 | 101 |

## Latest Summary Report

The newest report requested by the user is:

- `summaries/skip32_recent_top10_recent_only_raw_and_summary_20260429.md`

It contains two tables in one Markdown file:

1. Raw recent-only model performance under the top 10 `fdg_skip32 / recent` seeds.
2. Summary statistics by model across those 10 rows, including mean, sample variance, and sample standard deviation where included.

At snapshot time, `lstm / recent` was still running. The latest top10 report marks unfinished LSTM rows as `proxy`; specifically, requested seeds `512` and `79` use nearest completed LSTM recent seed `63` in that report.

## Notes

- `results/` remains ignored by `.gitignore`; this report directory is the Git-tracked snapshot.
- HIST had two stale error files locally, but the corresponding result JSON files existed before this snapshot was copied.
- Random FDG active results are the corrected per-seed random graph version; the old fixed-seed random run is not included here.
