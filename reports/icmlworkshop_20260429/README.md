# ICML Workshop Experiment Results Snapshot

Generated: 2026-04-29 11:54:09 CST
Updated: 2026-04-29 22:39:12 CST

This directory is the Git-tracked result package for the ICML workshop run. It intentionally excludes checkpoints, mlruns, caches, verbose logs, and nested folders explicitly marked invalid. The package keeps lightweight metric tables, curated 10-seed summaries, transpose-B inference results, and learned-B heatmap assets.

## Contents

- `summaries/skip32_recent_top10_recent_only_raw_and_summary_20260429.md`: current main recent-only top10 report with raw rows and summary statistics.
- `summaries/skip32_recent_top10_recent_only_raw_20260429.csv`: CSV export of the raw top10 table.
- `summaries/skip32_recent_top10_recent_only_summary_20260429.csv`: CSV export of the top10 summary table.
- `summaries/fdg_skip32_recent_top10_transposeB_inference_20260429.*`: transpose-B inference-only ablation for the same top10 seeds.
- `tables/`: all current lightweight JSON result tables from top-level local result roots.
- `tables_manifest.csv`: flat index over all copied JSON tables with model, split, seed, recorder id, and key metrics.
- `heatmaps/fdg_b_heatmap_20260429/`: learned-B heatmap figures plus B/asymmetry matrices and supporting CSV/NPZ data.
- `experiment_settings_log.md`: copied experiment-setting log for reproducibility notes.

## Current Top10 Summary State

- Top10 seeds are selected by `fdg_skip32 / recent` IC_mean: `512, 23, 37, 5, 40, 55, 25, 49, 17, 79`.
- The current main top10 table includes `fdg_skip32`, `fdg_skip32_transposeB`, corrected `fdg_skip32_random`, `fdg_skip32_symshare`, `fdg_skip32_sharew_freeb`, `fdg_skip32_sepw_symb`, `mlp`, `master`, `hist`, `lstm`, and `lgbm`.
- In this snapshot all top10 `lstm / recent` rows are exact; no proxy seed is used in the current main report.
- `mlp` uses `configs/qlib_recent_mlp.yaml` with QLib official Alpha158 handler mode (`dataset_mode: official`), `csi300_aligned`, local 2024H1 provider data, and CUDA.

## Result Table Counts

Total copied JSON result tables: 1489

| model | official tables | recent tables |
| --- | ---: | ---: |
| fdg_skip32 | 102 | 101 |
| fdg_skip32_random | 101 | 101 |
| fdg_skip32_sepw_symb | 10 | 10 |
| fdg_skip32_sharew_freeb | 13 | 12 |
| fdg_skip32_symshare | 101 | 101 |
| fdg_skip32_transposeB | 0 | 10 |
| hist | 102 | 101 |
| lgbm | 101 | 101 |
| lstm | 106 | 102 |
| master | 102 | 102 |
| mlp | 0 | 10 |

## Notes

- `results/` remains ignored by `.gitignore`; this report directory is the Git-tracked snapshot.
- Portfolio metrics for qworkflow models are recorder backtest outputs using `excess_return_without_cost`, matching the notes in the main report.
- Random FDG active results are the corrected per-seed random graph version. The nested `invalid_random_fixed_seed_20260426` local folder is intentionally not included in this package.
