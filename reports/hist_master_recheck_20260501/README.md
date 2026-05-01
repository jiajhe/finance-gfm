# HIST / MASTER Recent Top10 Recheck - 2026-05-01

Seeds: `512, 23, 37, 5, 40, 55, 25, 49, 17, 79`. Split: recent train `2010-01-01..2020-12-31`, valid `2021-01-01..2021-12-31`, test `2022-01-01..2025-09-17`.

## Key Fixes

- HIST: rerun with Alpha158 after `DropCol(VWAP0)`, cross-sectional feature z-score/fill, `CSZScoreNorm(label)`, official `qlib_csi300_stock2concept.npy`, and a lower/upper-case compatible stock-index map. The 20-dim official LSTM pretrained checkpoint was not used because it is incompatible with the 157-dim aligned feature input.
- MASTER: the original qworkflow sampler is invalid for this TSDataSampler order. It slices contiguous blocks while the index is not true date-major, so spatial attention can see future samples in the same pseudo-batch. A leaky diagnostic run hit IC `0.634249`; it is archived under `invalid_leaky_master/` and excluded here.
- MASTER fixed: uses `DailyBatchSamplerByDatetime`, true per-date batches, and prediction index reconstruction from the actual batch indices. Market handler dates and instruments are also capped/aligned to `csi300_aligned` / `2025-09-17`.

## Summary

| model | count | IC mean | IC std | IC min | IC max | ICIR mean | RankIC mean | RankICIR mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HIST dataproc-aligned | 10 | 0.032476 | 0.002022 | 0.029254 | 0.035165 | 0.277494 | 0.018202 | 0.158750 |
| MASTER sampler-fixed | 10 | 0.017470 | 0.001404 | 0.014796 | 0.019667 | 0.119093 | 0.040220 | 0.249909 |

## Raw Results

| model | seed | IC | ICIR | RankIC | RankICIR | n_days |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HIST dataproc-aligned | 512 | 0.032330 | 0.263387 | 0.018621 | 0.150643 | 899 |
| HIST dataproc-aligned | 23 | 0.031038 | 0.269084 | 0.019263 | 0.175591 | 899 |
| HIST dataproc-aligned | 37 | 0.035165 | 0.292032 | 0.019103 | 0.163564 | 899 |
| HIST dataproc-aligned | 5 | 0.034223 | 0.282851 | 0.018666 | 0.154622 | 899 |
| HIST dataproc-aligned | 40 | 0.031796 | 0.296037 | 0.017597 | 0.168527 | 899 |
| HIST dataproc-aligned | 55 | 0.033724 | 0.298053 | 0.015212 | 0.135114 | 899 |
| HIST dataproc-aligned | 25 | 0.032562 | 0.290533 | 0.020891 | 0.193323 | 899 |
| HIST dataproc-aligned | 49 | 0.029254 | 0.246841 | 0.016532 | 0.141627 | 899 |
| HIST dataproc-aligned | 17 | 0.035165 | 0.312734 | 0.021810 | 0.196943 | 899 |
| HIST dataproc-aligned | 79 | 0.029500 | 0.223386 | 0.014328 | 0.107548 | 899 |
| MASTER sampler-fixed | 512 | 0.018394 | 0.127639 | 0.042096 | 0.264991 | 899 |
| MASTER sampler-fixed | 23 | 0.017784 | 0.130964 | 0.040194 | 0.269400 | 899 |
| MASTER sampler-fixed | 37 | 0.017047 | 0.113021 | 0.038487 | 0.235371 | 899 |
| MASTER sampler-fixed | 5 | 0.016838 | 0.112817 | 0.034213 | 0.205100 | 899 |
| MASTER sampler-fixed | 40 | 0.018219 | 0.123653 | 0.041998 | 0.256981 | 899 |
| MASTER sampler-fixed | 55 | 0.017671 | 0.118232 | 0.037713 | 0.227140 | 899 |
| MASTER sampler-fixed | 25 | 0.014796 | 0.101688 | 0.037134 | 0.233354 | 899 |
| MASTER sampler-fixed | 49 | 0.015502 | 0.103962 | 0.035289 | 0.211834 | 899 |
| MASTER sampler-fixed | 17 | 0.019667 | 0.133800 | 0.046765 | 0.292462 | 899 |
| MASTER sampler-fixed | 79 | 0.018783 | 0.125155 | 0.048310 | 0.302453 | 899 |

## Top5 By IC

### HIST dataproc-aligned

| seed | IC | ICIR | RankIC |
| ---: | ---: | ---: | ---: |
| 17 | 0.035165 | 0.312734 | 0.021810 |
| 37 | 0.035165 | 0.292032 | 0.019103 |
| 5 | 0.034223 | 0.282851 | 0.018666 |
| 55 | 0.033724 | 0.298053 | 0.015212 |
| 25 | 0.032562 | 0.290533 | 0.020891 |

### MASTER sampler-fixed

| seed | IC | ICIR | RankIC |
| ---: | ---: | ---: | ---: |
| 17 | 0.019667 | 0.133800 | 0.046765 |
| 79 | 0.018783 | 0.125155 | 0.048310 |
| 512 | 0.018394 | 0.127639 | 0.042096 |
| 40 | 0.018219 | 0.123653 | 0.041998 |
| 23 | 0.017784 | 0.130964 | 0.040194 |

## Files

- Raw CSV: `hist_master_recheck_recent_top10_raw_20260501.csv`
- Summary CSV: `hist_master_recheck_recent_top10_summary_20260501.csv`
- Configs: `generated_configs/hist_recent_dataproc_aligned.yaml`, `generated_configs/master_recent_timefixed_samplerfixed.yaml`
- Fixed sampler code: `models/master_fixed_ts.py`
