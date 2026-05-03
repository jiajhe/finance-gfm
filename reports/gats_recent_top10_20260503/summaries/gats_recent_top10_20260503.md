# GATs Recent Top10 Seeds - 2026-05-03

Seeds selected by `fdg_skip32 / recent` top10 IC: `512, 23, 37, 5, 40, 55, 25, 49, 17, 79`.
Split: recent train `2010-01-01..2020-12-31`, valid `2021-01-01..2021-12-31`, test `2022-01-01..2025-09-17`.

Implementation: Qlib official `pytorch_gats_ts.GATs` model body with a fixed per-datetime sampler.
Data processing matches the current Qlib Alpha158 recent setup: `csi300_aligned`, `cn_data_2024h1`,
`DropCol(VWAP0)`, cross-sectional feature fill/normalization, normalized train label, and raw-label IC evaluation.

Completed rows: `2/10`.

## Raw Results

| seed | status | IC | ICIR | RankIC | RankICIR | Sharpe | AnnRet | AnnVol | MaxDD | Turnover | n_days |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | exact | 0.026689 | 0.260197 | 0.015226 | 0.139445 | 0.034782 | 0.006144 | 0.176653 | -0.388799 | 0.000000 | 899 |
| 23 | exact | 0.028578 | 0.264198 | 0.018305 | 0.158523 | 0.034782 | 0.006144 | 0.176653 | -0.388799 | 0.000000 | 899 |
| 37 | missing |  |  |  |  |  |  |  |  |  |  |
| 5 | missing |  |  |  |  |  |  |  |  |  |  |
| 40 | missing |  |  |  |  |  |  |  |  |  |  |
| 55 | missing |  |  |  |  |  |  |  |  |  |  |
| 25 | missing |  |  |  |  |  |  |  |  |  |  |
| 49 | missing |  |  |  |  |  |  |  |  |  |  |
| 17 | missing |  |  |  |  |  |  |  |  |  |  |
| 79 | missing |  |  |  |  |  |  |  |  |  |  |

## Summary Across Completed Seeds

| metric | n | mean | var | std | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IC_mean | 2 | 0.027634 | 0.000002 | 0.001336 | 0.026689 | 0.028578 |
| IC_std | 2 | 0.105371 | 0.000016 | 0.003957 | 0.102573 | 0.108169 |
| ICIR | 2 | 0.262197 | 0.000008 | 0.002830 | 0.260197 | 0.264198 |
| RankIC_mean | 2 | 0.016765 | 0.000005 | 0.002177 | 0.015226 | 0.018305 |
| RankIC_std | 2 | 0.112330 | 0.000020 | 0.004443 | 0.109189 | 0.115472 |
| RankICIR | 2 | 0.148984 | 0.000182 | 0.013490 | 0.139445 | 0.158523 |
| annual_return | 2 | 0.006144 | 0.000000 | 0.000000 | 0.006144 | 0.006144 |
| annual_vol | 2 | 0.176653 | 0.000000 | 0.000000 | 0.176653 | 0.176653 |
| sharpe | 2 | 0.034782 | 0.000000 | 0.000000 | 0.034782 | 0.034782 |
| max_drawdown | 2 | -0.388799 | 0.000000 | 0.000000 | -0.388799 | -0.388799 |
| turnover | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
