# Top5 skip32 recent seeds: 5-seed mean and variance

Generated: 2026-04-29 09:23:18 CST

Input: `skip32_recent_top5_all_experiments_20260429.csv`

Top seeds: `512, 23, 37, 5, 40` selected by `fdg_skip32 / recent` IC_mean.

Variance is sample variance across the five rows (`ddof=1`). `lstm / recent` still has one proxy row: requested seed 512 uses nearest completed seed 60.

## Summary

| model | split | n | proxy | IC mean | IC var | IC std | ICIR mean | ICIR var | RankIC mean | RankIC var | RankICIR mean | RankICIR var | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fdg_skip32 | official | 5 | 0 | 0.051255 | 0.000031 | 0.005607 | 0.450302 | 0.001142 | 0.046633 | 0.000023 | 0.413363 | 0.002108 |  |
| fdg_skip32 | recent | 5 | 0 | 0.043633 | 0.000001 | 0.001197 | 0.365233 | 0.000570 | 0.021174 | 0.000035 | 0.171323 | 0.002202 |  |
| fdg_skip32_random | official | 5 | 0 | 0.058366 | 0.000017 | 0.004139 | 0.498590 | 0.003028 | 0.047772 | 0.000005 | 0.432203 | 0.001175 |  |
| fdg_skip32_random | recent | 5 | 0 | 0.040705 | 0.000004 | 0.002084 | 0.351029 | 0.000671 | 0.019273 | 0.000073 | 0.175825 | 0.005947 |  |
| fdg_skip32_symshare | official | 5 | 0 | 0.055983 | 0.000015 | 0.003924 | 0.473685 | 0.001621 | 0.047742 | 0.000024 | 0.434582 | 0.001533 |  |
| fdg_skip32_symshare | recent | 5 | 0 | 0.037763 | 0.000005 | 0.002253 | 0.317485 | 0.000473 | 0.018366 | 0.000123 | 0.165002 | 0.011818 |  |
| master | official | 5 | 0 | 0.054717 | 0.000019 | 0.004367 | 0.522565 | 0.003130 | 0.060881 | 0.000020 | 0.587317 | 0.004496 |  |
| master | recent | 5 | 0 | 0.075871 | 0.000509 | 0.022553 | 0.733491 | 0.057631 | 0.083869 | 0.000527 | 0.771363 | 0.058282 |  |
| hist | official | 5 | 0 | 0.026809 | 0.000014 | 0.003675 | 0.217416 | 0.000979 | 0.038525 | 0.000002 | 0.303464 | 0.000230 |  |
| hist | recent | 5 | 0 | 0.015237 | 0.000000 | 0.000457 | 0.114929 | 0.000023 | 0.027978 | 0.000002 | 0.208518 | 0.000093 |  |
| lstm | official | 5 | 0 | 0.039395 | 0.000041 | 0.006416 | 0.368523 | 0.006391 | 0.047136 | 0.000007 | 0.461646 | 0.001856 |  |
| lstm | recent | 5 | 1 | 0.026971 | 0.000007 | 0.002733 | 0.215047 | 0.000956 | 0.039491 | 0.000015 | 0.307504 | 0.000289 | proxy: top_seed 512 -> used_seed 60 |
| lgbm | official | 5 | 0 | 0.055591 | 0.000003 | 0.001725 | 0.539287 | 0.000544 | 0.048279 | 0.000001 | 0.478550 | 0.000255 |  |
| lgbm | recent | 5 | 0 | 0.030848 | 0.000005 | 0.002174 | 0.268275 | 0.000402 | 0.015345 | 0.000002 | 0.133240 | 0.000121 |  |

## Notes

- `IC var`, `ICIR var`, `RankIC var`, and `RankICIR var` are sample variances over the five selected seed rows.
- `proxy > 0` means at least one requested seed was not available and a nearest completed seed was used, as requested.
- Sharpe statistics are included in the CSV; many qworkflow baselines do not emit Sharpe, so they are omitted from the markdown table for readability.
