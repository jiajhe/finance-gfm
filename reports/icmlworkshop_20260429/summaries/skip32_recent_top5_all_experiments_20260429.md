# FDG skip32 recent top5 seeds: all experiment performance

Generated: 2026-04-29 09:19:42 CST

Selection rule: sort `fdg_skip32 / recent` by `test_metrics.IC_mean` descending and take top 5 seeds.

Source result roots:
- FDG variants: `/home/jiajun/finance-gfm/results/seed100_gpu_fdg/tables`
- MASTER/HIST/LSTM: `/home/jiajun/finance-gfm/results/seed100_gpu_master_hist_fixed/tables`
- LGBM: `/home/jiajun/finance-gfm/results/seed100_grouped_cpu/tables`

Important note: `lstm / recent` is still running. Missing requested seeds are filled with the nearest completed LSTM recent seed and marked as `proxy`.

## Top5 seeds

| rank | seed | fdg_skip32 recent IC_mean | IC_std | ICIR | RankIC_mean | RankICIR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 0.045677 | 0.130493 | 0.350032 | 0.027799 | 0.220490 |
| 2 | 23 | 0.043556 | 0.129650 | 0.335952 | 0.013485 | 0.100271 |
| 3 | 37 | 0.043249 | 0.119608 | 0.361591 | 0.025994 | 0.201212 |
| 4 | 5 | 0.043116 | 0.111711 | 0.385961 | 0.021301 | 0.181790 |
| 5 | 40 | 0.042568 | 0.108419 | 0.392629 | 0.017289 | 0.152855 |

## Requested top seed 512

| model | split | used_seed | status | IC_mean | IC_std | ICIR | RankIC_mean | RankICIR | Sharpe | note |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fdg_skip32 | official | 512 | exact | 0.049306 | 0.114308 | 0.431345 | 0.047445 | 0.412039 | 1.712773 |  |
| fdg_skip32 | recent | 512 | exact | 0.045677 | 0.130493 | 0.350032 | 0.027799 | 0.220490 | 1.082962 |  |
| fdg_skip32_random | official | 512 | exact | 0.056735 | 0.118165 | 0.480132 | 0.050755 | 0.445854 | 1.700008 |  |
| fdg_skip32_random | recent | 512 | exact | 0.041338 | 0.126823 | 0.325951 | 0.031256 | 0.266608 | 1.122264 |  |
| fdg_skip32_symshare | official | 512 | exact | 0.052749 | 0.124677 | 0.423089 | 0.052863 | 0.483651 | 1.789203 |  |
| fdg_skip32_symshare | recent | 512 | exact | 0.039054 | 0.126747 | 0.308126 | 0.001054 | 0.008340 | 0.908866 |  |
| master | official | 512 | exact | 0.058301 | 0.101508 | 0.574344 | 0.063242 | 0.640414 | NA |  |
| master | recent | 512 | exact | 0.100100 | 0.102306 | 0.978437 | 0.107806 | 1.012593 | NA |  |
| hist | official | 512 | exact | 0.028366 | 0.123725 | 0.229262 | 0.039275 | 0.312479 | NA |  |
| hist | recent | 512 | exact | 0.014837 | 0.129299 | 0.114750 | 0.026779 | 0.202141 | NA |  |
| lstm | official | 512 | exact | 0.033574 | 0.119199 | 0.281660 | 0.046937 | 0.422745 | NA |  |
| lstm | recent | 60 | proxy | 0.028931 | 0.126046 | 0.229528 | 0.042394 | 0.328521 | NA | missing requested seed 512; using nearest completed LSTM recent seed 60 |
| lgbm | official | 512 | exact | 0.056695 | 0.102374 | 0.553804 | 0.048593 | 0.490530 | NA |  |
| lgbm | recent | 512 | exact | 0.029933 | 0.114985 | 0.260322 | 0.014471 | 0.125556 | NA |  |

## Requested top seed 23

| model | split | used_seed | status | IC_mean | IC_std | ICIR | RankIC_mean | RankICIR | Sharpe | note |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fdg_skip32 | official | 23 | exact | 0.056579 | 0.116814 | 0.484353 | 0.049488 | 0.478208 | 1.836723 |  |
| fdg_skip32 | recent | 23 | exact | 0.043556 | 0.129650 | 0.335952 | 0.013485 | 0.100271 | 1.058127 |  |
| fdg_skip32_random | official | 23 | exact | 0.064835 | 0.109117 | 0.594175 | 0.046303 | 0.464769 | 1.623399 |  |
| fdg_skip32_random | recent | 23 | exact | 0.041439 | 0.128866 | 0.321564 | 0.007902 | 0.059866 | 0.910430 |  |
| fdg_skip32_symshare | official | 23 | exact | 0.062383 | 0.116663 | 0.534730 | 0.045631 | 0.407931 | 1.707245 |  |
| fdg_skip32_symshare | recent | 23 | exact | 0.035481 | 0.118717 | 0.298872 | 0.013688 | 0.115456 | 1.240701 |  |
| master | official | 23 | exact | 0.050611 | 0.109693 | 0.461387 | 0.054350 | 0.486477 | NA |  |
| master | recent | 23 | exact | 0.045377 | 0.113071 | 0.401310 | 0.054903 | 0.455590 | NA |  |
| hist | official | 23 | exact | 0.023735 | 0.124475 | 0.190679 | 0.037021 | 0.287351 | NA |  |
| hist | recent | 23 | exact | 0.015810 | 0.130419 | 0.121222 | 0.028708 | 0.219430 | NA |  |
| lstm | official | 23 | exact | 0.040902 | 0.101652 | 0.402373 | 0.047888 | 0.500264 | NA |  |
| lstm | recent | 23 | exact | 0.025101 | 0.119739 | 0.209634 | 0.037246 | 0.314049 | NA |  |
| lgbm | official | 23 | exact | 0.054351 | 0.104863 | 0.518305 | 0.047953 | 0.462150 | NA |  |
| lgbm | recent | 23 | exact | 0.034655 | 0.114178 | 0.303514 | 0.017482 | 0.151770 | NA |  |

## Requested top seed 37

| model | split | used_seed | status | IC_mean | IC_std | ICIR | RankIC_mean | RankICIR | Sharpe | note |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fdg_skip32 | official | 37 | exact | 0.050604 | 0.121359 | 0.416981 | 0.047439 | 0.378986 | 1.624072 |  |
| fdg_skip32 | recent | 37 | exact | 0.043249 | 0.119608 | 0.361591 | 0.025994 | 0.201212 | 1.125344 |  |
| fdg_skip32_random | official | 37 | exact | 0.060036 | 0.121681 | 0.493391 | 0.048348 | 0.443535 | 1.647924 |  |
| fdg_skip32_random | recent | 37 | exact | 0.040007 | 0.107976 | 0.370522 | 0.020695 | 0.198499 | 1.028410 |  |
| fdg_skip32_symshare | official | 37 | exact | 0.053603 | 0.112464 | 0.476620 | 0.040677 | 0.392699 | 1.636004 |  |
| fdg_skip32_symshare | recent | 37 | exact | 0.039996 | 0.113599 | 0.352079 | 0.026224 | 0.240167 | 1.111308 |  |
| master | official | 37 | exact | 0.055524 | 0.103733 | 0.535265 | 0.060851 | 0.587802 | NA |  |
| master | recent | 37 | exact | 0.090013 | 0.099349 | 0.906030 | 0.097354 | 0.941707 | NA |  |
| hist | official | 37 | exact | 0.022177 | 0.123559 | 0.179486 | 0.036878 | 0.287088 | NA |  |
| hist | recent | 37 | exact | 0.014739 | 0.136889 | 0.107674 | 0.028853 | 0.208525 | NA |  |
| lstm | official | 37 | exact | 0.032621 | 0.112725 | 0.289383 | 0.044708 | 0.414414 | NA |  |
| lstm | recent | 37 | exact | 0.023119 | 0.140273 | 0.164818 | 0.044710 | 0.314791 | NA |  |
| lgbm | official | 37 | exact | 0.057543 | 0.103145 | 0.557879 | 0.049361 | 0.490713 | NA |  |
| lgbm | recent | 37 | exact | 0.030157 | 0.115862 | 0.260285 | 0.015009 | 0.129681 | NA |  |

## Requested top seed 5

| model | split | used_seed | status | IC_mean | IC_std | ICIR | RankIC_mean | RankICIR | Sharpe | note |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fdg_skip32 | official | 5 | exact | 0.043205 | 0.100561 | 0.429636 | 0.038375 | 0.362594 | 1.557153 |  |
| fdg_skip32 | recent | 5 | exact | 0.043116 | 0.111711 | 0.385961 | 0.021301 | 0.181790 | 1.264542 |  |
| fdg_skip32_random | official | 5 | exact | 0.055302 | 0.120370 | 0.459432 | 0.048359 | 0.432325 | 1.776144 |  |
| fdg_skip32_random | recent | 5 | exact | 0.043175 | 0.114101 | 0.378397 | 0.020968 | 0.205210 | 0.942697 |  |
| fdg_skip32_symshare | official | 5 | exact | 0.054143 | 0.117718 | 0.459943 | 0.051425 | 0.467461 | 1.727419 |  |
| fdg_skip32_symshare | recent | 5 | exact | 0.039100 | 0.128943 | 0.303230 | 0.023147 | 0.175586 | 1.130008 |  |
| master | official | 5 | exact | 0.059365 | 0.103179 | 0.575357 | 0.066289 | 0.654992 | NA |  |
| master | recent | 5 | exact | 0.083903 | 0.104029 | 0.806541 | 0.095075 | 0.867714 | NA |  |
| hist | official | 5 | exact | 0.030831 | 0.121131 | 0.254527 | 0.039509 | 0.319442 | NA |  |
| hist | recent | 5 | exact | 0.015552 | 0.135205 | 0.115025 | 0.029229 | 0.216286 | NA |  |
| lstm | official | 5 | exact | 0.048190 | 0.103582 | 0.465236 | 0.051285 | 0.508423 | NA |  |
| lstm | recent | 5 | exact | 0.028288 | 0.125552 | 0.225312 | 0.037351 | 0.292367 | NA |  |
| lgbm | official | 5 | exact | 0.056040 | 0.100654 | 0.556754 | 0.048259 | 0.489301 | NA |  |
| lgbm | recent | 5 | exact | 0.030329 | 0.114980 | 0.263777 | 0.015364 | 0.134128 | NA |  |

## Requested top seed 40

| model | split | used_seed | status | IC_mean | IC_std | ICIR | RankIC_mean | RankICIR | Sharpe | note |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fdg_skip32 | official | 40 | exact | 0.056581 | 0.115661 | 0.489195 | 0.050421 | 0.434988 | 1.908056 |  |
| fdg_skip32 | recent | 40 | exact | 0.042568 | 0.108419 | 0.392629 | 0.017289 | 0.152855 | 1.158640 |  |
| fdg_skip32_random | official | 40 | exact | 0.054924 | 0.117909 | 0.465819 | 0.045097 | 0.374535 | 1.662853 |  |
| fdg_skip32_random | recent | 40 | exact | 0.037566 | 0.104727 | 0.358710 | 0.015544 | 0.148942 | 1.041754 |  |
| fdg_skip32_symshare | official | 40 | exact | 0.057037 | 0.120319 | 0.474046 | 0.048113 | 0.421167 | 1.673220 |  |
| fdg_skip32_symshare | recent | 40 | exact | 0.035183 | 0.108215 | 0.325117 | 0.027718 | 0.285462 | 0.934924 |  |
| master | official | 40 | exact | 0.049785 | 0.106727 | 0.466473 | 0.059675 | 0.566902 | NA |  |
| master | recent | 40 | exact | 0.059961 | 0.104256 | 0.575136 | 0.064205 | 0.579209 | NA |  |
| hist | official | 40 | exact | 0.028934 | 0.124114 | 0.233125 | 0.039944 | 0.310961 | NA |  |
| hist | recent | 40 | exact | 0.015246 | 0.131462 | 0.115974 | 0.026319 | 0.196209 | NA |  |
| lstm | official | 40 | exact | 0.041687 | 0.103195 | 0.403965 | 0.044863 | 0.462385 | NA |  |
| lstm | recent | 40 | exact | 0.029414 | 0.119596 | 0.245945 | 0.035756 | 0.287790 | NA |  |
| lgbm | official | 40 | exact | 0.053326 | 0.104624 | 0.509693 | 0.047227 | 0.460056 | NA |  |
| lgbm | recent | 40 | exact | 0.029164 | 0.115055 | 0.253475 | 0.014396 | 0.125066 | NA |  |

## Proxy rows

| requested_seed | model | split | used_seed | reason |
| ---: | --- | --- | ---: | --- |
| 512 | lstm | recent | 60 | missing requested seed 512; using nearest completed LSTM recent seed 60 |

## Completion snapshot

| model | official count | recent count |
| --- | ---: | ---: |
| fdg_skip32 | 101 | 101 |
| fdg_skip32_random | 101 | 101 |
| fdg_skip32_symshare | 101 | 101 |
| master | 101 | 101 |
| hist | 101 | 101 |
| lstm | 101 | 47 |
| lgbm | 101 | 101 |
