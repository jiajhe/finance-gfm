# Skip32 Recent Top10 Seeds: Recent-Only Raw Data and Summary

Generated: 2026-04-29 09:30:11 CST
Updated: 2026-04-29 22:37 CST; all top10 LSTM rows are exact; includes MLP, transpose-B, and FDG ablations.

Top10 seeds selected by `fdg_skip32 / recent` IC_mean: `512, 23, 37, 5, 40, 55, 25, 49, 17, 79`.
Only `recent` split is included. Variance columns use sample variance across the selected seed rows (`ddof=1`).
For unfinished `lstm / recent` seeds, the nearest currently completed LSTM recent seed is used and marked as `proxy`; in this snapshot all top10 LSTM rows are exact.
`mlp` is run from `configs/qlib_recent_mlp.yaml` with QLib official Alpha158 handler mode (`dataset_mode: official`), `csi300_aligned`, the 2024H1 local provider, and CUDA.
Portfolio metrics for qworkflow models (`master`, `hist`, `lstm`, `lgbm`) are read from each mlruns recorder using `excess_return_without_cost`; `annual_vol` is daily std annualized by `sqrt(252)`.
Note: the qworkflow portfolio records currently have identical risk metrics and zero average turnover across these runs, so treat those portfolio columns as recorder backtest output rather than the FDG/MLP top-k portfolio metric.

## Raw Recent Data

| top_seed | model | used_seed | status | IC_mean | IC_std | ICIR | RankIC_mean | RankICIR | annual_return | annual_vol | sharpe | max_drawdown | turnover | note |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | fdg_skip32 | 512 | exact | 0.045677 | 0.130493 | 0.350032 | 0.027799 | 0.220490 | 0.269349 | 0.248902 | 1.082962 | -0.348837 | 0.638532 |  |
| 512 | fdg_skip32_transposeB | 512 | exact | 0.043926 | 0.119817 | 0.366609 | 0.025322 | 0.239879 | 0.259473 | 0.238074 | 1.088297 | -0.307380 | 0.636485 | inference only: use B.T; no retraining |
| 512 | fdg_skip32_random | 512 | exact | 0.041338 | 0.126823 | 0.325951 | 0.031256 | 0.266608 | 0.266869 | 0.235533 | 1.122264 | -0.280911 | 0.688632 | fully random graph baseline |
| 512 | fdg_skip32_symshare | 512 | exact | 0.039054 | 0.126747 | 0.308126 | 0.001054 | 0.008340 | 0.222501 | 0.257558 | 0.908866 | -0.364114 | 0.619377 | shared W_s/W_r; symmetric B; final A symmetrized |
| 512 | fdg_skip32_sharew_freeb | 512 | exact | 0.038305 | 0.128504 | 0.298085 | 0.008881 | 0.068310 | 0.251236 | 0.244459 | 1.039059 | -0.343814 | 0.675528 | shared W_s/W_r; free asymmetric B |
| 512 | fdg_skip32_sepw_symb | 512 | exact | 0.037891 | 0.107329 | 0.353041 | 0.013732 | 0.128282 | 0.232406 | 0.238839 | 0.994282 | -0.357769 | 0.675840 | separate W_s/W_r; symmetric B |
| 512 | mlp | 512 | exact | 0.039607 | 0.115176 | 0.343879 | 0.011371 | 0.107329 | 0.235274 | 0.250331 | 0.969332 | -0.321791 | 0.644961 | configs/qlib_recent_mlp.yaml; QLib official Alpha158 handler dataset_mode; csi300_aligned; CUDA |
| 512 | master | 512 | exact | 0.100100 | 0.102306 | 0.978437 | 0.107806 | 1.012593 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 512 | hist | 512 | exact | 0.014837 | 0.129299 | 0.114750 | 0.026779 | 0.202141 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 512 | lstm | 512 | exact | 0.026312 | 0.130355 | 0.201850 | 0.041839 | 0.324936 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 512 | lgbm | 512 | exact | 0.029933 | 0.114985 | 0.260322 | 0.014471 | 0.125556 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 23 | fdg_skip32 | 23 | exact | 0.043556 | 0.129650 | 0.335952 | 0.013485 | 0.100271 | 0.270525 | 0.257703 | 1.058127 | -0.312551 | 0.625451 |  |
| 23 | fdg_skip32_transposeB | 23 | exact | 0.043388 | 0.124969 | 0.347186 | 0.012101 | 0.092950 | 0.230087 | 0.250221 | 0.952802 | -0.372100 | 0.640756 | inference only: use B.T; no retraining |
| 23 | fdg_skip32_random | 23 | exact | 0.041439 | 0.128866 | 0.321564 | 0.007902 | 0.059866 | 0.220635 | 0.254563 | 0.910430 | -0.326603 | 0.611146 | fully random graph baseline |
| 23 | fdg_skip32_symshare | 23 | exact | 0.035481 | 0.118717 | 0.298872 | 0.013688 | 0.115456 | 0.305463 | 0.237633 | 1.240701 | -0.296785 | 0.632903 | shared W_s/W_r; symmetric B; final A symmetrized |
| 23 | fdg_skip32_sharew_freeb | 23 | exact | 0.037090 | 0.120319 | 0.308261 | 0.014132 | 0.115889 | 0.289629 | 0.238102 | 1.187535 | -0.301205 | 0.638799 | shared W_s/W_r; free asymmetric B |
| 23 | fdg_skip32_sepw_symb | 23 | exact | 0.044316 | 0.132008 | 0.335704 | 0.012764 | 0.092377 | 0.274313 | 0.258042 | 1.068538 | -0.330445 | 0.619244 | separate W_s/W_r; symmetric B |
| 23 | mlp | 23 | exact | 0.037080 | 0.114164 | 0.324798 | 0.021615 | 0.200973 | 0.215021 | 0.213916 | 1.017412 | -0.269845 | 0.654372 | configs/qlib_recent_mlp.yaml; QLib official Alpha158 handler dataset_mode; csi300_aligned; CUDA |
| 23 | master | 23 | exact | 0.045377 | 0.113071 | 0.401310 | 0.054903 | 0.455590 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 23 | hist | 23 | exact | 0.015810 | 0.130419 | 0.121222 | 0.028708 | 0.219430 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 23 | lstm | 23 | exact | 0.025101 | 0.119739 | 0.209634 | 0.037246 | 0.314049 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 23 | lgbm | 23 | exact | 0.034655 | 0.114178 | 0.303514 | 0.017482 | 0.151770 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 37 | fdg_skip32 | 37 | exact | 0.043249 | 0.119608 | 0.361591 | 0.025994 | 0.201212 | 0.261277 | 0.229788 | 1.125344 | -0.288745 | 0.678865 |  |
| 37 | fdg_skip32_transposeB | 37 | exact | 0.041743 | 0.117390 | 0.355596 | 0.023549 | 0.182503 | 0.284452 | 0.231639 | 1.196874 | -0.216942 | 0.665406 | inference only: use B.T; no retraining |
| 37 | fdg_skip32_random | 37 | exact | 0.040007 | 0.107976 | 0.370522 | 0.020695 | 0.198499 | 0.236735 | 0.233031 | 1.028410 | -0.308698 | 0.644716 | fully random graph baseline |
| 37 | fdg_skip32_symshare | 37 | exact | 0.039996 | 0.113599 | 0.352079 | 0.026224 | 0.240167 | 0.264056 | 0.235991 | 1.111308 | -0.322535 | 0.615706 | shared W_s/W_r; symmetric B; final A symmetrized |
| 37 | fdg_skip32_sharew_freeb | 37 | exact | 0.040207 | 0.119277 | 0.337093 | 0.026018 | 0.221106 | 0.250707 | 0.226188 | 1.102187 | -0.248675 | 0.662024 | shared W_s/W_r; free asymmetric B |
| 37 | fdg_skip32_sepw_symb | 37 | exact | 0.040757 | 0.115221 | 0.353734 | 0.021981 | 0.187747 | 0.277636 | 0.233271 | 1.167217 | -0.299765 | 0.677998 | separate W_s/W_r; symmetric B |
| 37 | mlp | 37 | exact | 0.038739 | 0.114977 | 0.336928 | 0.018523 | 0.170622 | 0.264706 | 0.241450 | 1.093502 | -0.333406 | 0.641513 | configs/qlib_recent_mlp.yaml; QLib official Alpha158 handler dataset_mode; csi300_aligned; CUDA |
| 37 | master | 37 | exact | 0.090013 | 0.099349 | 0.906030 | 0.097354 | 0.941707 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 37 | hist | 37 | exact | 0.014739 | 0.136889 | 0.107674 | 0.028853 | 0.208525 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 37 | lstm | 37 | exact | 0.023119 | 0.140273 | 0.164818 | 0.044710 | 0.314791 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 37 | lgbm | 37 | exact | 0.030157 | 0.115862 | 0.260285 | 0.015009 | 0.129681 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 5 | fdg_skip32 | 5 | exact | 0.043116 | 0.111711 | 0.385961 | 0.021301 | 0.181790 | 0.279945 | 0.213200 | 1.264542 | -0.229149 | 0.635106 |  |
| 5 | fdg_skip32_transposeB | 5 | exact | 0.043061 | 0.108627 | 0.396410 | 0.019808 | 0.172494 | 0.262869 | 0.217936 | 1.180049 | -0.232318 | 0.625451 | inference only: use B.T; no retraining |
| 5 | fdg_skip32_random | 5 | exact | 0.043175 | 0.114101 | 0.378397 | 0.020968 | 0.205210 | 0.201919 | 0.221049 | 0.942697 | -0.279352 | 0.647542 | fully random graph baseline |
| 5 | fdg_skip32_symshare | 5 | exact | 0.039100 | 0.128943 | 0.303230 | 0.023147 | 0.175586 | 0.285136 | 0.249688 | 1.130008 | -0.345987 | 0.627008 | shared W_s/W_r; symmetric B; final A symmetrized |
| 5 | fdg_skip32_sharew_freeb | 5 | exact | 0.035360 | 0.130757 | 0.270429 | 0.020870 | 0.160957 | 0.314850 | 0.248306 | 1.226769 | -0.324705 | 0.653726 | shared W_s/W_r; free asymmetric B |
| 5 | fdg_skip32_sepw_symb | 5 | exact | 0.041755 | 0.111556 | 0.374295 | 0.021346 | 0.182282 | 0.322439 | 0.213609 | 1.415426 | -0.222246 | 0.637642 | separate W_s/W_r; symmetric B |
| 5 | mlp | 5 | exact | 0.038185 | 0.107245 | 0.356054 | 0.015818 | 0.162448 | 0.226190 | 0.244244 | 0.956971 | -0.338044 | 0.597152 | configs/qlib_recent_mlp.yaml; QLib official Alpha158 handler dataset_mode; csi300_aligned; CUDA |
| 5 | master | 5 | exact | 0.083903 | 0.104029 | 0.806541 | 0.095075 | 0.867714 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 5 | hist | 5 | exact | 0.015552 | 0.135205 | 0.115025 | 0.029229 | 0.216286 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 5 | lstm | 5 | exact | 0.028288 | 0.125552 | 0.225312 | 0.037351 | 0.292367 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 5 | lgbm | 5 | exact | 0.030329 | 0.114980 | 0.263777 | 0.015364 | 0.134128 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 40 | fdg_skip32 | 40 | exact | 0.042568 | 0.108419 | 0.392629 | 0.017289 | 0.152855 | 0.265520 | 0.225135 | 1.158640 | -0.265997 | 0.645517 |  |
| 40 | fdg_skip32_transposeB | 40 | exact | 0.043216 | 0.107510 | 0.401968 | 0.013498 | 0.118893 | 0.260499 | 0.231005 | 1.117724 | -0.307754 | 0.623804 | inference only: use B.T; no retraining |
| 40 | fdg_skip32_random | 40 | exact | 0.037566 | 0.104727 | 0.358710 | 0.015544 | 0.148942 | 0.235277 | 0.227719 | 1.041754 | -0.320042 | 0.669588 | fully random graph baseline |
| 40 | fdg_skip32_symshare | 40 | exact | 0.035183 | 0.108215 | 0.325117 | 0.027718 | 0.285462 | 0.195707 | 0.216262 | 0.934924 | -0.270377 | 0.677620 | shared W_s/W_r; symmetric B; final A symmetrized |
| 40 | fdg_skip32_sharew_freeb | 40 | exact | 0.039090 | 0.139601 | 0.280009 | 0.013385 | 0.086828 | 0.237747 | 0.267854 | 0.930382 | -0.354443 | 0.565473 | shared W_s/W_r; free asymmetric B |
| 40 | fdg_skip32_sepw_symb | 40 | exact | 0.038198 | 0.103557 | 0.368857 | 0.015825 | 0.155272 | 0.237769 | 0.234790 | 1.026053 | -0.342189 | 0.637686 | separate W_s/W_r; symmetric B |
| 40 | mlp | 40 | exact | 0.038846 | 0.108248 | 0.358863 | 0.025147 | 0.236675 | 0.194395 | 0.221128 | 0.914000 | -0.266296 | 0.642225 | configs/qlib_recent_mlp.yaml; QLib official Alpha158 handler dataset_mode; csi300_aligned; CUDA |
| 40 | master | 40 | exact | 0.059961 | 0.104256 | 0.575136 | 0.064205 | 0.579209 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 40 | hist | 40 | exact | 0.015246 | 0.131462 | 0.115974 | 0.026319 | 0.196209 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 40 | lstm | 40 | exact | 0.029414 | 0.119596 | 0.245945 | 0.035756 | 0.287790 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 40 | lgbm | 40 | exact | 0.029164 | 0.115055 | 0.253475 | 0.014396 | 0.125066 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 55 | fdg_skip32 | 55 | exact | 0.042440 | 0.122064 | 0.347684 | 0.017277 | 0.139353 | 0.286240 | 0.238486 | 1.174860 | -0.320179 | 0.659800 |  |
| 55 | fdg_skip32_transposeB | 55 | exact | 0.042262 | 0.118707 | 0.356021 | 0.016067 | 0.136393 | 0.234634 | 0.239565 | 0.999641 | -0.352854 | 0.644449 | inference only: use B.T; no retraining |
| 55 | fdg_skip32_random | 55 | exact | 0.035360 | 0.120418 | 0.293640 | 0.010563 | 0.097190 | 0.164630 | 0.236666 | 0.762252 | -0.335125 | 0.628610 | fully random graph baseline |
| 55 | fdg_skip32_symshare | 55 | exact | 0.039064 | 0.121170 | 0.322388 | 0.007560 | 0.070611 | 0.220739 | 0.243973 | 0.939631 | -0.379028 | 0.610745 | shared W_s/W_r; symmetric B; final A symmetrized |
| 55 | fdg_skip32_sharew_freeb | 55 | exact | 0.036833 | 0.134826 | 0.273192 | 0.018944 | 0.140487 | 0.270906 | 0.262337 | 1.045273 | -0.382426 | 0.659711 | shared W_s/W_r; free asymmetric B |
| 55 | fdg_skip32_sepw_symb | 55 | exact | 0.042448 | 0.122075 | 0.347722 | 0.016118 | 0.129765 | 0.271149 | 0.242969 | 1.109058 | -0.307551 | 0.653793 | separate W_s/W_r; symmetric B |
| 55 | mlp | 55 | exact | 0.039295 | 0.109055 | 0.360321 | 0.020667 | 0.219189 | 0.262911 | 0.237204 | 1.102560 | -0.333649 | 0.650901 | configs/qlib_recent_mlp.yaml; QLib official Alpha158 handler dataset_mode; csi300_aligned; CUDA |
| 55 | master | 55 | exact | 0.058246 | 0.100668 | 0.578600 | 0.068460 | 0.648245 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 55 | hist | 55 | exact | 0.014620 | 0.139144 | 0.105073 | 0.030714 | 0.219251 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 55 | lstm | 55 | exact | 0.026986 | 0.111667 | 0.241668 | 0.031100 | 0.261516 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 55 | lgbm | 55 | exact | 0.030537 | 0.118485 | 0.257732 | 0.015711 | 0.133018 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 25 | fdg_skip32 | 25 | exact | 0.042343 | 0.116557 | 0.363281 | 0.020612 | 0.204485 | 0.247012 | 0.234554 | 1.058675 | -0.296212 | 0.662514 |  |
| 25 | fdg_skip32_transposeB | 25 | exact | 0.042547 | 0.118731 | 0.358345 | 0.022394 | 0.221668 | 0.260605 | 0.234119 | 1.106519 | -0.298613 | 0.658131 | inference only: use B.T; no retraining |
| 25 | fdg_skip32_random | 25 | exact | 0.040240 | 0.123629 | 0.325491 | 0.006521 | 0.052628 | 0.238763 | 0.246593 | 0.991867 | -0.327973 | 0.585829 | fully random graph baseline |
| 25 | fdg_skip32_symshare | 25 | exact | 0.035313 | 0.119448 | 0.295632 | 0.017314 | 0.147945 | 0.241732 | 0.239805 | 1.023058 | -0.329223 | 0.669566 | shared W_s/W_r; symmetric B; final A symmetrized |
| 25 | fdg_skip32_sharew_freeb | 25 | exact | 0.038329 | 0.121660 | 0.315048 | 0.019060 | 0.154436 | 0.313105 | 0.238474 | 1.261737 | -0.291323 | 0.653393 | shared W_s/W_r; free asymmetric B |
| 25 | fdg_skip32_sepw_symb | 25 | exact | 0.038323 | 0.133309 | 0.287478 | 0.010187 | 0.073625 | 0.239717 | 0.251587 | 0.980187 | -0.308572 | 0.554127 | separate W_s/W_r; symmetric B |
| 25 | mlp | 25 | exact | 0.039714 | 0.109417 | 0.362956 | 0.020062 | 0.204928 | 0.224434 | 0.245991 | 0.946279 | -0.340858 | 0.646652 | configs/qlib_recent_mlp.yaml; QLib official Alpha158 handler dataset_mode; csi300_aligned; CUDA |
| 25 | master | 25 | exact | 0.080676 | 0.099466 | 0.811091 | 0.086074 | 0.827632 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 25 | hist | 25 | exact | 0.014658 | 0.127052 | 0.115370 | 0.026035 | 0.202120 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 25 | lstm | 25 | exact | 0.025151 | 0.131063 | 0.191901 | 0.042643 | 0.333143 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 25 | lgbm | 25 | exact | 0.030297 | 0.117130 | 0.258659 | 0.014262 | 0.122305 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 49 | fdg_skip32 | 49 | exact | 0.042104 | 0.111046 | 0.379161 | 0.027409 | 0.251012 | 0.312797 | 0.236896 | 1.267674 | -0.303640 | 0.654461 |  |
| 49 | fdg_skip32_transposeB | 49 | exact | 0.041711 | 0.114868 | 0.363116 | 0.024148 | 0.211811 | 0.327962 | 0.242926 | 1.289438 | -0.292725 | 0.650590 | inference only: use B.T; no retraining |
| 49 | fdg_skip32_random | 49 | exact | 0.042079 | 0.118137 | 0.356185 | 0.023763 | 0.231274 | 0.285133 | 0.237577 | 1.174803 | -0.333003 | 0.662380 | fully random graph baseline |
| 49 | fdg_skip32_symshare | 49 | exact | 0.029942 | 0.112319 | 0.266579 | 0.020607 | 0.170245 | 0.209907 | 0.237923 | 0.920095 | -0.249403 | 0.743693 | shared W_s/W_r; symmetric B; final A symmetrized |
| 49 | fdg_skip32_sharew_freeb | 49 | exact | 0.039512 | 0.126457 | 0.312456 | 0.014081 | 0.108961 | 0.263468 | 0.249188 | 1.063056 | -0.321102 | 0.676863 | shared W_s/W_r; free asymmetric B |
| 49 | fdg_skip32_sepw_symb | 49 | exact | 0.036915 | 0.114024 | 0.323746 | 0.011836 | 0.106982 | 0.235009 | 0.244201 | 0.986681 | -0.343258 | 0.620067 | separate W_s/W_r; symmetric B |
| 49 | mlp | 49 | exact | 0.039187 | 0.107266 | 0.365326 | 0.016889 | 0.187138 | 0.242762 | 0.231348 | 1.055187 | -0.273217 | 0.627764 | configs/qlib_recent_mlp.yaml; QLib official Alpha158 handler dataset_mode; csi300_aligned; CUDA |
| 49 | master | 49 | exact | 0.046108 | 0.100411 | 0.459192 | 0.053362 | 0.498384 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 49 | hist | 49 | exact | 0.015166 | 0.129625 | 0.116998 | 0.026938 | 0.205668 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 49 | lstm | 49 | exact | 0.030221 | 0.123822 | 0.244068 | 0.039702 | 0.334547 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 49 | lgbm | 49 | exact | 0.029961 | 0.115463 | 0.259486 | 0.015857 | 0.137032 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 17 | fdg_skip32 | 17 | exact | 0.042026 | 0.128637 | 0.326699 | 0.020094 | 0.152543 | 0.286946 | 0.259514 | 1.102187 | -0.330959 | 0.592503 |  |
| 17 | fdg_skip32_transposeB | 17 | exact | 0.037947 | 0.131671 | 0.288197 | 0.019620 | 0.144828 | 0.258571 | 0.259308 | 1.016814 | -0.338337 | 0.574394 | inference only: use B.T; no retraining |
| 17 | fdg_skip32_random | 17 | exact | 0.039421 | 0.127019 | 0.310358 | 0.022023 | 0.174521 | 0.242191 | 0.254326 | 0.979801 | -0.318723 | 0.635729 | fully random graph baseline |
| 17 | fdg_skip32_symshare | 17 | exact | 0.037306 | 0.112080 | 0.332849 | 0.025612 | 0.221254 | 0.343275 | 0.235092 | 1.373165 | -0.286676 | 0.660489 | shared W_s/W_r; symmetric B; final A symmetrized |
| 17 | fdg_skip32_sharew_freeb | 17 | exact | 0.034036 | 0.127853 | 0.266215 | 0.017293 | 0.135657 | 0.234964 | 0.250746 | 0.967193 | -0.349544 | 0.622803 | shared W_s/W_r; free asymmetric B |
| 17 | fdg_skip32_sepw_symb | 17 | exact | 0.033828 | 0.120545 | 0.280625 | 0.019243 | 0.162751 | 0.307157 | 0.251966 | 1.189111 | -0.320083 | 0.634594 | separate W_s/W_r; symmetric B |
| 17 | mlp | 17 | exact | 0.039475 | 0.120103 | 0.328678 | 0.019077 | 0.163901 | 0.201875 | 0.255661 | 0.847253 | -0.295003 | 0.561980 | configs/qlib_recent_mlp.yaml; QLib official Alpha158 handler dataset_mode; csi300_aligned; CUDA |
| 17 | master | 17 | exact | 0.063312 | 0.105265 | 0.601457 | 0.067066 | 0.583323 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 17 | hist | 17 | exact | 0.014938 | 0.136997 | 0.109039 | 0.029893 | 0.215136 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 17 | lstm | 17 | exact | 0.024153 | 0.133055 | 0.181530 | 0.038838 | 0.301563 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 17 | lgbm | 17 | exact | 0.033539 | 0.114321 | 0.293377 | 0.016393 | 0.142981 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 79 | fdg_skip32 | 79 | exact | 0.041920 | 0.120753 | 0.347156 | 0.020209 | 0.159452 | 0.259682 | 0.226037 | 1.134468 | -0.235087 | 0.650790 |  |
| 79 | fdg_skip32_transposeB | 79 | exact | 0.040446 | 0.119998 | 0.337057 | 0.016504 | 0.132085 | 0.274556 | 0.231194 | 1.165081 | -0.256183 | 0.644605 | inference only: use B.T; no retraining |
| 79 | fdg_skip32_random | 79 | exact | 0.046521 | 0.118749 | 0.391762 | 0.017068 | 0.157724 | 0.254082 | 0.246157 | 1.042810 | -0.308800 | 0.631479 | fully random graph baseline |
| 79 | fdg_skip32_symshare | 79 | exact | 0.038961 | 0.127343 | 0.305955 | 0.014902 | 0.121073 | 0.231116 | 0.243157 | 0.976992 | -0.374448 | 0.645451 | shared W_s/W_r; symmetric B; final A symmetrized |
| 79 | fdg_skip32_sharew_freeb | 79 | exact | 0.036180 | 0.134153 | 0.269692 | 0.013804 | 0.099643 | 0.247520 | 0.252521 | 1.002286 | -0.365471 | 0.655751 | shared W_s/W_r; free asymmetric B |
| 79 | fdg_skip32_sepw_symb | 79 | exact | 0.038509 | 0.120762 | 0.318880 | 0.014353 | 0.117742 | 0.246140 | 0.237774 | 1.044527 | -0.318626 | 0.673571 | separate W_s/W_r; symmetric B |
| 79 | mlp | 79 | exact | 0.041990 | 0.116965 | 0.358994 | 0.019606 | 0.170320 | 0.248531 | 0.248686 | 1.017138 | -0.334080 | 0.657130 | configs/qlib_recent_mlp.yaml; QLib official Alpha158 handler dataset_mode; csi300_aligned; CUDA |
| 79 | master | 79 | exact | 0.110528 | 0.096569 | 1.144546 | 0.116929 | 1.166144 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 79 | hist | 79 | exact | 0.016285 | 0.128474 | 0.126755 | 0.027483 | 0.212614 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 79 | lstm | 79 | exact | 0.025231 | 0.131207 | 0.192298 | 0.038375 | 0.291974 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| 79 | lgbm | 79 | exact | 0.030267 | 0.115803 | 0.261369 | 0.014629 | 0.125999 | 0.006144 | 0.181775 | 0.034782 | -0.388799 | 0.000000 | portfolio metrics from qworkflow mlruns excess_return_without_cost |

## Summary Across Top10 Seeds

| model | n | exact | proxy | missing | IC_mean_mean | IC_mean_var | IC_mean_std | IC_std_mean | IC_std_var | IC_std_std | ICIR_mean | ICIR_var | ICIR_std | RankIC_mean_mean | RankIC_mean_var | RankIC_mean_std | RankICIR_mean | RankICIR_var | RankICIR_std | annual_return_mean | annual_return_var | annual_return_std | annual_vol_mean | annual_vol_var | annual_vol_std | sharpe_mean | sharpe_var | sharpe_std | max_drawdown_mean | max_drawdown_var | max_drawdown_std | turnover_mean | turnover_var | turnover_std | proxy_note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fdg_skip32 | 10 | 10 | 0 | 0 | 0.042900 | 0.000001 | 0.001120 | 0.119894 | 0.000064 | 0.008011 | 0.359015 | 0.000468 | 0.021635 | 0.021147 | 0.000022 | 0.004688 | 0.176346 | 0.001947 | 0.044125 | 0.273930 | 0.000340 | 0.018435 | 0.237021 | 0.000218 | 0.014775 | 1.142748 | 0.005727 | 0.075678 | -0.293136 | 0.001553 | 0.039414 | 0.644354 | 0.000563 | 0.023734 |  |
| fdg_skip32_transposeB | 10 | 10 | 0 | 0 | 0.042025 | 0.000003 | 0.001751 | 0.118229 | 0.000050 | 0.007082 | 0.357051 | 0.000993 | 0.031513 | 0.019301 | 0.000021 | 0.004613 | 0.165350 | 0.002333 | 0.048300 | 0.265371 | 0.000743 | 0.027263 | 0.237599 | 0.000131 | 0.011459 | 1.111324 | 0.010421 | 0.102081 | -0.297521 | 0.002552 | 0.050519 | 0.636407 | 0.000643 | 0.025363 |  |
| fdg_skip32_random | 10 | 10 | 0 | 0 | 0.040715 | 0.000009 | 0.003047 | 0.119044 | 0.000066 | 0.008140 | 0.343258 | 0.001041 | 0.032261 | 0.017630 | 0.000060 | 0.007715 | 0.159246 | 0.005089 | 0.071340 | 0.234623 | 0.001135 | 0.033687 | 0.239321 | 0.000121 | 0.011003 | 0.999709 | 0.013132 | 0.114597 | -0.313923 | 0.000397 | 0.019919 | 0.640565 | 0.000867 | 0.029449 |  |
| fdg_skip32_symshare | 10 | 10 | 0 | 0 | 0.036940 | 0.000009 | 0.003056 | 0.118858 | 0.000052 | 0.007236 | 0.311083 | 0.000549 | 0.023439 | 0.017783 | 0.000075 | 0.008677 | 0.155614 | 0.006746 | 0.082132 | 0.251963 | 0.002196 | 0.046860 | 0.239708 | 0.000116 | 0.010753 | 1.055875 | 0.024321 | 0.155953 | -0.321857 | 0.002025 | 0.044998 | 0.650256 | 0.001614 | 0.040180 |  |
| fdg_skip32_sharew_freeb | 10 | 10 | 0 | 0 | 0.037494 | 0.000004 | 0.001949 | 0.128341 | 0.000045 | 0.006695 | 0.293048 | 0.000600 | 0.024492 | 0.016647 | 0.000023 | 0.004820 | 0.129227 | 0.001914 | 0.043747 | 0.267413 | 0.000859 | 0.029305 | 0.247817 | 0.000145 | 0.012046 | 1.082548 | 0.012320 | 0.110995 | -0.328271 | 0.001577 | 0.039711 | 0.646407 | 0.001061 | 0.032577 |  |
| fdg_skip32_sepw_symb | 10 | 10 | 0 | 0 | 0.039294 | 0.000009 | 0.003048 | 0.118038 | 0.000094 | 0.009711 | 0.334408 | 0.001014 | 0.031845 | 0.015738 | 0.000016 | 0.003996 | 0.133683 | 0.001436 | 0.037890 | 0.264373 | 0.001006 | 0.031719 | 0.240705 | 0.000156 | 0.012470 | 1.098108 | 0.017767 | 0.133294 | -0.315050 | 0.001397 | 0.037382 | 0.638456 | 0.001363 | 0.036915 |  |
| mlp | 10 | 10 | 0 | 0 | 0.039212 | 0.000002 | 0.001254 | 0.112262 | 0.000021 | 0.004567 | 0.349680 | 0.000222 | 0.014916 | 0.018878 | 0.000014 | 0.003678 | 0.182352 | 0.001315 | 0.036262 | 0.231610 | 0.000569 | 0.023862 | 0.238996 | 0.000177 | 0.013302 | 0.991963 | 0.006518 | 0.080735 | -0.310619 | 0.000962 | 0.031022 | 0.632465 | 0.000911 | 0.030184 |  |
| master | 10 | 10 | 0 | 0 | 0.073823 | 0.000508 | 0.022536 | 0.102539 | 0.000021 | 0.004557 | 0.726234 | 0.057961 | 0.240751 | 0.081123 | 0.000509 | 0.022556 | 0.758054 | 0.057268 | 0.239307 | 0.006144 | 0.000000 | 0.000000 | 0.181775 | 0.000000 | 0.000000 | 0.034782 | 0.000000 | 0.000000 | -0.388799 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |
| hist | 10 | 10 | 0 | 0 | 0.015185 | 0.000000 | 0.000549 | 0.132457 | 0.000018 | 0.004226 | 0.114788 | 0.000041 | 0.006406 | 0.028095 | 0.000003 | 0.001605 | 0.209738 | 0.000065 | 0.008038 | 0.006144 | 0.000000 | 0.000000 | 0.181775 | 0.000000 | 0.000000 | 0.034782 | 0.000000 | 0.000000 | -0.388799 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |
| lstm | 10 | 10 | 0 | 0 | 0.026398 | 0.000005 | 0.002310 | 0.126633 | 0.000068 | 0.008245 | 0.209902 | 0.000805 | 0.028381 | 0.038756 | 0.000015 | 0.003844 | 0.305668 | 0.000529 | 0.023007 | 0.006144 | 0.000000 | 0.000000 | 0.181775 | 0.000000 | 0.000000 | 0.034782 | 0.000000 | 0.000000 | -0.388799 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |
| lgbm | 10 | 10 | 0 | 0 | 0.030884 | 0.000003 | 0.001753 | 0.115626 | 0.000002 | 0.001313 | 0.267200 | 0.000284 | 0.016850 | 0.015358 | 0.000001 | 0.001029 | 0.132754 | 0.000085 | 0.009199 | 0.006144 | 0.000000 | 0.000000 | 0.181775 | 0.000000 | 0.000000 | 0.034782 | 0.000000 | 0.000000 | -0.388799 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |

## Model Notes

| model | note |
| --- | --- |
| fdg_skip32 | - |
| fdg_skip32_transposeB | inference only: use B.T; no retraining |
| fdg_skip32_random | fully random graph baseline |
| fdg_skip32_symshare | shared W_s/W_r; symmetric B; final A symmetrized |
| fdg_skip32_sharew_freeb | shared W_s/W_r; free asymmetric B |
| fdg_skip32_sepw_symb | separate W_s/W_r; symmetric B |
| mlp | configs/qlib_recent_mlp.yaml; QLib official Alpha158 handler dataset_mode; csi300_aligned; CUDA |
| master | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| hist | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| lstm | portfolio metrics from qworkflow mlruns excess_return_without_cost |
| lgbm | portfolio metrics from qworkflow mlruns excess_return_without_cost |
