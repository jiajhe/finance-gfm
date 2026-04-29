# FDG Market-Grammar Transfer + HIST/MASTER Diagnostics - 2026-04-30

## Scope

- Cross-period: target is FDG skip32 csi300 recent top10 seed checkpoint; replace only fdg.B with same-seed csi300 official fdg.B, then evaluate on recent test.
- Cross-index: native csi100.txt and csi500.txt do not cover full recent 2022-2025, so the clean native-index test is official split. Target models are trained on csi100/csi500 official top10 seeds; replace only fdg.B with same-seed csi300 official fdg.B, then evaluate on target official test.
- All transfer rows keep target model encoder/head/S/R weights fixed and change only fdg.B. This is a strict B-only transfer test, but latent-axis alignment is still seed/training dependent.

## Diagnostic Notes

| item | finding |
| --- | --- |
| MASTER recent IC verification | pred.pkl/label.pkl recomputation exactly matches JSON; seed512 IC=0.100100. qworkflow portfolio metrics are not reliable here. |
| MASTER beta/market diagnostic | market=csi300 beta=5 seed512 rerun IC=0.100269 vs original 0.100100, so beta flag is not the cause. |
| HIST stock_index mapping | Qlib instruments are lower-case but stock_index.npy keys are upper-case; direct match 0/673, lower-case match 657/673. |
| HIST fixed-index seed512 | lower-case stock_index rerun IC=0.015915 vs original 0.014837, only a small improvement; HIST weakness is not fully explained by the mapping bug. |
| Recent split | train 2010-01-01..2020-12-31, valid 2021-01-01..2021-12-31, test 2022-01-01..2025-09-17 config cap, actual trading dates through 2025-09-16. |

## Summary Deltas

| target | metric | own mean | transfer mean | delta mean | delta std | delta min | delta max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| csi300_recent | IC_mean | 0.042900 | 0.040726 | -0.002174 | 0.001724 | -0.004889 | 0.000425 |
| csi300_recent | ICIR | 0.359015 | 0.348846 | -0.010169 | 0.014657 | -0.037268 | 0.015084 |
| csi300_recent | RankIC_mean | 0.021147 | 0.017896 | -0.003251 | 0.003186 | -0.009963 | 0.000735 |
| csi300_recent | RankICIR | 0.176346 | 0.153866 | -0.022481 | 0.024009 | -0.057081 | 0.013254 |
| csi300_recent | sharpe | 1.142748 | 1.080488 | -0.062260 | 0.078957 | -0.190626 | 0.075102 |
| csi100_official | IC_mean | 0.030344 | 0.030166 | -0.000178 | 0.001767 | -0.002121 | 0.003605 |
| csi100_official | ICIR | 0.165633 | 0.167492 | 0.001859 | 0.010388 | -0.013931 | 0.018567 |
| csi100_official | RankIC_mean | 0.037919 | 0.036950 | -0.000968 | 0.002180 | -0.004113 | 0.002164 |
| csi100_official | RankICIR | 0.218894 | 0.215460 | -0.003434 | 0.013701 | -0.020687 | 0.013300 |
| csi100_official | sharpe | 0.890586 | 0.883982 | -0.006604 | 0.047116 | -0.061956 | 0.095316 |
| csi500_official | IC_mean | 0.041128 | 0.038499 | -0.002629 | 0.003959 | -0.012069 | 0.001534 |
| csi500_official | ICIR | 0.374587 | 0.369188 | -0.005399 | 0.046795 | -0.109162 | 0.056934 |
| csi500_official | RankIC_mean | 0.040987 | 0.037865 | -0.003122 | 0.003998 | -0.008750 | 0.003271 |
| csi500_official | RankICIR | 0.377010 | 0.362960 | -0.014050 | 0.043059 | -0.072527 | 0.048480 |
| csi500_official | sharpe | 1.160074 | 1.083311 | -0.076762 | 0.118261 | -0.262352 | 0.076260 |

## Cross-Period Raw: csi300 official B -> csi300 recent target

| seed | own_IC | transfer_IC | delta_IC | own_RankIC | transfer_RankIC | delta_RankIC | own_sharpe | transfer_sharpe | delta_sharpe | B_cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0.045677 | 0.040787 | -0.004889 | 0.027799 | 0.017836 | -0.009963 | 1.082962 | 0.987307 | -0.095656 | 0.925954 |
| 23 | 0.043556 | 0.040910 | -0.002646 | 0.013485 | 0.009044 | -0.004441 | 1.058127 | 0.914579 | -0.143547 | 0.953754 |
| 37 | 0.043249 | 0.039358 | -0.003891 | 0.025994 | 0.022607 | -0.003387 | 1.125344 | 1.136234 | 0.010890 | 0.950378 |
| 5 | 0.043116 | 0.042437 | -0.000679 | 0.021301 | 0.020339 | -0.000963 | 1.264542 | 1.214263 | -0.050279 | 0.985597 |
| 40 | 0.042568 | 0.039075 | -0.003494 | 0.017289 | 0.011463 | -0.005825 | 1.158640 | 0.968014 | -0.190626 | 0.930594 |
| 55 | 0.042440 | 0.042864 | 0.000425 | 0.017277 | 0.017532 | 0.000255 | 1.174860 | 1.048237 | -0.126623 | 0.959168 |
| 25 | 0.042343 | 0.040932 | -0.001411 | 0.020612 | 0.021347 | 0.000735 | 1.058675 | 1.006760 | -0.051915 | 0.966601 |
| 49 | 0.042104 | 0.041510 | -0.000594 | 0.027409 | 0.024690 | -0.002719 | 1.267674 | 1.258729 | -0.008945 | 0.961765 |
| 17 | 0.042026 | 0.038655 | -0.003370 | 0.020094 | 0.018498 | -0.001596 | 1.102187 | 1.061184 | -0.041002 | 0.951269 |
| 79 | 0.041920 | 0.040733 | -0.001187 | 0.020209 | 0.015607 | -0.004602 | 1.134468 | 1.209570 | 0.075102 | 0.973592 |

## Cross-Index Raw: csi300 official B -> csi100 official target

| seed | own_IC | transfer_IC | delta_IC | own_RankIC | transfer_RankIC | delta_RankIC | own_sharpe | transfer_sharpe | delta_sharpe | B_cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0.024205 | 0.023988 | -0.000217 | 0.032862 | 0.028749 | -0.004113 | 0.828123 | 0.785726 | -0.042397 | 0.945654 |
| 23 | 0.029638 | 0.031478 | 0.001840 | 0.032305 | 0.034469 | 0.002164 | 0.973930 | 0.924545 | -0.049385 | 0.953683 |
| 37 | 0.031171 | 0.034776 | 0.003605 | 0.042727 | 0.044656 | 0.001929 | 0.887646 | 0.922196 | 0.034551 | 0.972167 |
| 5 | 0.023881 | 0.022116 | -0.001765 | 0.038104 | 0.036104 | -0.002000 | 0.858192 | 0.839788 | -0.018405 | 0.993745 |
| 40 | 0.028999 | 0.026878 | -0.002121 | 0.032735 | 0.030502 | -0.002233 | 0.729539 | 0.716309 | -0.013230 | 0.924674 |
| 55 | 0.031686 | 0.031049 | -0.000637 | 0.037498 | 0.035856 | -0.001641 | 0.937864 | 0.875908 | -0.061956 | 0.951191 |
| 25 | 0.030915 | 0.031365 | 0.000450 | 0.042753 | 0.042800 | 0.000047 | 0.758660 | 0.853976 | 0.095316 | 0.977918 |
| 49 | 0.031155 | 0.029440 | -0.001715 | 0.038628 | 0.034739 | -0.003889 | 0.990480 | 0.959672 | -0.030808 | 0.947056 |
| 17 | 0.032868 | 0.032456 | -0.000412 | 0.038069 | 0.038169 | 0.000100 | 0.954651 | 0.950761 | -0.003891 | 0.965209 |
| 79 | 0.038922 | 0.038116 | -0.000806 | 0.043507 | 0.043459 | -0.000048 | 0.986776 | 1.010940 | 0.024164 | 0.972813 |

## Cross-Index Raw: csi300 official B -> csi500 official target

| seed | own_IC | transfer_IC | delta_IC | own_RankIC | transfer_RankIC | delta_RankIC | own_sharpe | transfer_sharpe | delta_sharpe | B_cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0.041346 | 0.029276 | -0.012069 | 0.043706 | 0.034956 | -0.008750 | 1.039778 | 0.777426 | -0.262352 | 0.918838 |
| 23 | 0.038452 | 0.034529 | -0.003922 | 0.038562 | 0.032419 | -0.006142 | 1.196609 | 0.967252 | -0.229357 | 0.945601 |
| 37 | 0.042990 | 0.041641 | -0.001349 | 0.045857 | 0.045605 | -0.000252 | 1.239894 | 1.144067 | -0.095827 | 0.955603 |
| 5 | 0.043599 | 0.043062 | -0.000537 | 0.041951 | 0.043700 | 0.001748 | 1.177439 | 1.130399 | -0.047040 | 0.972282 |
| 40 | 0.046552 | 0.048086 | 0.001534 | 0.043024 | 0.040163 | -0.002861 | 1.190062 | 1.206787 | 0.016725 | 0.918300 |
| 55 | 0.043394 | 0.043249 | -0.000145 | 0.042911 | 0.040248 | -0.002663 | 1.493203 | 1.356274 | -0.136929 | 0.932706 |
| 25 | 0.034597 | 0.029453 | -0.005144 | 0.036101 | 0.030304 | -0.005796 | 0.852688 | 0.918092 | 0.065404 | 0.938910 |
| 49 | 0.041352 | 0.038480 | -0.002871 | 0.043686 | 0.035918 | -0.007768 | 1.241329 | 1.090220 | -0.151109 | 0.952714 |
| 17 | 0.043404 | 0.044555 | 0.001151 | 0.040266 | 0.043537 | 0.003271 | 1.253269 | 1.329529 | 0.076260 | 0.963622 |
| 79 | 0.035592 | 0.032655 | -0.002937 | 0.033808 | 0.031799 | -0.002009 | 0.916468 | 0.913068 | -0.003400 | 0.937750 |

## Artifact Paths

- Raw transfer JSONs: reports/fdg_market_grammar_transfer_20260430/tables/
- Cross-period and cross-index raw CSV: reports/fdg_market_grammar_transfer_20260430/summaries/fdg_market_grammar_transfer_raw_20260430.csv
- Summary CSV: reports/fdg_market_grammar_transfer_20260430/summaries/fdg_market_grammar_transfer_summary_20260430.csv
- Delta CSV: reports/fdg_market_grammar_transfer_20260430/summaries/fdg_market_grammar_transfer_deltas_20260430.csv
- Target checkpoints are kept local under results/fdg_cross_index_*_official_20260430/ckpts/ and are not included in this report package.
