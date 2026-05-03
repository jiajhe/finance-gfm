# GAT Rescue Diagnostics - 2026-05-03

Scope: CSI300 recent, seed 512, same Alpha158/Qlib data path and fixed per-datetime sampler.

| variant | IC | ICIR | RankIC | RankICIR | note |
| --- | ---: | ---: | ---: | ---: | --- |
| loss-stop, no pretrain | 0.026689 | 0.260197 | 0.015226 | 0.139445 | original completed GAT seed512 |
| RankICIR-stop, no pretrain | 0.020953 | 0.218247 | 0.018545 | 0.208734 | RankIC improves, IC drops |
| ICIR-stop, no pretrain | 0.022793 | 0.250788 | 0.014030 | 0.171028 | valid ICIR improves, test IC still lower |
| GRU pretrain + loss-stop | 0.008561 | 0.069756 | -0.009774 | -0.071078 | using same TSDatasetH daily-sampler pretrain |

Conclusion: the low GAT IC is not rescued by IC/RankIC checkpoint selection or by a GRU base pretrain. The current Qlib GAT should be treated as a weak external baseline under this experiment protocol, not as a model/config worth more top10 GPU time.

Notes:
- The Qlib official GRU pretrain class cannot directly consume `TSDatasetH` (`TSDataSampler has no dropna`), so this diagnostic adds a thin `GRUSeqFixedSampler` wrapper that trains GRU on the same daily batches as GAT.
- Portfolio metrics from these Qlib GAT runs are still suspicious and should not be used yet; IC/RankIC are recomputed directly from `pred.pkl` and `label.pkl`.
