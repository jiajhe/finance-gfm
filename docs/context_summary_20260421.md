# Context Summary (2026-04-21)

This document summarizes the main technical context, experiment results, and current conclusions from the recent development cycle on `finance-gfm`.

## Goal

The active goal is to turn `FDG` into a competitive stock-prediction graph model under a modern Qlib pipeline, with a specific focus on:

- matching or exceeding strong baselines on the standard Qlib split
- improving generalization on a harder recent split covering `2022-01-01 ~ 2025-12-31`
- exploring whether FDG can become a reusable graph component that plugs into stronger backbones

## Core FDG Design

The current standalone `FDG` pipeline is:

1. Input one trading-day cross section `X_t in R^{N_t x d}`
2. Optional feature bottleneck / encoder
3. Learn sender and receiver assignments
   - `S_t = softmax(Z_t W_s / tau)`
   - `R_t = softmax(Z_t W_r / tau)`
4. Build a factorized directed graph
   - `A_t = RowNorm(S_t B R_t^T)`
5. Run graph message passing and prediction head
   - `H_t = A_t (Z_t W_v) + Z_t W_skip`
   - `y_hat_t = MLP(H_t)`

Important code:

- model entry: `models/__init__.py`
- FDG core: `models/fdg.py`
- graph head: `models/gnn_head.py`
- bottleneck / residual blocks: `models/blocks.py`

## Dataset And Evaluation Protocol

We aligned the main training/evaluation path to the Qlib official `Alpha158 + DatasetH + processors` pipeline.

Two splits are now treated as mandatory for future experiments:

- `official split`
  - train `2008-01-01 ~ 2014-12-31`
  - valid `2015-01-01 ~ 2016-12-31`
  - test `2017-01-01 ~ 2020-08-01`
- `recent split`
  - train `2010-01-01 ~ 2020-12-31`
  - valid `2021-01-01 ~ 2021-12-31`
  - test `2022-01-01 ~ 2025-12-31`

Main reference config files:

- `configs/qlib_official_fdg.yaml`
- `configs/qlib_recent_fdg.yaml`
- `configs/qlib_official_mlp.yaml`
- `configs/qlib_recent_mlp.yaml`

## Main Baselines

The current benchmark summary is maintained locally in `results/tables/benchmark_compare_20260421.md` and is ignored by git. The most important numbers are copied here.

### Official split

- Qlib MLP: `IC 0.04329`
- Qlib LightGBM: `IC 0.04703`
- Our MLP: `IC 0.03968`
- Our FDG: `IC 0.04603`
- MASTER-quick: `IC 0.03363`
- HIST-Alpha158-quick: `IC 0.03102`

Takeaway:

- standalone FDG is competitive on the standard split
- FDG already reaches roughly the same level as strong Qlib tabular baselines on this split

### Recent split

- Qlib MLP: `IC 0.01466`
- Qlib LightGBM: `IC 0.01195`
- Our MLP: `IC 0.01653`
- Our FDG: `IC 0.02092`
- MASTER-quick: `IC 0.02847`
- HIST-Alpha158-quick: `IC 0.02193`

Takeaway:

- recent split is much harder than the official split
- our standalone FDG beats Qlib MLP and LightGBM on the recent split
- the strongest same-split reference so far is still `MASTER-quick`

## Paths Already Explored

### 1. Preprocessing transfer from MASTER

We tried feeding FDG with MASTER-style preprocessing. This did not help.

- official: `FDG + MASTER prep = 0.02739`, much worse than baseline FDG
- recent: `FDG + MASTER prep = 0.01835`, also worse than baseline FDG

Conclusion:

- MASTER's strength is not simply its preprocessing
- more likely it comes from the temporal backbone itself

### 2. MASTER + FDG residual

We implemented a first residual integration of FDG on top of MASTER.

- official: `0.03144`
- recent: `0.02664`

Conclusion:

- this first residual version did not beat raw MASTER
- directly attaching FDG at the end of MASTER is not enough

Relevant files:

- `configs/master_fdg_csi300_official_quick.yaml`
- `configs/master_fdg_csi300_recent_quick.yaml`

### 3. FDG structure variants

Several standalone FDG directions were screened:

- temporal FDG
- regularized FDG
- sparse rolling FDG
- slow graph / fast value
- smoothing and `IC/WPCC` hybrid loss

The pattern was consistent:

- some variants improved validation IC
- none of them beat the original standalone FDG on recent test

Conclusion:

- the current bottleneck is not lack of expressiveness
- the bottleneck is generalization under regime shift

Relevant files:

- `models/fdg_temporal.py`
- `models/fdg_regularized.py`
- `models/fdg_sparse.py`
- `models/fdg_slowfast.py`

### 4. Bottleneck ablations

The current best standalone FDG on recent split still uses the original bottleneck:

- baseline: `157 -> 64 -> 157`

Compared variants:

- `identity`
- `157 -> 32 -> 157`
- `157 -> 256 -> 157`
- `157 -> 64 -> 32 -> 64 -> 157`

Key result:

- recent best remains the original `157 -> 64 -> 157`
- official best is slightly better with `157 -> 256 -> 157`

Interpretation:

- higher-capacity encoders help the old regime more than the new regime
- recent split prefers a more conservative feature encoder

## Generalization Diagnosis

A dedicated recent-gap diagnosis was run and stored locally in:

- `results/tables/recent_gap_diagnostics_20260421.md`
- `results/tables/qlib_recent_feature_drift.json`

### Main finding

The major gap is:

- `early val -> late val`, not `late val -> test`

That means the dominant issue is time drift, not simply overfitting to the 2021 validation slice.

Five-seed summary:

- FDG-64
  - early val IC: `0.05536 +/- 0.00565`
  - late val IC: `0.02330 +/- 0.00274`
  - test IC: `0.01835 +/- 0.00131`
- FDG-bneck32
  - early val IC: `0.05380 +/- 0.00517`
  - late val IC: `0.02201 +/- 0.00650`
  - test IC: `0.01829 +/- 0.00082`

Implications:

- single-seed ranking differences such as `0.0209 vs 0.0191` should be treated cautiously
- future experiments should emphasize training regime and time-robustness rather than blind architecture growth

### Feature drift

The highest-drift features were:

- `IMXD60`
- `CNTN60`
- `CNTD60`
- `CORR30`
- `CORR60`
- `VSTD10`
- `VSTD20`

These are mostly regime-sensitive momentum / correlation / volatility style features.

## Drift-Mitigation Experiments

Three targeted interventions were run on top of the current best standalone FDG:

1. early stop on late validation only
2. rank-transform the 7 highest-PSI features per day
3. drop those 7 features entirely

Configs:

- `configs/qlib_recent_fdg_lateval.yaml`
- `configs/qlib_recent_fdg_lateval_rankpsi.yaml`
- `configs/qlib_recent_fdg_lateval_droppsi.yaml`
- `configs/qlib_official_fdg_lateval.yaml`
- `configs/qlib_official_fdg_lateval_rankpsi.yaml`
- `configs/qlib_official_fdg_lateval_droppsi.yaml`

Result:

- `late val` alone did not improve recent test
- `rank PSI` hurt recent test
- `drop PSI` hurt more

Conclusion:

- those drift-heavy features are not pure noise
- they still contain signal
- simple rank or hard removal is too crude

## 2026-04-22 Follow-Up: Stable-Graph Branch

A second round of recent-split diagnostics was run after adding support for:

- recency-weighted training loss
- train-window truncation
- separate graph/value feature paths for FDG
- graph-only input transforms
- optional graph gate

The diagnostic artifacts are stored locally under `results/tables/*.json` and remain gitignored.

### Recent single-seed follow-up results

- `recency504`
  - best epoch: `35`
  - early val IC: `0.03904`
  - late val IC: `0.00879`
  - test IC: `0.01380`
- `window5y`
  - best epoch: `20`
  - early val IC: `0.04661`
  - late val IC: `0.01884`
  - test IC: `0.01520`
- `stablegraph_rank`
  - best epoch: `51`
  - early val IC: `0.05662`
  - late val IC: `0.01829`
  - test IC: `0.02704`
  - test RankIC: `0.02874`
- `stablegraph_rank_gate`
  - best epoch: `17`
  - early val IC: `0.04860`
  - late val IC: `0.02239`
  - test IC: `0.01640`
  - test RankIC: `0.01456`
- `stablegraph_rank_w5y`
  - best epoch: `39`
  - early val IC: `0.04604`
  - late val IC: `0.02126`
  - test IC: `0.01418`
  - test RankIC: `0.01212`

### Main takeaway from the follow-up

The strongest gain came from changing who is allowed to control graph construction, not from changing the loss or shortening the train window.

The best follow-up variant was:

- keep full features for value / prediction
- exclude the 7 highest-drift features from graph assignment
- apply daily rank normalization only on the graph branch

This `stablegraph_rank` variant reached `test IC 0.02704`, which is:

- clearly above the previous standalone FDG recent baseline `0.02092`
- very close to the current `MASTER-quick` recent reference `0.02847`

### Negative follow-up findings

- recency-weighted loss alone hurt recent test materially
- truncating the training window to 5 years did not help once the stable-graph branch was already in place
- graph gate improved `late val` in one run, but hurt test noticeably

This is an important caution:

- `late val` improvements alone are not reliable
- under the current diagnostics script, checkpoint selection still uses `early_valid_ICIR`
- some variants that looked healthier on `late val` still generalized worse on test

## Practical Current Conclusion

The repo is now in a much clearer state than before:

- the standard benchmark story is established
- the recent split story is established
- the main failure mode has been diagnosed as time drift
- several obvious FDG architectural branches have already been screened
- the strongest current recent-split single-seed candidate is now `stablegraph_rank`

Right now the most promising next directions are likely:

1. multi-seed validation of `stablegraph_rank`
2. graph-branch normalization ablations such as `rank` vs `robust_zscore` vs `none`
3. walk-forward or rolling checkpoint selection that is more aligned with late-regime performance
4. stronger temporal backbones with a more native FDG integration, instead of a simple residual add-on

## Notes On Artifacts

- Raw experiment artifacts under `results/` are intentionally ignored by git.
- The repository push therefore focuses on code, configs, tests, and this written summary.
- Local result tables remain the source of truth for detailed numeric comparisons during active development.
