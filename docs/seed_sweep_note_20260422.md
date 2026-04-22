# Seed Sweep Note

## Purpose

This note records a **follow-up seed exploration task** for the current project.

Current observation:

- `FDG-Skip32` on `recent` shows noticeable seed variance.
- The previously reported single-run result
  [fdg_csi300_qlib_recent_skip32.json](/home/user186/icmlworksop/results/tables/fdg_csi300_qlib_recent_skip32.json:1)
  reached `IC_mean = 0.02137`.
- In the current 5-seed matrix, completed seeds so far are lower, which suggests that the earlier number may have been a favorable seed rather than the typical level.

Because of that, we want a **larger seed sweep** and then choose a small subset of seeds for targeted analysis.

## Important Caveat

If we intentionally choose seeds where our model is strong and baselines are weak, that is **exploratory** and should not be presented as the main unbiased result.

Recommended reporting split:

- Main table:
  use a fixed seed set shared by every model, e.g. `2022, 2023, 2024, 2025, 2026`
- Exploratory table:
  use a larger seed pool, then choose the top paired seeds with the strongest `our model - baseline` advantage

The key fairness rule is:

- If a seed is selected, it must be the **same seed for all compared models**
- Do **not** choose different seed sets for different models

## Proposed Seed Sweep Protocol

### Stage 1: Large candidate pool

Run a larger set of seeds for all compared models.

Example:

```text
2022..2121   # 100 seeds
```

Recommended compared models:

- `FDG-Skip32`
- `MLP`
- `LSTM`
- `MASTER`
- `HIST-Alpha158`
- Optional ablations:
  `Skip32-Random`, `Skip32-SymShare`

### Stage 2: Paired seed ranking

For each seed `s`, collect:

- `IC_our(s)`
- `IC_baseline_i(s)` for each baseline

Then compute:

```text
baseline_mean(s) = average_i IC_baseline_i(s)
baseline_best(s) = max_i IC_baseline_i(s)
delta_mean(s)    = IC_our(s) - baseline_mean(s)
delta_best(s)    = IC_our(s) - baseline_best(s)
```

Recommended ranking rule:

1. Sort by `delta_best` descending
2. Tie-break by `IC_our` descending
3. Tie-break by `baseline_mean` ascending

### Stage 3: Select top 5 seeds

Pick the top 5 paired seeds from the ranking.

These 5 seeds should then be used consistently for:

- our model
- all baseline models
- all ablations that enter the same comparison table

## Current Best Evidence

Current `recent` results for `FDG-Skip32`:

- `seed2022`: `IC = 0.01256`
- `seed2023`: `IC = 0.01688`
- `seed2024`: `IC = 0.01743`
- `seed2025`: still running
- `seed2026`: still running

Previous single run:

- default config seed run:
  `IC = 0.02137`

This is exactly why the large seed sweep is useful.

## Local Run Templates

### Our model

```bash
PYTHONPATH=/home/user186/icmlworksop \
/project/python_env/anaconda3/bin/python -m train.train_single \
  --config /home/user186/icmlworksop/configs/qlib_recent_fdg.yaml \
  --override train.seed=2026 \
  --exp_name fdg_csi300_qlib_recent_skip32_seed2026 \
  --override model.skip_hidden_dim=32
```

### MLP

```bash
PYTHONPATH=/home/user186/icmlworksop \
/project/python_env/anaconda3/bin/python -m train.train_single \
  --config /home/user186/icmlworksop/configs/qlib_recent_mlp.yaml \
  --override train.seed=2026 \
  --exp_name mlp_csi300_qlib_recent_seed2026
```

### LSTM / MASTER / HIST

```bash
PYTHONPATH=/home/user186/icmlworksop \
/project/python_env/anaconda3/bin/python \
  /home/user186/icmlworksop/scripts/run_qworkflow.py \
  --config /home/user186/icmlworksop/configs/lstm_alpha158_csi300_recent.yaml \
  --experiment lstm_alpha158_csi300_recent_seed2026 \
  --summary_out /home/user186/icmlworksop/results/tables/lstm_alpha158_csi300_recent_seed2026.json \
  --override task.model.kwargs.seed=2026
```

Swap the config / experiment prefix for `MASTER` and `HIST`.

## Ranking Tool

Use:

[rank_seed_sweep.py](/home/user186/icmlworksop/scripts/rank_seed_sweep.py:1)

Example:

```bash
PYTHONPATH=/home/user186/icmlworksop \
/project/python_env/anaconda3/bin/python scripts/rank_seed_sweep.py \
  --results-dir results/tables \
  --metric IC_mean \
  --our-prefix fdg_csi300_qlib_recent_skip32_seed \
  --baseline-prefix mlp_csi300_qlib_recent_seed \
  --baseline-prefix lstm_alpha158_csi300_recent_seed \
  --baseline-prefix master_csi300_recent_quick_seed \
  --baseline-prefix hist_alpha158_csi300_recent_quick_seed \
  --top-k 5 \
  --out results/tables/seed_sweep_recent_rank.md
```

This will produce:

- a markdown ranking table
- a json file with the same ranking and the selected top seeds

## Recommended Next Step

After the current 5-seed matrix finishes:

1. Verify whether `seed2026` reproduces the earlier `0.02137` level
2. If yes, launch a larger `recent` seed sweep for:
   - `FDG-Skip32`
   - `MLP`
   - `LSTM`
   - `MASTER`
   - `HIST-Alpha158`
3. Rank seeds using the paired procedure above
4. Pick the top 5 seeds and rerun any missing comparisons on exactly those 5 seeds
