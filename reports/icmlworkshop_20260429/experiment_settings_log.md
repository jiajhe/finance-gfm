# GPU FDG Experiment Settings - 2026-04-25

## Purpose

Run FDG-family PyTorch experiments on the local NVIDIA GeForce RTX 5060 Laptop GPU.
This intentionally deviates from the company-log reproduction environment because
`torch==2.6.0+cu124` cannot execute CUDA kernels on this GPU.

## Separation From Reproduction Environment

- CPU/company-log environment: `finance_gfm39`
- GPU experiment environment: `finance_gfm_gpu`
- CPU/company-log result root: `results/seed100_grouped_cpu`
- GPU result root: `results/seed100_gpu_fdg`

All GPU experiments in this project should use `finance_gfm_gpu` unless this file is updated.

## Hardware

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- VRAM: 8151 MiB
- Compute capability reported by `nvidia-smi`: 12.0
- Driver: 591.97

## GPU Runtime

- Conda env: `finance_gfm_gpu`
- Python: 3.10.20
- PyTorch: `2.11.0+cu128`
- PyQLib: `0.9.7`
- PyTorch source: official PyTorch CUDA 12.8 pip wheel index
- PyTorch index URL: `https://download.pytorch.org/whl/cu128`

Rationale: PyTorch official local install docs list Python 3.10+ for current stable
PyTorch, and CUDA 12.8 is an official Linux pip compute-platform option.

## Data

- Provider URI: `/home/jiajun/.qlib/qlib_data/cn_data_2024h1`
- Original local copy source: `/mnt/c/Users/Jiajun/.qlib/.qlib/qlib_data/cn_data_2024h1`
- Calendar last date: `2025-09-18`
- Backtest/date cap used by launcher: `2025-09-17`
- Market override: `csi300_aligned`

Reason for market override: the local `csi300` instrument file ends in 2020,
while `csi300_aligned` covers the recent split through the local data end date.

## Models In Scope

Initial GPU batch:

- `fdg_skip32`
- `fdg_skip32_random`
- `fdg_skip32_symshare`

Optional later, if needed:

- `mlp`

Excluded from this GPU FDG batch:

- `lgbm`: not a PyTorch GPU experiment in this repo setup.
- `lstm`, `master`, `hist`: qlib workflow/fork constraints differ from the FDG path.

## Launch Policy

- Keep the running CPU reproduction batch untouched.
- Use a separate result root for GPU experiments.
- Use one GPU training group at a time unless verified VRAM headroom is sufficient.
- Keep `PYTHONPATH=/home/jiajun/finance-gfm` so `sitecustomize.py` compatibility shims apply.
- Set `SEED100_TRAIN_DEVICE=cuda` for FDG GPU runs so generated configs record `train.device: cuda`.

## Verification Checklist

- [x] `finance_gfm_gpu` created.
- [x] `torch.cuda.is_available()` is true.
- [x] A CUDA tensor operation succeeds on RTX 5060.
- [x] `pip check` passes in `finance_gfm_gpu`.
- [x] Project focused tests pass in `finance_gfm_gpu`.
- [x] One FDG smoke seed completes on GPU.
- [x] 100-seed GPU FDG batch launched.

## Verification Results

CUDA probe:

```text
torch 2.11.0+cu128
cuda_available True
device NVIDIA GeForce RTX 5060 Laptop GPU
capability (12, 0)
cuda_probe 4.0
```

Focused tests:

```text
28 passed, 2 warnings
```

Initial smoke results, now invalid:

- `fdg_skip32_csi300_official_seed2026`: completed on `cuda`, test IC `-0.005456`
- `fdg_skip32_csi300_recent_seed2026`: completed on `cuda`, test IC `0.000000`

These initial numbers were invalidated on 2026-04-25. The official-handler tensor cache was passing unnormalized
finite outliers as large as about `1e20` into neural models, which caused `train_loss=nan` and collapsed validation
metrics. The old `results/seed100_gpu_fdg/tables/*.json` files were moved to:

```text
results/seed100_gpu_fdg/invalid_nan_20260425/tables/
```

Fix applied:

- `data/qlib_loader.py` now defaults official handler tensor payloads to per-day feature z-score normalization with `feature_clip: 10.0`.
- The tensor cache fingerprint includes the normalization/clip settings and a preprocessing version, so stale bad caches are not reused.
- `train/train_single.py` and `scripts/seed100_grouped_school.py` now fail fast on non-finite predictions, losses, or gradient norms.
- FDG/MLP official-handler configs explicitly record `normalize_features: true` and `feature_clip: 10.0`.

Post-fix FDG smoke:

- result root: `results/fdg_gpu_debug_norm`
- `fdg_skip32_csi300_official_seed2026`: completed on `cuda`, test IC `0.059854`, best epoch `9`
- no `nan` entries in `results/fdg_gpu_debug_norm/logs/fdg_skip32_csi300_official_seed2026.csv`
- `recent seed2026` cache build was killed during Qlib handler preprocessing while the CPU LSTM sweep was still consuming memory; it did not reach FDG training.

## Commands

Environment setup script:

```bash
bash scripts/setup_gpu_wsl_env.sh
```

CUDA verification:

```bash
PYTHONPATH=/home/jiajun/finance-gfm \
/home/jiajun/anaconda3/envs/finance_gfm_gpu/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); x=torch.ones(1, device='cuda'); print((x + 1).item())"
```

FDG GPU launch command will be recorded here after CUDA verification succeeds.

Smoke launch used before the full run:

```bash
PYTHONPATH=/home/jiajun/finance-gfm \
QLIB_NUM_WORKERS=4 \
SEED100_PROVIDER_URI=/home/jiajun/.qlib/qlib_data/cn_data_2024h1 \
SEED100_MARKET=csi300_aligned \
SEED100_RESULT_ROOT=results/seed100_gpu_fdg \
SEED100_TRAIN_DEVICE=cuda \
/home/jiajun/anaconda3/envs/finance_gfm_gpu/bin/python \
  scripts/seed100_grouped_school.py launch \
  --max-workers 1 \
  --models fdg_skip32 \
  --seeds 2026
```

Full GPU FDG batch launched:

```bash
PYTHONPATH=/home/jiajun/finance-gfm \
QLIB_NUM_WORKERS=4 \
SEED100_PROVIDER_URI=/home/jiajun/.qlib/qlib_data/cn_data_2024h1 \
SEED100_MARKET=csi300_aligned \
SEED100_RESULT_ROOT=results/seed100_gpu_fdg \
SEED100_TRAIN_DEVICE=cuda \
/home/jiajun/anaconda3/envs/finance_gfm_gpu/bin/python \
  scripts/seed100_grouped_school.py launch \
  --max-workers 1 \
  --models fdg_skip32 fdg_skip32_random fdg_skip32_symshare
```

The post-fix full batch was relaunched with the same command after quarantining invalid tables. Supervisor:

```text
results/seed100_gpu_fdg/manifests/launch_groups_max1.sh
```

After the 2026-04-26 machine restart, the same result root was resumed with
`QLIB_NUM_WORKERS=1` to reduce WSL memory pressure. Completed JSON files are
skipped by `seed100_grouped_school.py run-group`, so the resumed run appends
only missing seeds/groups.

## 2026-04-26 FDG Variant Audit

`fdg_skip32_symshare` is included in the FDG supervisor after
`fdg_skip32_random_recent`; it was not a finished group yet at the time of this
audit. The variant now uses these overrides:

```text
model.skip_hidden_dim=32
model.core_mode=symmetric
model.share_sr_weights=true
model.symmetrize_adjacency=true
```

The first two graph overrides make the pre-normalization FDG graph symmetric
(`S == R`, `B_core == B_core.T`). `model.symmetrize_adjacency=true` was added
before `symshare` started so the final adjacency passed into `GNNHead` is also
exactly symmetric after the normal FDG row-normalization step.

Original `fdg_skip32_random` configuration used only:

```text
model.skip_hidden_dim=32
model.graph_mode=random
```

That original version did not apply a per-seed `model.random_graph_seed`
override. The model default was `random_graph_seed=2026`, and
`FDGRegressor._random_adjacency` reinitialized a CPU generator with that seed
inside each forward pass before drawing a dense directed random adjacency and
row-normalizing it. The 100 experiment seeds therefore changed model
initialization/training order, not the random-graph seed itself. Those artifacts
were quarantined and should not be used as the random baseline.

Update after audit: the fixed-seed random run was stopped and quarantined before
being used in final summaries. Old random artifacts were moved to:

```text
results/seed100_gpu_fdg/invalid_random_fixed_seed_20260426/
```

The training launcher now applies a seed-dependent override for random FDG runs:
if `model.graph_mode=random` and no explicit `model.random_graph_seed` is set,
`model.random_graph_seed` is set to the experiment seed. The active completed
`fdg_skip32_random` tables under `results/seed100_gpu_fdg/tables/` are the rerun
version: 101 official plus 101 recent JSON files, each with
`model_config.random_graph_seed` equal to its experiment seed. These are
independent feature-free random graphs per seed.

Recent split used in generated GPU configs:

```text
train: 2010-01-01..2020-12-31
valid: 2021-01-01..2021-12-31
test:  2022-01-01..2025-09-17
```

The base config requests a test end of `2025-12-31`, but the launcher caps dates
to one trading day before the local provider calendar end (`2025-09-18`), so the
generated config uses `2025-09-17`.

The GPU run also overrides the market from `csi300` to `csi300_aligned`.
In this local provider, `csi300.txt` ends at `2020-09-25`, while
`csi300_aligned.txt` extends the same instrument list to `2025-09-18`.
This market-file difference is a likely source of large discrepancies versus
other devices or older company runs.

HIST scheduling update on 2026-04-27 23:43 CST:

The initial MASTER/HIST launcher used `--max-workers 1`, so HIST official and
recent were queued serially. Runtime checks during `hist_csi300_official_seed18`
showed about 1.6 GB / 8 GB GPU memory use, about 30% GPU utilization, and enough
host RAM headroom. The serial launcher parent was stopped after official was
already running, leaving the official child process alive, and
`hist_recent.sh` was started manually so HIST official and recent run in
parallel on the same GPU. The continuation watcher now only relaunches HIST if
both HIST processes disappear before both splits reach 101/101.

Night guard update on 2026-04-27 23:49 CST:

`scripts/hist_then_lstm_guard_20260428.sh` replaces the earlier simple HIST
continuation watcher. It watches HIST official and recent separately, relaunching
only the missing split if a process disappears before reaching 101 tables. After
both HIST splits are 101/101 and all HIST processes have exited, it starts LSTM
official and recent under the same GPU environment and result root:

```text
results/seed100_gpu_master_hist_fixed/
```

LSTM uses the qworkflow GPU override (`task.model.kwargs.GPU=0`) and the same
`csi300_aligned` provider setup as MASTER/HIST. The older CPU-root LSTM failures
were from missing qlib compiled extensions in that earlier environment and are
not reused for this GPU-root rerun.

Manual LSTM overlap update on 2026-04-28 22:39 CST:

With HIST recent still running, `lstm_official` was started early via
`scripts/lstm_official_manual_20260428.sh` to overlap one LSTM split with the
remaining HIST recent jobs. At launch, HIST recent was 87/101, LSTM official was
0/101, GPU memory was about 2.0 GB / 8.1 GB, and host memory still had about
4.9 GB available. LSTM recent remains under the guard and will be started after
HIST completion unless launched manually later.

Progress checks:

```bash
find results/seed100_gpu_fdg/tables -maxdepth 1 -type f | wc -l
find results/seed100_gpu_fdg/errors -maxdepth 1 -type f | wc -l
nvidia-smi
```

Summary command after the supervisor finishes:

```bash
PYTHONPATH=/home/jiajun/finance-gfm \
QLIB_NUM_WORKERS=4 \
SEED100_PROVIDER_URI=/home/jiajun/.qlib/qlib_data/cn_data_2024h1 \
SEED100_MARKET=csi300_aligned \
SEED100_RESULT_ROOT=results/seed100_gpu_fdg \
SEED100_TRAIN_DEVICE=cuda \
/home/jiajun/anaconda3/envs/finance_gfm_gpu/bin/python \
  scripts/seed100_grouped_school.py summarize \
  --models fdg_skip32 fdg_skip32_random fdg_skip32_symshare
```
