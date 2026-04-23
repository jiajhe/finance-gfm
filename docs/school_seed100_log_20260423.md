# School Seed100 Log (2026-04-23)

This note records the recent school-server attempts for the large seed sweep, including the migration work, launcher redesign, current `seed=2026` status, and why the school-side `skip32` result still differs from the company server.

## Goal

Run the following `8` model families on the ZJU CPU server:

- `lgbm`
- `lstm`
- `mlp`
- `master`
- `hist`
- `fdg_skip32`
- `fdg_skip32_random`
- `fdg_skip32_symshare`

Each family is evaluated on:

- `official`
- `recent`

Seed policy:

- run `2026` first as an environment sanity check
- then run `100` commonly used non-consecutive seeds

Total targets:

- `16` family/split groups
- `101` seeds each
- `1616` result files in total

## Migration And Environment

The company runtime was migrated to the school server as closely as possible:

- repo: `~/finance-gfm`
- qlib fork: `~/refs/qlib_sjtu`
- env: `finance_gfm39`
- qlib data snapshot copied from company:
  `~/.qlib_company_20260417/qlib_data/cn_data`

Key aligned versions on school:

- Python `3.9.25`
- torch `2.6.0`
- numpy `1.24.4`
- pandas `2.2.3`
- pyqlib `0.9.7`

Important caveat:

- the school server currently has no GPU
- this means all runs are effectively CPU-only

## Code And Config Checks

The critical FDG files were checked against the company repo and matched:

- `models/fdg.py`
- `models/gnn_head.py`
- `models/__init__.py`
- `configs/qlib_recent_fdg.yaml`

So the school-side `skip32` gap is not explained by a code fork in the plain FDG files.

## First Launcher Attempt

The original school launcher used many small per-seed tasks.

Observed problems:

- one failing seed could kill an entire slot shell
- qlib/data setup was repeated too often
- overall throughput was poor

That design was abandoned.

## Grouped Launcher Redesign

The runner was redesigned into `16` long-lived tasks:

- one task per `(family, split)`
- seeds are processed sequentially inside the task
- `2026` is always first

Current script:

- `scripts/seed100_grouped_school.py`

Design choice:

- `MLP / FDG` groups run in-process and reuse the already built dataset
- `LGBM / LSTM / MASTER / HIST` groups call `scripts/run_qworkflow.py` sequentially, one seed at a time

This keeps the matrix small while avoiding the previous slot-death behavior.

## Qlib Fork Note

One issue found during the redesign:

- if the grouped runner globally injected the qlib fork into `sys.path`, it interfered with the direct `train_single` path
- if it did not, `HIST` could fall back to site-packages `qlib`

The current compromise is:

- `seed100_grouped_school.py` does not globally own the qlib namespace
- `run_qworkflow.py` handles the qlib fork injection per subprocess

This is more stable for mixed `train_single + qworkflow` execution.

## Current `seed=2026` Status

At the latest check, `seed2026` had only partially finished.

Completed:

- `lgbm official`: `IC 0.0461148470`
- `lgbm recent`: `IC 0.0123069796`
- `mlp official`: `IC 0.0442085546`
- `fdg_skip32 official`: `IC 0.0412855646`
- `fdg_skip32 recent`: `IC 0.0168575414`
- `fdg_skip32_random official`: `IC 0.0431344111`
- `fdg_skip32_symshare official`: `IC 0.0427063745`

Still running at that moment:

- `lstm official`
- `lstm recent`
- `master official`
- `master recent`
- `mlp recent`
- `fdg_skip32_random recent`
- `fdg_skip32_symshare recent`

`HIST`:

- `hist official seed2026`: error
- `hist recent seed2026`: error

The grouped pipeline itself continued past failed seeds, so these errors did not stop the whole family.

## Why `skip32` Still Differs From Company

The company-side single run that motivated the school migration was:

- `fdg_skip32 recent seed2026 ~= 0.02137`

The school-side partial check currently gives:

- `fdg_skip32 recent seed2026 ~= 0.01686`

Most likely explanation:

- `skip32` is highly seed-sensitive and path-sensitive
- school runs are CPU-only, while the company runs used CUDA builds
- PyTorch/BLAS numeric paths differ across CPU vs GPU execution
- this task still uses dropout, shuffled batches, and noisy early stopping
- tiny numeric differences can change the selected checkpoint and final IC by several basis points

So the current evidence suggests:

- not a plain code mismatch
- more likely a training-path / hardware / environment sensitivity issue

## Practical Conclusion

Do not over-interpret a single `seed2026` result across machines.

For the paper-facing comparison, the right object is:

- the seed distribution
- mean / std
- top-k seed behavior

not one lucky single-seed point.

That is why the school machine is now left running the full `101`-seed grouped matrix.

## Paths To Watch On School

- summary markdown:
  `~/finance-gfm/results/seed100_grouped_cpu/seed100_grouped_summary.md`
- summary json:
  `~/finance-gfm/results/seed100_grouped_cpu/seed100_grouped_summary.json`
- manifest:
  `~/finance-gfm/results/seed100_grouped_cpu/manifests/seed100_grouped_root.json`

Manual refresh command:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate finance_gfm39
cd ~/finance-gfm
export PYTHONPATH=$HOME/finance-gfm
export QLIB_FORK_PATH=$HOME/refs/qlib_sjtu
export SEED100_PROVIDER_URI=$HOME/.qlib_company_20260417/qlib_data/cn_data
python scripts/seed100_grouped_school.py summarize
```
