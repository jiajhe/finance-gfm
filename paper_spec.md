# Finance GFM — Implementation Spec for Codex

> **Paper target**: GFM @ ICML 2026 Workshop, position paper (2–4 pages).
> **Title (working)**: *Low-Rank Directed Latent Graphs as a Structural Prior for Finance Graph Foundation Models*
> **Deadline**: 2026-04-24 23:59 AoE
> **Data constraint**: public only — Qlib + Alpha158 (Alpha360 optional).
> **Codex workflow**: follow phases in order. Do NOT skip ahead. Each phase has acceptance criteria that MUST pass before moving on.

---

## 0. Project Goal (what we're actually building)

A single-file clean implementation of **Factorized Directed Graph (FDG)** for cross-sectional stock return prediction, plus baselines, plus two transfer experiments, plus ablations. Output artifacts: 3 result tables + 1 ablation figure + 1 concept figure.

### Core math (everything derives from this)

For cross-section of $N$ stocks at time $t$ with features $X_t \in \mathbb{R}^{N \times d}$ and rank $r \ll N$:

$$
S_t = \mathrm{softmax}(X_t W_s / \tau), \quad R_t = \mathrm{softmax}(X_t W_r / \tau) \in \mathbb{R}^{N \times r}_{\ge 0}
$$

$$
A_t = S_t \, B \, R_t^\top \in \mathbb{R}^{N \times N}, \quad B \in \mathbb{R}^{r \times r}\ \text{(unconstrained, asymmetric)}
$$

Message passing (one layer, intentionally simple):

$$
H_t = A_t X_t W_v + X_t W_{\text{skip}}, \quad Z_t = \mathrm{LN}(\mathrm{GELU}(H_t)), \quad \hat{y}_t = \mathrm{MLP}(Z_t)
$$

Loss: cross-sectional IC loss, details in §3.4.

### Relation to HIST (write this into docstring verbatim)

HIST's hidden-concept branch is recovered by setting $W_s = W_r$ and $B = I_r$ (symmetric bipartite). FDG generalizes to the directed asymmetric case with a learnable core influence matrix $B$.

---

## 1. Repo Layout

```
finance-gfm/
├── README.md
├── requirements.txt
├── configs/
│   ├── source_csi300.yaml        # Phase 1 config
│   ├── source_csi500.yaml        # Phase 3
│   ├── transfer_time.yaml        # Phase 3
│   ├── transfer_market.yaml      # Phase 3
│   └── ablation_rank.yaml        # Phase 4
├── data/
│   ├── __init__.py
│   ├── qlib_loader.py            # Phase 1
│   └── entropy_cluster.py        # Phase 3
├── models/
│   ├── __init__.py
│   ├── fdg.py                    # Phase 1 (main contribution)
│   ├── gnn_head.py               # Phase 1
│   └── baselines/
│       ├── __init__.py
│       ├── mlp.py                # Phase 1
│       ├── gat.py                # Phase 2
│       ├── industry_gcn.py       # Phase 2
│       └── hist.py               # Phase 2 (use official repo)
├── train/
│   ├── __init__.py
│   ├── loss.py                   # Phase 1
│   ├── train_single.py           # Phase 1
│   └── train_transfer.py         # Phase 3
├── eval/
│   ├── __init__.py
│   ├── metrics.py                # Phase 1
│   └── portfolio.py              # Phase 1
├── scripts/
│   ├── run_table1.sh             # Phase 2 (source performance)
│   ├── run_table2.sh             # Phase 3 (transfer)
│   ├── run_ablation.sh           # Phase 4
│   └── make_figures.py           # Phase 4
├── results/                      # outputs go here, gitignored
│   ├── logs/
│   ├── ckpts/
│   └── tables/
└── tests/
    ├── test_fdg_shapes.py
    └── test_metrics.py
```

---

## 2. Environment

**Python**: 3.10.
**Hard deps** (`requirements.txt`):

```
pyqlib==0.9.5
torch==2.3.0
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
scikit-learn==1.5.0
pyyaml==6.0.1
tqdm==4.66.4
einops==0.8.0
```

**Qlib data setup** (MUST be fresh — we need data through 2025):

```bash
# 1. Pull latest community-maintained bundle (updated ~weekly, usually current to last trading day minus 1 week).
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data && tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2

# 2. Verify coverage reaches at least 2025-12-01.
python -c "
import qlib; from qlib.data import D
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')
cal = D.calendar(start_time='2010-01-01', end_time='2026-04-01')
print('latest date:', cal[-1])
assert str(cal[-1]) >= '2025-12-01', 'Data too old; re-download.'
"

# 3. Verify universe files for both markets.
python -c "
from qlib.data import D
print('csi300 sample:', list(D.instruments('csi300'))[:5])
print('csi500 sample:', list(D.instruments('csi500'))[:5])
"
```

**If the bundle is still behind, fall back to qlib's native fetcher** (slower but authoritative):
```bash
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --interval 1d
```

Hard requirement: latest calendar date ≥ 2025-12-01 before proceeding. If not, STOP and fix data before any coding.

---

## 3. PHASE 1 — Data + FDG + MLP + Training Loop (Day 1)

**Goal**: End-to-end trainable pipeline on CSI300 with FDG and MLP baseline. Should produce IC numbers on test set.

### 3.1 `data/qlib_loader.py`

Implement:

```python
class QlibCrossSectionalDataset(torch.utils.data.Dataset):
    """
    Yields (X_t, y_t, mask_t, date_t) per trading day.
    X_t: [N_t, d] float tensor, Alpha158 features, z-scored cross-sectionally per day.
    y_t: [N_t] float tensor, next-day return label (Ref($close, -2)/Ref($close, -1) - 1).
    mask_t: [N_t] bool, True for valid rows (no NaN).
    date_t: pandas.Timestamp.
    N_t varies across days (universe membership changes).
    """
    def __init__(
        self,
        market: str,              # "csi300" or "csi500"
        start_time: str,          # "2010-01-01"
        end_time: str,            # "2020-12-31"
        handler: str = "Alpha158",
        label: str = "Ref($close, -2) / Ref($close, -1) - 1",
        cache_dir: str = "~/.qlib_cache",
    ): ...
```

Requirements:
- Use `qlib.contrib.data.handler.Alpha158` (or `Alpha360` via arg).
- Cross-sectional z-score per day AFTER dropping all-NaN columns.
- Drop rows where label is NaN. If fewer than 30 valid stocks on a day, skip that day.
- Cache the pre-processed tensor dict to `cache_dir/{market}_{start}_{end}_{handler}.pt`. First call slow, subsequent fast.
- `__getitem__(i)` returns the i-th trading day; `__len__()` returns number of days.

**Collate function** (write in same file):
```python
def pad_collate(batch):
    """Pad variable-N batches to max N, return X [B, N_max, d], y [B, N_max], mask [B, N_max]."""
```

### 3.2 `models/fdg.py`

```python
class FDG(nn.Module):
    """
    Factorized Directed Graph.
    A = softmax(X W_s / tau) @ B @ softmax(X W_r / tau).T
    """
    def __init__(self, d_in: int, rank: int, tau: float = 1.0, b_init: str = "identity_perturbed"):
        # W_s, W_r: Linear(d_in, rank), no bias
        # B: nn.Parameter(rank, rank)
        #   - "identity_perturbed": B = I + N(0, 0.01)  (starts close to HIST, learns away)
        #   - "random": B = N(0, 1/sqrt(r))
        ...

    def forward(self, X, mask=None):
        """
        X: [B, N, d], mask: [B, N] bool.
        Returns:
          A: [B, N, N]  (masked rows/cols zeroed out)
          S: [B, N, r]
          R: [B, N, r]
        """
```

Requirements:
- When `mask` is given, set invalid rows/cols of $A$ to 0 AFTER computing softmax (not before — softmax denominator must exclude invalid entries). Do this by setting masked rows of `X W_s` to $-\infty$ pre-softmax.
- Row-normalize $A$ (each row sums to 1) to prevent magnitude blowup — document this in docstring.
- Expose $B$ as `self.B` for later visualization / regularization.

### 3.3 `models/gnn_head.py`

```python
class GNNHead(nn.Module):
    def __init__(self, d_in, d_hidden, d_out=1, dropout=0.1):
        # W_v: Linear(d_in, d_hidden)
        # W_skip: Linear(d_in, d_hidden)
        # LayerNorm, GELU, Dropout
        # MLP head: Linear(d_hidden, d_hidden) -> GELU -> Linear(d_hidden, d_out)
        ...

    def forward(self, A, X, mask=None):
        # H = A @ X @ W_v + X @ W_skip
        # Z = LN(GELU(H))
        # y_hat = MLP(Z).squeeze(-1)
        # return y_hat: [B, N]
```

### 3.4 `train/loss.py`

```python
def ic_loss(y_hat, y, mask):
    """
    Cross-sectional IC loss, averaged over batch.
    For each sample in batch:
      1. Select valid entries via mask
      2. Demean both y_hat and y
      3. Compute Pearson correlation
    Return: -mean(IC).  (Negative because we maximize IC.)
    """
```

Implementation detail: use `torch.where(mask, x, 0)` + count valid, compute means manually. Do NOT use `F.cosine_similarity` (doesn't handle mask).

Also implement `wpcc_loss(y_hat, y, mask)` (weighted Pearson, weights $\propto |\mathrm{rank}(y) - N/2|$) — available as flag.

### 3.5 `train/train_single.py`

Entry point for Phase 1. Interface:

```bash
python -m train.train_single --config configs/source_csi300.yaml
```

`configs/source_csi300.yaml`:
```yaml
market: csi300
handler: Alpha158
splits:
  train: [2010-01-01, 2020-12-31]
  valid: [2021-01-01, 2021-12-31]
  test:  [2022-01-01, 2025-12-31]

model:
  name: fdg            # or "mlp", "gat", "industry_gcn", "hist"
  rank: 32
  tau: 1.0
  d_hidden: 128
  dropout: 0.1
  b_init: identity_perturbed

train:
  batch_size: 8        # batch of trading days
  epochs: 50
  lr: 1e-3
  weight_decay: 1e-5
  loss: ic             # ic | wpcc
  early_stop_patience: 10
  grad_clip: 3.0
  seed: 2026

log:
  out_dir: results/
  exp_name: fdg_csi300_r32
```

Training loop MUST:
- Set seed everywhere (`torch`, `numpy`, `random`).
- Log per-epoch train IC, valid IC, valid RankIC to `results/logs/{exp_name}.csv`.
- Save best (on valid ICIR) checkpoint to `results/ckpts/{exp_name}.pt`.
- At end, load best ckpt, evaluate on test, save metrics to `results/tables/{exp_name}.json`.

### 3.6 `eval/metrics.py`

```python
def ic(preds, labels, masks) -> dict:
    """
    preds, labels, masks: list of numpy arrays (one per day), variable length.
    Returns:
      {
        "IC_mean": float,
        "IC_std": float,
        "ICIR": IC_mean / IC_std,
        "RankIC_mean": float,
        "RankICIR": float,
      }
    """
```

### 3.7 `eval/portfolio.py`

```python
def topk_portfolio(preds, labels, masks, dates, k=50):
    """
    Long-only top-K equal-weight portfolio, rebalanced daily.
    Returns:
      {
        "annual_return": float,   # (geometric, 252 days/yr)
        "annual_vol": float,
        "sharpe": float,
        "max_drawdown": float,
        "turnover": float,        # avg |position change| per day
      }
    """
```

No transaction costs for now. Document this limitation in paper discussion.

### 3.8 `models/baselines/mlp.py`

Same interface as FDG+GNNHead but no graph:
```python
class MLPBaseline(nn.Module):
    def __init__(self, d_in, d_hidden, dropout):
        # 3-layer MLP with LayerNorm + GELU + Dropout
    def forward(self, X, mask):
        # ignore inter-stock structure
        # return y_hat: [B, N]
```

### 3.9 PHASE 1 ACCEPTANCE CRITERIA

Run:
```bash
python -m train.train_single --config configs/source_csi300.yaml  # with model.name: fdg
python -m train.train_single --config configs/source_csi300.yaml  # with model.name: mlp
```

Both must complete without errors and produce:
- A valid `results/logs/*.csv` with monotone training curve (loss decreasing).
- A `results/tables/*.json` with **IC_mean > 0.03** on test for both models (sanity threshold — typical Alpha158 IC on CSI300 is 0.04–0.07).
- FDG IC > MLP IC by at least 0.005 (if not, something's wrong with FDG).

Only proceed to Phase 2 after these pass.

---

## 4. PHASE 2 — More baselines + Source performance table (Day 2)

### 4.1 `models/baselines/gat.py`

Sparse GAT with top-k attention (to avoid OOM on CSI500):

```python
class GATBaseline(nn.Module):
    def __init__(self, d_in, d_hidden, n_heads=4, topk=20, dropout=0.1):
        # Multi-head attention where for each query, only top-k keys contribute
        # (hard top-k selected via feature similarity, then softmax over those k)
    def forward(self, X, mask): ...
```

Key detail: for each row, compute attention scores vs all valid stocks, keep top-k, softmax over those, aggregate. Don't use `torch_geometric` — too heavy for this scope. Pure PyTorch with `torch.topk`.

### 4.2 `models/baselines/industry_gcn.py`

```python
class IndustryGCN(nn.Module):
    """
    Static graph from Qlib industry codes (stocks in same SW Level-1 industry are connected).
    Row-normalized adjacency, 1-hop GCN, then MLP head.
    """
    def __init__(self, d_in, d_hidden, industry_map, dropout=0.1):
        # industry_map: dict[str, int] loaded once from Qlib
        ...
```

Get industry map via:
```python
from qlib.data import D
D.features(D.instruments("csi300"), ["$industry"], ...)  # or appropriate API
```

If Qlib industry not available, fall back to SW industry CSV — put a TODO and use a synthetic industry (random assignment) as placeholder so code runs. Document this clearly.

### 4.3 `models/baselines/hist.py`

**Do not reimplement HIST.** Clone `https://github.com/Wentao-Xu/HIST` as a git submodule under `third_party/hist/`. Write a thin wrapper that:
- Imports their model
- Matches our data loader interface
- Uses their default hyperparameters (concept number = 60 for CSI300, 200 for CSI500, as in their paper)

Wrapper file:
```python
# models/baselines/hist.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../third_party/hist"))
from model import HIST as _HIST

class HISTBaseline(nn.Module):
    def __init__(self, d_in, d_hidden, n_concept, dropout):
        self.net = _HIST(d_feat=d_in, hidden_size=d_hidden, num_layers=2,
                         dropout=dropout, K=n_concept, base_model="GRU")
    def forward(self, X, mask):
        # HIST expects sequential input; we feed a single time step.
        # Unsqueeze time dim, call forward, squeeze output.
        ...
```

If HIST repo expects temporal sequences longer than 1 and refuses single-step, **fall back to Alpha360** (which has 60 lookback days built in) for HIST only. Keep FDG and others on Alpha158. Document this asymmetry honestly.

### 4.4 `scripts/run_table1.sh`

```bash
#!/bin/bash
for model in mlp gat industry_gcn hist fdg; do
  python -m train.train_single \
    --config configs/source_csi300.yaml \
    --override model.name=$model \
    --exp_name "${model}_csi300"
done
python scripts/aggregate_table1.py  # reads results/tables/*.json, outputs results/tables/table1.md
```

`aggregate_table1.py` produces a markdown table:

| Model | IC | ICIR | RankIC | Top50 AR | Sharpe |
|---|---|---|---|---|---|
| MLP | ... | ... | ... | ... | ... |
| GAT | ... | ... | ... | ... | ... |
| IndustryGCN | ... | ... | ... | ... | ... |
| HIST | ... | ... | ... | ... | ... |
| FDG (ours) | ... | ... | ... | ... | ... |

### 4.5 PHASE 2 ACCEPTANCE

- `results/tables/table1.md` exists and all 5 rows populated.
- FDG is within top-2 on at least 3 of 5 metrics.

---

## 5. PHASE 3 — Transfer experiments + Cluster warm-start (Day 3)

### 5.0 Parameter Transfer Matrix (READ THIS FIRST)

FDG has 5 parameter groups. Each has a DIFFERENT transfer role. Do not treat them uniformly.

| Parameter | Shape | Role | Transfer behavior in `cluster_warm` mode |
|---|---|---|---|
| `W_s` | `[d, r]` | Stock → sender-factor projection | **Reinitialized** via cluster warm-start on target data |
| `W_r` | `[d, r]` | Stock → receiver-factor projection | **Reinitialized** via cluster warm-start on target data |
| `B` | `[r, r]` | Factor-factor directed influence | **Fully inherited** from source checkpoint |
| `W_v`, `W_skip` | GNN layer | Message passing | Inherited, then fine-tuned end-to-end |
| `MLP` head | Output | Prediction | Inherited, then fine-tuned end-to-end |

**Rationale**: B captures market-invariant factor dynamics (momentum→reversal, etc.), while W_s/W_r encode market-specific stock compositions (what "momentum stocks" look like in CSI300 ≠ in CSI500).

**Training schedule for `cluster_warm`**:
1. Load source checkpoint.
2. Replace `W_s`, `W_r` with cluster-warm-start values (see §5.1).
3. **Freeze B for first 3 epochs** (let newly-init W_s, W_r align with the inherited B first).
4. Unfreeze all. Train with lr=1e-4 for remaining epochs.

Other modes:
- `zero_shot`: load source, evaluate on target test directly. No training.
- `finetune`: load source (all params), train on target train with lr=1e-4.
- `scratch` (baseline): random init, train on target train only.

### 5.1 `data/entropy_cluster.py`

Clustering is based on **factor fingerprints** (stock × factor cross-section), NOT return distributions. This aligns with W_s/W_r's role of projecting factor space to cluster space.

```python
def factor_fingerprint(X_window: np.ndarray) -> np.ndarray:
    """
    X_window: [T, N, d]  — rolling window of cross-sectional factors.
    Returns: [N, d] — time-average factor vector per stock (the fingerprint).
    Drop stocks present in < 50% of window days. Fill remaining NaN with cross-sectional median.
    """

def kmeans_cluster(fingerprints: np.ndarray, k: int, seed: int = 2026) -> np.ndarray:
    """
    fingerprints: [N, d].  Returns: [N] cluster labels in {0, ..., k-1}.
    Use sklearn.cluster.KMeans with n_init=10.  Standardize columns before clustering.
    """

def entropy_weighted_distance(
    X_window: np.ndarray,
    factor_ic: np.ndarray,   # [d] abs-IC of each factor, used as importance weight
    n_bins: int = 10,
) -> np.ndarray:
    """
    OPTIONAL entropy-based variant, kept for ablation.
    For each pair (i,j), compute weighted-sum of per-factor JS divergences between
    binned factor distributions, weighted by |IC|_k.  Returns: [N, N] distance matrix.
    Use when k-means-on-fingerprints underperforms.
    """

def cluster_warm_start(
    X_target_window: np.ndarray,  # [T, N_target, d]
    rank: int,
    ridge_lambda: float = 10.0,
    method: str = "kmeans_fingerprint",  # or "entropy_weighted"
    seed: int = 2026,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns W_s_init, W_r_init, both [d, rank] ndarray, for initializing FDG on target.
    Procedure:
      1. Compute fingerprints (or distance matrix).
      2. Cluster target stocks into `rank` groups.
      3. Build one-hot soft assignment C [N_target, rank].
      4. Ridge regression: W = argmin ||X_mean · W - C||_F^2 + lambda ||W||_F^2.
         Closed form: W = (X.T @ X + lambda * I)^-1 @ X.T @ C.
      5. Copy W → W_s_init; W_r_init = W_s_init + small Gaussian noise (std=0.01) to break symmetry.
    """
```

**Default**: `method="kmeans_fingerprint"`. Entropy variant is a Phase 4 ablation, don't implement eagerly.

### 5.2 `train/train_transfer.py`

```bash
python -m train.train_transfer \
  --source-config configs/source_csi300.yaml \
  --target-config configs/transfer_market.yaml \
  --mode cluster_warm  # one of: zero_shot, finetune, cluster_warm, scratch
```

Mode implementations (follow §5.0 parameter matrix strictly):

**`zero_shot`**:
```python
model = build_fdg(target_config)
model.load_state_dict(torch.load(source_ckpt))  # loads ALL params including W_s, W_r
# FDG architecture is N-agnostic: W_s, W_r are [d, r] not [N, r], so no shape issue.
# Just evaluate on target test. No training.
```

**`finetune`**:
```python
model = build_fdg(target_config)
model.load_state_dict(torch.load(source_ckpt))
# Train all params on target train with lr=1e-4.
```

**`cluster_warm`** (the money mode):
```python
model = build_fdg(target_config)
state = torch.load(source_ckpt)
# Inherit B, W_v, W_skip, MLP
model.load_state_dict(state)
# Overwrite W_s, W_r with cluster warm-start
W_s_init, W_r_init = cluster_warm_start(target_train_X_window, rank=cfg.model.rank, ...)
model.fdg.W_s.weight.data = torch.from_numpy(W_s_init.T).float()  # Linear stores [out, in]
model.fdg.W_r.weight.data = torch.from_numpy(W_r_init.T).float()
# Freeze B for first 3 epochs
for p in model.fdg.B.parameters(): p.requires_grad = False
# ... train for 3 epochs ...
for p in model.fdg.B.parameters(): p.requires_grad = True
# ... continue training remaining epochs with lr=1e-4 ...
```

**`scratch`** (baseline, required for honest comparison):
```python
model = build_fdg(target_config)  # random init
# Train on target train from scratch.
```

For `cluster_warm_start`, use `target_train_X_window = X[-60:]` of the target train split
(last 60 days of target training period, so no target test leakage).

### 5.3 Transfer configs

`configs/transfer_time.yaml` (same market CSI300, shifted time — low-resource target):
```yaml
source:
  market: csi300
  splits:
    train: [2010-01-01, 2018-12-31]
    valid: [2019-01-01, 2019-12-31]
target:
  market: csi300
  splits:
    train: [2022-01-01, 2023-06-30]   # deliberately short (1.5y) to showcase warm-start
    valid: [2023-07-01, 2023-12-31]
    test:  [2024-01-01, 2025-12-31]
```

`configs/transfer_market.yaml` (shifted market CSI300→CSI500, overlapping time):
```yaml
source:
  market: csi300
  splits:
    train: [2010-01-01, 2020-12-31]
    valid: [2021-01-01, 2021-12-31]
target:
  market: csi500
  splits:
    train: [2022-01-01, 2023-12-31]
    valid: [2024-01-01, 2024-06-30]
    test:  [2024-07-01, 2025-12-31]
```

Also need scratch-on-target configs (`configs/transfer_time_scratch.yaml`, `configs/transfer_market_scratch.yaml`): same target splits as above, no source, single-market training.

### 5.4 `scripts/run_table2.sh`

```bash
#!/bin/bash
for split in time market; do
  for mode in zero_shot finetune cluster_warm; do
    python -m train.train_transfer \
      --source-config configs/source_csi300.yaml \
      --target-config configs/transfer_${split}.yaml \
      --mode ${mode} \
      --exp_name "transfer_${split}_${mode}"
  done
done

# Also run "from scratch on target" as another baseline:
python -m train.train_single --config configs/transfer_time_scratch.yaml --exp_name "scratch_time"
python -m train.train_single --config configs/transfer_market_scratch.yaml --exp_name "scratch_market"

python scripts/aggregate_table2.py
```

Expected output table:

| Setting | Scratch | Zero-shot | Finetune | **Cluster warm (ours)** |
|---|---|---|---|---|
| Time transfer (IC) | ... | ... | ... | ... |
| Market transfer (IC) | ... | ... | ... | ... |

### 5.5 PHASE 3 ACCEPTANCE

- `results/tables/table2.md` exists with all 8 cells filled.
- **Cluster warm ≥ Finetune ≥ Zero-shot** on at least one of the two settings (ideally both). If cluster warm loses, investigate — see pitfalls §7.

---

## 6. PHASE 4 — Ablations + Figures (Day 4)

### 6.1 Rank sweep

Sweep `rank ∈ {4, 8, 16, 32, 64, 128}` on CSI300. Plot test IC vs rank as a line chart.

### 6.2 $B$ structure ablation

Variants:
- `b_diagonal`: constrain $B$ to diagonal (no factor-factor interaction).
- `b_symmetric`: $B = (B_0 + B_0^\top)/2$.
- `b_identity`: fix $B = I$ (never trained — this approximates HIST).
- `b_full` (default): unconstrained.

Compare IC on CSI300 test.

### 6.3 Warm-start ablation

Only on market transfer:
- Random init on target
- Inherit source $W_s, W_r$ (naive)
- Cluster warm-start (ours)

### 6.4 Figures

`scripts/make_figures.py` produces:

- **Fig 1** (concept figure): schematic of FDG showing $S_t$, $B$, $R_t^\top$ multiplication, with stocks colored by cluster. Make this a clean tikz-like matplotlib diagram, exported as PDF.
- **Fig 2** (rank sweep): line chart, rank on x-axis, IC on y-axis.
- **Fig 3** (learned $B$ heatmap): visualize the trained $B$ matrix for rank=32. This is the money shot for interpretability.

Save all as PDF to `results/figures/`.

### 6.5 PHASE 4 ACCEPTANCE

- Rank sweep shows clear sweet spot (probably r=16~32).
- Ablation table shows `b_full > b_symmetric ≈ b_diagonal > b_identity`. If not, we have a problem — the directedness claim is empirically weaker than hoped.
- Figure 3 $B$ heatmap shows clear non-symmetric structure (this is the visual proof of the directedness thesis).

---

## 7. Common Pitfalls — read these before debugging

1. **Qlib label alignment**: Alpha158 label formula `Ref($close, -2) / Ref($close, -1) - 1` is **next-day return**. If features are at day $t$, label is return from $t+1$ close to $t+2$ close. No look-ahead bias. Double-check by inspecting first few rows manually.

2. **Universe membership changes**: CSI300 constituents change twice a year. Always filter `D.instruments("csi300")` at each date, not just once. Qlib's handler does this correctly if you pass `instruments="csi300"`.

3. **NaN handling**: After `fillna(0)` on z-scored features is fine (since z-score makes 0 = mean). But do NOT fillna on labels — drop those stock-days entirely.

4. **Memory on CSI500**: N≈500, d=158, batch=8 → 500×500 attention is ~1M entries per sample. Fine for FDG (low-rank so you never materialize A for softmax). For dense GAT, use topk=20.

5. **IC loss numerical stability**: if all predictions are identical on some day, std is 0, IC is NaN. Add `+ 1e-8` to std in denominator. If count of valid stocks < 10, skip that day in loss (but still report in eval).

6. **Softmax temperature τ**: default 1.0. If $S_t, R_t$ collapse to one-hot (cluster degenerate), increase τ. If they're uniform (no structure), decrease.

7. **Cluster warm-start disappointment**: if cluster_warm < finetune, likely causes:
   - JS distance computed on too short a window (<30 days). Use >=60.
   - Clusters too balanced — try k-means instead of spectral.
   - Ridge regularization too weak (W too big) — increase `ridge_lambda` to 10 or 100.

8. **HIST baseline**: their code expects Alpha360 temporal features. If our FDG uses Alpha158, either (a) run HIST on Alpha360 and note the difference in paper, or (b) adapt HIST's GRU backbone to Alpha158 static features (loses some of HIST's story, but fair comparison).

9. **Random seed**: fix seed=2026 everywhere. Report mean ± std over 3 seeds for the main tables if time permits; if not, single seed is fine for a workshop paper, noted in limitations.

10. **Portfolio turnover**: high turnover = expensive in reality. Report it in table even though we don't model costs. Reviewers will ask.

---

## 8. Paper writing after experiments done

Do NOT start writing until Phase 3 acceptance passes. If Phase 3 numbers are weak, the paper's thesis needs adjustment (maybe reposition as "structure learning" rather than "transfer"). Keep writing decoupled from Codex work — this spec is implementation only.

Paper assets needed from code:
- Table 1 (source performance) → `results/tables/table1.md`
- Table 2 (transfer) → `results/tables/table2.md`
- Ablation table (compact) → `results/tables/ablation.md`
- Fig 1 concept, Fig 2 rank sweep, Fig 3 B heatmap

---

## 9. Summary of what to tell Codex in each session

**Session 1 (Phase 1)**: "Read PAPER_SPEC.md sections 0–3. Implement everything in Phase 1. Do not touch Phase 2+ files yet. Finish with running Phase 1 acceptance."

**Session 2 (Phase 2)**: "Read PAPER_SPEC.md section 4. Implement GAT, IndustryGCN, HIST wrapper baselines. Run `scripts/run_table1.sh`. Verify Phase 2 acceptance."

**Session 3 (Phase 3)**: "Read PAPER_SPEC.md section 5. Implement entropy_cluster.py and train_transfer.py. Run `scripts/run_table2.sh`. Verify Phase 3 acceptance."

**Session 4 (Phase 4)**: "Read PAPER_SPEC.md section 6. Run ablations and produce figures. Verify Phase 4 acceptance."

Do not let Codex batch across phases. Phase isolation is what keeps us sane.