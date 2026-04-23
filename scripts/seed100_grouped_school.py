from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import traceback
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from eval.metrics import ic as ic_metrics
from eval.portfolio import topk_portfolio
from models import build_model
from train.loss import build_loss
from train.train_single import (
    apply_overrides,
    apply_train_window_overrides,
    build_loaders,
    build_recency_weight_map,
    build_datasets,
    safe_torch_load,
    set_seed,
    to_serializable,
)


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
RUN_QWORKFLOW = ROOT / "scripts" / "run_qworkflow.py"
RESULT_ROOT = ROOT / "results" / "seed100_grouped_cpu"
CONFIG_DIR = RESULT_ROOT / "generated_configs"
LOG_DIR = RESULT_ROOT / "logs"
TABLES_DIR = RESULT_ROOT / "tables"
CKPT_DIR = RESULT_ROOT / "ckpts"
ERROR_DIR = RESULT_ROOT / "errors"
MANIFEST_DIR = RESULT_ROOT / "manifests"
SLOT_SCRIPT_DIR = MANIFEST_DIR / "slot_scripts"
SUMMARY_MD = RESULT_ROOT / "seed100_grouped_summary.md"
SUMMARY_JSON = RESULT_ROOT / "seed100_grouped_summary.json"

POPULAR_SEEDS_100 = [
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    29,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    40,
    41,
    42,
    43,
    44,
    47,
    49,
    50,
    52,
    55,
    57,
    60,
    63,
    64,
    66,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    111,
    123,
    128,
    137,
    149,
    168,
    169,
    188,
    199,
    200,
    202,
    233,
    256,
    314,
    512,
    666,
    777,
    888,
    999,
]
DEFAULT_SEEDS = [2026] + [seed for seed in POPULAR_SEEDS_100 if seed != 2026]
DEFAULT_MODELS = [
    "lgbm",
    "lstm",
    "mlp",
    "master",
    "hist",
    "fdg_skip32",
    "fdg_skip32_random",
    "fdg_skip32_symshare",
]
THREAD_ENV = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
}
PROVIDER_URI_OVERRIDE = os.environ.get("SEED100_PROVIDER_URI") or os.environ.get("FACTOR100_PROVIDER_URI")
QLIB_FORK_ROOT = Path(os.environ.get("QLIB_FORK_PATH", str(Path.home() / "refs" / "qlib_sjtu"))).expanduser()


@dataclass
class GroupSpec:
    family: str
    split: str
    runner: str
    config_key: str
    base_overrides: list[str]

    @property
    def group_name(self) -> str:
        return f"{self.family}_{self.split}"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _patch_train_single_config(base_path: Path, out_path: Path) -> Path:
    cfg = load_yaml(base_path)
    if PROVIDER_URI_OVERRIDE:
        cfg["provider_uri"] = PROVIDER_URI_OVERRIDE
    cfg.setdefault("log", {})["out_dir"] = str(RESULT_ROOT)
    save_yaml(out_path, cfg)
    return out_path


def _patch_lstm_or_hist_config(base_path: Path, out_path: Path) -> Path:
    cfg = load_yaml(base_path)
    if PROVIDER_URI_OVERRIDE:
        cfg.setdefault("qlib_init", {})["provider_uri"] = PROVIDER_URI_OVERRIDE
    model_kwargs = cfg["task"]["model"]["kwargs"]
    if "n_jobs" in model_kwargs:
        model_kwargs["n_jobs"] = 4
    if "model_path" in model_kwargs:
        model_kwargs["model_path"] = str(QLIB_FORK_ROOT / "examples" / "benchmarks" / "LSTM" / "csi300_lstm_ts.pkl")
    if "stock2concept" in model_kwargs:
        model_kwargs["stock2concept"] = str(
            QLIB_FORK_ROOT / "examples" / "benchmarks" / "HIST" / "qlib_csi300_stock2concept.npy"
        )
    if "stock_index" in model_kwargs:
        model_kwargs["stock_index"] = str(
            QLIB_FORK_ROOT / "examples" / "benchmarks" / "HIST" / "qlib_csi300_stock_index.npy"
        )
    save_yaml(out_path, cfg)
    return out_path


def _patch_master_config(base_path: Path, out_path: Path) -> Path:
    cfg = load_yaml(base_path)
    if PROVIDER_URI_OVERRIDE:
        cfg.setdefault("qlib_init", {})["provider_uri"] = PROVIDER_URI_OVERRIDE
    save_yaml(out_path, cfg)
    return out_path


def _patch_lgbm_config(base_path: Path, out_path: Path) -> Path:
    cfg = load_yaml(base_path)
    if PROVIDER_URI_OVERRIDE:
        cfg.setdefault("qlib_init", {})["provider_uri"] = PROVIDER_URI_OVERRIDE
    model_kwargs = cfg["task"]["model"]["kwargs"]
    if "num_threads" in model_kwargs:
        model_kwargs["num_threads"] = 4
    save_yaml(out_path, cfg)
    return out_path


def generate_configs() -> dict[str, Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}
    mapping = {
        "lgbm_official": (ROOT / "configs" / "lightgbm_alpha158_csi300_official.yaml", _patch_lgbm_config),
        "lgbm_recent": (ROOT / "configs" / "lightgbm_alpha158_csi300_recent.yaml", _patch_lgbm_config),
        "lstm_official": (ROOT / "configs" / "lstm_alpha158_csi300_official.yaml", _patch_lstm_or_hist_config),
        "lstm_recent": (ROOT / "configs" / "lstm_alpha158_csi300_recent.yaml", _patch_lstm_or_hist_config),
        "hist_official": (ROOT / "configs" / "hist_alpha158_csi300_official_quick.yaml", _patch_lstm_or_hist_config),
        "hist_recent": (ROOT / "configs" / "hist_alpha158_csi300_recent_quick.yaml", _patch_lstm_or_hist_config),
        "master_official": (ROOT / "configs" / "master_csi300_official_quick.yaml", _patch_master_config),
        "master_recent": (ROOT / "configs" / "master_csi300_recent_quick.yaml", _patch_master_config),
        "mlp_official": (ROOT / "configs" / "qlib_official_mlp.yaml", _patch_train_single_config),
        "mlp_recent": (ROOT / "configs" / "qlib_recent_mlp.yaml", _patch_train_single_config),
        "fdg_official": (ROOT / "configs" / "qlib_official_fdg.yaml", _patch_train_single_config),
        "fdg_recent": (ROOT / "configs" / "qlib_recent_fdg.yaml", _patch_train_single_config),
    }
    for key, (base_path, patch_fn) in mapping.items():
        out_path = CONFIG_DIR / f"{key}.yaml"
        generated[key] = patch_fn(base_path, out_path)
    return generated


def build_group_specs(models: list[str]) -> list[GroupSpec]:
    specs: list[GroupSpec] = []
    for family in models:
        if family == "lgbm":
            specs.extend(
                [
                    GroupSpec("lgbm", "official", "qworkflow", "lgbm_official", []),
                    GroupSpec("lgbm", "recent", "qworkflow", "lgbm_recent", []),
                ]
            )
        elif family == "lstm":
            specs.extend(
                [
                    GroupSpec("lstm", "official", "qworkflow", "lstm_official", []),
                    GroupSpec("lstm", "recent", "qworkflow", "lstm_recent", []),
                ]
            )
        elif family == "mlp":
            specs.extend(
                [
                    GroupSpec("mlp", "official", "train_single", "mlp_official", []),
                    GroupSpec("mlp", "recent", "train_single", "mlp_recent", []),
                ]
            )
        elif family == "master":
            specs.extend(
                [
                    GroupSpec("master", "official", "qworkflow", "master_official", ["task.model.kwargs.GPU=0"]),
                    GroupSpec("master", "recent", "qworkflow", "master_recent", ["task.model.kwargs.GPU=0"]),
                ]
            )
        elif family == "hist":
            specs.extend(
                [
                    GroupSpec(
                        "hist",
                        "official",
                        "qworkflow",
                        "hist_official",
                        ["task.model.kwargs.GPU=0", "task.model.kwargs.n_jobs=4"],
                    ),
                    GroupSpec(
                        "hist",
                        "recent",
                        "qworkflow",
                        "hist_recent",
                        ["task.model.kwargs.GPU=0", "task.model.kwargs.n_jobs=4"],
                    ),
                ]
            )
        elif family == "fdg_skip32":
            specs.extend(
                [
                    GroupSpec("fdg_skip32", "official", "train_single", "fdg_official", ["model.skip_hidden_dim=32"]),
                    GroupSpec("fdg_skip32", "recent", "train_single", "fdg_recent", ["model.skip_hidden_dim=32"]),
                ]
            )
        elif family == "fdg_skip32_random":
            specs.extend(
                [
                    GroupSpec(
                        "fdg_skip32_random",
                        "official",
                        "train_single",
                        "fdg_official",
                        ["model.skip_hidden_dim=32", "model.graph_mode=random"],
                    ),
                    GroupSpec(
                        "fdg_skip32_random",
                        "recent",
                        "train_single",
                        "fdg_recent",
                        ["model.skip_hidden_dim=32", "model.graph_mode=random"],
                    ),
                ]
            )
        elif family == "fdg_skip32_symshare":
            specs.extend(
                [
                    GroupSpec(
                        "fdg_skip32_symshare",
                        "official",
                        "train_single",
                        "fdg_official",
                        ["model.skip_hidden_dim=32", "model.core_mode=symmetric", "model.share_sr_weights=true"],
                    ),
                    GroupSpec(
                        "fdg_skip32_symshare",
                        "recent",
                        "train_single",
                        "fdg_recent",
                        ["model.skip_hidden_dim=32", "model.core_mode=symmetric", "model.share_sr_weights=true"],
                    ),
                ]
            )
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["launch", "run-group", "summarize"])
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--family", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    return parser.parse_args()


def _task_env_exports() -> str:
    exports = {
        **THREAD_ENV,
        "PYTHONPATH": os.environ.get("PYTHONPATH", str(ROOT)),
        "QLIB_FORK_PATH": os.environ.get("QLIB_FORK_PATH", str(Path.home() / "refs" / "qlib_sjtu")),
    }
    if PROVIDER_URI_OVERRIDE:
        exports["SEED100_PROVIDER_URI"] = PROVIDER_URI_OVERRIDE
    return "\n".join(f"export {k}={shlex.quote(v)}" for k, v in exports.items())


def write_group_scripts(specs: list[GroupSpec], seeds: list[int]) -> list[Path]:
    SLOT_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    scripts: list[Path] = []
    export_block = _task_env_exports()
    seed_args = " ".join(str(seed) for seed in seeds)
    for spec in specs:
        script_path = SLOT_SCRIPT_DIR / f"{spec.group_name}.sh"
        log_path = LOG_DIR / f"{spec.group_name}.log"
        lines = [
            "#!/usr/bin/env bash",
            "set -u",
            f"cd {shlex.quote(str(ROOT))}",
            export_block,
            (
                f"stdbuf -oL -eL {shlex.quote(str(PYTHON))} {shlex.quote(str(ROOT / 'scripts' / 'seed100_grouped_school.py'))} "
                f"run-group --family {shlex.quote(spec.family)} --split {shlex.quote(spec.split)} --seeds {seed_args} "
                f"> {shlex.quote(str(log_path))} 2>&1"
            ),
        ]
        script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script_path.chmod(0o755)
        scripts.append(script_path)
    return scripts


def launch_group_scripts(scripts: list[Path]) -> None:
    launcher_log_dir = RESULT_ROOT / "launcher_logs"
    launcher_log_dir.mkdir(parents=True, exist_ok=True)
    for script_path in scripts:
        log_path = launcher_log_dir / f"{script_path.stem}.launcher.log"
        subprocess.Popen(
            ["nohup", "bash", str(script_path)],
            stdout=log_path.open("ab"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(ROOT),
        )


def write_manifest(specs: list[GroupSpec], seeds: list[int]) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "python": str(PYTHON),
        "provider_uri": PROVIDER_URI_OVERRIDE,
        "seeds": seeds,
        "groups": [
            {
                "family": spec.family,
                "split": spec.split,
                "runner": spec.runner,
                "config_key": spec.config_key,
                "base_overrides": spec.base_overrides,
            }
            for spec in specs
        ],
    }
    path = MANIFEST_DIR / "seed100_grouped_root.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _save_error(exp_name: str) -> Path:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    path = ERROR_DIR / f"{exp_name}.txt"
    path.write_text(traceback.format_exc(), encoding="utf-8")
    return path


def _collect_batch_arrays(
    preds,
    labels,
    raw_labels,
    masks,
    dates,
    instruments,
    pred_store,
    label_store,
    mask_store,
    date_store,
    instrument_store,
):
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    raw_labels_np = raw_labels.detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy().astype(bool)
    for idx, date in enumerate(dates):
        pred_store.append(preds_np[idx])
        label_store.append(raw_labels_np[idx] if raw_labels_np is not None else labels_np[idx])
        mask_store.append(masks_np[idx])
        date_store.append(date)
        instrument_store.append(instruments[idx])


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip, day_weight_map=None):
    model.train()
    pred_losses, reg_losses, total_losses = [], [], []
    pred_store, label_store, mask_store, date_store, instrument_store = [], [], [], [], []

    for X, y, raw_y, mask, dates, instruments, history in tqdm(loader, desc="train", leave=False):
        X = X.to(device)
        y = y.to(device)
        raw_y = raw_y.to(device)
        mask = mask.to(device)
        history = history.to(device) if history is not None else None
        sample_weight = None
        if day_weight_map is not None:
            sample_weight = torch.tensor(
                [day_weight_map[pd.Timestamp(date).strftime("%Y-%m-%d")] for date in dates],
                dtype=X.dtype,
                device=device,
            )

        optimizer.zero_grad(set_to_none=True)
        preds = model(X, mask, history=history)
        pred_loss = loss_fn(preds, y, mask, sample_weight=sample_weight)
        reg_loss = (
            model.regularization_loss()
            if hasattr(model, "regularization_loss")
            else pred_loss.detach().new_zeros(())
        )
        total_loss = pred_loss + reg_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        pred_losses.append(float(pred_loss.detach().cpu()))
        reg_losses.append(float(reg_loss.detach().cpu()))
        total_losses.append(float(total_loss.detach().cpu()))
        _collect_batch_arrays(
            preds,
            y,
            raw_y,
            mask,
            dates,
            instruments,
            pred_store,
            label_store,
            mask_store,
            date_store,
            instrument_store,
        )

    metrics = ic_metrics(pred_store, label_store, mask_store)
    return {
        "pred_loss": float(np.mean(pred_losses)) if pred_losses else 0.0,
        "reg_loss": float(np.mean(reg_losses)) if reg_losses else 0.0,
        "total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
        "metrics": metrics,
    }


@torch.no_grad()
def evaluate(model, loader, device, compute_portfolio=False):
    model.eval()
    pred_store, label_store, mask_store, date_store, instrument_store = [], [], [], [], []

    for X, y, raw_y, mask, dates, instruments, history in tqdm(loader, desc="eval", leave=False):
        X = X.to(device)
        y = y.to(device)
        raw_y = raw_y.to(device)
        mask = mask.to(device)
        history = history.to(device) if history is not None else None
        preds = model(X, mask, history=history)
        _collect_batch_arrays(
            preds,
            y,
            raw_y,
            mask,
            dates,
            instruments,
            pred_store,
            label_store,
            mask_store,
            date_store,
            instrument_store,
        )

    metrics = ic_metrics(pred_store, label_store, mask_store)
    if compute_portfolio:
        metrics.update(
            topk_portfolio(
                pred_store,
                label_store,
                mask_store,
                date_store,
                k=50,
                instrument_lists=instrument_store,
            )
        )
    return metrics


def run_train_group(spec: GroupSpec, config_path: Path, seeds: list[int]) -> None:
    cfg = apply_train_window_overrides(apply_overrides(load_yaml(config_path), spec.base_overrides))
    cfg.setdefault("log", {})["out_dir"] = str(RESULT_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, valid_ds, test_ds = build_datasets(cfg)
    train_day_weight_map = build_recency_weight_map(cfg["train"], train_ds)
    print(
        f"[{spec.group_name}] datasets loaded once: "
        f"train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)} d_in={train_ds.feature_dim}"
    )

    for seed in seeds:
        exp_name = f"{spec.family}_csi300_{spec.split}_seed{seed}"
        result_path = TABLES_DIR / f"{exp_name}.json"
        if result_path.exists():
            print(f"[{spec.group_name}] SKIP {exp_name}")
            continue

        try:
            local_cfg = deepcopy(cfg)
            local_cfg["train"]["seed"] = int(seed)
            local_cfg["log"]["exp_name"] = exp_name
            set_seed(int(seed))

            train_loader, valid_loader, test_loader = build_loaders(local_cfg, train_ds, valid_ds, test_ds)
            model = build_model(local_cfg["model"], d_in=train_ds.feature_dim, train_dataset=train_ds).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(local_cfg["train"]["lr"]),
                weight_decay=float(local_cfg["train"]["weight_decay"]),
            )
            loss_fn = build_loss(
                loss_name=str(local_cfg["train"]["loss"]).lower(),
                drop_extreme_pct=float(local_cfg["train"].get("drop_extreme_pct", 0.0)),
                wpcc_weight=float(local_cfg["train"].get("wpcc_weight", 0.0)),
                ic_weight=float(local_cfg["train"].get("ic_weight", 0.0)),
            )

            max_epochs = int(local_cfg["train"]["epochs"])
            patience = int(local_cfg["train"]["early_stop_patience"])
            grad_clip = float(local_cfg["train"]["grad_clip"])
            best_icir = -float("inf")
            best_epoch = 0
            bad_epochs = 0
            ckpt_path = CKPT_DIR / f"{exp_name}.pt"
            log_path = LOG_DIR / f"{exp_name}.csv"
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            TABLES_DIR.mkdir(parents=True, exist_ok=True)

            with log_path.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(
                    fp,
                    fieldnames=[
                        "epoch",
                        "train_loss",
                        "train_reg_loss",
                        "train_total_loss",
                        "train_IC",
                        "valid_IC",
                        "valid_ICIR",
                        "valid_RankIC",
                    ],
                )
                writer.writeheader()
                for epoch in range(1, max_epochs + 1):
                    train_out = train_one_epoch(
                        model=model,
                        loader=train_loader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        device=device,
                        grad_clip=grad_clip,
                        day_weight_map=train_day_weight_map,
                    )
                    valid_metrics = evaluate(model=model, loader=valid_loader, device=device, compute_portfolio=False)
                    writer.writerow(
                        {
                            "epoch": epoch,
                            "train_loss": train_out["pred_loss"],
                            "train_reg_loss": train_out["reg_loss"],
                            "train_total_loss": train_out["total_loss"],
                            "train_IC": train_out["metrics"]["IC_mean"],
                            "valid_IC": valid_metrics["IC_mean"],
                            "valid_ICIR": valid_metrics["ICIR"],
                            "valid_RankIC": valid_metrics["RankIC_mean"],
                        }
                    )
                    fp.flush()
                    if valid_metrics["ICIR"] > best_icir:
                        best_icir = valid_metrics["ICIR"]
                        best_epoch = epoch
                        bad_epochs = 0
                        torch.save(
                            {
                                "model_state": model.state_dict(),
                                "config": local_cfg,
                                "feature_dim": train_ds.feature_dim,
                                "feature_names": train_ds.feature_names,
                                "epoch": epoch,
                                "valid_metrics": valid_metrics,
                            },
                            ckpt_path,
                        )
                    else:
                        bad_epochs += 1
                    if bad_epochs >= patience:
                        break

            checkpoint = safe_torch_load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            test_metrics = evaluate(model=model, loader=test_loader, device=device, compute_portfolio=True)
            payload = {
                "exp_name": exp_name,
                "model": local_cfg["model"]["name"],
                "market": local_cfg["market"],
                "handler": local_cfg.get("handler", "Alpha158"),
                "best_epoch": best_epoch,
                "valid_best_ICIR": best_icir,
                "test_metrics": test_metrics,
            }
            result_path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")
            print(f"[{spec.group_name}] DONE {exp_name} IC={test_metrics['IC_mean']:.6f}")
        except Exception:
            err_path = _save_error(exp_name)
            print(f"[{spec.group_name}] FAILED {exp_name} -> {err_path}")


def run_qworkflow_group(spec: GroupSpec, config_path: Path, seeds: list[int]) -> None:
    for seed in seeds:
        exp_name = f"{spec.family}_csi300_{spec.split}_seed{seed}"
        result_path = TABLES_DIR / f"{exp_name}.json"
        if result_path.exists():
            print(f"[{spec.group_name}] SKIP {exp_name}")
            continue
        try:
            overrides = list(spec.base_overrides) + [f"task.model.kwargs.seed={seed}"]
            if spec.family == "master":
                save_dir = RESULT_ROOT / "model_ckpt" / "master"
                save_dir.mkdir(parents=True, exist_ok=True)
                overrides.extend(
                    [
                        f"task.model.kwargs.save_path={str(save_dir)}/",
                        f"task.model.kwargs.save_prefix={spec.split}_seed{seed}_",
                    ]
                )

            cmd = [
                str(PYTHON),
                str(RUN_QWORKFLOW),
                "--config",
                str(config_path),
                "--experiment",
                exp_name,
                "--summary_out",
                str(result_path),
            ]
            for override in overrides:
                cmd.extend(["--override", override])

            env = os.environ.copy()
            env.update(THREAD_ENV)
            env.setdefault("PYTHONPATH", str(ROOT))
            env.setdefault("QLIB_FORK_PATH", str(QLIB_FORK_ROOT))
            if PROVIDER_URI_OVERRIDE:
                env["SEED100_PROVIDER_URI"] = PROVIDER_URI_OVERRIDE

            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"run_qworkflow failed with code {proc.returncode}\n"
                    + proc.stdout[-4000:]
                )

            payload = json.loads(result_path.read_text(encoding="utf-8"))
            print(f"[{spec.group_name}] DONE {exp_name} IC={payload['test_metrics']['IC_mean']:.6f}")
        except Exception:
            err_path = _save_error(exp_name)
            print(f"[{spec.group_name}] FAILED {exp_name} -> {err_path}")


def summarize(specs: list[GroupSpec], seeds: list[int]) -> tuple[Path, Path]:
    from collections import defaultdict
    import statistics as st

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    completed = 0
    for spec in specs:
        for seed in seeds:
            result_path = TABLES_DIR / f"{spec.family}_csi300_{spec.split}_seed{seed}.json"
            if not result_path.exists():
                continue
            completed += 1
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            metrics = payload.get("test_metrics", {})
            if "IC_mean" in metrics:
                grouped[(spec.family, spec.split)].append(float(metrics["IC_mean"]))

    rows = []
    for (family, split), values in sorted(grouped.items()):
        rows.append(
            {
                "family": family,
                "split": split,
                "count": len(values),
                "ic_mean": float(st.mean(values)),
                "ic_std": float(st.pstdev(values)) if len(values) > 1 else 0.0,
                "ic_best": float(max(values)),
                "ic_worst": float(min(values)),
            }
        )

    payload = {"completed": completed, "total": len(specs) * len(seeds), "rows": rows}
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Seed100 Grouped CPU Summary",
        "",
        f"- completed: `{completed}/{len(specs) * len(seeds)}`",
        "",
        "| Family | Split | Count | IC Mean | IC Std | Best | Worst |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['family']} | {row['split']} | {row['count']} | "
            f"{row['ic_mean']:.6f} | {row['ic_std']:.6f} | "
            f"{row['ic_best']:.6f} | {row['ic_worst']:.6f} |"
        )
    SUMMARY_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return SUMMARY_MD, SUMMARY_JSON


def main() -> None:
    args = parse_args()
    seeds = [int(seed) for seed in args.seeds]
    specs = build_group_specs(args.models)
    cfgs = generate_configs()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)

    if args.action == "launch":
        scripts = write_group_scripts(specs, seeds)
        manifest = write_manifest(specs, seeds)
        launch_group_scripts(scripts)
        print(f"Launched {len(specs)} grouped tasks.")
        print(f"Manifest: {manifest}")
        return

    if args.action == "run-group":
        if not args.family or not args.split:
            raise ValueError("run-group requires --family and --split")
        target = None
        for spec in specs:
            if spec.family == args.family and spec.split == args.split:
                target = spec
                break
        if target is None:
            raise ValueError(f"Unknown group: {args.family}/{args.split}")
        config_path = cfgs[target.config_key]
        print(f"RUN_GROUP {target.group_name} seeds={len(seeds)} first_seed={seeds[0]}")
        if target.runner == "train_single":
            run_train_group(target, config_path, seeds)
        else:
            run_qworkflow_group(target, config_path, seeds)
        return

    md, js = summarize(specs, seeds)
    print(f"Summary written to {md}")
    print(f"Summary json: {js}")


if __name__ == "__main__":
    main()
