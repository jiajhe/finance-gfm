from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.qlib_loader import (
    InMemoryCrossSectionalDataset,
    build_qlib_handler_bundle,
    build_qlib_handler_datasets,
    pad_collate,
)
from eval.metrics import ic as ic_metrics
from eval.portfolio import topk_portfolio
from models import build_model
from train.loss import build_loss
from train.train_single import evaluate, safe_torch_load, to_serializable, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Base recent config to diagnose.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for this run.")
    parser.add_argument("--exp_name", type=str, default=None, help="Output experiment name.")
    parser.add_argument(
        "--compute_drift",
        action="store_true",
        help="Also compute train/early/late/test feature drift for the processed inputs.",
    )
    parser.add_argument(
        "--drift_output",
        type=str,
        default=None,
        help="Optional explicit JSON path for the drift report.",
    )
    parser.add_argument(
        "--drift_only",
        action="store_true",
        help="Only build the diagnostic splits and compute feature drift.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_loss(train_cfg: dict):
    return build_loss(
        loss_name=str(train_cfg["loss"]).lower(),
        drop_extreme_pct=float(train_cfg.get("drop_extreme_pct", 0.0)),
        wpcc_weight=float(train_cfg.get("wpcc_weight", 0.0)),
        ic_weight=float(train_cfg.get("ic_weight", 0.0)),
    )


def _loader(dataset, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=pad_collate,
        pin_memory=torch.cuda.is_available(),
    )


def _recent_diagnostic_splits(cfg: dict) -> tuple[dict, dict]:
    _, valid_ds, _ = build_qlib_handler_datasets(cfg)
    valid_dates = [day["date"] for day in valid_ds.days]
    if len(valid_dates) < 20:
        raise ValueError("Validation split is too short to split into early/late halves.")

    midpoint = len(valid_dates) // 2
    valid_start = valid_dates[0].strftime("%Y-%m-%d")
    early_end = valid_dates[midpoint - 1].strftime("%Y-%m-%d")
    late_start = valid_dates[midpoint].strftime("%Y-%m-%d")
    valid_end = valid_dates[-1].strftime("%Y-%m-%d")

    train_start, train_end = cfg["splits"]["train"]
    test_start, test_end = cfg["splits"]["test"]
    diag_splits = {
        "train": [str(train_start), str(train_end)],
        "valid_early": [valid_start, early_end],
        "valid_late": [late_start, valid_end],
        "test": [str(test_start), str(test_end)],
    }
    split_meta = {
        "valid_total_days": len(valid_dates),
        "valid_early_days": midpoint,
        "valid_late_days": len(valid_dates) - midpoint,
        "valid_start": valid_start,
        "valid_early_end": early_end,
        "valid_late_start": late_start,
        "valid_end": valid_end,
    }
    return diag_splits, split_meta


def build_recent_diagnostic_datasets(cfg: dict):
    diag_splits, split_meta = _recent_diagnostic_splits(cfg)
    bundle = build_qlib_handler_bundle(cfg, splits=diag_splits)
    feature_names = bundle["feature_names"]
    train_ds = InMemoryCrossSectionalDataset(bundle["splits"]["train"], feature_names)
    early_ds = InMemoryCrossSectionalDataset(bundle["splits"]["valid_early"], feature_names)
    late_ds = InMemoryCrossSectionalDataset(bundle["splits"]["valid_late"], feature_names)
    test_ds = InMemoryCrossSectionalDataset(bundle["splits"]["test"], feature_names)
    return train_ds, early_ds, late_ds, test_ds, split_meta


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def _gap_payload(early_metrics: dict, late_metrics: dict, test_metrics: dict) -> dict:
    early_ic = float(early_metrics["IC_mean"])
    late_ic = float(late_metrics["IC_mean"])
    test_ic = float(test_metrics["IC_mean"])
    return {
        "late_minus_early_ic": late_ic - early_ic,
        "test_minus_late_ic": test_ic - late_ic,
        "test_minus_early_ic": test_ic - early_ic,
        "late_over_early_ic": _safe_ratio(late_ic, early_ic),
        "test_over_late_ic": _safe_ratio(test_ic, late_ic),
        "late_rankic_minus_early": float(late_metrics["RankIC_mean"]) - float(early_metrics["RankIC_mean"]),
        "test_rankic_minus_late": float(test_metrics["RankIC_mean"]) - float(late_metrics["RankIC_mean"]),
    }


def _sample_train_rows(dataset, *, max_rows: int = 200000, seed: int = 2026) -> np.ndarray:
    total_rows = sum(int(day["X"].shape[0]) for day in dataset.days)
    keep_prob = min(1.0, max_rows / max(1, total_rows))
    rng = np.random.default_rng(seed)
    chunks = []
    count = 0
    for day in dataset.days:
        array = day["X"].detach().cpu().numpy()
        mask = rng.random(array.shape[0]) < keep_prob
        if mask.any():
            chosen = array[mask]
            chunks.append(chosen)
            count += chosen.shape[0]
    if not chunks:
        return dataset.days[0]["X"].detach().cpu().numpy()
    sample = np.concatenate(chunks, axis=0)
    if sample.shape[0] > max_rows:
        indices = rng.choice(sample.shape[0], size=max_rows, replace=False)
        sample = sample[indices]
    return sample


def _train_reference_bins(train_ds, *, quantiles: int = 10, seed: int = 2026) -> list[np.ndarray]:
    sample = _sample_train_rows(train_ds, max_rows=200000, seed=seed)
    bins = []
    q = np.linspace(0.0, 1.0, quantiles + 1)
    for feature_idx in range(sample.shape[1]):
        col = sample[:, feature_idx]
        edges = np.quantile(col, q)
        edges = np.unique(edges)
        if edges.size < 2:
            center = float(col[0]) if col.size else 0.0
            edges = np.array([center - 1e-6, center + 1e-6], dtype=np.float64)
        else:
            edges[0] -= 1e-6
            edges[-1] += 1e-6
        bins.append(edges.astype(np.float64))
    return bins


def _stream_split_stats(dataset, bins: list[np.ndarray]) -> dict:
    d_in = len(bins)
    counts = [np.zeros(len(feature_bins) - 1, dtype=np.float64) for feature_bins in bins]
    sum_x = np.zeros(d_in, dtype=np.float64)
    sum_x2 = np.zeros(d_in, dtype=np.float64)
    n_obs = np.zeros(d_in, dtype=np.float64)

    for day in dataset.days:
        array = day["X"].detach().cpu().numpy().astype(np.float64, copy=False)
        sum_x += array.sum(axis=0)
        sum_x2 += np.square(array).sum(axis=0)
        n_obs += array.shape[0]
        for feature_idx in range(d_in):
            hist, _ = np.histogram(array[:, feature_idx], bins=bins[feature_idx])
            counts[feature_idx] += hist

    means = sum_x / np.maximum(n_obs, 1.0)
    variances = np.maximum(sum_x2 / np.maximum(n_obs, 1.0) - np.square(means), 0.0)
    stds = np.sqrt(variances)
    return {
        "counts": counts,
        "means": means,
        "stds": stds,
        "n_obs": n_obs,
    }


def _drift_from_counts(train_counts: np.ndarray, other_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-8
    psi_values = []
    kl_values = []
    for train_hist, other_hist in zip(train_counts, other_counts):
        p = train_hist + eps
        q = other_hist + eps
        p = p / p.sum()
        q = q / q.sum()
        psi_values.append(float(((p - q) * np.log(p / q)).sum()))
        kl_values.append(float((p * np.log(p / q)).sum()))
    return np.asarray(psi_values, dtype=np.float64), np.asarray(kl_values, dtype=np.float64)


def compute_feature_drift(train_ds, early_ds, late_ds, test_ds, *, seed: int) -> dict:
    feature_names = list(train_ds.feature_names)
    bins = _train_reference_bins(train_ds, seed=seed)
    train_stats = _stream_split_stats(train_ds, bins)
    reports = {}
    for split_name, dataset in [
        ("valid_early", early_ds),
        ("valid_late", late_ds),
        ("test", test_ds),
    ]:
        split_stats = _stream_split_stats(dataset, bins)
        psi, kl = _drift_from_counts(train_stats["counts"], split_stats["counts"])
        mean_shift = np.abs(split_stats["means"] - train_stats["means"])
        std_ratio = split_stats["stds"] / np.maximum(train_stats["stds"], 1e-8)
        order = np.argsort(-psi)
        top_features = [
            {
                "feature": feature_names[idx],
                "psi": float(psi[idx]),
                "kl": float(kl[idx]),
                "train_mean": float(train_stats["means"][idx]),
                "split_mean": float(split_stats["means"][idx]),
                "train_std": float(train_stats["stds"][idx]),
                "split_std": float(split_stats["stds"][idx]),
            }
            for idx in order[:10]
        ]
        reports[split_name] = {
            "median_psi": float(np.median(psi)),
            "mean_psi": float(np.mean(psi)),
            "max_psi": float(np.max(psi)),
            "median_kl": float(np.median(kl)),
            "mean_kl": float(np.mean(kl)),
            "median_abs_mean_shift": float(np.median(mean_shift)),
            "mean_abs_mean_shift": float(np.mean(mean_shift)),
            "median_std_ratio": float(np.median(std_ratio)),
            "mean_std_ratio": float(np.mean(std_ratio)),
            "top_features_by_psi": top_features,
        }
    return {
        "reference": "processed model inputs after Qlib handler pipeline",
        "feature_dim": len(feature_names),
        "feature_names": feature_names,
        "splits": reports,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["train"]["seed"] = int(args.seed)
    exp_name = args.exp_name or f"{cfg['log']['exp_name']}_diag_seed{args.seed}"

    out_dir = Path(cfg["log"]["out_dir"]).expanduser()
    logs_dir = out_dir / "logs"
    ckpt_dir = out_dir / "ckpts"
    tables_dir = out_dir / "tables"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, early_ds, late_ds, test_ds, split_meta = build_recent_diagnostic_datasets(cfg)
    train_loader = _loader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True)
    early_loader = _loader(early_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    late_loader = _loader(late_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    test_loader = _loader(test_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)

    print(
        "Loaded diagnostic datasets: "
        f"train={len(train_ds)} early={len(early_ds)} late={len(late_ds)} "
        f"test={len(test_ds)} d_in={train_ds.feature_dim}"
    )
    print(
        "Valid split: "
        f"{split_meta['valid_start']}..{split_meta['valid_early_end']} "
        f"({split_meta['valid_early_days']} days) | "
        f"{split_meta['valid_late_start']}..{split_meta['valid_end']} "
        f"({split_meta['valid_late_days']} days)"
    )

    drift_payload = None
    if args.compute_drift or args.drift_only:
        drift_payload = compute_feature_drift(train_ds, early_ds, late_ds, test_ds, seed=int(args.seed))
        drift_output = (
            Path(args.drift_output)
            if args.drift_output
            else tables_dir / f"{Path(args.config).stem}_recent_feature_drift.json"
        )
        drift_output.parent.mkdir(parents=True, exist_ok=True)
        with drift_output.open("w", encoding="utf-8") as fp:
            json.dump(to_serializable(drift_payload), fp, indent=2)
        print(f"Saved drift report to {drift_output}")

        if args.drift_only:
            result_payload = {
                "base_config": args.config,
                "seed": int(args.seed),
                "split_meta": split_meta,
                "feature_drift": drift_payload,
            }
            result_path = tables_dir / f"{exp_name}.json"
            with result_path.open("w", encoding="utf-8") as fp:
                json.dump(to_serializable(result_payload), fp, indent=2)
            print(f"Saved drift-only payload to {result_path}")
            return

    model = build_model(cfg["model"], d_in=train_ds.feature_dim, train_dataset=train_ds).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    loss_fn = choose_loss(cfg["train"])

    max_epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"]["early_stop_patience"])
    grad_clip = float(cfg["train"]["grad_clip"])
    best_icir = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    ckpt_path = ckpt_dir / f"{exp_name}.pt"
    log_path = logs_dir / f"{exp_name}.csv"

    with log_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_reg_loss",
                "train_total_loss",
                "train_IC",
                "early_valid_IC",
                "early_valid_ICIR",
                "early_valid_RankIC",
                "late_valid_IC",
                "late_valid_ICIR",
                "late_valid_RankIC",
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
            )
            early_metrics = evaluate(model=model, loader=early_loader, device=device, compute_portfolio=False)
            late_metrics = evaluate(model=model, loader=late_loader, device=device, compute_portfolio=False)

            row = {
                "epoch": epoch,
                "train_loss": train_out["pred_loss"],
                "train_reg_loss": train_out["reg_loss"],
                "train_total_loss": train_out["total_loss"],
                "train_IC": train_out["metrics"]["IC_mean"],
                "early_valid_IC": early_metrics["IC_mean"],
                "early_valid_ICIR": early_metrics["ICIR"],
                "early_valid_RankIC": early_metrics["RankIC_mean"],
                "late_valid_IC": late_metrics["IC_mean"],
                "late_valid_ICIR": late_metrics["ICIR"],
                "late_valid_RankIC": late_metrics["RankIC_mean"],
            }
            writer.writerow(row)
            fp.flush()

            if early_metrics["ICIR"] > best_icir:
                best_icir = early_metrics["ICIR"]
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": cfg,
                        "feature_dim": train_ds.feature_dim,
                        "feature_names": train_ds.feature_names,
                        "epoch": epoch,
                        "early_metrics": early_metrics,
                        "late_metrics": late_metrics,
                    },
                    ckpt_path,
                )
            else:
                bad_epochs += 1

            print(
                f"epoch={epoch:03d} "
                f"train_IC={train_out['metrics']['IC_mean']:.4f} "
                f"early_IC={early_metrics['IC_mean']:.4f} "
                f"late_IC={late_metrics['IC_mean']:.4f} "
                f"early_ICIR={early_metrics['ICIR']:.4f}"
            )

            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}).")
                break

    checkpoint = safe_torch_load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    early_metrics = evaluate(model=model, loader=early_loader, device=device, compute_portfolio=False)
    late_metrics = evaluate(model=model, loader=late_loader, device=device, compute_portfolio=False)
    test_metrics = evaluate(model=model, loader=test_loader, device=device, compute_portfolio=True)

    result_payload = {
        "exp_name": exp_name,
        "base_config": args.config,
        "model": cfg["model"]["name"],
        "market": cfg["market"],
        "handler": cfg.get("handler", "Alpha158"),
        "seed": int(args.seed),
        "best_epoch": best_epoch,
        "early_valid_best_ICIR": best_icir,
        "split_meta": split_meta,
        "early_valid_metrics": early_metrics,
        "late_valid_metrics": late_metrics,
        "test_metrics": test_metrics,
        "gap_metrics": _gap_payload(early_metrics, late_metrics, test_metrics),
    }

    if args.compute_drift:
        result_payload["feature_drift"] = drift_payload

    result_path = tables_dir / f"{exp_name}.json"
    with result_path.open("w", encoding="utf-8") as fp:
        json.dump(to_serializable(result_payload), fp, indent=2)

    print(f"Saved log to {log_path}")
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Saved diagnostic metrics to {result_path}")


if __name__ == "__main__":
    main()
