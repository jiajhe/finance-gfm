from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.qlib_loader import QlibCrossSectionalDataset, build_qlib_handler_datasets, pad_collate
from eval.metrics import ic as ic_metrics
from eval.portfolio import topk_portfolio
from models import build_model
from train.loss import build_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--exp_name", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    cfg = copy.deepcopy(config)
    for item in overrides:
        key, raw_value = item.split("=", maxsplit=1)
        value = yaml.safe_load(raw_value)
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return cfg


def apply_train_window_overrides(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    train_cfg = cfg.get("train", {})
    window_years = train_cfg.get("train_window_years")
    if window_years is None:
        return cfg

    train_end = pd.Timestamp(cfg["splits"]["train"][1])
    train_start = train_end - pd.DateOffset(years=int(window_years)) + pd.Timedelta(days=1)
    train_start_str = train_start.strftime("%Y-%m-%d")
    train_end_str = train_end.strftime("%Y-%m-%d")
    cfg["splits"]["train"] = [train_start_str, train_end_str]
    cfg["fit_start_time"] = train_start_str
    cfg["fit_end_time"] = train_end_str
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_torch_load(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_datasets(cfg: dict):
    dataset_mode = str(cfg.get("dataset_mode", "raw")).lower()
    if dataset_mode in {"qlib_handler", "official"}:
        return build_qlib_handler_datasets(cfg)

    common_kwargs = {
        "market": cfg["market"],
        "handler": cfg.get("handler", "Alpha158"),
        "cache_dir": cfg.get("cache_dir", "~/.qlib_cache"),
        "provider_uri": cfg.get("provider_uri"),
        "label": cfg.get("label", "Ref($close, -2) / Ref($close, -1) - 1"),
        "chunk_size_days": int(cfg.get("chunk_size_days", 252)),
        "feature_limit": cfg.get("feature_limit"),
        "history_window": int(cfg.get("history_window", cfg.get("model", {}).get("history_window", 0))),
    }
    splits = cfg["splits"]
    train_ds = QlibCrossSectionalDataset(
        start_time=splits["train"][0],
        end_time=splits["train"][1],
        **common_kwargs,
    )
    valid_ds = QlibCrossSectionalDataset(
        start_time=splits["valid"][0],
        end_time=splits["valid"][1],
        **common_kwargs,
    )
    test_ds = QlibCrossSectionalDataset(
        start_time=splits["test"][0],
        end_time=splits["test"][1],
        **common_kwargs,
    )
    return train_ds, valid_ds, test_ds


def build_loaders(cfg: dict, train_ds, valid_ds, test_ds):
    batch_size = int(cfg["train"]["batch_size"])
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "collate_fn": pad_collate,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, valid_loader, test_loader


def choose_loss(train_cfg: dict):
    return build_loss(
        loss_name=str(train_cfg["loss"]).lower(),
        drop_extreme_pct=float(train_cfg.get("drop_extreme_pct", 0.0)),
        wpcc_weight=float(train_cfg.get("wpcc_weight", 0.0)),
        ic_weight=float(train_cfg.get("ic_weight", 0.0)),
    )


def _date_key(date) -> str:
    return pd.Timestamp(date).strftime("%Y-%m-%d")


def build_recency_weight_map(train_cfg: dict, train_dataset) -> dict[str, float] | None:
    spec = train_cfg.get("recency_weighting") or {}
    mode = str(spec.get("mode", "none")).lower()
    if mode in {"none", "off", "false"}:
        return None
    if mode not in {"exp", "exponential"}:
        raise ValueError(f"Unsupported recency weighting mode: {mode}")

    lambda_days = int(spec.get("lambda_days", 504))
    if lambda_days <= 0:
        raise ValueError("recency_weighting.lambda_days must be positive.")

    num_days = len(train_dataset.days)
    day_positions = np.arange(num_days, dtype=np.float64)
    weights = np.exp(-(num_days - 1 - day_positions) / float(lambda_days))
    weights = weights / max(weights.mean(), 1e-12)
    return {
        _date_key(day["date"]): float(weight)
        for day, weight in zip(train_dataset.days, weights)
    }


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
                [day_weight_map[_date_key(date)] for date in dates],
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
            preds=preds,
            labels=y,
            raw_labels=raw_y,
            masks=mask,
            dates=dates,
            instruments=instruments,
            pred_store=pred_store,
            label_store=label_store,
            mask_store=mask_store,
            date_store=date_store,
            instrument_store=instrument_store,
        )

    metrics = ic_metrics(pred_store, label_store, mask_store)
    return {
        "pred_loss": float(np.mean(pred_losses)) if pred_losses else 0.0,
        "reg_loss": float(np.mean(reg_losses)) if reg_losses else 0.0,
        "total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
        "metrics": metrics,
    }


@torch.no_grad()
def evaluate(model, loader, device, compute_portfolio: bool = False):
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
            preds=preds,
            labels=y,
            raw_labels=raw_y,
            masks=mask,
            dates=dates,
            instruments=instruments,
            pred_store=pred_store,
            label_store=label_store,
            mask_store=mask_store,
            date_store=date_store,
            instrument_store=instrument_store,
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


def to_serializable(obj):
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(value) for value in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main() -> None:
    args = parse_args()
    cfg = apply_train_window_overrides(apply_overrides(load_config(args.config), args.override))
    exp_name = args.exp_name or cfg["log"]["exp_name"]
    out_dir = Path(cfg["log"]["out_dir"]).expanduser()
    logs_dir = out_dir / "logs"
    ckpt_dir = out_dir / "ckpts"
    tables_dir = out_dir / "tables"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, valid_ds, test_ds = build_datasets(cfg)
    print(
        f"Loaded datasets: train={len(train_ds)} days, "
        f"valid={len(valid_ds)} days, test={len(test_ds)} days, "
        f"d_in={train_ds.feature_dim}"
    )
    train_loader, valid_loader, test_loader = build_loaders(cfg, train_ds, valid_ds, test_ds)
    train_day_weight_map = build_recency_weight_map(cfg["train"], train_ds)

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

            row = {
                "epoch": epoch,
                "train_loss": train_out["pred_loss"],
                "train_reg_loss": train_out["reg_loss"],
                "train_total_loss": train_out["total_loss"],
                "train_IC": train_out["metrics"]["IC_mean"],
                "valid_IC": valid_metrics["IC_mean"],
                "valid_ICIR": valid_metrics["ICIR"],
                "valid_RankIC": valid_metrics["RankIC_mean"],
            }
            writer.writerow(row)
            fp.flush()

            if valid_metrics["ICIR"] > best_icir:
                best_icir = valid_metrics["ICIR"]
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": cfg,
                        "feature_dim": train_ds.feature_dim,
                        "feature_names": train_ds.feature_names,
                        "epoch": epoch,
                        "valid_metrics": valid_metrics,
                    },
                    ckpt_path,
                )
            else:
                bad_epochs += 1

            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_out['pred_loss']:.6f} "
                f"train_reg_loss={train_out['reg_loss']:.6f} "
                f"train_IC={train_out['metrics']['IC_mean']:.4f} "
                f"valid_IC={valid_metrics['IC_mean']:.4f} "
                f"valid_ICIR={valid_metrics['ICIR']:.4f}"
            )

            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}).")
                break

    checkpoint = safe_torch_load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model=model, loader=test_loader, device=device, compute_portfolio=True)
    result_payload = {
        "exp_name": exp_name,
        "model": cfg["model"]["name"],
        "market": cfg["market"],
        "handler": cfg.get("handler", "Alpha158"),
        "best_epoch": best_epoch,
        "valid_best_ICIR": best_icir,
        "test_metrics": test_metrics,
    }
    result_path = tables_dir / f"{exp_name}.json"
    with result_path.open("w", encoding="utf-8") as fp:
        json.dump(to_serializable(result_payload), fp, indent=2)

    print(f"Saved log to {log_path}")
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Saved test metrics to {result_path}")


if __name__ == "__main__":
    main()
