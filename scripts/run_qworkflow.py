from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _inject_qlib_fork() -> None:
    candidates: list[Path] = []
    env_path = os.environ.get("QLIB_FORK_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            Path("/project/user186_refs/qlib_sjtu"),
            Path.home() / "refs" / "qlib_sjtu",
        ]
    )
    for candidate in candidates:
        if (candidate / "qlib").exists():
            sys.path.insert(0, str(candidate))
            return


_inject_qlib_fork()

# Older dask/lightgbm stacks still reach for this pandas symbol.
try:
    from pandas.core.strings.accessor import StringMethods as _PandasStringMethods

    if not hasattr(pd.core.strings, "StringMethods"):
        pd.core.strings.StringMethods = _PandasStringMethods
except Exception:
    pass

import qlib
import yaml
from qlib.utils import init_instance_by_config
from qlib.workflow import R


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a Qlib workflow yaml.")
    parser.add_argument("--experiment", default="workflow", help="Experiment name for the recorder.")
    parser.add_argument("--override", action="append", default=[], help="YAML-style key=value override.")
    parser.add_argument("--summary_out", default=None, help="Optional path to write a compact result json.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    cfg = deepcopy(config)
    for item in overrides:
        key, raw_value = item.split("=", maxsplit=1)
        value = yaml.safe_load(raw_value)
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return cfg


def _resolve_placeholders(value, *, model, dataset):
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v, model=model, dataset=dataset) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(v, model=model, dataset=dataset) for v in value]
    if value == "<MODEL>":
        return model
    if value == "<DATASET>":
        return dataset
    return value


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom < 1e-12:
        return np.nan
    return float((x * y).sum() / denom)


def _series_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ir": 0.0, "count": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ir = float(mean / std) if std > 1e-12 else 0.0
    return {"mean": mean, "std": std, "ir": ir, "count": int(arr.size)}


def _mlruns_root(recorder) -> Path:
    uri = getattr(recorder, "uri", None)
    if isinstance(uri, str) and uri.startswith("file:"):
        return Path(uri[5:])
    return Path.cwd() / "mlruns"


def _load_recorder_object(recorder, object_name: str):
    try:
        return recorder.load_object(object_name)
    except Exception:
        artifacts = sorted(_mlruns_root(recorder).glob(f"*/{recorder.id}/artifacts/{object_name}"))
        if not artifacts:
            raise
        return pd.read_pickle(artifacts[-1])


def _safe_list_metrics(recorder) -> dict:
    try:
        return recorder.list_metrics()
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


def _compute_signal_metrics(recorder) -> dict:
    pred = _load_recorder_object(recorder, "pred.pkl")
    label = _load_recorder_object(recorder, "label.pkl")

    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    if isinstance(label, pd.Series):
        label = label.to_frame("label")

    pred = pred.rename(columns={pred.columns[0]: "score"})
    label = label.rename(columns={label.columns[0]: "label"})
    joined = pred.join(label, how="inner")
    if joined.empty:
        return {
            "IC_mean": 0.0,
            "IC_std": 0.0,
            "ICIR": 0.0,
            "RankIC_mean": 0.0,
            "RankIC_std": 0.0,
            "RankICIR": 0.0,
            "n_days": 0,
        }

    ic_values = []
    rankic_values = []
    for _, frame in joined.groupby(level=0):
        score = frame["score"].to_numpy(dtype=np.float64)
        target = frame["label"].to_numpy(dtype=np.float64)
        mask = np.isfinite(score) & np.isfinite(target)
        if mask.sum() < 2:
            continue
        score = score[mask]
        target = target[mask]
        ic_values.append(_safe_corr(score, target))
        score_rank = pd.Series(score).rank(method="average").to_numpy(dtype=np.float64)
        target_rank = pd.Series(target).rank(method="average").to_numpy(dtype=np.float64)
        rankic_values.append(_safe_corr(score_rank, target_rank))

    ic_stats = _series_stats(ic_values)
    rankic_stats = _series_stats(rankic_values)
    return {
        "IC_mean": ic_stats["mean"],
        "IC_std": ic_stats["std"],
        "ICIR": ic_stats["ir"],
        "RankIC_mean": rankic_stats["mean"],
        "RankIC_std": rankic_stats["std"],
        "RankICIR": rankic_stats["ir"],
        "n_days": ic_stats["count"],
    }


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(value) for value in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.override)

    qlib_cfg = deepcopy(config.get("qlib_init", {}))
    qlib.init(**qlib_cfg)

    task = config["task"]
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    summary = None
    with R.start(experiment_name=args.experiment):
        recorder = R.get_recorder()
        if hasattr(model, "fit"):
            model_save_path = task.get("model_save_path")
            if model_save_path:
                model.fit(dataset, save_path=model_save_path)
            else:
                model.fit(dataset)

        for record_cfg in task.get("record", []):
            resolved_cfg = deepcopy(record_cfg)
            resolved_cfg["kwargs"] = _resolve_placeholders(
                resolved_cfg.get("kwargs", {}),
                model=model,
                dataset=dataset,
            )
            record = init_instance_by_config(resolved_cfg, recorder=recorder)
            record.generate()

        if args.summary_out:
            payload = {
                "experiment": args.experiment,
                "config": args.config,
                "overrides": args.override,
                "recorder_id": recorder.id,
                "recorder_name": recorder.name,
                "source_run": getattr(recorder, "uri", None),
                "logged_metrics": _safe_list_metrics(recorder),
                "test_metrics": _compute_signal_metrics(recorder),
            }
            summary = _to_serializable(payload)

    if args.summary_out and summary is not None:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        print(f"Saved qworkflow summary to {summary_path}")


if __name__ == "__main__":
    main()
