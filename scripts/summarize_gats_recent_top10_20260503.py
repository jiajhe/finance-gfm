from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


SEEDS = [512, 23, 37, 5, 40, 55, 25, 49, 17, 79]
METRICS = [
    "IC_mean",
    "IC_std",
    "ICIR",
    "RankIC_mean",
    "RankIC_std",
    "RankICIR",
    "annual_return",
    "annual_vol",
    "sharpe",
    "max_drawdown",
    "turnover",
]


def _fmt(value) -> str:
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def _read_metric_file(path: Path) -> float | None:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return None
    if not lines:
        return None
    parts = lines[-1].split()
    if len(parts) < 2:
        return None
    try:
        value = float(parts[1])
    except ValueError:
        return None
    if math.isnan(value):
        return None
    return value


def _find_run_dir(payload: dict) -> Path | None:
    recorder_id = payload.get("recorder_id")
    source_run = payload.get("source_run")
    if not recorder_id or not isinstance(source_run, str) or not source_run.startswith("file:"):
        return None
    root = Path(source_run[5:])
    matches = sorted(root.glob(f"*/{recorder_id}"), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def _load_portfolio_metrics(payload: dict) -> dict:
    run_dir = _find_run_dir(payload)
    if run_dir is None:
        return {}

    metrics_dir = run_dir / "metrics"
    raw = {}
    if metrics_dir.exists():
        for path in metrics_dir.iterdir():
            if path.is_file():
                raw[path.name] = _read_metric_file(path)

    base = "1day.excess_return_without_cost."
    annual_return = raw.get(base + "annualized_return")
    qlib_ir = raw.get(base + "information_ratio")
    daily_std = raw.get(base + "std")
    annual_vol = None
    if annual_return is not None and qlib_ir is not None and abs(qlib_ir) > 1e-12:
        annual_vol = abs(annual_return / qlib_ir)
    elif daily_std is not None:
        annual_vol = float(daily_std) * math.sqrt(252.0)

    turnover = None
    try:
        import pandas as pd

        report_path = run_dir / "artifacts" / "portfolio_analysis" / "report_normal_1day.pkl"
        if report_path.exists():
            report = pd.read_pickle(report_path)
            if "total_turnover" in report.columns:
                turnover = float(report["total_turnover"].replace([np.inf, -np.inf], np.nan).dropna().mean())
    except Exception:
        turnover = None

    return {
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": qlib_ir,
        "max_drawdown": raw.get(base + "max_drawdown"),
        "turnover": turnover,
        "daily_return_mean": raw.get(base + "mean"),
        "daily_return_std": daily_std,
        "recorder_run_dir": str(run_dir),
    }


def _read_rows(results_dir: Path) -> list[dict]:
    rows = []
    for seed in SEEDS:
        path = results_dir / "tables" / f"gats_samplerfixed_csi300_recent_seed{seed}.json"
        if not path.exists():
            rows.append({"seed": seed, "status": "missing", "path": str(path)})
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))
        test_metrics = payload.get("test_metrics", {})
        portfolio_metrics = _load_portfolio_metrics(payload)
        row = {
            "seed": seed,
            "status": "exact",
            "recorder_id": payload.get("recorder_id"),
            "path": str(path),
        }
        for metric in ["IC_mean", "IC_std", "ICIR", "RankIC_mean", "RankIC_std", "RankICIR", "n_days"]:
            row[metric] = test_metrics.get(metric)
        row.update(portfolio_metrics)
        rows.append(row)
    return rows


def _summary(rows: list[dict]) -> list[dict]:
    exact_rows = [row for row in rows if row.get("status") == "exact"]
    output = []
    for metric in METRICS:
        values = []
        for row in exact_rows:
            value = row.get(metric)
            if value is None:
                continue
            value = float(value)
            if not math.isnan(value):
                values.append(value)
        if not values:
            output.append({"metric": metric, "n": 0, "mean": None, "var": None, "std": None, "min": None, "max": None})
            continue
        arr = np.asarray(values, dtype=np.float64)
        output.append(
            {
                "metric": metric,
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "var": float(arr.var(ddof=1)) if arr.size > 1 else 0.0,
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_md(path: Path, rows: list[dict], summary_rows: list[dict]) -> None:
    exact_count = sum(1 for row in rows if row.get("status") == "exact")
    lines = [
        "# GATs Recent Top10 Seeds - 2026-05-03",
        "",
        "Seeds selected by `fdg_skip32 / recent` top10 IC: `512, 23, 37, 5, 40, 55, 25, 49, 17, 79`.",
        "Split: recent train `2010-01-01..2020-12-31`, valid `2021-01-01..2021-12-31`, test `2022-01-01..2025-09-17`.",
        "",
        "Implementation: Qlib official `pytorch_gats_ts.GATs` model body with a fixed per-datetime sampler.",
        "Data processing matches the current Qlib Alpha158 recent setup: `csi300_aligned`, `cn_data_2024h1`,",
        "`DropCol(VWAP0)`, cross-sectional feature fill/normalization, normalized train label, and raw-label IC evaluation.",
        "",
        f"Completed rows: `{exact_count}/10`.",
        "",
        "## Raw Results",
        "",
        "| seed | status | IC | ICIR | RankIC | RankICIR | Sharpe | AnnRet | AnnVol | MaxDD | Turnover | n_days |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {status} | {IC_mean} | {ICIR} | {RankIC_mean} | {RankICIR} | "
            "{sharpe} | {annual_return} | {annual_vol} | {max_drawdown} | {turnover} | {n_days} |".format(
                seed=row["seed"],
                status=row["status"],
                IC_mean=_fmt(row.get("IC_mean")),
                ICIR=_fmt(row.get("ICIR")),
                RankIC_mean=_fmt(row.get("RankIC_mean")),
                RankICIR=_fmt(row.get("RankICIR")),
                sharpe=_fmt(row.get("sharpe")),
                annual_return=_fmt(row.get("annual_return")),
                annual_vol=_fmt(row.get("annual_vol")),
                max_drawdown=_fmt(row.get("max_drawdown")),
                turnover=_fmt(row.get("turnover")),
                n_days="" if row.get("n_days") is None else row.get("n_days"),
            )
        )

    lines.extend(
        [
            "",
            "## Summary Across Completed Seeds",
            "",
            "| metric | n | mean | var | std | min | max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {metric} | {n} | {mean} | {var} | {std} | {min} | {max} |".format(
                metric=row["metric"],
                n=row["n"],
                mean=_fmt(row.get("mean")),
                var=_fmt(row.get("var")),
                std=_fmt(row.get("std")),
                min=_fmt(row.get("min")),
                max=_fmt(row.get("max")),
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/seed100_gpu_gats"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/gats_recent_top10_20260503"))
    args = parser.parse_args()

    rows = _read_rows(args.results_dir)
    summary_rows = _summary(rows)
    summaries_dir = args.report_dir / "summaries"
    raw_fields = [
        "seed",
        "status",
        "recorder_id",
        "IC_mean",
        "IC_std",
        "ICIR",
        "RankIC_mean",
        "RankIC_std",
        "RankICIR",
        "annual_return",
        "annual_vol",
        "sharpe",
        "max_drawdown",
        "turnover",
        "n_days",
        "daily_return_mean",
        "daily_return_std",
        "path",
        "recorder_run_dir",
    ]
    _write_csv(summaries_dir / "gats_recent_top10_raw_20260503.csv", rows, raw_fields)
    _write_csv(
        summaries_dir / "gats_recent_top10_summary_20260503.csv",
        summary_rows,
        ["metric", "n", "mean", "var", "std", "min", "max"],
    )
    _write_md(summaries_dir / "gats_recent_top10_20260503.md", rows, summary_rows)
    print(f"Wrote {summaries_dir / 'gats_recent_top10_20260503.md'}")


if __name__ == "__main__":
    main()
