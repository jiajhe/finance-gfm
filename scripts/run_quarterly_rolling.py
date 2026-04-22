from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path("/home/user186/icmlworksop")
PYTHON = Path("/project/python_env/anaconda3/bin/python")
QSTU_PYTHONPATH = f"/project/user186_refs/qlib_sjtu:{ROOT}"
LOCAL_PYTHONPATH = str(ROOT)
CALENDAR_PATH = Path("~/.qlib/qlib_data/cn_data/calendars/day.txt").expanduser()


@dataclass(frozen=True)
class ModelSpec:
    key: str
    runner: str
    config: str
    display_name: str
    pythonpath: str
    fixed_result: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "fdg": ModelSpec(
        key="fdg",
        runner="train_single",
        config="configs/qlib_recent_fdg.yaml",
        display_name="FDG",
        pythonpath=LOCAL_PYTHONPATH,
        fixed_result="results/tables/fdg_csi300_qlib_recent.json",
    ),
    "mlp": ModelSpec(
        key="mlp",
        runner="train_single",
        config="configs/qlib_recent_mlp.yaml",
        display_name="MLP",
        pythonpath=LOCAL_PYTHONPATH,
        fixed_result="results/tables/mlp_csi300_qlib_recent.json",
    ),
    "master": ModelSpec(
        key="master",
        runner="qworkflow",
        config="configs/master_csi300_recent_quick.yaml",
        display_name="MASTER-quick",
        pythonpath=QSTU_PYTHONPATH,
        fixed_result="results/tables/master_csi300_recent_quick.json",
    ),
    "hist_alpha158": ModelSpec(
        key="hist_alpha158",
        runner="qworkflow",
        config="configs/hist_alpha158_csi300_recent_quick.yaml",
        display_name="HIST-Alpha158-quick",
        pythonpath=QSTU_PYTHONPATH,
        fixed_result="results/tables/hist_alpha158_csi300_recent_quick.json",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="fdg,mlp,master,hist_alpha158")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--start-quarter", default="2022Q1")
    parser.add_argument("--end-quarter", default="2025Q4")
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--log-dir", default="results/launcher_logs/quarterly_roll_20260422")
    parser.add_argument("--summary-json", default="results/tables/quarterly_roll_20260422.json")
    parser.add_argument("--summary-md", default="results/tables/quarterly_roll_20260422.md")
    return parser.parse_args()


def load_calendar() -> pd.DatetimeIndex:
    cal = pd.read_csv(CALENDAR_PATH, header=None)[0]
    return pd.DatetimeIndex(pd.to_datetime(cal))


def build_quarter_window(calendar: pd.DatetimeIndex, quarter: pd.Period, train_start: str) -> dict[str, str | int]:
    test_mask = (calendar >= quarter.start_time) & (calendar <= quarter.end_time)
    test_days = calendar[test_mask]
    if test_days.empty:
        raise ValueError(f"No trading days for quarter {quarter}")

    valid_quarter = quarter - 1
    valid_mask = (calendar >= valid_quarter.start_time) & (calendar <= valid_quarter.end_time)
    valid_days = calendar[valid_mask]
    if valid_days.empty:
        raise ValueError(f"No trading days for validation quarter {valid_quarter}")

    train_end_candidates = calendar[calendar < valid_days[0]]
    if train_end_candidates.empty:
        raise ValueError(f"No training days before validation start {valid_days[0]}")

    train_start_ts = pd.Timestamp(train_start)
    train_days = calendar[(calendar >= train_start_ts) & (calendar <= train_end_candidates[-1])]
    if train_days.empty:
        raise ValueError(f"No training days for window ending {train_end_candidates[-1]}")

    return {
        "quarter": str(quarter),
        "train_start": train_days[0].strftime("%Y-%m-%d"),
        "train_end": train_days[-1].strftime("%Y-%m-%d"),
        "valid_start": valid_days[0].strftime("%Y-%m-%d"),
        "valid_end": valid_days[-1].strftime("%Y-%m-%d"),
        "test_start": test_days[0].strftime("%Y-%m-%d"),
        "test_end": test_days[-1].strftime("%Y-%m-%d"),
        "n_test_days": int(len(test_days)),
    }


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def train_single_command(spec: ModelSpec, window: dict[str, str | int], exp_name: str) -> list[str]:
    cmd = [
        str(PYTHON),
        "-m",
        "train.train_single",
        "--config",
        str(ROOT / spec.config),
        "--exp_name",
        exp_name,
        "--override",
        f"fit_start_time={window['train_start']}",
        "--override",
        f"fit_end_time={window['train_end']}",
        "--override",
        f"splits.train=[\"{window['train_start']}\",\"{window['train_end']}\"]",
        "--override",
        f"splits.valid=[\"{window['valid_start']}\",\"{window['valid_end']}\"]",
        "--override",
        f"splits.test=[\"{window['test_start']}\",\"{window['test_end']}\"]",
    ]
    return cmd


def qworkflow_command(spec: ModelSpec, window: dict[str, str | int], exp_name: str, summary_path: Path) -> list[str]:
    overrides = [
        f"data_handler_config.start_time={window['train_start']}",
        f"data_handler_config.end_time={window['test_end']}",
        f"data_handler_config.fit_start_time={window['train_start']}",
        f"data_handler_config.fit_end_time={window['train_end']}",
        f"task.dataset.kwargs.handler.kwargs.start_time={window['train_start']}",
        f"task.dataset.kwargs.handler.kwargs.end_time={window['test_end']}",
        f"task.dataset.kwargs.handler.kwargs.fit_start_time={window['train_start']}",
        f"task.dataset.kwargs.handler.kwargs.fit_end_time={window['train_end']}",
        f"task.dataset.kwargs.segments.train=[\"{window['train_start']}\",\"{window['train_end']}\"]",
        f"task.dataset.kwargs.segments.valid=[\"{window['valid_start']}\",\"{window['valid_end']}\"]",
        f"task.dataset.kwargs.segments.test=[\"{window['test_start']}\",\"{window['test_end']}\"]",
        f"port_analysis_config.backtest.start_time={window['test_start']}",
        f"port_analysis_config.backtest.end_time={window['test_end']}",
        "task.model.kwargs.GPU=0",
    ]
    if spec.key == "master":
        overrides.extend(
            [
                f"market_data_handler_config.start_time={window['train_start']}",
                f"market_data_handler_config.end_time={window['test_end']}",
                f"market_data_handler_config.fit_start_time={window['train_start']}",
                f"market_data_handler_config.fit_end_time={window['train_end']}",
                f"task.dataset.kwargs.market_data_handler_config.start_time={window['train_start']}",
                f"task.dataset.kwargs.market_data_handler_config.end_time={window['test_end']}",
                f"task.dataset.kwargs.market_data_handler_config.fit_start_time={window['train_start']}",
                f"task.dataset.kwargs.market_data_handler_config.fit_end_time={window['train_end']}",
                f"task.model.kwargs.save_prefix={exp_name}_",
                f"task.model.kwargs.save_path={str(ROOT / 'results' / 'rolling_models' / 'master') + '/'}",
            ]
        )

    cmd = [
        str(PYTHON),
        str(ROOT / "scripts" / "run_qworkflow.py"),
        "--config",
        str(ROOT / spec.config),
        "--experiment",
        f"quarterly_roll_{spec.key}",
        "--summary_out",
        str(summary_path),
    ]
    for item in overrides:
        cmd.extend(["--override", item])
    return cmd


def metric_block(mean: float, std: float, count: int) -> dict[str, float | int]:
    ir = float(mean / std) if std > 1e-12 else 0.0
    return {"mean": float(mean), "std": float(std), "ir": ir, "count": int(count)}


def resolve_std(metrics: dict, mean_key: str, std_key: str, ir_key: str) -> float:
    if std_key in metrics:
        return float(metrics[std_key])
    ir = float(metrics.get(ir_key, 0.0))
    mean = float(metrics.get(mean_key, 0.0))
    if abs(ir) > 1e-12:
        return abs(mean / ir)
    return 0.0


def combine_metric_blocks(blocks: list[dict[str, float | int]]) -> dict[str, float | int]:
    total_n = int(sum(int(block["count"]) for block in blocks))
    if total_n == 0:
        return metric_block(0.0, 0.0, 0)

    global_mean = sum(int(block["count"]) * float(block["mean"]) for block in blocks) / total_n
    if total_n <= 1:
        return metric_block(global_mean, 0.0, total_n)

    numerator = 0.0
    for block in blocks:
        n = int(block["count"])
        mean = float(block["mean"])
        std = float(block["std"])
        numerator += (n - 1) * (std**2) + n * ((mean - global_mean) ** 2)
    global_std = math.sqrt(max(numerator / (total_n - 1), 0.0))
    return metric_block(global_mean, global_std, total_n)


def load_fixed_ic(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.load(path.open("r", encoding="utf-8"))
    return data.get("test_metrics", {}).get("IC_mean")


def build_jobs(args: argparse.Namespace, calendar: pd.DatetimeIndex) -> list[dict]:
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    quarters = pd.period_range(args.start_quarter, args.end_quarter, freq="Q")
    jobs = []
    for model_key in models:
        spec = MODEL_SPECS[model_key]
        for quarter in quarters:
            window = build_quarter_window(calendar, quarter, args.train_start)
            exp_name = f"{spec.key}_rollq_{str(quarter).lower()}"
            result_path = ROOT / "results" / "tables" / f"{exp_name}.json"
            if spec.runner == "train_single":
                cmd = train_single_command(spec, window, exp_name)
            else:
                cmd = qworkflow_command(spec, window, exp_name, result_path)
            jobs.append(
                {
                    "model_key": spec.key,
                    "display_name": spec.display_name,
                    "runner": spec.runner,
                    "quarter": str(quarter),
                    "window": window,
                    "exp_name": exp_name,
                    "result_path": result_path,
                    "cmd": cmd,
                    "pythonpath": spec.pythonpath,
                    "fixed_result": str(ROOT / spec.fixed_result),
                }
            )
    return jobs


def launch_jobs(jobs: list[dict], gpus: list[int], log_dir: Path) -> list[dict]:
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    active: dict[int, dict] = {}
    finished: list[dict] = []

    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            log_path = log_dir / f"{job['exp_name']}.log"
            env = os.environ.copy()
            env["PYTHONPATH"] = job["pythonpath"]
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            fp = open(log_path, "w", encoding="utf-8")
            process = subprocess.Popen(
                job["cmd"],
                cwd=ROOT,
                env=env,
                stdout=fp,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[gpu] = {
                "job": job,
                "proc": process,
                "fp": fp,
                "log_path": str(log_path),
                "start": time.time(),
            }
            print(f"[START][GPU {gpu}] {job['exp_name']}")

        time.sleep(5)
        done_gpus = []
        for gpu, slot in active.items():
            returncode = slot["proc"].poll()
            if returncode is None:
                continue
            slot["fp"].close()
            duration = time.time() - slot["start"]
            record = {
                "gpu": gpu,
                "returncode": int(returncode),
                "duration_sec": float(duration),
                "log_path": slot["log_path"],
                **slot["job"],
            }
            finished.append(record)
            state = "DONE" if returncode == 0 else "FAIL"
            print(f"[{state}][GPU {gpu}] {slot['job']['exp_name']} ({duration/60:.1f} min)")
            done_gpus.append(gpu)
        for gpu in done_gpus:
            active.pop(gpu, None)

    return finished


def summarize(jobs: list[dict], finished: list[dict], summary_json: Path, summary_md: Path) -> None:
    finished_map = {item["exp_name"]: item for item in finished}
    model_rows: dict[str, list[dict]] = {}

    for job in jobs:
        result = None
        if job["result_path"].exists():
            result = json.load(job["result_path"].open("r", encoding="utf-8"))
        model_rows.setdefault(job["model_key"], []).append(
            {
                "quarter": job["quarter"],
                "window": job["window"],
                "result": result,
                "run": finished_map.get(job["exp_name"]),
            }
        )

    payload = {"models": {}, "jobs": finished}
    md_lines = [
        "# Quarterly Rolling Results 2026-04-22",
        "",
        "## Setup",
        "",
        "- Scheme: expanding train, previous quarter as validation, current quarter as test.",
        "- Test horizon: 2022Q1 ~ 2025Q4.",
        "- Models: " + ", ".join(MODEL_SPECS[key].display_name for key in model_rows),
        "",
        "## Overall",
        "",
        "| Model | Fixed Recent IC | Rolling IC | Rolling RankIC | Rolling ICIR | Quarters | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for model_key, rows in model_rows.items():
        ic_blocks = []
        rank_blocks = []
        quarter_table = []
        complete_quarters = 0
        for row in sorted(rows, key=lambda item: item["quarter"]):
            result = row["result"]
            if result is None:
                quarter_table.append({"quarter": row["quarter"], "status": "missing"})
                continue
            tm = result.get("test_metrics", {})
            n_days = int(tm.get("n_days", row["window"]["n_test_days"]))
            ic_blocks.append(
                metric_block(
                    float(tm.get("IC_mean", 0.0)),
                    resolve_std(tm, "IC_mean", "IC_std", "ICIR"),
                    n_days,
                )
            )
            rank_blocks.append(
                metric_block(
                    float(tm.get("RankIC_mean", 0.0)),
                    resolve_std(tm, "RankIC_mean", "RankIC_std", "RankICIR"),
                    n_days,
                )
            )
            complete_quarters += 1
            quarter_table.append(
                {
                    "quarter": row["quarter"],
                    "IC_mean": float(tm.get("IC_mean", 0.0)),
                    "RankIC_mean": float(tm.get("RankIC_mean", 0.0)),
                    "ICIR": float(tm.get("ICIR", 0.0)),
                    "n_days": n_days,
                }
            )

        ic_summary = combine_metric_blocks(ic_blocks)
        rank_summary = combine_metric_blocks(rank_blocks)
        fixed_ic = load_fixed_ic(Path(rows[0]["run"]["fixed_result"]) if rows and rows[0]["run"] else Path(rows[0]["result_path"]))
        if fixed_ic is None:
            fixed_ic = load_fixed_ic(Path(rows[0]["result_path"]).parent / Path(MODEL_SPECS[model_key].fixed_result).name)
        status = "complete" if complete_quarters == len(rows) else "partial"
        payload["models"][model_key] = {
            "display_name": MODEL_SPECS[model_key].display_name,
            "fixed_recent_ic": fixed_ic,
            "rolling_ic": ic_summary,
            "rolling_rankic": rank_summary,
            "quarters": quarter_table,
        }
        md_lines.append(
            f"| {MODEL_SPECS[model_key].display_name} | "
            f"{fixed_ic if fixed_ic is not None else 0.0:.5f} | "
            f"{float(ic_summary['mean']):.5f} | {float(rank_summary['mean']):.5f} | {float(ic_summary['ir']):.5f} | "
            f"{complete_quarters}/{len(rows)} | {status} |"
        )

    md_lines.extend(["", "## Per Quarter", ""])
    for model_key, rows in model_rows.items():
        md_lines.append(f"### {MODEL_SPECS[model_key].display_name}")
        md_lines.append("")
        md_lines.append("| Quarter | Train | Valid | Test | IC | RankIC | ICIR | |")
        md_lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | --- |")
        for row in sorted(rows, key=lambda item: item["quarter"]):
            window = row["window"]
            result = row["result"]
            if result is None:
                md_lines.append(
                    f"| {row['quarter']} | {window['train_start']}~{window['train_end']} | "
                    f"{window['valid_start']}~{window['valid_end']} | {window['test_start']}~{window['test_end']} | "
                    f"- | - | - | missing |"
                )
                continue
            tm = result.get("test_metrics", {})
            md_lines.append(
                f"| {row['quarter']} | {window['train_start']}~{window['train_end']} | "
                f"{window['valid_start']}~{window['valid_end']} | {window['test_start']}~{window['test_end']} | "
                f"{float(tm.get('IC_mean', 0.0)):.5f} | {float(tm.get('RankIC_mean', 0.0)):.5f} | "
                f"{float(tm.get('ICIR', 0.0)):.5f} | ok |"
            )
        md_lines.append("")

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    calendar = load_calendar()
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    (ROOT / "results" / "rolling_models" / "master").mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args, calendar)
    finished = launch_jobs(jobs, gpus, ROOT / args.log_dir)
    summarize(jobs, finished, ROOT / args.summary_json, ROOT / args.summary_md)


if __name__ == "__main__":
    main()
