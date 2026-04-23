from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
RESULT_ROOT = ROOT / "results" / "seed100_cpu"
CONFIG_DIR = RESULT_ROOT / "generated_configs"
LOG_DIR = RESULT_ROOT / "logs"
MANIFEST_DIR = RESULT_ROOT / "manifests"
SLOT_SCRIPT_DIR = MANIFEST_DIR / "slot_scripts"
SUMMARY_MD = RESULT_ROOT / "seed100_summary.md"
SUMMARY_JSON = RESULT_ROOT / "seed100_summary.json"

POPULAR_SEEDS_100 = [
    0, 1, 2, 3, 4, 5, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 29, 31, 32, 33,
    34, 35, 36, 37, 40, 41, 42, 43, 44, 47,
    49, 50, 52, 55, 57, 60, 63, 64, 66, 69,
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 111, 123, 128, 137, 149, 168, 169, 188, 199,
    200, 202, 233, 256, 314, 512, 666, 777, 888, 999,
]
DEFAULT_SEEDS = POPULAR_SEEDS_100
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
class Task:
    name: str
    family: str
    split: str
    seed: int
    result_path: str
    log_path: str
    command: list[str]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _patch_train_single_config(base_path: Path, out_path: Path) -> Path:
    cfg = load_yaml(base_path)
    if PROVIDER_URI_OVERRIDE:
        cfg["provider_uri"] = PROVIDER_URI_OVERRIDE
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


def train_single_task(
    *,
    name: str,
    family: str,
    split: str,
    seed: int,
    config: Path,
    overrides: list[str] | None = None,
) -> Task:
    result_path = RESULT_ROOT / "tables" / f"{name}.json"
    log_path = LOG_DIR / f"{name}.log"
    command = [
        str(PYTHON),
        "-m",
        "train.train_single",
        "--config",
        str(config),
        "--override",
        f"train.seed={seed}",
        "--exp_name",
        name,
    ]
    for item in overrides or []:
        command.extend(["--override", item])
    return Task(name, family, split, seed, str(result_path), str(log_path), command)


def qworkflow_task(
    *,
    name: str,
    family: str,
    split: str,
    seed: int,
    config: Path,
    overrides: list[str] | None = None,
) -> Task:
    result_path = RESULT_ROOT / "tables" / f"{name}.json"
    log_path = LOG_DIR / f"{name}.log"
    command = [
        str(PYTHON),
        str(ROOT / "scripts" / "run_qworkflow.py"),
        "--config",
        str(config),
        "--experiment",
        name,
        "--summary_out",
        str(result_path),
        "--override",
        f"task.model.kwargs.seed={seed}",
    ]
    for item in overrides or []:
        command.extend(["--override", item])
    return Task(name, family, split, seed, str(result_path), str(log_path), command)


def build_tasks(models: list[str], seeds: list[int], cfgs: dict[str, Path]) -> list[Task]:
    tasks: list[Task] = []
    master_save_dir = RESULT_ROOT / "model_ckpt" / "master"

    for seed in seeds:
        if "lgbm" in models:
            tasks.append(
                qworkflow_task(
                    name=f"lgbm_csi300_official_seed{seed}",
                    family="lgbm",
                    split="official",
                    seed=seed,
                    config=cfgs["lgbm_official"],
                )
            )
            tasks.append(
                qworkflow_task(
                    name=f"lgbm_csi300_recent_seed{seed}",
                    family="lgbm",
                    split="recent",
                    seed=seed,
                    config=cfgs["lgbm_recent"],
                )
            )

        if "lstm" in models:
            tasks.append(
                qworkflow_task(
                    name=f"lstm_csi300_official_seed{seed}",
                    family="lstm",
                    split="official",
                    seed=seed,
                    config=cfgs["lstm_official"],
                )
            )
            tasks.append(
                qworkflow_task(
                    name=f"lstm_csi300_recent_seed{seed}",
                    family="lstm",
                    split="recent",
                    seed=seed,
                    config=cfgs["lstm_recent"],
                )
            )

        if "mlp" in models:
            tasks.append(
                train_single_task(
                    name=f"mlp_csi300_official_seed{seed}",
                    family="mlp",
                    split="official",
                    seed=seed,
                    config=cfgs["mlp_official"],
                )
            )
            tasks.append(
                train_single_task(
                    name=f"mlp_csi300_recent_seed{seed}",
                    family="mlp",
                    split="recent",
                    seed=seed,
                    config=cfgs["mlp_recent"],
                )
            )

        if "master" in models:
            common = [
                "task.model.kwargs.GPU=0",
                f"task.model.kwargs.save_path={str(master_save_dir)}/",
            ]
            tasks.append(
                qworkflow_task(
                    name=f"master_csi300_official_seed{seed}",
                    family="master",
                    split="official",
                    seed=seed,
                    config=cfgs["master_official"],
                    overrides=common + [f"task.model.kwargs.save_prefix=official_seed{seed}_"],
                )
            )
            tasks.append(
                qworkflow_task(
                    name=f"master_csi300_recent_seed{seed}",
                    family="master",
                    split="recent",
                    seed=seed,
                    config=cfgs["master_recent"],
                    overrides=common + [f"task.model.kwargs.save_prefix=recent_seed{seed}_"],
                )
            )

        if "hist" in models:
            overrides = ["task.model.kwargs.GPU=0", "task.model.kwargs.n_jobs=4"]
            tasks.append(
                qworkflow_task(
                    name=f"hist_csi300_official_seed{seed}",
                    family="hist",
                    split="official",
                    seed=seed,
                    config=cfgs["hist_official"],
                    overrides=overrides,
                )
            )
            tasks.append(
                qworkflow_task(
                    name=f"hist_csi300_recent_seed{seed}",
                    family="hist",
                    split="recent",
                    seed=seed,
                    config=cfgs["hist_recent"],
                    overrides=overrides,
                )
            )

        if "fdg_skip32" in models:
            extra = ["model.skip_hidden_dim=32"]
            tasks.append(
                train_single_task(
                    name=f"fdg_skip32_csi300_official_seed{seed}",
                    family="fdg_skip32",
                    split="official",
                    seed=seed,
                    config=cfgs["fdg_official"],
                    overrides=extra,
                )
            )
            tasks.append(
                train_single_task(
                    name=f"fdg_skip32_csi300_recent_seed{seed}",
                    family="fdg_skip32",
                    split="recent",
                    seed=seed,
                    config=cfgs["fdg_recent"],
                    overrides=extra,
                )
            )

        if "fdg_skip32_random" in models:
            extra = ["model.skip_hidden_dim=32", "model.graph_mode=random"]
            tasks.append(
                train_single_task(
                    name=f"fdg_skip32_random_csi300_official_seed{seed}",
                    family="fdg_skip32_random",
                    split="official",
                    seed=seed,
                    config=cfgs["fdg_official"],
                    overrides=extra,
                )
            )
            tasks.append(
                train_single_task(
                    name=f"fdg_skip32_random_csi300_recent_seed{seed}",
                    family="fdg_skip32_random",
                    split="recent",
                    seed=seed,
                    config=cfgs["fdg_recent"],
                    overrides=extra,
                )
            )

        if "fdg_skip32_symshare" in models:
            extra = [
                "model.skip_hidden_dim=32",
                "model.core_mode=symmetric",
                "model.share_sr_weights=true",
            ]
            tasks.append(
                train_single_task(
                    name=f"fdg_skip32_symshare_csi300_official_seed{seed}",
                    family="fdg_skip32_symshare",
                    split="official",
                    seed=seed,
                    config=cfgs["fdg_official"],
                    overrides=extra,
                )
            )
            tasks.append(
                train_single_task(
                    name=f"fdg_skip32_symshare_csi300_recent_seed{seed}",
                    family="fdg_skip32_symshare",
                    split="recent",
                    seed=seed,
                    config=cfgs["fdg_recent"],
                    overrides=extra,
                )
            )

    return tasks


def _task_env_exports() -> str:
    exports = {
        **THREAD_ENV,
        "PYTHONPATH": os.environ.get("PYTHONPATH", str(ROOT)),
        "QLIB_FORK_PATH": os.environ.get("QLIB_FORK_PATH", str(Path.home() / "refs" / "qlib_sjtu")),
    }
    if PROVIDER_URI_OVERRIDE:
        exports["SEED100_PROVIDER_URI"] = PROVIDER_URI_OVERRIDE
    lines = [f"export {key}={shlex.quote(value)}" for key, value in exports.items()]
    return "\n".join(lines)


def write_slot_scripts(tasks: list[Task], max_workers: int) -> list[Path]:
    SLOT_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    slot_scripts: list[Path] = []
    slots: list[list[Task]] = [[] for _ in range(max_workers)]
    for idx, task in enumerate(tasks):
        slots[idx % max_workers].append(task)

    export_block = _task_env_exports()
    for slot_idx, slot_tasks in enumerate(slots):
        if not slot_tasks:
            continue
        script_path = SLOT_SCRIPT_DIR / f"seed100_slot_{slot_idx:02d}.sh"
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(ROOT))}",
            export_block,
        ]
        for task in slot_tasks:
            log_path = Path(task.log_path)
            lines.append(f"mkdir -p {shlex.quote(str(log_path.parent))}")
            lines.append(f"if [ -f {shlex.quote(task.result_path)} ]; then echo SKIP {shlex.quote(task.name)}; else")
            lines.append(f"  stdbuf -oL -eL {shlex.join(task.command)} > {shlex.quote(task.log_path)} 2>&1")
            lines.append("fi")
        script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script_path.chmod(0o755)
        slot_scripts.append(script_path)
    return slot_scripts


def launch_slots(slot_scripts: list[Path]) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    launcher_log_dir = RESULT_ROOT / "launcher_logs"
    launcher_log_dir.mkdir(parents=True, exist_ok=True)
    for script_path in slot_scripts:
        log_path = launcher_log_dir / f"{script_path.stem}.launcher.log"
        subprocess.Popen(
            ["nohup", "bash", str(script_path)],
            stdout=log_path.open("ab"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(ROOT),
        )


def write_manifest(tasks: list[Task], max_workers: int) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "python": str(PYTHON),
        "max_workers": max_workers,
        "provider_uri": PROVIDER_URI_OVERRIDE,
        "tasks": [asdict(task) for task in tasks],
    }
    path = MANIFEST_DIR / "seed100_root.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def summarize(tasks: list[Task]) -> tuple[Path, Path]:
    from collections import defaultdict
    import statistics as st

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    completed = 0
    for task in tasks:
        result_path = Path(task.result_path)
        if not result_path.exists():
            continue
        completed += 1
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = payload.get("test_metrics", {})
        if "IC_mean" in metrics:
            grouped[(task.family, task.split)].append(float(metrics["IC_mean"]))

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

    payload = {"completed": completed, "total": len(tasks), "rows": rows}
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Seed100 CPU Summary",
        "",
        f"- completed: `{completed}/{len(tasks)}`",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["launch", "summarize"])
    parser.add_argument("--max-workers", type=int, default=18)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfgs = generate_configs()
    tasks = build_tasks(args.models, args.seeds, cfgs)

    if args.action == "launch":
        slot_scripts = write_slot_scripts(tasks, args.max_workers)
        manifest_path = write_manifest(tasks, args.max_workers)
        launch_slots(slot_scripts)
        print(f"Launched {len(tasks)} tasks with {len(slot_scripts)} workers.")
        print(f"Manifest: {manifest_path}")
        return

    md_path, json_path = summarize(tasks)
    print(f"Summary written to {md_path}")
    print(f"Summary json: {json_path}")


if __name__ == "__main__":
    main()
