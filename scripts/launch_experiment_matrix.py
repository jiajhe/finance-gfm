from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/project/python_env/anaconda3/bin/python")
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
MANIFEST_DIR = RESULTS_DIR / "run_manifests"
LOG_DIR = RESULTS_DIR / "seed_matrix_logs"
MODEL_SAVE_DIR = RESULTS_DIR / "seed_matrix_models"
SLOT_SCRIPT_DIR = RESULTS_DIR / "run_manifests" / "slot_scripts"
SEEDS = [2022, 2023, 2024, 2025, 2026]


@dataclass
class Task:
    name: str
    family: str
    split: str
    seed: int
    gpu: int | None
    result_path: str
    log_path: str
    command: list[str]


def train_single_task(
    *,
    name: str,
    family: str,
    split: str,
    seed: int,
    config: str,
    overrides: list[str] | None = None,
) -> Task:
    result_path = TABLES_DIR / f"{name}.json"
    log_path = LOG_DIR / f"{name}.log"
    command = [
        str(PYTHON),
        "-m",
        "train.train_single",
        "--config",
        config,
        "--override",
        f"train.seed={seed}",
        "--exp_name",
        name,
    ]
    for item in overrides or []:
        command.extend(["--override", item])
    return Task(
        name=name,
        family=family,
        split=split,
        seed=seed,
        gpu=None,
        result_path=str(result_path),
        log_path=str(log_path),
        command=command,
    )


def qworkflow_task(
    *,
    name: str,
    family: str,
    split: str,
    seed: int,
    config: str,
    overrides: list[str] | None = None,
) -> Task:
    result_path = TABLES_DIR / f"{name}.json"
    log_path = LOG_DIR / f"{name}.log"
    command = [
        str(PYTHON),
        str(ROOT / "scripts" / "run_qworkflow.py"),
        "--config",
        config,
        "--experiment",
        name,
        "--summary_out",
        str(result_path),
        "--override",
        f"task.model.kwargs.seed={seed}",
    ]
    for item in overrides or []:
        command.extend(["--override", item])
    return Task(
        name=name,
        family=family,
        split=split,
        seed=seed,
        gpu=None,
        result_path=str(result_path),
        log_path=str(log_path),
        command=command,
    )


def build_tasks() -> list[Task]:
    tasks: list[Task] = []

    for seed in SEEDS:
        tasks.append(
            qworkflow_task(
                name=f"lstm_alpha158_csi300_official_seed{seed}",
                family="lstm",
                split="official",
                seed=seed,
                config=str(ROOT / "configs" / "lstm_alpha158_csi300_official.yaml"),
            )
        )
        tasks.append(
            qworkflow_task(
                name=f"lstm_alpha158_csi300_recent_seed{seed}",
                family="lstm",
                split="recent",
                seed=seed,
                config=str(ROOT / "configs" / "lstm_alpha158_csi300_recent.yaml"),
            )
        )
        tasks.append(
            train_single_task(
                name=f"mlp_csi300_qlib_official_seed{seed}",
                family="mlp",
                split="official",
                seed=seed,
                config=str(ROOT / "configs" / "qlib_official_mlp.yaml"),
            )
        )
        tasks.append(
            train_single_task(
                name=f"mlp_csi300_qlib_recent_seed{seed}",
                family="mlp",
                split="recent",
                seed=seed,
                config=str(ROOT / "configs" / "qlib_recent_mlp.yaml"),
            )
        )

        master_save_dir = MODEL_SAVE_DIR / "master"
        tasks.append(
            qworkflow_task(
                name=f"master_csi300_official_quick_seed{seed}",
                family="master",
                split="official",
                seed=seed,
                config=str(ROOT / "configs" / "master_csi300_official_quick.yaml"),
                overrides=[
                    f"task.model.kwargs.save_prefix={f'master_official_seed{seed}_'}",
                    f"task.model.kwargs.save_path={str(master_save_dir)}/",
                ],
            )
        )
        tasks.append(
            qworkflow_task(
                name=f"master_csi300_recent_quick_seed{seed}",
                family="master",
                split="recent",
                seed=seed,
                config=str(ROOT / "configs" / "master_csi300_recent_quick.yaml"),
                overrides=[
                    f"task.model.kwargs.save_prefix={f'master_recent_seed{seed}_'}",
                    f"task.model.kwargs.save_path={str(master_save_dir)}/",
                ],
            )
        )

        tasks.append(
            qworkflow_task(
                name=f"hist_alpha158_csi300_official_quick_seed{seed}",
                family="hist",
                split="official",
                seed=seed,
                config=str(ROOT / "configs" / "hist_alpha158_csi300_official_quick.yaml"),
            )
        )
        tasks.append(
            qworkflow_task(
                name=f"hist_alpha158_csi300_recent_quick_seed{seed}",
                family="hist",
                split="recent",
                seed=seed,
                config=str(ROOT / "configs" / "hist_alpha158_csi300_recent_quick.yaml"),
            )
        )

        tasks.append(
            train_single_task(
                name=f"fdg_csi300_qlib_official_skip32_seed{seed}",
                family="fdg_skip32",
                split="official",
                seed=seed,
                config=str(ROOT / "configs" / "qlib_official_fdg.yaml"),
                overrides=["model.skip_hidden_dim=32"],
            )
        )
        tasks.append(
            train_single_task(
                name=f"fdg_csi300_qlib_recent_skip32_seed{seed}",
                family="fdg_skip32",
                split="recent",
                seed=seed,
                config=str(ROOT / "configs" / "qlib_recent_fdg.yaml"),
                overrides=["model.skip_hidden_dim=32"],
            )
        )

        tasks.append(
            train_single_task(
                name=f"fdg_csi300_qlib_official_skip32_random_seed{seed}",
                family="fdg_skip32_random",
                split="official",
                seed=seed,
                config=str(ROOT / "configs" / "qlib_official_fdg.yaml"),
                overrides=["model.skip_hidden_dim=32", "model.graph_mode=random"],
            )
        )
        tasks.append(
            train_single_task(
                name=f"fdg_csi300_qlib_recent_skip32_random_seed{seed}",
                family="fdg_skip32_random",
                split="recent",
                seed=seed,
                config=str(ROOT / "configs" / "qlib_recent_fdg.yaml"),
                overrides=["model.skip_hidden_dim=32", "model.graph_mode=random"],
            )
        )

        tasks.append(
            train_single_task(
                name=f"fdg_csi300_qlib_official_skip32_symshare_seed{seed}",
                family="fdg_skip32_symshare",
                split="official",
                seed=seed,
                config=str(ROOT / "configs" / "qlib_official_fdg.yaml"),
                overrides=[
                    "model.skip_hidden_dim=32",
                    "model.core_mode=symmetric",
                    "model.share_sr_weights=true",
                ],
            )
        )
        tasks.append(
            train_single_task(
                name=f"fdg_csi300_qlib_recent_skip32_symshare_seed{seed}",
                family="fdg_skip32_symshare",
                split="recent",
                seed=seed,
                config=str(ROOT / "configs" / "qlib_recent_fdg.yaml"),
                overrides=[
                    "model.skip_hidden_dim=32",
                    "model.core_mode=symmetric",
                    "model.share_sr_weights=true",
                ],
            )
        )

    return tasks


def build_slot_assignment(tasks: list[Task], slots_per_gpu: int, num_gpus: int) -> list[list[Task]]:
    slot_gpus = []
    for gpu in range(num_gpus):
        for _ in range(slots_per_gpu):
            slot_gpus.append(gpu)

    buckets: list[list[Task]] = [[] for _ in slot_gpus]
    for idx, task in enumerate(tasks):
        slot_idx = idx % len(slot_gpus)
        task.gpu = slot_gpus[slot_idx]
        buckets[slot_idx].append(task)
    return buckets


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def write_slot_script(slot_idx: int, gpu: int, tasks: list[Task]) -> Path:
    SLOT_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    script_path = SLOT_SCRIPT_DIR / f"seed_matrix_slot{slot_idx}.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        f"export CUDA_VISIBLE_DEVICES={gpu}",
        f"export PYTHONPATH={shlex.quote(str(ROOT))}",
        "",
    ]
    for task in tasks:
        task_log = shlex.quote(task.log_path)
        task_result = shlex.quote(task.result_path)
        cmd = shell_join(task.command)
        lines.extend(
            [
                f"if [ -f {task_result} ]; then",
                f"  echo '[skip] {task.name}' >> {task_log}",
                "else",
                f"  echo '[launch] gpu={gpu} slot={slot_idx} name={task.name}' >> {task_log}",
                f"  {cmd} >> {task_log} 2>&1",
                f"  echo '[done:$?] {task.name}' >> {task_log}",
                "fi",
                "",
            ]
        )
    script_path.write_text("\n".join(lines), encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def run_worker(slot_idx: int, gpu: int, tasks: list[Task], manifest_path: Path) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(ROOT)

    status = {
        "slot_idx": slot_idx,
        "gpu": gpu,
        "tasks_total": len(tasks),
        "tasks": [],
    }

    for task in tasks:
        log_path = Path(task.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        result_path = Path(task.result_path)
        task_record = asdict(task)
        if result_path.exists():
            task_record["status"] = "skipped_existing"
            status["tasks"].append(task_record)
            write_manifest(manifest_path, status)
            continue

        task_record["status"] = "running"
        status["tasks"].append(task_record)
        write_manifest(manifest_path, status)

        with log_path.open("a", encoding="utf-8") as log_fp:
            log_fp.write(f"[launch] gpu={gpu} slot={slot_idx} name={task.name}\n")
            log_fp.flush()
            proc = subprocess.Popen(
                task.command,
                cwd=str(ROOT),
                env=env,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return_code = proc.wait()

        task_record["return_code"] = return_code
        task_record["status"] = "done" if return_code == 0 else "failed"
        write_manifest(manifest_path, status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks()
    buckets = build_slot_assignment(tasks, slots_per_gpu=args.slots_per_gpu, num_gpus=args.num_gpus)

    index_payload = {
        "num_gpus": args.num_gpus,
        "slots_per_gpu": args.slots_per_gpu,
        "total_tasks": len(tasks),
        "slots": [
            {
                "slot_idx": slot_idx,
                "gpu": bucket[0].gpu if bucket else slot_idx // max(1, args.slots_per_gpu),
                "tasks": [task.name for task in bucket],
                "manifest": str(MANIFEST_DIR / f"seed_matrix_20260422_slot{slot_idx}.json"),
            }
            for slot_idx, bucket in enumerate(buckets)
        ],
    }
    write_manifest(MANIFEST_DIR / "seed_matrix_20260422_index.json", index_payload)

    children: list[subprocess.Popen] = []
    for slot_idx, bucket in enumerate(buckets):
        if not bucket:
            continue
        gpu = bucket[0].gpu if bucket[0].gpu is not None else 0
        if args.dry_run:
            continue
        manifest_path = MANIFEST_DIR / f"seed_matrix_20260422_slot{slot_idx}.json"
        write_manifest(
            manifest_path,
            {
                "slot_idx": slot_idx,
                "gpu": gpu,
                "tasks_total": len(bucket),
                "tasks": [asdict(task) | {"status": "queued"} for task in bucket],
            },
        )
        script_path = write_slot_script(slot_idx, gpu, bucket)
        cmd = ["bash", str(script_path)]
        slot_log = LOG_DIR / f"seed_matrix_slot{slot_idx}.launcher.log"
        with slot_log.open("a", encoding="utf-8") as log_fp:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                text=True,
            )
        children.append(proc)

    root_manifest = {
        "status": "dry_run" if args.dry_run else "launched",
        "children": [proc.pid for proc in children],
        "num_children": len(children),
    }
    write_manifest(MANIFEST_DIR / "seed_matrix_20260422_root.json", root_manifest)
    print(json.dumps(root_manifest, indent=2))


if __name__ == "__main__":
    main()
