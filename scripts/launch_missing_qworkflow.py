from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from scripts.launch_experiment_matrix import LOG_DIR, ROOT, Task, build_tasks


PYTHON = Path("/project/python_env/anaconda3/bin/python")
FAMILIES = {"lstm", "master", "hist"}
RUN_MANIFEST_DIR = ROOT / "results" / "run_manifests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    return parser.parse_args()


def active_task_names() -> set[str]:
    ps = subprocess.run(
        ["ps", "-eo", "cmd="],
        capture_output=True,
        text=True,
        check=False,
    )
    active: set[str] = set()
    for task in build_tasks():
        if task.family not in FAMILIES:
            continue
        if any(task.name in line for line in ps.stdout.splitlines()):
            active.add(task.name)
    return active


def missing_qworkflow_tasks() -> list[Task]:
    active = active_task_names()
    missing: list[Task] = []
    for task in build_tasks():
        if task.family not in FAMILIES:
            continue
        if Path(task.result_path).exists():
            continue
        if task.name in active:
            continue
        missing.append(task)
    return missing


def assign_buckets(tasks: list[Task], num_gpus: int, slots_per_gpu: int) -> list[tuple[int, list[Task]]]:
    slot_gpus: list[int] = []
    for gpu in range(num_gpus):
        for _ in range(slots_per_gpu):
            slot_gpus.append(gpu)
    buckets: list[list[Task]] = [[] for _ in slot_gpus]
    for idx, task in enumerate(tasks):
        buckets[idx % len(slot_gpus)].append(task)
    return list(zip(slot_gpus, buckets))


def run_slot(slot_idx: int, gpu: int, tasks: list[Task]) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(ROOT)
    failures = 0
    for task in tasks:
        log_path = Path(task.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_fp:
            log_fp.write(f"[relaunch] gpu={gpu} slot={slot_idx} name={task.name}\n")
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
            log_fp.write(f"[relaunch_done:{return_code}] {task.name}\n")
            log_fp.flush()
        if return_code != 0:
            failures += 1
    return failures


def main() -> None:
    args = parse_args()
    tasks = missing_qworkflow_tasks()
    print(f"[missing-qworkflow] pending={len(tasks)}", flush=True)
    if not tasks:
        return

    buckets = assign_buckets(tasks, num_gpus=args.num_gpus, slots_per_gpu=args.slots_per_gpu)
    procs: list[subprocess.Popen] = []
    log_files = []
    for slot_idx, (gpu, bucket) in enumerate(buckets):
        if not bucket:
            continue
        slot_log = LOG_DIR / f"missing_qworkflow_slot{slot_idx}.launcher.log"
        log_fp = slot_log.open("a", encoding="utf-8")
        cmd = [
            str(PYTHON),
            "-c",
            (
                "from scripts.launch_missing_qworkflow import run_slot; "
                f"raise SystemExit(run_slot({slot_idx}, {gpu}, __import__('pickle').loads({repr(__import__('pickle').dumps(bucket))})))"
            ),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        proc._log_fp = log_fp  # type: ignore[attr-defined]
        procs.append(proc)
        print(f"[missing-qworkflow] slot={slot_idx} gpu={gpu} tasks={len(bucket)} pid={proc.pid}", flush=True)

    total_failures = 0
    for proc in procs:
        total_failures += proc.wait()
        proc._log_fp.close()  # type: ignore[attr-defined]
    print(f"[missing-qworkflow] done failures={total_failures}", flush=True)


if __name__ == "__main__":
    main()
