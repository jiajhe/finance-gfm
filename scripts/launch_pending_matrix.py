from __future__ import annotations

import argparse
import os
import subprocess
from collections import defaultdict
from pathlib import Path

from scripts.launch_experiment_matrix import LOG_DIR, ROOT, Task, build_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=8)
    return parser.parse_args()


def active_task_names(tasks: list[Task]) -> set[str]:
    ps = subprocess.run(
        ["ps", "-eo", "cmd="],
        capture_output=True,
        text=True,
        check=False,
    )
    active: set[str] = set()
    lines = ps.stdout.splitlines()
    for task in tasks:
        if any(task.name in line for line in lines):
            active.add(task.name)
    return active


def pending_tasks(tasks: list[Task]) -> list[Task]:
    active = active_task_names(tasks)
    pending: list[Task] = []
    for task in tasks:
        if Path(task.result_path).exists():
            continue
        if task.name in active:
            continue
        pending.append(task)
    return pending


def current_gpu_load(num_gpus: int) -> dict[int, int]:
    load = {gpu: 0 for gpu in range(num_gpus)}
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if query.returncode != 0:
        return load

    uuid_to_index: dict[str, int] = {}
    gpu_info = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,gpu_uuid",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in gpu_info.stdout.splitlines():
        if not line.strip():
            continue
        idx_raw, uuid_raw = [part.strip() for part in line.split(",", 1)]
        try:
            uuid_to_index[uuid_raw] = int(idx_raw)
        except ValueError:
            continue

    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        uuid_raw = line.split(",", 1)[0].strip()
        gpu_idx = uuid_to_index.get(uuid_raw)
        if gpu_idx is None or gpu_idx not in load:
            continue
        load[gpu_idx] += 1
    return load


def assign_gpus(tasks: list[Task], num_gpus: int) -> dict[int, list[Task]]:
    gpu_load = current_gpu_load(num_gpus)
    buckets: dict[int, list[Task]] = defaultdict(list)
    for task in tasks:
        gpu = min(gpu_load, key=lambda g: (gpu_load[g], g))
        buckets[gpu].append(task)
        gpu_load[gpu] += 1
    return buckets


def run_slot(gpu: int, tasks: list[Task]) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(ROOT)

    failures = 0
    for task in tasks:
        log_path = Path(task.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_fp:
            log_fp.write(f"[pending-launch] gpu={gpu} name={task.name}\n")
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
            log_fp.write(f"[pending-done:{return_code}] {task.name}\n")
            log_fp.flush()
        if return_code != 0:
            failures += 1
    return failures


def main() -> None:
    args = parse_args()
    tasks = pending_tasks(build_tasks())
    print(f"[pending-matrix] pending={len(tasks)}", flush=True)
    if not tasks:
        return

    buckets = assign_gpus(tasks, args.num_gpus)
    procs: list[subprocess.Popen] = []
    for gpu, bucket in sorted(buckets.items()):
        if not bucket:
            continue
        slot_log = LOG_DIR / f"pending_matrix_gpu{gpu}.launcher.log"
        log_fp = slot_log.open("a", encoding="utf-8")
        code = (
            "from scripts.launch_pending_matrix import run_slot; "
            f"raise SystemExit(run_slot({gpu}, __import__('pickle').loads({repr(__import__('pickle').dumps(bucket))})))"
        )
        proc = subprocess.Popen(
            ["/project/python_env/anaconda3/bin/python", "-c", code],
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
        proc._log_fp = log_fp  # type: ignore[attr-defined]
        procs.append(proc)
        print(f"[pending-matrix] gpu={gpu} tasks={len(bucket)} pid={proc.pid}", flush=True)

    total_failures = 0
    for proc in procs:
        total_failures += proc.wait()
        proc._log_fp.close()  # type: ignore[attr-defined]
    print(f"[pending-matrix] done failures={total_failures}", flush=True)


if __name__ == "__main__":
    main()
