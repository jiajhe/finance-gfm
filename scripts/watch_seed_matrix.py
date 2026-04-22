from __future__ import annotations

import subprocess
import time
from pathlib import Path

from scripts.launch_experiment_matrix import build_tasks


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "seed_matrix_logs"
SLOT_SCRIPT_DIR = ROOT / "results" / "run_manifests" / "slot_scripts"
PYTHON = Path("/project/python_env/anaconda3/bin/python")
TASKS = build_tasks()
EXPECTED_TOTAL = len(TASKS)
TASK_NAMES = [task.name for task in TASKS]


def active_slot_pids() -> list[int]:
    ps = subprocess.run(
        ["ps", "-eo", "pid=,cmd="],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: list[int] = []
    for line in ps.stdout.splitlines():
        if "bash results/run_manifests/slot_scripts/seed_matrix_slot" not in line:
            continue
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pids.append(int(parts[0]))
        except ValueError:
            continue
    return pids


def result_count() -> int:
    return sum(1 for task in TASKS if Path(task.result_path).exists())


def active_matrix_task_count() -> int:
    ps = subprocess.run(
        ["ps", "-eo", "cmd="],
        capture_output=True,
        text=True,
        check=False,
    )
    count = 0
    for line in ps.stdout.splitlines():
        if not ("/home/user186/icmlworksop/configs" in line or "train.train_single" in line):
            continue
        if any(task_name in line for task_name in TASK_NAMES):
            count += 1
    return count


def run_one_round() -> None:
    procs: list[subprocess.Popen] = []
    for slot_script in sorted(SLOT_SCRIPT_DIR.glob("seed_matrix_slot*.sh")):
        slot_name = slot_script.stem.replace("seed_matrix_", "")
        launcher_log = LOG_DIR / f"{slot_name}.launcher.log"
        log_fp = launcher_log.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            ["bash", str(slot_script)],
            cwd=str(ROOT),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
        proc._log_fp = log_fp  # type: ignore[attr-defined]
        procs.append(proc)
    for proc in procs:
        proc.wait()
        proc._log_fp.close()  # type: ignore[attr-defined]


def main() -> None:
    round_idx = 0
    while True:
        done = result_count()
        active_slots = len(active_slot_pids())
        active_tasks = active_matrix_task_count()
        print(
            f"[watch] finished={done}/{EXPECTED_TOTAL} active_slots={active_slots} active_tasks={active_tasks}",
            flush=True,
        )
        if done >= EXPECTED_TOTAL:
            subprocess.run(
                [str(PYTHON), str(ROOT / "scripts" / "summarize_seed_matrix.py")],
                cwd=str(ROOT),
                check=False,
            )
            print("[watch] all runs finished; summary generated.", flush=True)
            return

        if active_slots or active_tasks:
            time.sleep(180)
            continue

        round_idx += 1
        print(f"[watch] launching retry round {round_idx}", flush=True)
        run_one_round()
        time.sleep(10)


if __name__ == "__main__":
    main()
