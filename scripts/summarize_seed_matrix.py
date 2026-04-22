from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from scripts.launch_experiment_matrix import build_tasks


ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "results" / "tables" / "seed_matrix_summary_20260422.md"
OUT_JSON = ROOT / "results" / "tables" / "seed_matrix_summary_20260422.json"

FAMILY_LABELS = {
    "lstm": "LSTM",
    "mlp": "MLP",
    "master": "MASTER",
    "hist": "HIST-Alpha158",
    "fdg_skip32": "FDG-Skip32",
    "fdg_skip32_random": "Skip32-Random",
    "fdg_skip32_symshare": "Skip32-SymShare",
}


@dataclass
class SeedResult:
    task_name: str
    seed: int
    model: str
    split: str
    ic: float | None
    rank_ic: float | None
    ann_ret: float | None
    ir_or_sharpe: float | None
    result_path: str


def _safe_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def _metric(payload: dict, key: str) -> float | None:
    metrics = payload.get("test_metrics", {})
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _pick_metric(payload: dict, keys: list[str]) -> float | None:
    for key in keys:
        value = _metric(payload, key)
        if value is not None:
            return value
    return None


def _fmt(mean_value: float | None, std_value: float | None, n: int) -> str:
    if mean_value is None or n == 0:
        return "-"
    if std_value is None:
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f} ± {std_value:.4f}"


def _agg(values: list[float | None]) -> tuple[float | None, float | None, int]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None, 0
    if len(clean) == 1:
        return clean[0], 0.0, 1
    return mean(clean), pstdev(clean), len(clean)


def main() -> None:
    tasks = build_tasks()
    grouped: dict[tuple[str, str], list[SeedResult]] = {}
    missing: list[dict] = []

    for task in tasks:
        result_path = Path(task.result_path)
        payload = _safe_load(result_path)
        model = FAMILY_LABELS.get(task.family, task.family)
        key = (model, task.split)
        grouped.setdefault(key, [])

        if payload is None:
            missing.append(
                {
                    "task": task.name,
                    "model": model,
                    "split": task.split,
                    "seed": task.seed,
                    "result_path": str(result_path),
                }
            )
            continue

        grouped[key].append(
            SeedResult(
                task_name=task.name,
                seed=task.seed,
                model=model,
                split=task.split,
                ic=_metric(payload, "IC_mean"),
                rank_ic=_metric(payload, "RankIC_mean"),
                ann_ret=_pick_metric(payload, ["annual_return", "annualized_return_without_cost"]),
                ir_or_sharpe=_pick_metric(payload, ["sharpe", "information_ratio_without_cost"]),
                result_path=str(result_path),
            )
        )

    rows: list[dict] = []
    model_order = [
        "LSTM",
        "MLP",
        "MASTER",
        "HIST-Alpha158",
        "FDG-Skip32",
        "Skip32-Random",
        "Skip32-SymShare",
    ]
    split_order = ["official", "recent"]
    for model in model_order:
        for split in split_order:
            results = grouped.get((model, split), [])
            ic_mean, ic_std, n_ic = _agg([r.ic for r in results])
            ric_mean, ric_std, n_ric = _agg([r.rank_ic for r in results])
            ret_mean, ret_std, n_ret = _agg([r.ann_ret for r in results])
            ir_mean, ir_std, n_ir = _agg([r.ir_or_sharpe for r in results])
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "n": len(results),
                    "IC_mean": ic_mean,
                    "IC_std": ic_std,
                    "RankIC_mean": ric_mean,
                    "RankIC_std": ric_std,
                    "AnnRet_mean": ret_mean,
                    "AnnRet_std": ret_std,
                    "IR_mean": ir_mean,
                    "IR_std": ir_std,
                }
            )

    lines = [
        "# Seed Matrix Summary (2026-04-22)",
        "",
        f"- Expected runs: {len(tasks)}",
        f"- Finished runs: {len(tasks) - len(missing)}",
        f"- Missing runs: {len(missing)}",
        "",
        "| Model | Split | Finished Seeds | IC | RankIC | AnnRet | IR/Sharpe |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            "| {model} | {split} | {n} | {ic} | {ric} | {ret} | {ir} |".format(
                model=row["model"],
                split=row["split"],
                n=row["n"],
                ic=_fmt(row["IC_mean"], row["IC_std"], row["n"]),
                ric=_fmt(row["RankIC_mean"], row["RankIC_std"], row["n"]),
                ret=_fmt(row["AnnRet_mean"], row["AnnRet_std"], row["n"]),
                ir=_fmt(row["IR_mean"], row["IR_std"], row["n"]),
            )
        )

    if missing:
        lines.extend(
            [
                "",
                "## Missing Runs",
                "",
            ]
        )
        for item in missing:
            lines.append(
                f"- `{item['task']}` ({item['model']}, {item['split']}, seed={item['seed']})"
            )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with OUT_JSON.open("w", encoding="utf-8") as fp:
        json.dump({"rows": rows, "missing": missing}, fp, indent=2, ensure_ascii=False)

    print(f"Saved summary markdown to {OUT_MD}")
    print(f"Saved summary json to {OUT_JSON}")


if __name__ == "__main__":
    main()
