from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SeedMetric:
    seed: int
    name: str
    metric: float
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank paired seed runs across models.")
    parser.add_argument("--our-prefix", required=True, help="Filename prefix for our model, e.g. fdg_csi300_qlib_recent_skip32_seed")
    parser.add_argument(
        "--baseline-prefix",
        action="append",
        default=[],
        help="Filename prefix for a baseline model. Can be repeated.",
    )
    parser.add_argument(
        "--results-dir",
        default="results/tables",
        help="Directory containing per-seed result json files.",
    )
    parser.add_argument(
        "--metric",
        default="IC_mean",
        help="Metric key inside test_metrics to rank by. Default: IC_mean",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many seeds to print at the top of the ranking.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional markdown output path.",
    )
    return parser.parse_args()


def load_seed_metrics(results_dir: Path, prefix: str, metric_key: str) -> dict[int, SeedMetric]:
    pattern = re.compile(re.escape(prefix) + r"(?P<seed>\d+)\.json$")
    records: dict[int, SeedMetric] = {}
    for path in sorted(results_dir.glob(f"{prefix}*.json")):
        match = pattern.match(path.name)
        if not match:
            continue
        seed = int(match.group("seed"))
        try:
            with path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception:
            continue
        metrics = payload.get("test_metrics", {})
        if metric_key not in metrics:
            continue
        try:
            metric = float(metrics[metric_key])
        except Exception:
            continue
        records[seed] = SeedMetric(seed=seed, name=path.stem, metric=metric, path=path)
    return records


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    our = load_seed_metrics(results_dir, args.our_prefix, args.metric)
    baselines = {
        prefix: load_seed_metrics(results_dir, prefix, args.metric)
        for prefix in args.baseline_prefix
    }

    candidate_seeds = set(our)
    for mapping in baselines.values():
        candidate_seeds &= set(mapping)
    rows = []
    for seed in sorted(candidate_seeds):
        our_metric = our[seed].metric
        baseline_values = {prefix: mapping[seed].metric for prefix, mapping in baselines.items()}
        if baseline_values:
            baseline_mean = sum(baseline_values.values()) / len(baseline_values)
            baseline_best = max(baseline_values.values())
            delta_mean = our_metric - baseline_mean
            delta_best = our_metric - baseline_best
        else:
            baseline_mean = None
            baseline_best = None
            delta_mean = None
            delta_best = None
        rows.append(
            {
                "seed": seed,
                "our_metric": our_metric,
                "baseline_metrics": baseline_values,
                "baseline_mean": baseline_mean,
                "baseline_best": baseline_best,
                "delta_vs_mean": delta_mean,
                "delta_vs_best": delta_best,
                "our_path": str(our[seed].path),
            }
        )

    rows.sort(
        key=lambda row: (
            row["delta_vs_best"] if row["delta_vs_best"] is not None else row["our_metric"],
            row["our_metric"],
            -(row["baseline_mean"] or 0.0),
        ),
        reverse=True,
    )

    header = [
        f"# Seed Sweep Ranking",
        "",
        f"- Our prefix: `{args.our_prefix}`",
        f"- Baselines: {', '.join(f'`{p}`' for p in args.baseline_prefix) if args.baseline_prefix else '(none)'}",
        f"- Metric: `{args.metric}`",
        f"- Candidate seeds with complete paired results: `{len(rows)}`",
        "",
        "| Rank | Seed | Our | Baseline Mean | Baseline Best | Delta vs Mean | Delta vs Best |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        header.append(
            "| {rank} | {seed} | {our:.4f} | {bmean} | {bbest} | {dmean} | {dbest} |".format(
                rank=rank,
                seed=row["seed"],
                our=row["our_metric"],
                bmean=f"{row['baseline_mean']:.4f}" if row["baseline_mean"] is not None else "-",
                bbest=f"{row['baseline_best']:.4f}" if row["baseline_best"] is not None else "-",
                dmean=f"{row['delta_vs_mean']:.4f}" if row["delta_vs_mean"] is not None else "-",
                dbest=f"{row['delta_vs_best']:.4f}" if row["delta_vs_best"] is not None else "-",
            )
        )

    top_rows = rows[: args.top_k]
    summary = {
        "top_k": args.top_k,
        "top_seeds": [row["seed"] for row in top_rows],
        "rows": rows,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(header) + "\n", encoding="utf-8")
        with out_path.with_suffix(".json").open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
        print(f"Saved markdown ranking to {out_path}")
        print(f"Saved json ranking to {out_path.with_suffix('.json')}")
    else:
        print("\n".join(header[: min(len(header), args.top_k + 9)]))


if __name__ == "__main__":
    main()
