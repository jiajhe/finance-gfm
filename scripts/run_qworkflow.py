from __future__ import annotations

import argparse
from copy import deepcopy

import qlib
import yaml
from qlib.utils import init_instance_by_config
from qlib.workflow import R


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a Qlib workflow yaml.")
    parser.add_argument("--experiment", default="workflow", help="Experiment name for the recorder.")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    qlib_cfg = deepcopy(config.get("qlib_init", {}))
    qlib.init(**qlib_cfg)

    task = config["task"]
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    with R.start(experiment_name=args.experiment):
        recorder = R.get_recorder()
        if hasattr(model, "fit"):
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


if __name__ == "__main__":
    main()
