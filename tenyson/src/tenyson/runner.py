import argparse
import importlib
import os
from typing import Any, Dict

import yaml

from tenyson.jobs.eval import EvalJob
from tenyson.jobs.rl import RLJob
from tenyson.jobs.sft import SFTJob


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            import json

            return json.load(f)
        return yaml.safe_load(f)


def _load_task(spec: str):
    """
    Load a TaskPlugin implementation given an import spec of the form
    'module.path:ClassName'.
    """
    if ":" not in spec:
        raise ValueError(
            f"task-module must be of form 'module.path:ClassName', got: {spec}"
        )
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    task_cls = getattr(module, class_name)
    return task_cls()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tenyson remote job runner")
    parser.add_argument("--job-type", required=True, choices=["sft", "rl", "eval"])
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    parser.add_argument(
        "--task-module",
        required=True,
        help="TaskPlugin import path, e.g. 'tenyson.examples.wordle.wordle_task:WordleTask'",
    )

    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config = _load_config(config_path)
    task = _load_task(args.task_module)

    if args.job_type == "sft":
        job = SFTJob(config=config, task=task)
    elif args.job_type == "rl":
        job = RLJob(config=config, task=task)
    else:
        job = EvalJob(config=config, task=task)

    job.run()


if __name__ == "__main__":
    main()

