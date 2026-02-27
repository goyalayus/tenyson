import argparse
import os

from tenyson.jobs.eval import EvalJob
from tenyson.jobs.rl import RLJob
from tenyson.jobs.sft import SFTJob
from tenyson.loader import load_config, load_task, load_task_from_spec


def _resolve_task(spec: str):
    """Resolve task from a file path or a module:ClassName spec."""
    if spec.endswith(".py") or "/" in spec or "\\" in spec:
        return load_task(spec)
    return load_task_from_spec(spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tenyson remote job runner")
    parser.add_argument("--job-type", required=True, choices=["sft", "rl", "eval"])
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    parser.add_argument(
        "--task-module",
        required=True,
        help="Path to task .py file or module.path:ClassName spec",
    )

    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    task = _resolve_task(args.task_module)

    if args.job_type == "sft":
        job = SFTJob(config=config, task=task)
    elif args.job_type == "rl":
        job = RLJob(config=config, task=task)
    else:
        job = EvalJob(config=config, task=task)

    job.run()


if __name__ == "__main__":
    main()

