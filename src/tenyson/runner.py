import argparse
import os
import sys

from tenyson.jobs.eval import EvalJob
from tenyson.jobs.rl import RLJob
from tenyson.jobs.result import JobResult
from tenyson.jobs.sft import SFTJob
from tenyson.core.execution_policy import require_gpu_provider_runtime
from tenyson.core.stage_templates import bind_stage_templates_from_config
from tenyson.loader import load_config, load_task, load_task_from_spec


def _resolve_task(spec: str):
    """Resolve task from a file path or a module:ClassName spec."""
    if spec.endswith(".py") or "/" in spec or "\\" in spec:
        return load_task(spec)
    return load_task_from_spec(spec)


def _maybe_add_task_module_parent_to_sys_path(spec: str) -> None:
    """Make sibling imports from a task file importable on remote workers.

    Example:
    `examples/arithmetic/experiment.py` may import hooks with
    `from functional import build_three_digit_addition_dataset`.
    Those hook refs serialize as module `functional`, so the remote runner needs
    the task file's parent directory on `sys.path` before it rebinds templates.
    """
    if not (spec.endswith(".py") or "/" in spec or "\\" in spec):
        return

    task_path = os.path.abspath(spec)
    task_dir = os.path.dirname(task_path)
    if task_dir and task_dir not in sys.path:
        sys.path.insert(0, task_dir)


def _maybe_fast_exit_after_cloud_rl_job(job_type: str, result: object) -> None:
    if job_type != "rl":
        return
    if str(os.getenv("TENYSON_EXECUTION_MODE") or "").strip().lower() != "cloud":
        return
    if not isinstance(result, JobResult):
        return
    # RL cloud workers can hit an Unsloth/vLLM/Torch teardown crash after the
    # terminal JobResult is already recorded. Flush logs, then exit before
    # Python starts interpreter shutdown on those objects.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main() -> None:
    require_gpu_provider_runtime()
    parser = argparse.ArgumentParser(description="Tenyson remote job runner")
    parser.add_argument("--job-type", required=True, choices=["sft", "rl", "eval"])
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    parser.add_argument(
        "--task-module",
        required=True,
        help="Path to task .py file or module.path:ClassName spec",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Hugging Face reference repo_id:revision to resume training (SFT/RL only)",
    )

    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    if args.resume_from_checkpoint and args.job_type in ("sft", "rl"):
        config.setdefault("training", {})["resume_from_checkpoint"] = args.resume_from_checkpoint
    _maybe_add_task_module_parent_to_sys_path(args.task_module)
    task = _resolve_task(args.task_module)
    task = bind_stage_templates_from_config(task, config)

    if args.job_type == "sft":
        job = SFTJob(config=config, task=task)
    elif args.job_type == "rl":
        job = RLJob(config=config, task=task)
    else:
        job = EvalJob(config=config, task=task)

    result = job.run()
    _maybe_fast_exit_after_cloud_rl_job(args.job_type, result)


if __name__ == "__main__":
    main()
