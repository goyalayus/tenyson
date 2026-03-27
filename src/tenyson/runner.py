import argparse
import os
import sys

from tenyson.jobs.eval import EvalJob
from tenyson.jobs.rl import RLJob
from tenyson.jobs.result import JobResult
from tenyson.jobs.sft import SFTJob
from tenyson.core.execution_policy import require_gpu_provider_runtime
from tenyson.loader import load_config, load_task, load_task_from_spec


def _resolve_task(spec: str):
    """Resolve task from a file path or a module:ClassName spec."""
    if spec.endswith(".py") or "/" in spec or "\\" in spec:
        return load_task(spec)
    return load_task_from_spec(spec)


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
    task = _resolve_task(args.task_module)

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
