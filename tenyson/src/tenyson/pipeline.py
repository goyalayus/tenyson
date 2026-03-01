"""
Pipeline runner with human-in-the-loop on failure: wait for user to choose
resume from checkpoint, restart from scratch, or abort.
"""

import glob
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tenyson.cloud.base import _red_print
from tenyson.core.notify import notify_failure
from tenyson.jobs.result import JobResult
from tenyson.reporting.builder import ReportBuilder


def get_latest_checkpoint_dir(output_dir: str, job_type: str = "sft") -> Optional[str]:
    """
    Find the latest checkpoint directory under output_dir.
    For SFT: output_dir/checkpoint-*.
    For RL: output_dir/trainer_out/checkpoint-*.
    Returns the path with the largest step number, or None if no checkpoints.
    """
    if job_type == "rl":
        search_dir = os.path.join(output_dir, "trainer_out")
    else:
        search_dir = output_dir
    pattern = os.path.join(search_dir, "checkpoint-*")
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    best_path = None
    best_step = -1
    for d in dirs:
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[1])
                if step > best_step:
                    best_step = step
                    best_path = d
            except (IndexError, ValueError):
                continue
    return best_path


def _report_update_data(label: str, result: JobResult) -> Dict[str, Any]:
    """Build placeholder update dict for report from step label and result."""
    wandb_link = (
        f"[{label} run (WandB)]({result.wandb_url})"
        if result.wandb_url
        else "n/a"
    )
    data: Dict[str, Any] = {
        f"{label}_status": result.status,
        f"{label}_wandb_link": wandb_link,
    }
    if result.metrics:
        for k, v in result.metrics.items():
            data[f"{label}_{k}"] = v if isinstance(v, str) else str(v)
    return data


def _run_step(
    label: str,
    config: dict,
    job_class: type,
    task: Any,
    cloud: Any,
) -> JobResult:
    """Build and run a single step."""
    job = job_class(config=config, task=task)
    return cloud.run(job)


def run_pipeline(
    steps: List[Tuple[str, dict, type, Any]],
    cloud: Any,
    on_failure: str = "wait",
    failure_log_dir: Optional[str] = None,
    failure_webhook_url: Optional[str] = None,
    db_url: Optional[str] = None,
    report_template_path: Optional[str] = None,
    report_output_path: Optional[str] = None,
    report_initial_data: Optional[Dict[str, Any]] = None,
    before_step: Optional[
        Callable[[str, int, dict, List[JobResult]], None]
    ] = None,
) -> List[JobResult]:
    """
    Run a sequence of steps. Each step is (label, config, JobClass, task).
    Optional before_step(label, step_index, config, previous_results) is called
    before each step so the caller can mutate config (e.g. init_adapter_repo).
    When reporting is enabled (report_template_path + report_output_path),
    a report file is created at start and updated after each step.

    When a step returns status "failed", do not exit: print failure in red,
    call notify_failure, then enter a wait loop (resume / restart / abort).
    Works with both AWS and Modal.
    """
    if (report_template_path is None) != (report_output_path is None):
        raise ValueError(
            "report_template_path and report_output_path must both be set or both unset"
        )
    report_enabled = report_template_path is not None and report_output_path is not None
    if report_enabled and report_initial_data is None:
        raise ValueError(
            "report_initial_data is required when report_template_path and report_output_path are set"
        )

    results: List[JobResult] = []
    job_type_from_class = {
        "SFTJob": "sft",
        "RLJob": "rl",
        "EvalJob": "eval",
    }

    if report_enabled:
        report = ReportBuilder(
            template_path=report_template_path,
            output_path=report_output_path,
        )
        report.fill(report_initial_data)
        report.generate()

    for step_index, (label, config, job_class, step_task) in enumerate(steps):
        if before_step is not None:
            before_step(label, step_index, config, results)

        job_type = job_type_from_class.get(job_class.__name__, "sft")
        while True:
            result = _run_step(label, config, job_class, step_task, cloud)
            results.append(result)

            if report_enabled:
                report.update(_report_update_data(label, result))

            if result.status == "success":
                break

            # Failed: notify and optionally wait for user.
            _red_print(
                f"[TENYSON] Step \"{label}\" failed: {getattr(result, 'failure_reason', 'unknown')}"
            )
            notify_failure(
                step_label=label,
                result=result,
                failure_log_dir=failure_log_dir,
                failure_webhook_url=failure_webhook_url,
                db_url=db_url,
            )

            if on_failure != "wait":
                return results

            # Wait loop: resume, restart, or abort.
            train_cfg = config.get("training", {})
            output_dir = train_cfg.get("output_dir", "")
            can_resume = job_type in ("sft", "rl") and output_dir

            while True:
                if can_resume:
                    sys.stderr.write(
                        f"  [resume] Resume from last checkpoint\n"
                        f"  [restart] Restart step from scratch\n"
                    )
                else:
                    sys.stderr.write(f"  [restart] Restart step from scratch\n")
                sys.stderr.write("  [abort] Abort pipeline\n")
                sys.stderr.write("Choice (resume/restart/abort): ")
                sys.stderr.flush()
                try:
                    choice = sys.stdin.readline().strip().lower()
                except (EOFError, KeyboardInterrupt):
                    choice = "abort"
                if choice == "abort":
                    return results
                if choice == "restart":
                    train_cfg.pop("resume_from_checkpoint", None)
                    break
                if choice == "resume" and can_resume:
                    abs_output = os.path.abspath(output_dir)
                    latest = get_latest_checkpoint_dir(abs_output, job_type)
                    if latest:
                        train_cfg["resume_from_checkpoint"] = latest
                        break
                    sys.stderr.write("  No checkpoint found; choose restart or abort.\n")
                    continue
                sys.stderr.write("  Invalid choice.\n")

    return results
