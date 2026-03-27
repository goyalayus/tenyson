"""
Pipeline runner with human-in-the-loop on failure: wait for user to choose
resume from checkpoint, restart from scratch, continue after a manual stop,
or abort.
"""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tenyson.cloud.base import _red_print
from tenyson.core.control import request_stop
from tenyson.core.notify import notify_failure
from tenyson.core.run_name import resolve_required_run_name_for_job_class
from tenyson.core.telemetry import (
    TelemetryClient,
    record_run_result,
    record_run_summary,
    resolve_experiment_id,
    resolve_telemetry_context,
)
from tenyson.jobs.result import JobResult
from tenyson.reporting.builder import ReportBuilder

StepTuple = Tuple[str, dict, type, Any]
ParallelStage = Dict[str, Any]
PipelineStep = Union[StepTuple, ParallelStage]


_FAILURE_PROMPT_LOCK = threading.Lock()
_FAILURE_PROMPT_NO_INPUT_WAIT_SECONDS = 5.0


def _normalize_on_failure_policy(on_failure: Optional[str]) -> str:
    """
    Tenyson now always waits for an operator decision after a failure or
    manual stop. Keep accepting legacy inputs such as "abort", but normalize
    them to the only supported policy.
    """
    del on_failure
    return "wait"


def _is_terminal_nonfailure(result: JobResult) -> bool:
    return str(getattr(result, "status", "") or "").lower() in {
        "success",
        "partial",
    }


def _report_update_data(label: str, result: JobResult) -> Dict[str, Any]:
    """Build placeholder update dict for report from step label and result."""
    return ReportBuilder.result_placeholder_data(
        label,
        result,
        metric_precision=None,
        wandb_text=f"{label} run (WandB)",
    )


def _accept_stopped_result(
    result: JobResult,
    *,
    config: dict,
    job_type: str,
) -> None:
    """
    Mark a manually stopped run as an accepted partial result so downstream
    orchestration can move on while still preserving stopped_early metadata.
    When telemetry is configured, mirror that accepted state back into the
    canonical run summary/result store so reports and controller lookups agree.
    """
    result.status = "partial"
    try:
        backend_ref, experiment_id = resolve_telemetry_context(config)
        if not backend_ref or not experiment_id:
            return
        client = TelemetryClient(db_url=backend_ref)
        record_run_summary(
            client=client,
            experiment_id=experiment_id,
            phase=job_type,
            result=result,
        )
        record_run_result(
            client=client,
            experiment_id=experiment_id,
            run_id=result.run_id,
            phase=job_type,
            results_payload=result,
            job_result_payload=result,
        )
    except Exception as exc:  # noqa: BLE001
        _red_print(
            "[TENYSON] Warning: accepted stopped result was promoted to partial "
            f"locally, but telemetry sync failed: {exc}"
        )


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


def _validate_step_tuple(step: Any) -> None:
    if not isinstance(step, tuple) or len(step) != 4:
        raise TypeError(
            "Each sequential step must be a 4-tuple: (label, config, job_class, task)."
        )
    label, config, _job_class, _task = step
    if not isinstance(label, str):
        raise TypeError("Step label must be a string.")
    if not isinstance(config, dict):
        raise TypeError(f"Step config for label '{label}' must be a dict.")


def _is_parallel_stage(step: Any) -> bool:
    return isinstance(step, dict) and "parallel" in step


def _validate_parallel_stage(stage: ParallelStage) -> None:
    label = stage.get("label")
    if label is not None and not isinstance(label, str):
        raise TypeError("Parallel stage 'label' must be a string when provided.")
    branches = stage.get("parallel")
    if not isinstance(branches, list) or not branches:
        raise ValueError("Parallel stage requires a non-empty 'parallel' list.")
    for branch in branches:
        _validate_step_tuple(branch)


def _prompt_failure_action(
    step_label: str,
    config: dict,
    job_type: str,
    on_failure: str,
    last_result: JobResult,
) -> str:
    on_failure = _normalize_on_failure_policy(on_failure)

    train_cfg = config.get("training", {})
    hf_repo_id = str(getattr(last_result, "hf_repo_id", "") or "").strip()
    hf_revision = str(getattr(last_result, "hf_revision", "") or "").strip()
    run_id = str(getattr(last_result, "run_id", "") or "").strip() or "unknown"
    status = str(getattr(last_result, "status", "") or "").strip().lower()
    can_resume = job_type in ("sft", "rl") and bool(hf_repo_id and hf_revision)
    can_continue = status == "stopped" and can_resume

    with _FAILURE_PROMPT_LOCK:
        while True:
            sys.stderr.write(
                "[TENYSON] Awaiting failure action for "
                f'step "{step_label}" (run_id={run_id}, job_type={job_type}).\n'
            )
            if can_resume:
                sys.stderr.write(
                    f"  [resume] Resume from HF checkpoint\n"
                    f"  [restart] Restart step from scratch\n"
                )
            else:
                sys.stderr.write(f"  [restart] Restart step from scratch\n")
            if can_continue:
                sys.stderr.write(
                    "  [continue] Accept the stopped checkpoint and move to the next stage\n"
                )
            sys.stderr.write("  [abort] Abort pipeline\n")
            choices = ["resume", "restart"] if can_resume else ["restart"]
            if can_continue:
                choices.append("continue")
            choices.append("abort")
            sys.stderr.write(f"Choice ({'/'.join(choices)}): ")
            sys.stderr.flush()
            try:
                raw_choice = sys.stdin.readline()
            except EOFError:
                raw_choice = ""
            except KeyboardInterrupt:
                return "abort"
            if raw_choice == "":
                sys.stderr.write(
                    "  No operator input available. Waiting 5s before retrying.\n"
                )
                sys.stderr.flush()
                time.sleep(_FAILURE_PROMPT_NO_INPUT_WAIT_SECONDS)
                continue
            choice = raw_choice.strip().lower()
            if not choice:
                sys.stderr.write("  Empty choice.\n")
                continue
            if choice == "abort":
                return "abort"
            if choice == "restart":
                train_cfg.pop("resume_from_checkpoint", None)
                return "restart"
            if choice == "resume" and can_resume:
                train_cfg["resume_from_checkpoint"] = f"{hf_repo_id}:{hf_revision}"
                return "resume"
            if choice == "continue" and can_continue:
                train_cfg.pop("resume_from_checkpoint", None)
                return "continue"
            sys.stderr.write("  Invalid choice.\n")


def _run_branch_once(
    branch_index: int,
    label: str,
    config: dict,
    job_class: type,
    task: Any,
    cloud: Any,
) -> Tuple[int, JobResult]:
    return branch_index, _run_step(label, config, job_class, task, cloud)


def _abort_parallel_stage_runs(
    branches: List[StepTuple],
    *,
    source_run_id: Optional[str],
) -> None:
    source_run_id = str(source_run_id or "").strip()
    seen_run_ids: set[str] = set()
    for label, config, job_class, _task in branches:
        try:
            _job_type, run_id = resolve_required_run_name_for_job_class(
                config, job_class
            )
        except Exception:
            continue

        run_id = str(run_id).strip()
        if not run_id or run_id == source_run_id or run_id in seen_run_ids:
            continue
        seen_run_ids.add(run_id)

        try:
            db_url, experiment_id = resolve_telemetry_context(config)
        except Exception as exc:  # noqa: BLE001
            _red_print(
                f'[TENYSON] Warning: failed to resolve stop target for parallel branch "{label}": {exc}'
            )
            continue
        if not db_url or not experiment_id:
            continue

        attempt_token = str(
            config.get("telemetry", {}).get("attempt_token") or ""
        ).strip() or None

        try:
            request_stop(
                db_url=db_url,
                run_id=run_id,
                experiment_id=experiment_id,
                phase=_job_type,
                create_if_missing=True,
                attempt_token=attempt_token,
            )
            _red_print(
                f'[TENYSON] Requested stop for parallel sibling "{label}" (run_id={run_id}).'
            )
        except Exception as exc:  # noqa: BLE001
            _red_print(
                f'[TENYSON] Warning: failed to stop parallel sibling "{label}" '
                f"(run_id={run_id}): {exc}"
            )


def _validate_pipeline_run_names(steps: List[PipelineStep]) -> None:
    """
    Preflight: enforce explicit non-default run_name and uniqueness.
    """
    sightings: Dict[str, List[str]] = {}

    def _record(label: str, config: dict, job_class: type) -> None:
        _job_type, run_name = resolve_required_run_name_for_job_class(config, job_class)
        sightings.setdefault(run_name, []).append(label)

    for step in steps:
        if _is_parallel_stage(step):
            _validate_parallel_stage(step)
            for branch in step["parallel"]:
                _validate_step_tuple(branch)
                label, config, job_class, _task = branch
                _record(label, config, job_class)
        else:
            _validate_step_tuple(step)
            label, config, job_class, _task = step
            _record(label, config, job_class)

    duplicates = {
        run_name: labels for run_name, labels in sightings.items() if len(labels) > 1
    }
    if duplicates:
        details = "; ".join(
            f"run_name='{run_name}' used by steps {labels}"
            for run_name, labels in duplicates.items()
        )
        raise ValueError(
            "Duplicate run_name values detected in pipeline. "
            f"Each step must use a unique run_name. {details}"
        )


def run_pipeline(
    steps: List[PipelineStep],
    cloud: Any,
    on_failure: str = "wait",
    failure_log_dir: Optional[str] = None,
    failure_webhook_url: Optional[str] = None,
    db_url: Optional[str] = None,
    report_template_path: Optional[str] = None,
    report_output_path: Optional[str] = None,
    report_initial_data: Optional[Dict[str, Any]] = None,
    before_step: Optional[Callable[[str, int, dict, List[JobResult]], None]] = None,
) -> List[JobResult]:
    """
    Run a sequence of steps.
    - Sequential step: (label, config, JobClass, task)
    - Parallel stage: {"label": "...", "parallel": [step, step, ...]}

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
    on_failure = _normalize_on_failure_policy(on_failure)
    report_enabled = report_template_path is not None and report_output_path is not None
    if report_enabled and report_initial_data is None:
        raise ValueError(
            "report_initial_data is required when report_template_path and report_output_path are set"
        )
    _validate_pipeline_run_names(steps)

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

    for step_index, step in enumerate(steps):
        # Parallel stage support: {"label": "...", "parallel": [step_tuple, ...]}.
        if _is_parallel_stage(step):
            stage = step
            _validate_parallel_stage(stage)
            branches: List[StepTuple] = stage["parallel"]
            prior_results = list(results)

            if before_step is not None:
                for label, config, _job_class, _task in branches:
                    before_step(label, step_index, config, prior_results)

            abort_parallel_stage = False
            executor = ThreadPoolExecutor(max_workers=len(branches))
            try:
                future_map = {
                    executor.submit(
                        _run_branch_once,
                        idx,
                        label,
                        config,
                        job_class,
                        step_task,
                        cloud,
                    ): idx
                    for idx, (label, config, job_class, step_task) in enumerate(
                        branches
                    )
                }
                for future in as_completed(future_map):
                    idx = future_map[future]
                    completed_idx, result = future.result()
                    if completed_idx != idx:
                        idx = completed_idx
                    label, config, job_class, step_task = branches[idx]
                    job_type = job_type_from_class.get(job_class.__name__, "sft")

                    while True:
                        results.append(result)
                        if report_enabled:
                            report.update(_report_update_data(label, result))

                        if _is_terminal_nonfailure(result):
                            break

                        _red_print(
                            f'[TENYSON] Step "{label}" failed: '
                            f"{getattr(result, 'failure_reason', 'unknown')}"
                        )
                        experiment_id = resolve_experiment_id(config)
                        notify_failure(
                            step_label=label,
                            result=result,
                            failure_log_dir=failure_log_dir,
                            failure_webhook_url=failure_webhook_url,
                            db_url=db_url,
                            experiment_id=experiment_id,
                            phase=job_type,
                        )

                        action = _prompt_failure_action(
                            step_label=label,
                            config=config,
                            job_type=job_type,
                            on_failure=on_failure,
                            last_result=result,
                        )
                        if action == "abort":
                            abort_parallel_stage = True
                            _abort_parallel_stage_runs(
                                branches,
                                source_run_id=getattr(result, "run_id", None),
                            )
                            return results
                        if action == "continue":
                            _accept_stopped_result(
                                result,
                                config=config,
                                job_type=job_type,
                            )
                            break
                        result = _run_step(label, config, job_class, step_task, cloud)
            finally:
                executor.shutdown(
                    wait=not abort_parallel_stage,
                    cancel_futures=abort_parallel_stage,
                )
            continue

        _validate_step_tuple(step)
        label, config, job_class, step_task = step
        if before_step is not None:
            before_step(label, step_index, config, results)

        job_type = job_type_from_class.get(job_class.__name__, "sft")
        while True:
            result = _run_step(label, config, job_class, step_task, cloud)
            results.append(result)

            if report_enabled:
                report.update(_report_update_data(label, result))

            if _is_terminal_nonfailure(result):
                break

            # Failed: notify and optionally wait for user.
            _red_print(
                f'[TENYSON] Step "{label}" failed: {getattr(result, "failure_reason", "unknown")}'
            )
            experiment_id = resolve_experiment_id(config)
            notify_failure(
                step_label=label,
                result=result,
                failure_log_dir=failure_log_dir,
                failure_webhook_url=failure_webhook_url,
                db_url=db_url,
                experiment_id=experiment_id,
                phase=job_type,
            )

            action = _prompt_failure_action(
                step_label=label,
                config=config,
                job_type=job_type,
                on_failure=on_failure,
                last_result=result,
            )
            if action == "abort":
                return results
            if action == "continue":
                _accept_stopped_result(
                    result,
                    config=config,
                    job_type=job_type,
                )
                break

    return results
