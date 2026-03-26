from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import threading
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from tenyson.cloud.base import _red_print
from tenyson.core.environment import bind_environment_run
from tenyson.core.control import request_stop
from tenyson.core.telemetry import (
    TelemetryClient,
    get_run_result,
    get_run_metadata_wandb_url,
    resolve_telemetry_context,
    telemetry_project_url,
)
from tenyson.jobs.eval import EvalJob
from tenyson.jobs.result import JobResult
from tenyson.jobs.rl import RLJob
from tenyson.jobs.sft import SFTJob
from tenyson.loader import load_config
from tenyson.pipeline import (
    _accept_stopped_result,
    _normalize_on_failure_policy,
    run_pipeline,
)
from tenyson.reporting.builder import ReportBuilder


_DEFAULT_ABORT_MESSAGE = (
    "[TENYSON] Experiment aborted. Skipping remaining stages and final outputs."
)
_RECOVERY_PROMPT_LOCK = threading.Lock()


def _deep_merge(base: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _is_retryable_failure(result: JobResult) -> bool:
    return str(getattr(result, "status", "") or "").lower() not in {
        "success",
        "partial",
    }


class ExperimentAborted(RuntimeError):
    pass


@dataclass(frozen=True)
class AdapterRef:
    repo_id: str
    revision: str


@dataclass(frozen=True)
class StageSpec:
    id: str
    config: Dict[str, Any]
    job_class: type
    task: Any
    run_type: str
    run_name: str
    environment_run: Optional[str] = None
    variant: Optional[str] = None

    def as_pipeline_step(self) -> Tuple[str, Dict[str, Any], type, Any]:
        return (self.id, self.config, self.job_class, self.task)


def _ensure_stage_attempt_token(stage: StageSpec) -> str:
    telemetry_cfg = stage.config.setdefault("telemetry", {})
    attempt_token = str(telemetry_cfg.get("attempt_token") or "").strip()
    if attempt_token:
        return attempt_token
    attempt_token = uuid4().hex
    telemetry_cfg["attempt_token"] = attempt_token
    return attempt_token


def _prompt_recovery_action(
    *,
    step_label: str,
    run_id: str,
    job_type: str,
    last_result: JobResult,
) -> str:
    hf_repo_id = str(getattr(last_result, "hf_repo_id", "") or "").strip()
    hf_revision = str(getattr(last_result, "hf_revision", "") or "").strip()
    can_resume = job_type in ("sft", "rl") and bool(hf_repo_id and hf_revision)
    can_continue = str(getattr(last_result, "status", "") or "").strip().lower() == "stopped"
    choices = []
    if can_resume:
        choices.append("resume")
    if can_continue:
        choices.append("continue")
    choices.append("restart")

    with _RECOVERY_PROMPT_LOCK:
        while True:
            sys.stderr.write(
                "[TENYSON] Found a previous stopped stage for "
                f'"{step_label}" (run_id={run_id}, job_type={job_type}).\n'
            )
            if can_resume:
                sys.stderr.write("  [resume] Resume from the latest saved HF checkpoint\n")
            if can_continue:
                sys.stderr.write(
                    "  [continue] Accept the stopped checkpoint and move to the next stage\n"
                )
            sys.stderr.write("  [restart] Restart this stage from scratch\n")
            sys.stderr.write(f"Choice ({'/'.join(choices)}): ")
            sys.stderr.flush()
            try:
                choice = sys.stdin.readline().strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "restart"
            if not choice:
                choice = "restart"

            normalized_choice = choice.replace("_", "-").strip()
            if normalized_choice == "resume" and can_resume:
                return "resume"
            if normalized_choice in {
                "continue",
                "next",
                "move-next",
                "stop-and-move-next",
            } and can_continue:
                return "continue"
            if normalized_choice == "restart":
                return "restart"
            sys.stderr.write("  Invalid choice.\n")


class ConfigTemplates:
    def __init__(self, templates: Mapping[str, Dict[str, Any]]) -> None:
        self._templates = {name: deepcopy(config) for name, config in templates.items()}

    @classmethod
    def from_paths(cls, **named_paths: str | Path) -> "ConfigTemplates":
        templates = {name: load_config(str(path)) for name, path in named_paths.items()}
        return cls(templates)

    @classmethod
    def from_directory(
        cls,
        config_dir: str | Path,
        *,
        sft: str = "sft.yaml",
        rl: str = "rl.yaml",
        eval: str = "eval.yaml",
    ) -> "ConfigTemplates":
        config_root = Path(config_dir)
        return cls.from_paths(
            sft=config_root / sft,
            rl=config_root / rl,
            eval=config_root / eval,
        )

    def clone(self, name: str) -> Dict[str, Any]:
        if name not in self._templates:
            raise KeyError(f"Unknown config template: {name}")
        return deepcopy(self._templates[name])


@dataclass(frozen=True)
class _ActiveRun:
    run_id: str
    backend_ref: Optional[str]
    experiment_id: Optional[str]
    phase: Optional[str]
    attempt_token: Optional[str]
    label: str


class _ExperimentAbortController:
    def __init__(self) -> None:
        import threading

        self._event = threading.Event()
        self._lock = threading.Lock()
        self._active_runs: Dict[str, _ActiveRun] = {}

    def is_set(self) -> bool:
        return self._event.is_set()

    def register_stage(self, stage: StageSpec) -> Optional[str]:
        run_id = _resolve_stage_run_name(stage)
        if not run_id:
            return None
        backend_ref, experiment_id = resolve_telemetry_context(stage.config)
        with self._lock:
            self._active_runs[run_id] = _ActiveRun(
                run_id=run_id,
                backend_ref=backend_ref,
                experiment_id=experiment_id,
                phase=_resolve_stage_phase(stage),
                attempt_token=str(
                    stage.config.get("telemetry", {}).get("attempt_token") or ""
                ).strip()
                or None,
                label=stage.id,
            )
        return run_id

    def unregister_run(self, run_id: Optional[str]) -> None:
        if not run_id:
            return
        with self._lock:
            self._active_runs.pop(run_id, None)

    def request_abort(
        self,
        *,
        source_run_id: Optional[str],
        source_label: str,
    ) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            active_runs = [
                active_run
                for run_id, active_run in self._active_runs.items()
                if run_id != str(source_run_id or "")
            ]

        _red_print(
            f'[TENYSON] Abort selected in stage "{source_label}". '
            f"Stopping experiment and signalling {len(active_runs)} active sibling run(s)."
        )
        for active_run in active_runs:
            if not active_run.backend_ref or not active_run.experiment_id:
                continue
            try:
                request_stop(
                    db_url=active_run.backend_ref,
                    run_id=active_run.run_id,
                    experiment_id=active_run.experiment_id,
                    phase=active_run.phase,
                    attempt_token=active_run.attempt_token,
                    create_if_missing=True,
                )
                _red_print(
                    f'[TENYSON] Requested stop for sibling stage "{active_run.label}" '
                    f"(run_id={active_run.run_id})."
                )
            except Exception as exc:  # noqa: BLE001
                _red_print(
                    f"[TENYSON] Warning: failed to request stop for sibling stage "
                    f'"{active_run.label}" (run_id={active_run.run_id}): {exc}'
                )


def _resolve_stage_run_name(stage: StageSpec) -> str:
    run_id = stage.config.get("training", {}).get("run_name") or stage.config.get(
        "evaluation", {}
    ).get("run_name")
    run_id = str(run_id or "").strip()
    if not run_id:
        raise ValueError(f'Stage "{stage.id}" is missing a run_name in its config.')
    return run_id


def _resolve_stage_phase(stage: StageSpec) -> str:
    job_name = getattr(stage.job_class, "__name__", "")
    if job_name == "EvalJob":
        return "eval"
    if job_name == "RLJob":
        return "rl"
    return "sft"


@dataclass
class _ReportWatch:
    stop_event: threading.Event
    thread: threading.Thread


class _ExperimentReportController:
    def __init__(
        self,
        report: Any,
        *,
        metric_precision: Optional[int] = 4,
        wandb_text: str = "run",
        missing: str = "n/a",
        poll_interval_seconds: float = 2.0,
    ) -> None:
        self.report = report
        self.metric_precision = metric_precision
        self.wandb_text = wandb_text
        self.missing = missing
        self.poll_interval_seconds = max(0.1, float(poll_interval_seconds))
        self._lock = threading.Lock()
        self._watches: Dict[str, _ReportWatch] = {}
        self._clients: Dict[str, TelemetryClient] = {}

    def _client_for(self, db_url: str) -> TelemetryClient:
        with self._lock:
            client = self._clients.get(db_url)
            if client is None:
                client = TelemetryClient(db_url=db_url)
                self._clients[db_url] = client
            return client

    def start_stage(self, stage: StageSpec) -> None:
        stage_id = str(stage.id)
        backend_ref, experiment_id = resolve_telemetry_context(stage.config)
        self._sync_report_context(stage, backend_ref=backend_ref, experiment_id=experiment_id)
        self._register_stage(stage)
        if hasattr(self.report, "mark_stage_running"):
            self.report.mark_stage_running(stage_id)
        else:
            self.report.update(
                {
                    f"{stage_id}_status": "running",
                    f"{stage_id}_wandb_link": self.missing,
                }
            )
        if not backend_ref or not experiment_id:
            return

        self.stop_stage(stage_id)
        run_id = _resolve_stage_run_name(stage)
        phase = _resolve_stage_phase(stage)
        min_attempt_updated_at = datetime.now(timezone.utc) - timedelta(seconds=30)
        attempt_token = str(
            stage.config.get("telemetry", {}).get("attempt_token") or ""
        ).strip() or None
        stop_event = threading.Event()
        watcher = threading.Thread(
            target=self._poll_wandb_url,
            args=(
                stage_id,
                run_id,
                phase,
                str(backend_ref),
                str(experiment_id),
                min_attempt_updated_at,
                attempt_token,
                stop_event,
            ),
            daemon=True,
            name=f"tenyson-report-{stage_id}",
        )
        with self._lock:
            self._watches[stage_id] = _ReportWatch(
                stop_event=stop_event,
                thread=watcher,
            )
        watcher.start()

    def _sync_report_context(
        self,
        stage: StageSpec,
        *,
        backend_ref: Optional[str],
        experiment_id: Optional[str],
    ) -> None:
        if not hasattr(self.report, "set_context"):
            return
        environment_name = None
        if hasattr(stage.task, "get_environment_name"):
            try:
                environment_name = stage.task.get_environment_name()
            except Exception:  # noqa: BLE001
                environment_name = None

        project_url = None
        if backend_ref:
            try:
                project_url = telemetry_project_url(self._client_for(str(backend_ref)))
            except Exception:  # noqa: BLE001
                project_url = None

        self.report.set_context(
            environment_name=environment_name,
            experiment_id=experiment_id,
            telemetry_backend_ref=backend_ref,
            telemetry_project_url=project_url,
        )

    def _register_stage(self, stage: StageSpec) -> None:
        if not hasattr(self.report, "register_stage"):
            return
        self.report.register_stage(
            stage_id=stage.id,
            run_type=stage.run_type,
            run_name=stage.run_name,
            variant=stage.environment_run or stage.variant,
        )

    def _poll_wandb_url(
        self,
        stage_id: str,
        run_id: str,
        phase: str,
        backend_ref: str,
        experiment_id: str,
        min_attempt_updated_at: datetime,
        attempt_token: Optional[str],
        stop_event: threading.Event,
    ) -> None:
        warned = False
        while not stop_event.is_set():
            try:
                client = self._client_for(backend_ref)
                run_url = get_run_metadata_wandb_url(
                    client=client,
                    experiment_id=experiment_id,
                    run_id=run_id,
                    min_attempt_updated_at=min_attempt_updated_at,
                    phase=phase,
                    attempt_token=attempt_token,
                )
                if run_url:
                    self.report.update_wandb_link(
                        stage_id,
                        run_url,
                        text=self.wandb_text,
                        missing=self.missing,
                    )
                    return
                warned = False
            except Exception as exc:  # noqa: BLE001
                if not warned:
                    _red_print(
                        f'[TENYSON] Warning: live report WandB polling failed for '
                        f'stage "{stage_id}" (run_id={run_id}): {exc}'
                    )
                    warned = True
            if stop_event.wait(self.poll_interval_seconds):
                return

    def stop_stage(self, stage_id: str) -> None:
        watch = None
        with self._lock:
            watch = self._watches.pop(stage_id, None)
        if watch is not None:
            watch.stop_event.set()

    def finish_stage(self, stage_id: str, result: JobResult) -> None:
        self.stop_stage(stage_id)
        self.report.update_result(
            stage_id,
            result,
            metric_precision=self.metric_precision,
            wandb_text=self.wandb_text,
            missing=self.missing,
        )

    def record_stage_result(self, stage: StageSpec, result: JobResult) -> None:
        backend_ref, experiment_id = resolve_telemetry_context(stage.config)
        self._sync_report_context(
            stage,
            backend_ref=backend_ref,
            experiment_id=experiment_id,
        )
        self._register_stage(stage)
        self.finish_stage(stage.id, result)

    def close(self) -> None:
        with self._lock:
            watches = list(self._watches.values())
            self._watches.clear()
        for watch in watches:
            watch.stop_event.set()


class ExperimentBranch:
    def __init__(
        self, session: "ExperimentSession", *, cloud: Optional[Any] = None
    ) -> None:
        self.session = session
        self._cloud = cloud
        self._results: Dict[str, JobResult] = {}

    def sft(self, stage_id: str, **kwargs: Any) -> StageSpec:
        return self.session.sft(stage_id, **kwargs)

    def rl(self, stage_id: str, **kwargs: Any) -> StageSpec:
        return self.session.rl(stage_id, **kwargs)

    def eval(self, stage_id: str, **kwargs: Any) -> StageSpec:
        return self.session.eval(stage_id, **kwargs)

    def run(self, stage: StageSpec) -> JobResult:
        result = self.session.run_stage(stage, cloud=self._resolve_cloud())
        self._store_result(stage.id, result)
        self.session.raise_if_aborted()
        return result

    def run_parallel(
        self,
        label: str,
        stages: Sequence[StageSpec],
    ) -> Dict[str, JobResult]:
        if not self.session.parallel:
            results: Dict[str, JobResult] = {}
            for stage in stages:
                result = self.session.run_stage(stage, cloud=self._resolve_cloud())
                results[stage.id] = result
                self._store_result(stage.id, result)
            self.session.raise_if_aborted()
            return results

        results = self.session.run_parallel(label, stages, cloud=self._resolve_cloud())
        for stage in stages:
            self._store_result(stage.id, results[stage.id])
        self.session.raise_if_aborted()
        return results

    def result(self, stage_id: str) -> JobResult:
        if stage_id not in self._results:
            raise KeyError(f'No result recorded for stage "{stage_id}".')
        return self._results[stage_id]

    def require_adapter(self, stage_id: str) -> AdapterRef:
        return self.session.require_adapter(self.result(stage_id), stage_id)

    def results(self) -> Dict[str, JobResult]:
        return dict(self._results)

    def _resolve_cloud(self) -> Any:
        if self._cloud is None:
            self._cloud = self.session.create_cloud()
        return self._cloud

    def _store_result(self, stage_id: str, result: JobResult) -> None:
        if stage_id in self._results:
            raise ValueError(f'Duplicate branch result for stage "{stage_id}".')
        self._results[stage_id] = result


class ExperimentSession:
    def __init__(
        self,
        *,
        task: Any,
        templates: ConfigTemplates,
        cloud_factory: Optional[Callable[[], Any]] = None,
        on_failure: str = "wait",
        shared_overrides: Optional[Mapping[str, Any]] = None,
        abort_message: str = _DEFAULT_ABORT_MESSAGE,
        parallel: bool = True,
        report: Optional[Any] = None,
        report_builder: Optional[ReportBuilder] = None,
        report_metric_precision: Optional[int] = 4,
        report_wandb_text: str = "run",
        report_missing: str = "n/a",
        report_poll_interval_seconds: float = 2.0,
        recovery_experiment_id: Optional[str] = None,
        recovery_restart_stages: Optional[Sequence[str]] = None,
    ) -> None:
        if report is not None and report_builder is not None:
            raise ValueError("Pass either report or report_builder, not both.")
        self.task = task
        self.templates = templates
        self.cloud_factory = cloud_factory
        self.on_failure = _normalize_on_failure_policy(on_failure)
        self.shared_overrides = deepcopy(shared_overrides) if shared_overrides else None
        self.abort_message = abort_message
        self.parallel = parallel
        self.recovery_experiment_id = (
            str(recovery_experiment_id or "").strip() or None
        )
        self.recovery_restart_stages = {
            str(stage_name).strip()
            for stage_name in (recovery_restart_stages or [])
            if str(stage_name).strip()
        }
        self._abort_controller = _ExperimentAbortController()
        resolved_report = report if report is not None else report_builder
        self._report_controller = (
            _ExperimentReportController(
                resolved_report,
                metric_precision=report_metric_precision,
                wandb_text=report_wandb_text,
                missing=report_missing,
                poll_interval_seconds=report_poll_interval_seconds,
            )
            if resolved_report is not None
            else None
        )

    @property
    def aborted(self) -> bool:
        return self._abort_controller.is_set()

    def create_cloud(self) -> Any:
        if self.cloud_factory is None:
            raise RuntimeError("This experiment session has no cloud_factory.")
        return self.cloud_factory()

    def branch(self, *, cloud: Optional[Any] = None) -> ExperimentBranch:
        self.raise_if_aborted()
        return ExperimentBranch(self, cloud=cloud)

    def raise_if_aborted(self) -> None:
        if self.aborted:
            raise ExperimentAborted(self.abort_message)

    def close(self) -> None:
        if self._report_controller is not None:
            self._report_controller.close()

    def _resolve_recovery_context(self, stage: StageSpec) -> Optional[Tuple[str, str]]:
        recovery_experiment_id = self.recovery_experiment_id
        if recovery_experiment_id is None:
            return None
        try:
            backend_ref, stage_experiment_id = resolve_telemetry_context(stage.config)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f'Stage "{stage.id}" cannot recover prior results because its telemetry '
                f"context could not be resolved: {exc}"
            ) from exc
        if not backend_ref:
            raise ValueError(
                f'Stage "{stage.id}" cannot recover prior results because telemetry is not configured.'
            )
        if stage_experiment_id and str(stage_experiment_id) != recovery_experiment_id:
            raise ValueError(
                f'Stage "{stage.id}" is configured for experiment_id "{stage_experiment_id}" '
                f'but recovery_experiment_id is "{recovery_experiment_id}".'
            )
        return str(backend_ref), recovery_experiment_id

    def _load_recovered_result(self, stage: StageSpec) -> Optional[JobResult]:
        recovery_context = self._resolve_recovery_context(stage)
        if recovery_context is None:
            return None
        backend_ref, experiment_id = recovery_context
        client = TelemetryClient(db_url=backend_ref)
        run_id = _resolve_stage_run_name(stage)
        phase = _resolve_stage_phase(stage)
        row = get_run_result(
            client,
            experiment_id=experiment_id,
            run_id=run_id,
            phase=phase,
            include_results_payload=False,
        )
        if row is None:
            return None
        _results_payload, job_result_payload = row
        if not isinstance(job_result_payload, dict) or not job_result_payload:
            return None
        try:
            return JobResult.from_dict(job_result_payload)
        except TypeError as exc:
            raise ValueError(
                f'Stage "{stage.id}" recovered an invalid telemetry payload for '
                f'run_id "{run_id}": {exc}'
            ) from exc

    def _should_restart_recovered_stage(self, stage: StageSpec) -> bool:
        if not self.recovery_restart_stages:
            return False
        stage_id = str(stage.id).strip()
        run_name = str(stage.run_name).strip()
        return stage_id in self.recovery_restart_stages or run_name in self.recovery_restart_stages

    def _resolve_recovered_stage_result(self, stage: StageSpec) -> Optional[JobResult]:
        prior_result = self._load_recovered_result(stage)
        if prior_result is None:
            return None

        status = str(getattr(prior_result, "status", "") or "").strip().lower()
        if self._should_restart_recovered_stage(stage):
            train_cfg = stage.config.get("training")
            if isinstance(train_cfg, dict):
                train_cfg.pop("resume_from_checkpoint", None)
            _red_print(
                f'[TENYSON] Recovery restart requested for stage "{stage.id}". '
                f'Ignoring prior {status or "unknown"} result and rerunning it.'
            )
            return None
        if status in {"success", "partial"}:
            _red_print(
                f'[TENYSON] Reusing prior {status} result for stage "{stage.id}" '
                f"(run_id={prior_result.run_id})."
            )
            return prior_result

        if status != "stopped":
            return None

        train_cfg = stage.config.setdefault("training", {})
        action = _prompt_recovery_action(
            step_label=stage.id,
            run_id=str(getattr(prior_result, "run_id", "") or _resolve_stage_run_name(stage)),
            job_type=_resolve_stage_phase(stage),
            last_result=prior_result,
        )
        if action == "resume":
            train_cfg["resume_from_checkpoint"] = (
                f"{prior_result.hf_repo_id}:{prior_result.hf_revision}"
            )
            prior_attempt_token = str(getattr(prior_result, "attempt_token", "") or "").strip()
            if prior_attempt_token:
                stage.config.setdefault("telemetry", {})["attempt_token"] = prior_attempt_token
            _red_print(
                f'[TENYSON] Resuming stage "{stage.id}" from '
                f'{train_cfg["resume_from_checkpoint"]}.'
            )
            return None

        train_cfg.pop("resume_from_checkpoint", None)
        if action == "continue":
            _accept_stopped_result(
                prior_result,
                config=stage.config,
                job_type=_resolve_stage_phase(stage),
            )
            _red_print(
                f'[TENYSON] Accepted prior stopped result for stage "{stage.id}" '
                "and will move to the next stage."
            )
            return prior_result

        _red_print(
            f'[TENYSON] Restarting stage "{stage.id}" from scratch instead of '
            "reusing the prior stopped result."
        )
        return None

    def run_branches(
        self,
        branches: Mapping[str, Callable[[ExperimentBranch], None]],
    ) -> Dict[str, Dict[str, JobResult]]:
        self.raise_if_aborted()
        if not branches:
            return {}

        def _run_branch(
            runner: Callable[[ExperimentBranch], None],
        ) -> Dict[str, JobResult]:
            branch = self.branch()
            runner(branch)
            return branch.results()

        if not self.parallel:
            completed: Dict[str, Dict[str, JobResult]] = {}
            for label, runner in branches.items():
                completed[label] = _run_branch(runner)
            return completed

        completed: Dict[str, Dict[str, JobResult]] = {}
        with ThreadPoolExecutor(max_workers=len(branches)) as executor:
            future_to_label = {
                executor.submit(_run_branch, runner): label
                for label, runner in branches.items()
            }
            for future in as_completed(future_to_label):
                label = future_to_label[future]
                completed[label] = future.result()
        return {label: completed[label] for label in branches}

    @staticmethod
    def combine_results(
        *result_groups: Mapping[str, JobResult],
    ) -> Dict[str, JobResult]:
        combined: Dict[str, JobResult] = {}
        for group in result_groups:
            for stage_id, result in group.items():
                if stage_id in combined:
                    raise ValueError(
                        f'Duplicate stage id "{stage_id}" while combining experiment results.'
                    )
                combined[stage_id] = result
        return combined

    def sft(
        self,
        stage_id: str,
        *,
        base: str = "sft",
        run: Optional[str] = None,
        variant: Optional[str] = None,
        run_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> StageSpec:
        return self._build_stage(
            stage_id=stage_id,
            base=base,
            job_class=SFTJob,
            run_type="sft",
            environment_run=run,
            variant=variant,
            run_name=run_name,
            output_dir=output_dir,
            overrides=overrides,
            adapter=None,
            run_section="training",
        )

    def rl(
        self,
        stage_id: str,
        *,
        adapter: AdapterRef,
        base: str = "rl",
        run: Optional[str] = None,
        variant: Optional[str] = None,
        run_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> StageSpec:
        return self._build_stage(
            stage_id=stage_id,
            base=base,
            job_class=RLJob,
            run_type="rl",
            environment_run=run,
            variant=variant,
            run_name=run_name,
            output_dir=output_dir,
            overrides=overrides,
            adapter=adapter,
            run_section="training",
        )

    def eval(
        self,
        stage_id: str,
        *,
        adapter: AdapterRef,
        base: str = "eval",
        run: Optional[str] = None,
        variant: Optional[str] = None,
        run_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> StageSpec:
        return self._build_stage(
            stage_id=stage_id,
            base=base,
            job_class=EvalJob,
            run_type="eval",
            environment_run=run,
            variant=variant,
            run_name=run_name,
            output_dir=output_dir,
            overrides=overrides,
            adapter=adapter,
            run_section="evaluation",
        )

    def run_stage(self, stage: StageSpec, *, cloud: Optional[Any] = None) -> JobResult:
        self.raise_if_aborted()
        recovered_result = self._resolve_recovered_stage_result(stage)
        if recovered_result is not None:
            if self._report_controller is not None:
                self._report_controller.record_stage_result(stage, recovered_result)
            return recovered_result
        run_id = _resolve_stage_run_name(stage)
        _ensure_stage_attempt_token(stage)
        active_run_id = self._abort_controller.register_stage(stage)
        matched_result = None
        if self._report_controller is not None:
            self._report_controller.start_stage(stage)
        try:
            cloud_instance = cloud if cloud is not None else self.create_cloud()
            results = run_pipeline(
                [stage.as_pipeline_step()],
                cloud_instance,
                on_failure=self.on_failure,
            )
            matched_result = None
            for result in reversed(results):
                if result.run_id == run_id:
                    matched_result = result
                    break
            if matched_result is None:
                returned_run_ids = [result.run_id for result in results]
                raise RuntimeError(
                    f'Stage "{stage.id}" expected run_id "{run_id}" but no exact result was returned. '
                    f"Returned run_ids: {returned_run_ids!r}"
                )
        finally:
            self._abort_controller.unregister_run(active_run_id)
            if matched_result is None and self._report_controller is not None:
                self._report_controller.stop_stage(stage.id)

        if self._report_controller is not None:
            self._report_controller.finish_stage(stage.id, matched_result)

        if _is_retryable_failure(matched_result):
            self._abort_controller.request_abort(
                source_run_id=run_id,
                source_label=stage.id,
            )
        return matched_result

    def run_parallel(
        self,
        label: str,
        stages: Sequence[StageSpec],
        *,
        cloud: Optional[Any] = None,
    ) -> Dict[str, JobResult]:
        self.raise_if_aborted()
        if not stages:
            return {}

        output: Dict[str, JobResult] = {}
        stages_to_run: list[StageSpec] = []
        for stage in stages:
            recovered_result = self._resolve_recovered_stage_result(stage)
            if recovered_result is not None:
                output[stage.id] = recovered_result
                if self._report_controller is not None:
                    self._report_controller.record_stage_result(stage, recovered_result)
                continue
            stages_to_run.append(stage)

        if not stages_to_run:
            return {stage.id: output[stage.id] for stage in stages}

        for stage in stages_to_run:
            _ensure_stage_attempt_token(stage)
        active_run_ids = [
            active_run_id
            for active_run_id in (
                self._abort_controller.register_stage(stage) for stage in stages_to_run
            )
            if active_run_id
        ]
        if self._report_controller is not None:
            for stage in stages_to_run:
                self._report_controller.start_stage(stage)
        results = None
        try:
            cloud_instance = cloud if cloud is not None else self.create_cloud()
            results = run_pipeline(
                [
                    {
                        "label": label,
                        "parallel": [stage.as_pipeline_step() for stage in stages_to_run],
                    }
                ],
                cloud_instance,
                on_failure=self.on_failure,
            )
        finally:
            for active_run_id in active_run_ids:
                self._abort_controller.unregister_run(active_run_id)
            if results is None and self._report_controller is not None:
                for stage in stages_to_run:
                    self._report_controller.stop_stage(stage.id)

        try:
            for stage in stages_to_run:
                run_id = _resolve_stage_run_name(stage)
                matched_result = None
                for result in reversed(results):
                    if result.run_id == run_id:
                        matched_result = result
                        break
                if matched_result is None:
                    returned_run_ids = [result.run_id for result in results]
                    raise RuntimeError(
                        f'Parallel stage "{label}" expected run_id "{run_id}" for stage "{stage.id}" '
                        f"but no exact result was returned. Returned run_ids: {returned_run_ids!r}"
                    )
                output[stage.id] = matched_result
                if self._report_controller is not None:
                    self._report_controller.finish_stage(stage.id, matched_result)
        except Exception:
            if self._report_controller is not None:
                for stage in stages_to_run:
                    self._report_controller.stop_stage(stage.id)
            raise

        for stage in stages_to_run:
            result = output[stage.id]
            if _is_retryable_failure(result):
                self._abort_controller.request_abort(
                    source_run_id=result.run_id,
                    source_label=stage.id,
                )
                break
        return {stage.id: output[stage.id] for stage in stages}

    def require_adapter(self, result: JobResult, stage_id: str) -> AdapterRef:
        repo = getattr(result, "hf_repo_id", None)
        if not repo:
            raise RuntimeError(
                f"{stage_id} finished without hf_repo_id. "
                "Set training.hf_repo_base to a valid Hugging Face namespace/repo prefix."
            )
        revision = getattr(result, "hf_revision", None)
        if not revision:
            raise RuntimeError(
                f"{stage_id} finished without hf_revision. "
                "This run cannot seed later stages safely because its exact adapter revision is unknown."
            )
        return AdapterRef(repo_id=str(repo), revision=str(revision))

    def _build_stage(
        self,
        *,
        stage_id: str,
        base: str,
        job_class: type,
        run_type: str,
        environment_run: Optional[str],
        variant: Optional[str],
        run_name: Optional[str],
        output_dir: Optional[str],
        overrides: Optional[Mapping[str, Any]],
        adapter: Optional[AdapterRef],
        run_section: str,
    ) -> StageSpec:
        config = self.templates.clone(base)
        task_overrides = None
        resolved_environment_run = str(environment_run or "").strip() or None
        if resolved_environment_run is not None:
            named_run_type = None
            if hasattr(self.task, "get_named_run_type"):
                named_run_type = self.task.get_named_run_type(resolved_environment_run)
            if named_run_type is not None and str(named_run_type).strip().lower() != str(
                run_type
            ).strip().lower():
                raise ValueError(
                    f'Stage "{stage_id}" requested run "{resolved_environment_run}" '
                    f'with run_type "{named_run_type}", expected "{run_type}".'
                )
            if not hasattr(self.task, "get_named_run_config_overrides"):
                raise ValueError(
                    f'Task "{type(self.task).__name__}" does not support named runs '
                    f'but stage "{stage_id}" requested run "{resolved_environment_run}".'
                )
            task_overrides = self.task.get_named_run_config_overrides(
                resolved_environment_run
            )
        elif hasattr(self.task, "get_run_config_overrides"):
            task_overrides = self.task.get_run_config_overrides(
                run_type,
                variant=variant,
            )
        if task_overrides:
            _deep_merge(config, task_overrides)
        if self.shared_overrides:
            _deep_merge(config, self.shared_overrides)
        if overrides:
            _deep_merge(config, overrides)

        section_cfg = config.setdefault(run_section, {})
        resolved_run_name = str(run_name or stage_id)
        section_cfg["run_name"] = resolved_run_name
        if output_dir is not None:
            section_cfg["output_dir"] = output_dir
        elif not section_cfg.get("output_dir"):
            section_cfg["output_dir"] = f"./outputs/{resolved_run_name}"

        if adapter is not None:
            model_cfg = config.setdefault("model", {})
            model_cfg["init_adapter_repo"] = adapter.repo_id
            model_cfg["init_adapter_revision"] = adapter.revision
        if resolved_environment_run is not None:
            bind_environment_run(config, resolved_environment_run)

        return StageSpec(
            id=stage_id,
            config=config,
            job_class=job_class,
            task=self.task,
            run_type=run_type,
            run_name=resolved_run_name,
            environment_run=resolved_environment_run,
            variant=variant,
        )
