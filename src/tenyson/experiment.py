from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from tenyson.cloud.base import _red_print
from tenyson.core.controller_runtime import (
    boundary_stop_requested,
    clear_stop_at_boundary_request,
    controller_metadata_path_from_env,
    update_controller_runtime_state,
)
from tenyson.core.environment import bind_environment_run
from tenyson.core.stage_templates import (
    EvalDatasetTemplate,
    EvalDatasetLike,
    EvalMetricsTemplate,
    EvalMetricsLike,
    RLRewardTemplate,
    RLDatasetTemplate,
    SFTDatasetTemplate,
    STAGE_TEMPLATE_CONFIG_KEY,
    bind_stage_templates,
    coerce_eval_dataset_template,
    coerce_eval_metrics_template,
    has_explicit_stage_templates,
    serialize_stage_templates,
)
from tenyson.core.control import list_live_runs, prime_stop_target, request_stop
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
_RECOVERY_LIVE_RUN_MAX_AGE_SECONDS = 180
_PARALLEL_RESULT_TELEMETRY_WAIT_TIMEOUT_SECONDS = 120.0
_PARALLEL_RESULT_TELEMETRY_POLL_SECONDS = 5.0
_PARALLEL_RESULT_TELEMETRY_EMPTY_LIVE_POLLS = 3
_RECOVERY_FILE_LOCKS: Dict[str, Tuple[Any, int]] = {}
_RECOVERY_FILE_LOCKS_LOCK = threading.Lock()
_ADAPTER_ARTIFACT_TYPE = "adapter"
_FULL_MODEL_ARTIFACT_TYPE = "full_model"
_SUPPORTED_ARTIFACT_TYPES = {
    _ADAPTER_ARTIFACT_TYPE,
    _FULL_MODEL_ARTIFACT_TYPE,
}


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


def _safe_recovery_lock_name(experiment_id: str) -> str:
    text = str(experiment_id or "").strip().lower()
    if not text:
        return "recovery"
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "-"
        for char in text
    )
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "recovery"


class ExperimentAborted(RuntimeError):
    pass


def _normalize_artifact_type(value: Optional[str]) -> str:
    artifact_type = str(value or _ADAPTER_ARTIFACT_TYPE).strip().lower()
    if not artifact_type:
        artifact_type = _ADAPTER_ARTIFACT_TYPE
    if artifact_type not in _SUPPORTED_ARTIFACT_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_ARTIFACT_TYPES))
        raise ValueError(
            f'Unsupported artifact_type "{value}". Expected one of: {supported}.'
        )
    return artifact_type


@dataclass(frozen=True)
class AdapterRef:
    repo_id: str
    revision: str
    artifact_type: str = _ADAPTER_ARTIFACT_TYPE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_type",
            _normalize_artifact_type(self.artifact_type),
        )

    @property
    def is_adapter(self) -> bool:
        return self.artifact_type == _ADAPTER_ARTIFACT_TYPE

    @property
    def is_full_model(self) -> bool:
        return self.artifact_type == _FULL_MODEL_ARTIFACT_TYPE


ArtifactRef = AdapterRef


@dataclass(frozen=True)
class _DeferredStageArtifactRef:
    stage_id: str


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
    deferred_artifact: Optional[_DeferredStageArtifactRef] = None

    def as_pipeline_step(self) -> Tuple[str, Dict[str, Any], type, Any]:
        if self.deferred_artifact is not None:
            raise RuntimeError(
                f'Stage "{self.id}" still has an unresolved artifact dependency on '
                f'"{self.deferred_artifact.stage_id}".'
            )
        return (self.id, self.config, self.job_class, self.task)


def _resolve_stage_artifact(
    *,
    stage_id: str,
    run_type: str,
    adapter: Optional[AdapterRef | _DeferredStageArtifactRef],
    artifact: Optional[AdapterRef | _DeferredStageArtifactRef],
) -> Optional[AdapterRef | _DeferredStageArtifactRef]:
    if adapter is not None and artifact is not None:
        raise ValueError(
            f'Stage "{stage_id}" passed both adapter= and artifact= to {run_type}().'
        )

    return artifact if artifact is not None else adapter


def _is_deferred_stage_artifact_ref(
    value: object,
) -> bool:
    return isinstance(value, _DeferredStageArtifactRef)


def _apply_stage_artifact_config(
    config: Dict[str, Any],
    artifact: Optional[AdapterRef],
) -> None:
    if artifact is None:
        return

    model_cfg = config.setdefault("model", {})
    model_cfg.pop("init_artifact_type", None)
    model_cfg.pop("init_model_repo", None)
    model_cfg.pop("init_model_revision", None)

    if artifact.is_full_model:
        model_cfg.pop("init_adapter_repo", None)
        model_cfg.pop("init_adapter_revision", None)
        model_cfg["init_artifact_type"] = _FULL_MODEL_ARTIFACT_TYPE
        model_cfg["init_model_repo"] = artifact.repo_id
        model_cfg["init_model_revision"] = artifact.revision
        return

    model_cfg["init_adapter_repo"] = artifact.repo_id
    model_cfg["init_adapter_revision"] = artifact.revision


def _resolve_deferred_stage_artifact(
    stage: StageSpec,
    artifact: AdapterRef,
) -> StageSpec:
    if stage.deferred_artifact is None:
        return stage

    resolved_config = deepcopy(stage.config)
    _apply_stage_artifact_config(resolved_config, artifact)
    _require_stage_model_source(
        config=resolved_config,
        stage_id=stage.id,
        run_type=stage.run_type,
    )
    return replace(
        stage,
        config=resolved_config,
        deferred_artifact=None,
    )


def _has_stage_model_source(config: Dict[str, Any]) -> bool:
    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, Mapping):
        return False

    model_name = str(model_cfg.get("name") or "").strip()
    init_adapter_repo = str(model_cfg.get("init_adapter_repo") or "").strip()
    init_model_repo = str(model_cfg.get("init_model_repo") or "").strip()
    return bool(model_name or init_adapter_repo or init_model_repo)


def _require_stage_model_source(
    *,
    config: Dict[str, Any],
    stage_id: str,
    run_type: str,
) -> None:
    if _has_stage_model_source(config):
        return
    raise TypeError(
        f'Stage "{stage_id}" has no model source for {run_type}(). '
        "Either pass adapter=/artifact= or set model.name / model.init_* in the config."
    )


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
    can_continue = (
        str(getattr(last_result, "status", "") or "").strip().lower() == "stopped"
        and (job_type == "eval" or can_resume)
    )
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
        try:
            backend_ref, experiment_id = resolve_telemetry_context(stage.config)
        except Exception:
            backend_ref, experiment_id = None, None
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
        try:
            backend_ref, experiment_id = resolve_telemetry_context(stage.config)
        except Exception:
            backend_ref, experiment_id = None, None
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
        try:
            backend_ref, experiment_id = resolve_telemetry_context(stage.config)
        except Exception:
            backend_ref, experiment_id = None, None
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
        resolved_stage = self._resolve_stage_dependencies(stage)
        result = self.session.run_stage(resolved_stage, cloud=self._resolve_cloud())
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
                resolved_stage = self._resolve_stage_dependencies(stage)
                result = self.session.run_stage(
                    resolved_stage,
                    cloud=self._resolve_cloud(),
                )
                results[stage.id] = result
                self._store_result(stage.id, result)
            self.session.raise_if_aborted()
            return results

        resolved_stages = [self._resolve_stage_dependencies(stage) for stage in stages]
        results = self.session.run_parallel(
            label,
            resolved_stages,
            cloud=self._resolve_cloud(),
        )
        for stage in stages:
            self._store_result(stage.id, results[stage.id])
        self.session.raise_if_aborted()
        return results

    def result(self, stage_id: str) -> JobResult:
        if stage_id not in self._results:
            raise KeyError(f'No result recorded for stage "{stage_id}".')
        return self._results[stage_id]

    def artifact(self, stage_id: str) -> AdapterRef:
        return self.require_artifact(stage_id)

    def require_artifact(self, stage_id: str) -> AdapterRef:
        return self.session.require_artifact(self.result(stage_id), stage_id)

    def adapter(self, stage_id: str) -> AdapterRef:
        return self.require_adapter(stage_id)

    def require_adapter(self, stage_id: str) -> AdapterRef:
        return self.require_artifact(stage_id)

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

    def _resolve_stage_dependencies(self, stage: StageSpec) -> StageSpec:
        deferred_artifact = stage.deferred_artifact
        if deferred_artifact is None:
            return stage

        try:
            artifact = self.require_artifact(deferred_artifact.stage_id)
        except KeyError as exc:
            raise RuntimeError(
                f'Stage "{stage.id}" depends on artifact from stage '
                f'"{deferred_artifact.stage_id}", but that result is not available '
                "in this branch yet."
            ) from exc

        return _resolve_deferred_stage_artifact(stage, artifact)


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
        self._owned_clouds: list[Any] = []
        self._owned_cloud_ids: set[int] = set()
        self._owned_cloud_lock = threading.Lock()
        self._recovery_launch_preflight_lock = threading.Lock()
        self._recovery_launch_preflight_done = False
        self._recovery_lock_key: Optional[str] = None
        self._acquire_recovery_controller_lock()
        update_controller_runtime_state(
            state="starting",
            active_stage_ids=[],
            last_completed_stage_ids=[],
        )

    @property
    def aborted(self) -> bool:
        return self._abort_controller.is_set()

    def create_cloud(self) -> Any:
        if self.cloud_factory is None:
            raise RuntimeError("This experiment session has no cloud_factory.")
        cloud = self.cloud_factory()
        self._register_cloud(cloud)
        return cloud

    def branch(self, *, cloud: Optional[Any] = None) -> ExperimentBranch:
        self.raise_if_aborted()
        if cloud is not None:
            self._register_cloud(cloud)
        return ExperimentBranch(self, cloud=cloud)

    def raise_if_aborted(self) -> None:
        if boundary_stop_requested():
            clear_stop_at_boundary_request()
            self._abort_controller.request_abort(
                source_run_id=None,
                source_label="controller boundary stop",
            )
        if self.aborted:
            raise ExperimentAborted(self.abort_message)

    def close(self) -> None:
        with self._owned_cloud_lock:
            clouds = list(self._owned_clouds)
        for cloud in clouds:
            close_method = getattr(cloud, "close", None)
            if not callable(close_method):
                continue
            try:
                close_method()
            except Exception as exc:  # noqa: BLE001
                _red_print(
                    "[TENYSON] Warning: failed to close cloud manager during "
                    f"session shutdown: {exc}"
                )
        if self._report_controller is not None:
            self._report_controller.close()
        self._release_recovery_controller_lock()
        update_controller_runtime_state(
            state="closed",
            active_stage_ids=[],
            last_completed_stage_ids=[],
        )

    def _register_cloud(self, cloud: Any) -> None:
        cloud_id = id(cloud)
        with self._owned_cloud_lock:
            if cloud_id in self._owned_cloud_ids:
                return
            self._owned_cloud_ids.add(cloud_id)
            self._owned_clouds.append(cloud)

    def _recovery_lock_path(self) -> Optional[Path]:
        if self.recovery_experiment_id is None:
            return None
        configured_root = str(
            os.getenv("TENYSON_RECOVERY_LOCK_DIR", ".tenyson_runs/recovery_locks")
        ).strip()
        lock_root = Path(configured_root or ".tenyson_runs/recovery_locks").expanduser()
        safe_name = _safe_recovery_lock_name(self.recovery_experiment_id)
        return (lock_root / f"{safe_name}.lock").resolve()

    def _acquire_recovery_controller_lock(self) -> None:
        lock_path = self._recovery_lock_path()
        if lock_path is None or fcntl is None or self._recovery_lock_key is not None:
            return
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_key = str(lock_path)
        with _RECOVERY_FILE_LOCKS_LOCK:
            existing = _RECOVERY_FILE_LOCKS.get(lock_key)
            if existing is not None:
                handle, refcount = existing
                _RECOVERY_FILE_LOCKS[lock_key] = (handle, refcount + 1)
                self._recovery_lock_key = lock_key
                return

            handle = lock_path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.seek(0)
                owner = handle.read().strip()
                handle.close()
                owner_suffix = f" Holder: {owner}" if owner else ""
                raise RuntimeError(
                    f'Recovery controller for experiment_id "{self.recovery_experiment_id}" '
                    f"is already running.{owner_suffix} Wait for it to finish or stop "
                    "it before starting another recovery controller."
                ) from exc
            handle.seek(0)
            handle.truncate()
            handle.write(
                f"pid={os.getpid()} started_at={datetime.now(timezone.utc).isoformat()}\n"
            )
            handle.flush()
            _RECOVERY_FILE_LOCKS[lock_key] = (handle, 1)
            self._recovery_lock_key = lock_key

    def _release_recovery_controller_lock(self) -> None:
        lock_key = self._recovery_lock_key
        if lock_key is None or fcntl is None:
            self._recovery_lock_key = None
            return

        handle = None
        with _RECOVERY_FILE_LOCKS_LOCK:
            existing = _RECOVERY_FILE_LOCKS.get(lock_key)
            if existing is None:
                self._recovery_lock_key = None
                return
            handle, refcount = existing
            if refcount > 1:
                _RECOVERY_FILE_LOCKS[lock_key] = (handle, refcount - 1)
                self._recovery_lock_key = None
                return
            _RECOVERY_FILE_LOCKS.pop(lock_key, None)
            self._recovery_lock_key = None

        try:
            handle.seek(0)
            handle.truncate()
            handle.flush()
        except Exception:
            pass
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass

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

    def _ensure_recovery_launch_preflight(self, stage: StageSpec) -> None:
        if self.recovery_experiment_id is None:
            return
        recovery_context = self._resolve_recovery_context(stage)
        if recovery_context is None:
            return
        backend_ref, experiment_id = recovery_context
        with self._recovery_launch_preflight_lock:
            if self._recovery_launch_preflight_done:
                return
            live_rows = list_live_runs(
                db_url=backend_ref,
                experiment_id=experiment_id,
                max_age_seconds=_RECOVERY_LIVE_RUN_MAX_AGE_SECONDS,
            )
            if live_rows:
                formatted_rows = ", ".join(
                    f"{row.run_id} (phase={row.phase})" for row in live_rows
                )
                raise RuntimeError(
                    f'Stage "{stage.id}" cannot start recovery for experiment_id '
                    f'"{experiment_id}" because live runs are still active: '
                    f"{formatted_rows}. Stop or let them finish before starting "
                    "another recovery controller."
                )
            self._recovery_launch_preflight_done = True

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

    def _prime_stage_stop_target(self, stage: StageSpec) -> None:
        if self._report_controller is None and controller_metadata_path_from_env() is None:
            return
        try:
            backend_ref, experiment_id = resolve_telemetry_context(stage.config)
        except Exception as exc:  # noqa: BLE001
            _red_print(
                f'[TENYSON] Warning: failed to resolve telemetry while priming '
                f'stop control for stage "{stage.id}": {exc}'
            )
            return
        if not backend_ref or not experiment_id:
            return
        try:
            prime_stop_target(
                db_url=str(backend_ref),
                run_id=_resolve_stage_run_name(stage),
                experiment_id=str(experiment_id),
                phase=_resolve_stage_phase(stage),
                attempt_token=str(
                    stage.config.get("telemetry", {}).get("attempt_token") or ""
                ).strip()
                or None,
            )
        except Exception as exc:  # noqa: BLE001
            _red_print(
                f'[TENYSON] Warning: failed to pre-arm stop control for stage '
                f'"{stage.id}": {exc}'
            )

    @staticmethod
    def _mark_controller_running(stages: Sequence[StageSpec]) -> None:
        update_controller_runtime_state(
            state="running",
            active_stage_ids=[stage.id for stage in stages],
            last_completed_stage_ids=[],
        )

    @staticmethod
    def _mark_controller_between_stages(stage_ids: Sequence[str]) -> None:
        update_controller_runtime_state(
            state="between-stages",
            active_stage_ids=[],
            last_completed_stage_ids=list(stage_ids),
        )

    def _load_stage_result_from_telemetry(self, stage: StageSpec) -> Optional[JobResult]:
        try:
            backend_ref, experiment_id = resolve_telemetry_context(stage.config)
        except Exception:
            return None
        if not backend_ref or not experiment_id:
            return None

        attempt_token = str(
            stage.config.get("telemetry", {}).get("attempt_token") or ""
        ).strip() or None
        try:
            client = TelemetryClient(db_url=str(backend_ref))
            row = get_run_result(
                client,
                experiment_id=str(experiment_id),
                run_id=_resolve_stage_run_name(stage),
                phase=_resolve_stage_phase(stage),
                attempt_token=attempt_token,
                include_results_payload=False,
            )
        except Exception:
            return None
        if row is None:
            return None
        _results_payload, job_result_payload = row
        if not isinstance(job_result_payload, dict) or not job_result_payload:
            return None
        try:
            return JobResult.from_dict(job_result_payload)
        except TypeError:
            return None

    def _wait_for_stage_result_from_telemetry(self, stage: StageSpec) -> Optional[JobResult]:
        result = self._load_stage_result_from_telemetry(stage)
        if result is not None:
            return result

        try:
            backend_ref, experiment_id = resolve_telemetry_context(stage.config)
        except Exception:
            return None
        if not backend_ref or not experiment_id:
            return None

        deadline = time.monotonic() + _PARALLEL_RESULT_TELEMETRY_WAIT_TIMEOUT_SECONDS
        empty_live_polls = 0
        while time.monotonic() < deadline:
            try:
                live_rows = list_live_runs(
                    db_url=str(backend_ref),
                    experiment_id=str(experiment_id),
                    max_age_seconds=_RECOVERY_LIVE_RUN_MAX_AGE_SECONDS,
                )
            except Exception:
                live_rows = []

            if live_rows:
                empty_live_polls = 0
            else:
                empty_live_polls += 1

            time.sleep(
                max(0.1, float(_PARALLEL_RESULT_TELEMETRY_POLL_SECONDS))
            )
            result = self._load_stage_result_from_telemetry(stage)
            if result is not None:
                return result
            if empty_live_polls >= _PARALLEL_RESULT_TELEMETRY_EMPTY_LIVE_POLLS:
                break

        return self._load_stage_result_from_telemetry(stage)

    def _reconcile_stage_report_from_telemetry(self, stage: StageSpec) -> None:
        if self._report_controller is None:
            return
        recovered_result = self._load_stage_result_from_telemetry(stage)
        if recovered_result is not None:
            self._report_controller.record_stage_result(stage, recovered_result)
            return
        self._report_controller.stop_stage(stage.id)

    def run_branches(
        self,
        branches: Mapping[str, Callable[[ExperimentBranch], None] | Sequence[StageSpec]],
    ) -> Dict[str, Dict[str, JobResult]]:
        self.raise_if_aborted()
        if not branches:
            return {}

        def _run_branch(
            plan: Callable[[ExperimentBranch], None] | Sequence[StageSpec],
        ) -> Dict[str, JobResult]:
            branch = self.branch()
            if callable(plan):
                plan(branch)
            else:
                for stage in plan:
                    branch.run(stage)
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
        dataset: Optional[SFTDatasetTemplate] = None,
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
            artifact=None,
            run_section="training",
            sft_dataset=dataset,
            rl_dataset=None,
            rl_reward=None,
            eval_dataset=None,
            eval_metrics=None,
        )

    def rl(
        self,
        stage_id: str,
        *,
        adapter: Optional[AdapterRef] = None,
        artifact: Optional[AdapterRef] = None,
        base: str = "rl",
        run: Optional[str] = None,
        variant: Optional[str] = None,
        run_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        overrides: Optional[Mapping[str, Any]] = None,
        dataset: Optional[RLDatasetTemplate] = None,
        reward: Optional[RLRewardTemplate] = None,
    ) -> StageSpec:
        resolved_artifact = _resolve_stage_artifact(
            stage_id=stage_id,
            run_type="rl",
            adapter=adapter,
            artifact=artifact,
        )
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
            artifact=resolved_artifact,
            run_section="training",
            sft_dataset=None,
            rl_dataset=dataset,
            rl_reward=reward,
            eval_dataset=None,
            eval_metrics=None,
        )

    def eval(
        self,
        stage_id: str,
        *,
        adapter: Optional[AdapterRef] = None,
        artifact: Optional[AdapterRef] = None,
        base: str = "eval",
        run: Optional[str] = None,
        variant: Optional[str] = None,
        run_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        overrides: Optional[Mapping[str, Any]] = None,
        dataset: Optional[EvalDatasetLike] = None,
        metrics: Optional[EvalMetricsLike] = None,
    ) -> StageSpec:
        resolved_artifact = _resolve_stage_artifact(
            stage_id=stage_id,
            run_type="eval",
            adapter=adapter,
            artifact=artifact,
        )
        resolved_dataset = coerce_eval_dataset_template(dataset)
        resolved_metrics = coerce_eval_metrics_template(metrics)
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
            artifact=resolved_artifact,
            run_section="evaluation",
            sft_dataset=None,
            rl_dataset=None,
            rl_reward=None,
            eval_dataset=resolved_dataset,
            eval_metrics=resolved_metrics,
        )

    def run_stage(self, stage: StageSpec, *, cloud: Optional[Any] = None) -> JobResult:
        self.raise_if_aborted()
        recovered_result = self._resolve_recovered_stage_result(stage)
        if recovered_result is not None:
            if self._report_controller is not None:
                self._report_controller.record_stage_result(stage, recovered_result)
            return recovered_result
        self._ensure_recovery_launch_preflight(stage)
        run_id = _resolve_stage_run_name(stage)
        _ensure_stage_attempt_token(stage)
        self._prime_stage_stop_target(stage)
        self._mark_controller_running([stage])
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
                matched_result = self._load_stage_result_from_telemetry(stage)
            if matched_result is None:
                returned_run_ids = [result.run_id for result in results]
                raise RuntimeError(
                    f'Stage "{stage.id}" expected run_id "{run_id}" but no exact result was returned. '
                    f"Returned run_ids: {returned_run_ids!r}"
                )
        finally:
            self._abort_controller.unregister_run(active_run_id)
            self._mark_controller_between_stages([stage.id])
            if matched_result is None and self._report_controller is not None:
                self._reconcile_stage_report_from_telemetry(stage)

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

        self._ensure_recovery_launch_preflight(stages_to_run[0])
        for stage in stages_to_run:
            _ensure_stage_attempt_token(stage)
            self._prime_stage_stop_target(stage)
        self._mark_controller_running(stages_to_run)
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
            self._mark_controller_between_stages([stage.id for stage in stages_to_run])
            if results is None and self._report_controller is not None:
                for stage in stages_to_run:
                    self._reconcile_stage_report_from_telemetry(stage)

        try:
            for stage in stages_to_run:
                run_id = _resolve_stage_run_name(stage)
                matched_result = None
                for result in reversed(results):
                    if result.run_id == run_id:
                        matched_result = result
                        break
                if matched_result is None:
                    matched_result = self._wait_for_stage_result_from_telemetry(stage)
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
                    self._reconcile_stage_report_from_telemetry(stage)
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

    def require_artifact(self, result: JobResult, stage_id: str) -> AdapterRef:
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
                "This run cannot seed later stages safely because its exact artifact revision is unknown."
            )
        return AdapterRef(
            repo_id=str(repo),
            revision=str(revision),
            artifact_type=(
                getattr(result, "hf_artifact_type", None)
                or getattr(result, "artifact_type", None)
            ),
        )

    def require_adapter(self, result: JobResult, stage_id: str) -> AdapterRef:
        return self.require_artifact(result, stage_id)

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
        artifact: Optional[AdapterRef | _DeferredStageArtifactRef],
        run_section: str,
        sft_dataset: Optional[SFTDatasetTemplate],
        rl_dataset: Optional[RLDatasetTemplate],
        rl_reward: Optional[RLRewardTemplate],
        eval_dataset: Optional[EvalDatasetTemplate],
        eval_metrics: Optional[EvalMetricsTemplate],
    ) -> StageSpec:
        config = self.templates.clone(base)
        deferred_artifact = (
            artifact
            if _is_deferred_stage_artifact_ref(artifact)
            else None
        )
        resolved_artifact = (
            None
            if deferred_artifact is not None
            else artifact
        )
        task_overrides = None
        resolved_environment_run = str(environment_run or "").strip() or None
        explicit_stage_templates = has_explicit_stage_templates(
            sft_dataset=sft_dataset,
            rl_dataset=rl_dataset,
            rl_reward=rl_reward,
            eval_dataset=eval_dataset,
            eval_metrics=eval_metrics,
        )
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
        elif not explicit_stage_templates and hasattr(self.task, "get_run_config_overrides"):
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

        _apply_stage_artifact_config(config, resolved_artifact)
        if deferred_artifact is None:
            _require_stage_model_source(
                config=config,
                stage_id=stage_id,
                run_type=run_type,
            )
        if resolved_environment_run is not None:
            bind_environment_run(config, resolved_environment_run)
        serialized_stage_templates = serialize_stage_templates(
            sft_dataset=sft_dataset,
            rl_dataset=rl_dataset,
            rl_reward=rl_reward,
            eval_dataset=eval_dataset,
            eval_metrics=eval_metrics,
        )
        if serialized_stage_templates is not None:
            config[STAGE_TEMPLATE_CONFIG_KEY] = serialized_stage_templates

        stage_task = bind_stage_templates(
            self.task,
            sft_dataset=sft_dataset,
            rl_dataset=rl_dataset,
            rl_reward=rl_reward,
            eval_dataset=eval_dataset,
            eval_metrics=eval_metrics,
        )

        return StageSpec(
            id=stage_id,
            config=config,
            job_class=job_class,
            task=stage_task,
            run_type=run_type,
            run_name=resolved_run_name,
            environment_run=resolved_environment_run,
            variant=variant,
            deferred_artifact=deferred_artifact,
        )
