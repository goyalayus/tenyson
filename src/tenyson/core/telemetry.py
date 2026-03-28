from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from tenyson.core import wandb_store

try:
    from transformers.trainer_callback import (
        TrainerCallback,
        TrainerControl,
        TrainerState,
    )
except ImportError:  # pragma: no cover - local controller path fallback
    # Local controller processes may import telemetry without a full training stack.
    # Remote GPU workers still use the real Transformers callback classes.
    class TrainerCallback:  # type: ignore[no-redef]
        pass

    class TrainerControl:  # type: ignore[no-redef]
        should_training_stop: bool = False
        should_save: bool = False

    class TrainerState:  # type: ignore[no-redef]
        global_step: int = 0

def _validate_shared_db_url(db_url: str) -> None:
    """
    Validate telemetry backend refs for distributed cloud execution.

    Telemetry is W&B-only. Backend refs must use:
      wandb://<entity>/<project>
    """
    backend_ref = str(db_url or "").strip()
    if not backend_ref:
        raise ValueError(
            "Missing telemetry backend ref. Configure W&B via telemetry.entity / "
            "telemetry.project or set telemetry.db_url to wandb://<entity>/<project>."
        )
    if not wandb_store.is_wandb_backend_ref(backend_ref):
        raise ValueError(
            "Telemetry only supports W&B refs. Use "
            "wandb://<entity>/<project>."
        )
    try:
        wandb_store.parse_backend_ref(backend_ref)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Invalid telemetry backend ref '{backend_ref}'. Use "
            "wandb://<entity>/<project>."
        ) from exc

@dataclass
class TelemetryClient:
    db_url: str

    def __post_init__(self) -> None:
        _validate_shared_db_url(self.db_url)
        self.backend = "wandb"
        self.wandb_target = wandb_store.parse_backend_ref(self.db_url)


def ensure_wandb_telemetry_run(
    client: TelemetryClient,
    *,
    experiment_id: str,
    phase: str,
    run_name: str,
    config: Optional[Dict[str, Any]] = None,
    attempt_token: Optional[str] = None,
) -> Any:
    return wandb_store.ensure_run(
        client.db_url,
        experiment_id=experiment_id,
        phase=phase,
        run_name=run_name,
        config=config,
        attempt_token=attempt_token,
    )


def telemetry_project_url(client: TelemetryClient) -> Optional[str]:
    if client.wandb_target is None:
        return None
    return client.wandb_target.project_url


@dataclass(frozen=True)
class RLRolloutWindow:
    rollout_step: int
    rollout_batch_id: str
    generation_global_step: int


class RLRolloutTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rollout_counter = 0
        self._current_rollout: Optional[RLRolloutWindow] = None

    def start_rollout(self, generation_global_step: int) -> RLRolloutWindow:
        with self._lock:
            self._rollout_counter += 1
            rollout = RLRolloutWindow(
                rollout_step=self._rollout_counter,
                rollout_batch_id=str(uuid4()),
                generation_global_step=int(generation_global_step),
            )
            self._current_rollout = rollout
            return rollout

    def current_rollout(self) -> Optional[RLRolloutWindow]:
        with self._lock:
            return self._current_rollout


@dataclass(frozen=True)
class LiveRunInfo:
    run_id: str
    phase: str
    provider: Optional[str]
    status: str
    is_active: bool
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    attempt_token: Optional[str] = None


_WANDB_RECORD_LOCK = threading.RLock()


def _normalize_timestamp(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _heartbeat_provider(provider: Optional[str] = None) -> Optional[str]:
    resolved = str(provider or os.getenv("TENYSON_GPU_PROVIDER") or "").strip()
    return resolved or None


def _upsert_run_heartbeat(
    client: TelemetryClient,
    experiment_id: str,
    run_id: str,
    phase: str,
    *,
    status: str,
    is_active: bool,
    provider: Optional[str] = None,
    reset_created_at: bool = False,
) -> None:
    now = datetime.now(timezone.utc)
    _unused = reset_created_at
    run = wandb_store.active_run()
    if run is None:
        return
    wandb_store.update_run_summary(
        run,
        {
            wandb_store.SUMMARY_EXPERIMENT_ID: str(experiment_id),
            wandb_store.SUMMARY_PHASE: str(phase),
            wandb_store.SUMMARY_RUN_NAME: str(run_id),
            wandb_store.SUMMARY_STATUS: str(status),
            wandb_store.SUMMARY_IS_ACTIVE: bool(is_active),
            wandb_store.SUMMARY_PROVIDER: _heartbeat_provider(provider),
            wandb_store.SUMMARY_HEARTBEAT_AT: now.isoformat(),
        },
    )


def start_run_heartbeat(
    client: TelemetryClient,
    experiment_id: str,
    run_id: str,
    phase: str,
    *,
    provider: Optional[str] = None,
) -> None:
    _upsert_run_heartbeat(
        client,
        experiment_id,
        run_id,
        phase,
        status="running",
        is_active=True,
        provider=provider,
        reset_created_at=True,
    )


def beat_run_heartbeat(
    client: TelemetryClient,
    experiment_id: str,
    run_id: str,
    phase: str,
    *,
    provider: Optional[str] = None,
) -> None:
    _upsert_run_heartbeat(
        client,
        experiment_id,
        run_id,
        phase,
        status="running",
        is_active=True,
        provider=provider,
        reset_created_at=False,
    )


def finish_run_heartbeat(
    client: TelemetryClient,
    experiment_id: str,
    run_id: str,
    phase: str,
    *,
    status: str,
    provider: Optional[str] = None,
) -> None:
    _upsert_run_heartbeat(
        client,
        experiment_id,
        run_id,
        phase,
        status=status,
        is_active=False,
        provider=provider,
        reset_created_at=False,
    )


def list_live_run_heartbeats(
    client: TelemetryClient,
    experiment_id: str,
    *,
    max_age_seconds: int = 90,
) -> List[LiveRunInfo]:
    rows = wandb_store.list_live_runs(
        client.db_url,
        experiment_id=experiment_id,
        max_age_seconds=max_age_seconds,
    )
    return [
        LiveRunInfo(
            run_id=str(row.get("run_id") or ""),
            phase=str(row.get("phase") or ""),
            provider=row.get("provider"),
            status=str(row.get("status") or "running"),
            is_active=bool(row.get("is_active")),
            created_at=_normalize_timestamp(row.get("created_at")),
            updated_at=_normalize_timestamp(row.get("updated_at")),
            attempt_token=str(row.get("attempt_token") or "").strip() or None,
        )
        for row in rows
    ]


def begin_run_attempt(
    client: TelemetryClient,
    experiment_id: str,
    run_id: str,
    phase: Optional[str] = None,
    attempt_token: Optional[str] = None,
) -> bool:
    """
    Start a fresh control epoch for a logical run.

    On the W&B-backed path, this clears any stale stop request and resets the
    canonical result fields for the next attempt.

    Returns True when the newest previous attempt had stop_requested=True.
    """
    experiment_id = str(experiment_id)
    run_id = str(run_id)
    phase_name = str(phase or "").strip().lower()
    now = datetime.now(timezone.utc)
    run = wandb_store.ensure_run(
        client.db_url,
        experiment_id=experiment_id,
        phase=phase_name or "run",
        run_name=run_id,
        attempt_token=attempt_token,
    )
    cleared_stale_stop = False
    existing_result = None
    preserved_stop_requested = False
    preserved_stop_requested_at = None
    try:
        preserved_stop_requested, preserved_stop_requested_at = (
            wandb_store.fetch_stop_request_state(
                client.db_url,
                experiment_id=experiment_id,
                phase=phase_name or "run",
                run_name=run_id,
                attempt_token=attempt_token,
            )
        )
    except Exception:  # noqa: BLE001
        preserved_stop_requested = False
        preserved_stop_requested_at = None
    try:
        _results_payload, existing_result = get_run_result(
            client,
            experiment_id=experiment_id,
            run_id=run_id,
            phase=phase_name or "run",
            attempt_token=attempt_token,
            include_results_payload=False,
        )
        del _results_payload
    except Exception:  # noqa: BLE001
        existing_result = None

    if existing_result is not None:
        prior_status = str(existing_result.get("status") or "").strip().lower()
        if prior_status in {"success", "failed", "stopped", "partial"}:
            cleared_stale_stop = bool(preserved_stop_requested)
            wandb_store.set_stop_requested(
                client.db_url,
                experiment_id=experiment_id,
                phase=phase_name or "run",
                run_name=run_id,
                requested=False,
                when_iso=None,
                create_if_missing=False,
                attempt_token=attempt_token,
            )
            preserved_stop_requested = False
            preserved_stop_requested_at = None
    wandb_store.update_run_summary(
        run,
        {
            wandb_store.SUMMARY_ATTEMPT_TOKEN: str(attempt_token or ""),
            wandb_store.SUMMARY_STATUS: "running",
            wandb_store.SUMMARY_IS_ACTIVE: True,
            wandb_store.SUMMARY_JOB_RESULT_JSON: None,
            wandb_store.SUMMARY_RESULTS_JSON: None,
            wandb_store.SUMMARY_RESULT_ARTIFACT: None,
            wandb_store.SUMMARY_FAILURE_REASON: None,
            wandb_store.SUMMARY_WANDB_URL: getattr(run, "url", None),
            wandb_store.SUMMARY_STOP_REQUESTED: bool(preserved_stop_requested),
            wandb_store.SUMMARY_STOP_REQUESTED_AT: (
                preserved_stop_requested_at if preserved_stop_requested else None
            ),
            wandb_store.SUMMARY_HEARTBEAT_AT: now.isoformat(),
        },
    )
    return bool(cleared_stale_stop)


def resolve_experiment_id(config: Dict[str, Any]) -> Optional[str]:
    """
    Resolve experiment_id from config first, then TENYSON_EXPERIMENT_ID env var.
    """
    telemetry_cfg = config.get("telemetry", {}) if isinstance(config, dict) else {}
    from_config = str(telemetry_cfg.get("experiment_id", "")).strip()
    if from_config:
        return from_config
    from_env = str(os.getenv("TENYSON_EXPERIMENT_ID", "")).strip()
    return from_env or None


def resolve_telemetry_context(
    config: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (backend_ref, experiment_id) for W&B telemetry.

    Supported backend ref format:
    - wandb://<entity>/<project>
    """
    telemetry_cfg = config.get("telemetry", {}) if isinstance(config, dict) else {}
    wandb_cfg = telemetry_cfg.get("wandb", {}) if isinstance(telemetry_cfg, dict) else {}
    backend = str(telemetry_cfg.get("backend") or "").strip().lower()
    if backend and backend != "wandb":
        raise ValueError(
            "Telemetry is W&B-only. Remove telemetry.backend or leave it as 'wandb'."
        )

    db_url_ref = str(telemetry_cfg.get("db_url") or "").strip()
    if db_url_ref and not wandb_store.is_wandb_backend_ref(db_url_ref):
        raise ValueError(
            "SQL telemetry is no longer supported. Configure W&B via "
            "telemetry.entity/project (or set telemetry.db_url to "
            "wandb://<entity>/<project>)."
        )
    if db_url_ref:
        parsed_target = wandb_store.parse_backend_ref(db_url_ref)
        parsed_entity = parsed_target.entity
        parsed_project = parsed_target.project
    else:
        parsed_entity = ""
        parsed_project = ""

    entity = str(
        telemetry_cfg.get("entity")
        or wandb_cfg.get("entity")
        or parsed_entity
        or os.getenv("TENYSON_WANDB_ENTITY")
        or os.getenv("WANDB_ENTITY")
        or ""
    ).strip()
    project = str(
        telemetry_cfg.get("project")
        or wandb_cfg.get("project")
        or parsed_project
        or os.getenv("TENYSON_WANDB_PROJECT")
        or os.getenv("WANDB_PROJECT")
        or ""
    ).strip()

    telemetry_requested = backend == "wandb" or bool(entity or project or db_url_ref)
    if not telemetry_requested:
        return None, None

    if not entity or not project:
        raise ValueError(
            "W&B telemetry requires telemetry.entity and telemetry.project "
            "(or telemetry.wandb.entity / telemetry.wandb.project)."
        )
    backend_ref = f"wandb://{entity}/{project}"
    experiment_id = resolve_experiment_id(config)
    if not experiment_id:
        raise ValueError(
            "Telemetry enabled but experiment_id is missing. "
            "Set telemetry.experiment_id or TENYSON_EXPERIMENT_ID."
        )
    return str(backend_ref), experiment_id


def resolve_required_telemetry_context(config: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return required telemetry context. Raises when W&B config or experiment_id is missing.
    """
    backend_ref, experiment_id = resolve_telemetry_context(config)
    if not backend_ref:
        raise ValueError(
            "Missing telemetry configuration. Set telemetry.entity/project or "
            "TENYSON_WANDB_ENTITY / TENYSON_WANDB_PROJECT."
        )
    if not experiment_id:
        raise ValueError(
            "Missing telemetry.experiment_id. Set telemetry.experiment_id or "
            "TENYSON_EXPERIMENT_ID."
        )
    return str(backend_ref), str(experiment_id)


def record_run_summary(
    client: TelemetryClient,
    experiment_id: str,
    phase: str,
    result: Any,
) -> None:
    """
    Upsert final run summary keyed by (experiment_id, run_id, phase).
    """
    with _WANDB_RECORD_LOCK:
        run_id = str(getattr(result, "run_id", "unknown"))
        metrics = getattr(result, "metrics", {}) or {}
        attempt_token = getattr(result, "attempt_token", None)
        run = wandb_store.resolve_run(
            client.db_url,
            experiment_id=experiment_id,
            phase=phase,
            run_name=run_id,
            create_if_missing=True,
            attempt_token=attempt_token,
        )
        wandb_store.update_run_summary(
            run,
            {
                wandb_store.SUMMARY_EXPERIMENT_ID: experiment_id,
                wandb_store.SUMMARY_PHASE: phase,
                wandb_store.SUMMARY_RUN_NAME: run_id,
                wandb_store.SUMMARY_STATUS: str(getattr(result, "status", "unknown")),
                wandb_store.SUMMARY_TOTAL_TIME: getattr(
                    result, "total_time_seconds", None
                ),
                wandb_store.SUMMARY_METRICS_JSON: json.dumps(
                    metrics,
                    ensure_ascii=False,
                    default=str,
                ),
                wandb_store.SUMMARY_HF_REPO_ID: getattr(result, "hf_repo_id", None),
                wandb_store.SUMMARY_HF_REVISION: getattr(
                    result, "hf_revision", None
                ),
                wandb_store.SUMMARY_WANDB_URL: getattr(run, "url", None)
                or getattr(result, "wandb_url", None),
                wandb_store.SUMMARY_FAILURE_REASON: getattr(
                    result, "failure_reason", None
                ),
                wandb_store.SUMMARY_INSTANCE_ID: getattr(result, "instance_id", None),
                wandb_store.SUMMARY_SPOT_INTERRUPTION: getattr(
                    result, "spot_interruption", None
                ),
                wandb_store.SUMMARY_IS_ACTIVE: False,
                wandb_store.SUMMARY_ATTEMPT_TOKEN: attempt_token,
                wandb_store.SUMMARY_HEARTBEAT_AT: datetime.now(timezone.utc).isoformat(),
            },
        )


def _as_payload_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(getattr(value, "__dict__"))
    return {"value": value}


def record_run_result(
    client: TelemetryClient,
    experiment_id: str,
    run_id: str,
    phase: str,
    results_payload: Any,
    job_result_payload: Any,
) -> None:
    """
    Upsert canonical per-run payloads in run_results.
    """
    with _WANDB_RECORD_LOCK:
        results_payload_dict = _as_payload_dict(results_payload)
        job_result_payload_dict = _as_payload_dict(job_result_payload)
        attempt_token = job_result_payload_dict.get("attempt_token")
        run = wandb_store.ensure_run(
            client.db_url,
            experiment_id=experiment_id,
            phase=phase,
            run_name=str(run_id),
            attempt_token=attempt_token,
        )
        serialized_results = json.dumps(
            results_payload_dict,
            ensure_ascii=False,
            default=str,
        )
        artifact_name = None
        if len(serialized_results) > 50_000:
            artifact_name = wandb_store.log_result_payload(
                run,
                experiment_id=experiment_id,
                phase=phase,
                run_name=str(run_id),
                results_payload=results_payload_dict,
                job_result_payload=job_result_payload_dict,
                attempt_token=attempt_token,
            )
            serialized_results = ""
        wandb_store.update_run_summary(
            run,
            {
                wandb_store.SUMMARY_JOB_RESULT_JSON: json.dumps(
                    job_result_payload_dict,
                    ensure_ascii=False,
                    default=str,
                ),
                wandb_store.SUMMARY_RESULTS_JSON: serialized_results or None,
                wandb_store.SUMMARY_RESULT_ARTIFACT: artifact_name,
                wandb_store.SUMMARY_IS_ACTIVE: False,
                wandb_store.SUMMARY_WANDB_URL: getattr(run, "url", None),
                wandb_store.SUMMARY_ATTEMPT_TOKEN: job_result_payload_dict.get(
                    "attempt_token"
                ),
            },
        )


def get_run_result(
    client: TelemetryClient,
    experiment_id: str,
    run_id: str,
    phase: str,
    *,
    attempt_token: Optional[str] = None,
    min_attempt_updated_at: Optional[datetime] = None,
    include_results_payload: bool = True,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Read canonical per-run payloads from run_results.
    """
    return wandb_store.fetch_run_result(
        client.db_url,
        experiment_id=experiment_id,
        phase=phase,
        run_name=run_id,
        attempt_token=attempt_token,
        min_attempt_updated_at=min_attempt_updated_at,
        include_results_payload=include_results_payload,
    )


def get_run_metadata_wandb_url(
    client: TelemetryClient,
    experiment_id: str,
    run_id: str,
    *,
    min_attempt_updated_at: Optional[datetime] = None,
    phase: Optional[str] = None,
    attempt_token: Optional[str] = None,
) -> Optional[str]:
    """
    Read the latest WandB URL for a run from run_metadata, if available.
    """
    return wandb_store.fetch_run_url(
        client.db_url,
        experiment_id=experiment_id,
        phase=str(phase or ""),
        run_name=run_id,
        attempt_token=attempt_token,
        min_attempt_updated_at=min_attempt_updated_at,
    )


def run_stop_requested(
    client: TelemetryClient,
    *,
    experiment_id: str,
    run_id: str,
    phase: str,
    attempt_token: Optional[str] = None,
) -> bool:
    return wandb_store.fetch_stop_requested(
        client.db_url,
        experiment_id=experiment_id,
        phase=phase,
        run_name=run_id,
        attempt_token=attempt_token,
    )


def wait_for_run_result(
    client: TelemetryClient,
    experiment_id: str,
    run_id: str,
    phase: str,
    timeout_seconds: int = 120,
    poll_interval_seconds: float = 2.0,
    attempt_token: Optional[str] = None,
    min_attempt_updated_at: Optional[datetime] = None,
    include_results_payload: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Poll run_results until the canonical payload is available or timeout.
    """
    deadline = time.time() + max(1, int(timeout_seconds))
    while True:
        row = get_run_result(
            client=client,
            experiment_id=experiment_id,
            run_id=run_id,
            phase=phase,
            attempt_token=attempt_token,
            min_attempt_updated_at=min_attempt_updated_at,
            include_results_payload=include_results_payload,
        )
        if row is not None:
            return row
        if time.time() >= deadline:
            break
        time.sleep(max(0.1, float(poll_interval_seconds)))
    raise TimeoutError(
        f"Timed out waiting for run_results row "
        f"(experiment_id={experiment_id}, run_id={run_id}, phase={phase})."
    )


class GRPOEpochTelemetryCallback(TrainerCallback):
    """
    Placeholder callback for RL update metrics.

    RL loss/KL telemetry is intentionally disabled for now because GRPO reuses
    rollout batches on an internal clock that the callback surface does not expose
    precisely enough to label update-side metrics without overclaiming accuracy.
    """

    def __init__(
        self,
        run_id: str,
        experiment_id: str,
        client: TelemetryClient,
        rollout_tracker: Optional[RLRolloutTracker] = None,
    ):
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.client = client
        self.rollout_tracker = rollout_tracker
        self.current_step: Optional[int] = None
        self.current_epoch = 0

    def on_step_begin(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        return None

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        return None


class SFTTelemetryCallback(TrainerCallback):
    """
    Retained as a lightweight no-op for backwards compatibility.

    SFT metrics now live in W&B run history directly.
    """

    def __init__(self, run_id: str, experiment_id: str, client: TelemetryClient):
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.client = client

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        _unused = (args, state, control, logs, kwargs)
        return None


class RunHeartbeatTelemetryCallback(TrainerCallback):
    """
    Periodically marks a run as alive so local control commands can discover
    currently-running jobs without guessing from metrics tables.
    """

    def __init__(
        self,
        run_id: str,
        experiment_id: str,
        phase: str,
        client: TelemetryClient,
        *,
        provider: Optional[str] = None,
        min_interval_seconds: float = 10.0,
    ):
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.phase = phase
        self.client = client
        self.provider = provider
        self.min_interval_seconds = max(0.1, float(min_interval_seconds))
        self._last_beat_at = 0.0
        self._warned = False

    def _maybe_beat(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_beat_at) < self.min_interval_seconds:
            return
        try:
            beat_run_heartbeat(
                client=self.client,
                experiment_id=self.experiment_id,
                run_id=self.run_id,
                phase=self.phase,
                provider=self.provider,
            )
            self._last_beat_at = now
            self._warned = False
        except Exception as exc:  # noqa: BLE001
            if not self._warned:
                print(
                    "[RunHeartbeatTelemetryCallback] Warning: heartbeat update "
                    f"failed; continuing training. {exc}",
                    flush=True,
                )
                self._warned = True

    def on_train_begin(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        self._maybe_beat(force=True)
        return control

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._maybe_beat(force=False)
        return control


class ManualStopTelemetryCallback(TrainerCallback):
    """
    Polls W&B-backed control state for a given run_id and requests a graceful
    stop when stop_requested is set.
    """

    def __init__(
        self,
        run_id: str,
        experiment_id: str,
        phase: str,
        client: TelemetryClient,
        attempt_token: Optional[str] = None,
        check_every_n_steps: int = 1,
    ):
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.phase = str(phase)
        self.client = client
        self.attempt_token = str(attempt_token or "").strip() or None
        self.check_every_n_steps = max(1, int(check_every_n_steps))
        self.stop_requested = False
        self.stop_step: Optional[int] = None

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Optionally throttle checks; default is every step for responsiveness.
        if state.global_step % self.check_every_n_steps != 0:
            return control
        try:
            if run_stop_requested(
                self.client,
                experiment_id=self.experiment_id,
                run_id=self.run_id,
                phase=self.phase,
                attempt_token=self.attempt_token,
            ):
                print(
                    f"[ManualStopTelemetryCallback] Stop requested for run_id="
                    f"{self.run_id} at step {state.global_step}",
                    flush=True,
                )
                self.stop_requested = True
                self.stop_step = int(state.global_step)
                control.should_training_stop = True
                control.should_save = True
        except Exception as exc:  # noqa: BLE001
            print(
                "[ManualStopTelemetryCallback] Warning: stop polling failed; "
                f"continuing training. {exc}",
                flush=True,
            )

        return control


class WandBUrlTelemetryCallback(TrainerCallback):
    """
    Lightweight no-op shim.

    W&B URLs now come straight from the W&B-backed telemetry store, so there is
    no separate metadata sink to update here.
    """

    def __init__(self, run_id: str, experiment_id: str, client: TelemetryClient):
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.client = client
        self._written = False

    def _maybe_write_url(self) -> None:
        if self._written:
            return
        try:
            import wandb  # type: ignore[import-not-found]

            run = getattr(wandb, "run", None)
            if run is None:
                return
            url = getattr(run, "url", None)
            if not url:
                return
        except Exception:  # noqa: BLE001
            return
        self._written = True

    def on_train_begin(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        self._maybe_write_url()
        return control

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        self._maybe_write_url()
        return control
