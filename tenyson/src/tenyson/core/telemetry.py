from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError

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

Base = declarative_base()


def _validate_shared_db_url(db_url: str) -> None:
    """
    Validate telemetry DB URLs for distributed cloud execution.

    The same DB endpoint must be reachable by both remote workers and local control
    commands; local-only SQLite files and localhost hosts are rejected.
    """
    try:
        parsed = make_url(str(db_url))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Invalid telemetry.db_url '{db_url}'. Provide a valid hosted SQL connection string."
        ) from exc

    driver = (parsed.drivername or "").lower()
    host = (parsed.host or "").strip().lower()

    if driver.startswith("sqlite"):
        raise ValueError(
            "SQLite telemetry URLs are not supported. Use a hosted SQL database URL "
            "(for example: postgresql+psycopg://user:pass@host:5432/dbname)."
        )
    if host in {"", "localhost", "127.0.0.1"}:
        raise ValueError(
            "Telemetry DB host must be network-reachable by both local control and "
            "remote workers. Avoid localhost/127.0.0.1."
        )


class Rollout(Base):
    __tablename__ = "rollouts"
    id = Column(String, primary_key=True)
    experiment_id = Column(String, index=True)
    run_id = Column(String, index=True)
    global_step = Column(Integer, index=True)
    rollout_step = Column(Integer, index=True, nullable=True)
    rollout_batch_id = Column(String, index=True, nullable=True)
    prompt_text = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Generation(Base):
    """
    Stores individual completions and rewards for RL / Eval runs.
    """

    __tablename__ = "generations"
    id = Column(String, primary_key=True)
    experiment_id = Column(String, index=True)
    run_id = Column(String, index=True)
    global_step = Column(Integer, index=True)
    rollout_step = Column(Integer, index=True, nullable=True)
    rollout_batch_id = Column(String, index=True, nullable=True)
    phase = Column(String, index=True)  # e.g. "rl" or "eval"
    prompt_text = Column(String)
    completion_text = Column(String)
    reward = Column(Float, nullable=True)
    reward_components_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class EpochMetric(Base):
    """
    Stores per-epoch metrics for GRPO RL training.
    """

    __tablename__ = "epoch_metrics"
    id = Column(String, primary_key=True)
    experiment_id = Column(String, index=True)
    run_id = Column(String, index=True)
    global_step = Column(Integer, index=True)
    epoch_number = Column(Integer)
    loss = Column(Float)
    kl = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SFTMetric(Base):
    """
    Stores logged metrics for SFT training.
    """

    __tablename__ = "sft_metrics"
    id = Column(String, primary_key=True)
    experiment_id = Column(String, index=True)
    run_id = Column(String, index=True)
    global_step = Column(Integer, index=True)
    loss = Column(Float, nullable=True)
    eval_loss = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class RunControl(Base):
    """
    Simple control row for a run, used to request a graceful stop.
    """

    __tablename__ = "run_controls"
    id = Column(String, primary_key=True)
    experiment_id = Column(String, index=True)
    run_id = Column(String, index=True)
    stop_requested = Column(Boolean, default=False, nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class RunMetadata(Base):
    """
    Stores run metadata (e.g. WandB URL) so clients can poll for early URLs
    before the run finishes. Requires a shared DB that both worker and client can reach.
    """

    __tablename__ = "run_metadata"
    id = Column(String, primary_key=True)
    experiment_id = Column(String, index=True)
    run_id = Column(String, index=True)
    wandb_url = Column(String, nullable=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class RunFailure(Base):
    """
    Records run failures for queryable history (e.g. pipeline step failed).
    """

    __tablename__ = "run_failures"
    id = Column(String, primary_key=True)
    experiment_id = Column(String, index=True)
    run_id = Column(String, index=True)
    step_label = Column(String, index=True)
    failure_reason = Column(String)
    instance_id = Column(String, nullable=True)
    spot_interruption = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class RunSummary(Base):
    """
    Final per-job summary row for querying run outcomes by experiment.
    """

    __tablename__ = "run_summaries"
    id = Column(String, primary_key=True)
    experiment_id = Column(String, index=True)
    run_id = Column(String, index=True)
    phase = Column(String, index=True)  # "sft" | "rl" | "eval"
    status = Column(String, index=True)  # "success" | "failed"
    total_time_seconds = Column(Float, nullable=True)
    metrics_json = Column(String, nullable=True)
    hf_repo_id = Column(String, nullable=True)
    hf_revision = Column(String, nullable=True)
    wandb_url = Column(String, nullable=True)
    failure_reason = Column(String, nullable=True)
    instance_id = Column(String, nullable=True)
    spot_interruption = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class RunResult(Base):
    """
    Canonical per-job payloads keyed by (experiment_id, run_id, phase).

    - results_json: detailed payload (e.g. eval compute_metrics output)
    - job_result_json: serialized JobResult-compatible dictionary
    """

    __tablename__ = "run_results"
    id = Column(String, primary_key=True)
    experiment_id = Column(String, index=True)
    run_id = Column(String, index=True)
    phase = Column(String, index=True)
    results_json = Column(Text, nullable=False)
    job_result_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


@dataclass
class TelemetryClient:
    db_url: str

    def __post_init__(self) -> None:
        _validate_shared_db_url(self.db_url)
        self.engine = create_engine(
            self.db_url,
            pool_pre_ping=True,
            pool_recycle=300,
        )
        Base.metadata.create_all(self.engine)
        self._ensure_schema_columns()
        self.Session = sessionmaker(bind=self.engine)

    def _ensure_schema_columns(self) -> None:
        self._ensure_table_columns(
            "generations",
            {
                "reward_components_json": "TEXT",
                "rollout_step": "INTEGER",
                "rollout_batch_id": "TEXT",
            },
        )
        self._ensure_table_columns(
            "rollouts",
            {
                "rollout_step": "INTEGER",
                "rollout_batch_id": "TEXT",
            },
        )

    def _ensure_table_columns(self, table_name: str, columns: Dict[str, str]) -> None:
        inspector = inspect(self.engine)
        if table_name not in inspector.get_table_names():
            return
        existing_columns = {
            column["name"] for column in inspector.get_columns(table_name)
        }
        missing_columns = {
            name: ddl for name, ddl in columns.items() if name not in existing_columns
        }
        if not missing_columns:
            return
        for column_name, ddl in missing_columns.items():
            try:
                with self.engine.begin() as connection:
                    connection.execute(
                        text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}")
                    )
            except Exception as exc:  # noqa: BLE001
                message = str(exc).lower()
                if (
                    "duplicate column" not in message
                    and "already exists" not in message
                ):
                    raise


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


def begin_run_attempt(
    client: TelemetryClient,
    experiment_id: str,
    run_id: str,
    phase: Optional[str] = None,
) -> bool:
    """
    Start a fresh control epoch for a logical run.

    This inserts a new RunControl row with stop_requested=False so a stale stop
    from a previous attempt does not immediately terminate a restarted run. The
    latest row continues to be the source of truth for later stop requests.

    Returns True when the newest previous control row had stop_requested=True.
    """
    experiment_id = str(experiment_id)
    run_id = str(run_id)
    phase_name = str(phase or "").strip().lower()
    now = datetime.now(timezone.utc)
    reset_session = client.Session()
    try:
        reset_session.query(RunFailure).filter(
            RunFailure.experiment_id == experiment_id
        ).filter(RunFailure.run_id == run_id).delete(synchronize_session=False)
        reset_session.query(RunMetadata).filter(
            RunMetadata.experiment_id == experiment_id
        ).filter(RunMetadata.run_id == run_id).delete(synchronize_session=False)
        if phase_name:
            reset_session.query(RunSummary).filter(
                RunSummary.experiment_id == experiment_id
            ).filter(RunSummary.run_id == run_id).filter(
                RunSummary.phase == phase_name
            ).delete(synchronize_session=False)
            reset_session.query(RunResult).filter(
                RunResult.experiment_id == experiment_id
            ).filter(RunResult.run_id == run_id).filter(
                RunResult.phase == phase_name
            ).delete(synchronize_session=False)
            if phase_name == "sft":
                reset_session.query(SFTMetric).filter(
                    SFTMetric.experiment_id == experiment_id
                ).filter(SFTMetric.run_id == run_id).delete(synchronize_session=False)
            elif phase_name == "rl":
                reset_session.query(EpochMetric).filter(
                    EpochMetric.experiment_id == experiment_id
                ).filter(EpochMetric.run_id == run_id).delete(synchronize_session=False)
                reset_session.query(Rollout).filter(
                    Rollout.experiment_id == experiment_id
                ).filter(Rollout.run_id == run_id).delete(synchronize_session=False)
                reset_session.query(Generation).filter(
                    Generation.experiment_id == experiment_id
                ).filter(Generation.run_id == run_id).filter(
                    Generation.phase == phase_name
                ).delete(synchronize_session=False)
            elif phase_name == "eval":
                reset_session.query(Generation).filter(
                    Generation.experiment_id == experiment_id
                ).filter(Generation.run_id == run_id).filter(
                    Generation.phase == phase_name
                ).delete(synchronize_session=False)
        reset_session.commit()
    finally:
        reset_session.close()
    session = client.Session()
    try:
        latest = (
            session.query(RunControl)
            .filter(RunControl.run_id == run_id)
            .filter(RunControl.experiment_id == experiment_id)
            .order_by(RunControl.updated_at.desc())
            .first()
        )
        cleared_stale_stop = bool(latest and latest.stop_requested)
        session.add(
            RunControl(
                id=str(uuid4()),
                experiment_id=experiment_id,
                run_id=run_id,
                stop_requested=False,
                updated_at=now,
            )
        )
        session.commit()
        return cleared_stale_stop
    finally:
        session.close()


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
    Return (db_url, experiment_id). If db_url is set, experiment_id is required.
    """
    telemetry_cfg = config.get("telemetry", {}) if isinstance(config, dict) else {}
    db_url = telemetry_cfg.get("db_url")
    if not db_url:
        return None, None
    experiment_id = resolve_experiment_id(config)
    if not experiment_id:
        raise ValueError(
            "Telemetry enabled but experiment_id is missing. "
            "Set telemetry.experiment_id or TENYSON_EXPERIMENT_ID."
        )
    return db_url, experiment_id


def resolve_required_telemetry_context(config: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return required telemetry context. Raises when db_url or experiment_id is missing.
    """
    db_url, experiment_id = resolve_telemetry_context(config)
    if not db_url:
        raise ValueError(
            "Missing telemetry.db_url. Tenyson requires a hosted SQL telemetry DB "
            "for all runs."
        )
    if not experiment_id:
        raise ValueError(
            "Missing telemetry.experiment_id. Set telemetry.experiment_id or "
            "TENYSON_EXPERIMENT_ID."
        )
    return str(db_url), str(experiment_id)


def record_run_summary(
    client: TelemetryClient,
    experiment_id: str,
    phase: str,
    result: Any,
) -> None:
    """
    Upsert final run summary keyed by (experiment_id, run_id, phase).
    """
    run_id = str(getattr(result, "run_id", "unknown"))
    metrics = getattr(result, "metrics", {}) or {}
    now = datetime.now(timezone.utc)
    session = client.Session()
    try:
        existing = (
            session.query(RunSummary)
            .filter(RunSummary.experiment_id == experiment_id)
            .filter(RunSummary.run_id == run_id)
            .filter(RunSummary.phase == phase)
            .one_or_none()
        )
        if existing is None:
            existing = RunSummary(
                id=str(uuid4()),
                experiment_id=experiment_id,
                run_id=run_id,
                phase=phase,
                created_at=now,
            )
            session.add(existing)

        existing.status = str(getattr(result, "status", "unknown"))
        total_time = getattr(result, "total_time_seconds", None)
        existing.total_time_seconds = (
            float(total_time) if total_time is not None else None
        )
        existing.metrics_json = json.dumps(metrics, ensure_ascii=False, default=str)
        existing.hf_repo_id = getattr(result, "hf_repo_id", None)
        existing.hf_revision = getattr(result, "hf_revision", None)
        existing.wandb_url = getattr(result, "wandb_url", None)
        existing.failure_reason = getattr(result, "failure_reason", None)
        existing.instance_id = getattr(result, "instance_id", None)
        existing.spot_interruption = getattr(result, "spot_interruption", None)
        existing.updated_at = now
        session.commit()
    finally:
        session.close()


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
    now = datetime.now(timezone.utc)
    session = client.Session()
    try:
        existing = (
            session.query(RunResult)
            .filter(RunResult.experiment_id == experiment_id)
            .filter(RunResult.run_id == str(run_id))
            .filter(RunResult.phase == phase)
            .order_by(RunResult.updated_at.desc())
            .first()
        )
        if existing is None:
            existing = RunResult(
                id=str(uuid4()),
                experiment_id=experiment_id,
                run_id=str(run_id),
                phase=phase,
                created_at=now,
            )
            session.add(existing)

        existing.results_json = json.dumps(
            _as_payload_dict(results_payload), ensure_ascii=False, default=str
        )
        existing.job_result_json = json.dumps(
            _as_payload_dict(job_result_payload), ensure_ascii=False, default=str
        )
        existing.updated_at = now
        session.commit()
    finally:
        session.close()


def get_run_result(
    client: TelemetryClient,
    experiment_id: str,
    run_id: str,
    phase: str,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Read canonical per-run payloads from run_results.
    """
    session = client.Session()
    try:
        row = (
            session.query(RunResult)
            .filter(RunResult.experiment_id == experiment_id)
            .filter(RunResult.run_id == str(run_id))
            .filter(RunResult.phase == phase)
            .order_by(RunResult.updated_at.desc())
            .first()
        )
        if row is None:
            return None

        results_payload = json.loads(row.results_json) if row.results_json else {}
        job_result_payload = (
            json.loads(row.job_result_json) if row.job_result_json else {}
        )
        if not isinstance(results_payload, dict):
            results_payload = {"value": results_payload}
        if not isinstance(job_result_payload, dict):
            job_result_payload = {"value": job_result_payload}
        return results_payload, job_result_payload
    finally:
        session.close()


def wait_for_run_result(
    client: TelemetryClient,
    experiment_id: str,
    run_id: str,
    phase: str,
    timeout_seconds: int = 120,
    poll_interval_seconds: float = 2.0,
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
    Logs loss and eval_loss for SFT runs.
    """

    def __init__(self, run_id: str, experiment_id: str, client: TelemetryClient):
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.client = client

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        if not logs:
            return

        loss_val = logs.get("loss")
        eval_loss_val = logs.get("eval_loss")
        if loss_val is None and eval_loss_val is None:
            return

        session = self.client.Session()
        try:
            metric = SFTMetric(
                id=str(uuid4()),
                experiment_id=self.experiment_id,
                run_id=self.run_id,
                global_step=int(state.global_step),
                loss=float(loss_val) if loss_val is not None else None,
                eval_loss=float(eval_loss_val) if eval_loss_val is not None else None,
            )
            session.add(metric)
            session.commit()
        finally:
            session.close()


class ManualStopTelemetryCallback(TrainerCallback):
    """
    Polls the RunControl table for a given run_id and requests a graceful stop
    when stop_requested is set.
    """

    def __init__(
        self,
        run_id: str,
        experiment_id: str,
        client: TelemetryClient,
        check_every_n_steps: int = 1,
    ):
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.client = client
        self.check_every_n_steps = max(1, int(check_every_n_steps))
        self.stop_requested = False
        self.stop_step: Optional[int] = None

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Optionally throttle checks; default is every step for responsiveness.
        if state.global_step % self.check_every_n_steps != 0:
            return control

        session = self.client.Session()
        try:
            query = session.query(RunControl).filter(RunControl.run_id == self.run_id)
            if self.experiment_id:
                query = query.filter(RunControl.experiment_id == self.experiment_id)
            control_row = query.order_by(RunControl.updated_at.desc()).first()
            if control_row and control_row.stop_requested:
                print(
                    f"[ManualStopTelemetryCallback] Stop requested for run_id="
                    f"{self.run_id} at step {state.global_step}",
                    flush=True,
                )
                self.stop_requested = True
                self.stop_step = int(state.global_step)
                control.should_training_stop = True
                control.should_save = True
        except SQLAlchemyError as exc:
            session.rollback()
            print(
                "[ManualStopTelemetryCallback] Warning: stop polling failed; "
                f"continuing training. {exc}",
                flush=True,
            )
        finally:
            session.close()

        return control


class WandBUrlTelemetryCallback(TrainerCallback):
    """
    When WandB is enabled and telemetry uses a shared DB, upserts the run's
    WandB URL to RunMetadata as soon as WandB is inited so clients can poll
    for the link before the run finishes.
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
        session = self.client.Session()
        try:
            existing = (
                session.query(RunMetadata)
                .filter(RunMetadata.run_id == self.run_id)
                .filter(RunMetadata.experiment_id == self.experiment_id)
                .order_by(RunMetadata.updated_at.desc())
                .first()
            )
            now = datetime.now(timezone.utc)
            if existing:
                existing.wandb_url = url
                existing.updated_at = now
            else:
                session.add(
                    RunMetadata(
                        id=str(uuid4()),
                        experiment_id=self.experiment_id,
                        run_id=self.run_id,
                        wandb_url=url,
                        updated_at=now,
                    )
                )
            session.commit()
            self._written = True
        finally:
            session.close()

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
