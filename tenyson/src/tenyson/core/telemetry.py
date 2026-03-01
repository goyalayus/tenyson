from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

Base = declarative_base()


class Rollout(Base):
    __tablename__ = "rollouts"
    id = Column(String, primary_key=True)
    experiment_id = Column(String, index=True)
    run_id = Column(String, index=True)
    global_step = Column(Integer, index=True)
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
    phase = Column(String, index=True)  # e.g. "rl" or "eval"
    prompt_text = Column(String)
    completion_text = Column(String)
    reward = Column(Float, nullable=True)
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
    loss = Column(Float)
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
    local_output_dir = Column(String, nullable=True)
    failure_reason = Column(String, nullable=True)
    instance_id = Column(String, nullable=True)
    spot_interruption = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


@dataclass
class TelemetryClient:
    db_url: str

    def __post_init__(self) -> None:
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)


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


def resolve_telemetry_context(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
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
        existing.total_time_seconds = float(total_time) if total_time is not None else None
        existing.metrics_json = json.dumps(metrics, ensure_ascii=False, default=str)
        existing.hf_repo_id = getattr(result, "hf_repo_id", None)
        existing.hf_revision = getattr(result, "hf_revision", None)
        existing.wandb_url = getattr(result, "wandb_url", None)
        existing.local_output_dir = getattr(result, "local_output_dir", None)
        existing.failure_reason = getattr(result, "failure_reason", None)
        existing.instance_id = getattr(result, "instance_id", None)
        existing.spot_interruption = getattr(result, "spot_interruption", None)
        existing.updated_at = now
        session.commit()
    finally:
        session.close()


class GRPOEpochTelemetryCallback(TrainerCallback):
    """
    Logs per-epoch GRPO metrics (loss, KL) into the SQL database.
    """

    def __init__(self, run_id: str, experiment_id: str, client: TelemetryClient):
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.client = client
        self.current_step: Optional[int] = None
        self.current_epoch = 0

    def on_step_begin(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if self.current_step != state.global_step:
            self.current_step = state.global_step
            self.current_epoch = 0

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        if not logs:
            return
        self.current_epoch += 1

        loss_val = float(logs.get("loss", 0.0))
        # TRL may log KL under different names; fall back if needed.
        kl_val = float(
            logs.get("kl", logs.get("kl_divergence", logs.get("approx_kl", 0.0)))
        )

        session = self.client.Session()
        try:
            metric = EpochMetric(
                id=str(uuid4()),
                experiment_id=self.experiment_id,
                run_id=self.run_id,
                global_step=int(state.global_step),
                epoch_number=int(self.current_epoch),
                loss=loss_val,
                kl=kl_val,
            )
            session.add(metric)
            session.commit()
        finally:
            session.close()


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
                loss=float(loss_val) if loss_val is not None else 0.0,
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

    def on_step_end(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
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
                control.should_training_stop = True
                control.should_save = True
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

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._maybe_write_url()
        return control

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        self._maybe_write_url()
        return control
