from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

Base = declarative_base()


class Rollout(Base):
    __tablename__ = "rollouts"
    id = Column(String, primary_key=True)
    run_id = Column(String, index=True)
    global_step = Column(Integer, index=True)
    prompt_text = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class EpochMetric(Base):
    __tablename__ = "epoch_metrics"
    id = Column(String, primary_key=True)
    run_id = Column(String, index=True)
    global_step = Column(Integer, index=True)
    epoch_number = Column(Integer)
    loss = Column(Float)
    kl = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


@dataclass
class TelemetryClient:
    db_url: str

    def __post_init__(self) -> None:
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)


class GRPOEpochTelemetryCallback(TrainerCallback):
    def __init__(self, run_id: str, client: TelemetryClient):
        self.run_id = run_id
        self.client = client
        self.current_step: Optional[int] = None
        self.current_epoch = 0

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.current_step != state.global_step:
            self.current_step = state.global_step
            self.current_epoch = 0

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return
        self.current_epoch += 1
