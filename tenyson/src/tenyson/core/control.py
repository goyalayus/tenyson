from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from .telemetry import RunControl, TelemetryClient


def request_stop(
    db_url: str,
    run_id: str,
    experiment_id: Optional[str] = None,
    create_if_missing: bool = True,
) -> None:
    """
    Set stop_requested = True for the given run_id in the RunControl table.
    """
    client = TelemetryClient(db_url=db_url)
    session = client.Session()
    try:
        query = session.query(RunControl).filter(RunControl.run_id == run_id)
        if experiment_id:
            query = query.filter(RunControl.experiment_id == experiment_id)
        control: Optional[RunControl] = (
            query.order_by(RunControl.updated_at.desc()).first()
        )
        if control is None and create_if_missing:
            control = RunControl(
                id=str(uuid4()),
                experiment_id=experiment_id,
                run_id=run_id,
                stop_requested=True,
                updated_at=datetime.now(timezone.utc),
            )
            session.add(control)
        elif control is not None:
            control.stop_requested = True
            control.updated_at = datetime.now(timezone.utc)
        session.commit()
    finally:
        session.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Request a manual stop for a running tenyson job."
    )
    parser.add_argument(
        "--db-url",
        required=True,
        help="SQLAlchemy database URL used for telemetry (e.g. sqlite:///tenyson.db).",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run identifier to stop (matches training.run_name / evaluation.run_name).",
    )
    parser.add_argument(
        "--experiment-id",
        default=os.getenv("TENYSON_EXPERIMENT_ID"),
        help=(
            "Optional experiment identifier. When provided, stop is scoped to "
            "(experiment_id, run_id)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    request_stop(
        db_url=args.db_url,
        run_id=args.run_id,
        experiment_id=args.experiment_id,
        create_if_missing=True,
    )


if __name__ == "__main__":
    main()
