from __future__ import annotations

import argparse
from typing import Optional

from .telemetry import RunControl, TelemetryClient


def request_stop(db_url: str, run_id: str, create_if_missing: bool = True) -> None:
    """
    Set stop_requested = True for the given run_id in the RunControl table.
    """
    client = TelemetryClient(db_url=db_url)
    session = client.Session()
    try:
        control: Optional[RunControl] = (
            session.query(RunControl).filter(RunControl.run_id == run_id).one_or_none()
        )
        if control is None and create_if_missing:
            control = RunControl(run_id=run_id, stop_requested=True)
            session.add(control)
        elif control is not None:
            control.stop_requested = True
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    request_stop(db_url=args.db_url, run_id=args.run_id, create_if_missing=True)


if __name__ == "__main__":
    main()

