import argparse
import os
import sys
from datetime import datetime, timezone
from uuid import uuid4

from tenyson.core.telemetry import RunControl, TelemetryClient


def _cmd_stop(args: argparse.Namespace) -> None:
    """Mark a run as stop_requested in the telemetry database."""
    db_url = args.db_url or os.getenv("TENYSON_DB_URL")
    if not db_url:
        print(
            "Error: --db-url is required (or set TENYSON_DB_URL).",
            file=sys.stderr,
        )
        sys.exit(1)

    client = TelemetryClient(db_url=db_url)
    session = client.Session()
    try:
        query = session.query(RunControl).filter(RunControl.run_id == args.run_id)
        if args.experiment_id:
            query = query.filter(RunControl.experiment_id == args.experiment_id)
        control_row = query.order_by(RunControl.updated_at.desc()).first()
        now = datetime.now(timezone.utc)
        if control_row is None:
            control_row = RunControl(
                id=str(uuid4()),
                experiment_id=args.experiment_id,
                run_id=args.run_id,
                stop_requested=True,
                updated_at=now,
            )
            session.add(control_row)
        else:
            control_row.stop_requested = True
            control_row.updated_at = now

        session.commit()
    finally:
        session.close()

    print(
        (
            f"[tenyson.ctl] Marked run_id={args.run_id}"
            f"{f' experiment_id={args.experiment_id}' if args.experiment_id else ''}"
            f" as stop_requested in {db_url}"
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="tenyson ctl", description="Tenyson control")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stop_parser = subparsers.add_parser(
        "stop",
        help="Request a graceful stop for a running SFT job.",
    )
    stop_parser.add_argument(
        "--run-id",
        required=True,
        help="Run identifier (typically training.run_name from the config).",
    )
    stop_parser.add_argument(
        "--db-url",
        default=None,
        help="Telemetry database URL (overrides TENYSON_DB_URL).",
    )
    stop_parser.add_argument(
        "--experiment-id",
        default=os.getenv("TENYSON_EXPERIMENT_ID"),
        help=(
            "Optional experiment identifier. When provided, stop is scoped to "
            "(experiment_id, run_id)."
        ),
    )
    stop_parser.set_defaults(func=_cmd_stop)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
