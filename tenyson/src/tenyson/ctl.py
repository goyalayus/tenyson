import argparse
import os
import sys

from tenyson.core.control import request_stop


def _cmd_stop(args: argparse.Namespace) -> None:
    """Mark a run as stop_requested in the telemetry database."""
    db_url = args.db_url or os.getenv("TENYSON_DB_URL")
    if not db_url:
        print(
            "Error: --db-url is required (or set TENYSON_DB_URL).",
            file=sys.stderr,
        )
        sys.exit(1)
    experiment_id = str(args.experiment_id or "").strip()
    if not experiment_id:
        print(
            "Error: --experiment-id is required (or set TENYSON_EXPERIMENT_ID).",
            file=sys.stderr,
        )
        sys.exit(1)

    request_stop(
        db_url=db_url,
        run_id=args.run_id,
        experiment_id=experiment_id,
        create_if_missing=True,
    )

    print(
        (
            f"[tenyson.ctl] Marked run_id={args.run_id}"
            f" experiment_id={experiment_id}"
            f" as stop_requested in {db_url}"
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="tenyson ctl", description="Tenyson control")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stop_parser = subparsers.add_parser(
        "stop",
        help="Request a graceful stop for a running job.",
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
            "Experiment identifier for the target run. Required unless "
            "TENYSON_EXPERIMENT_ID is set."
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
