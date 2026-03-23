import argparse
from datetime import datetime, timezone
import os
import sys

from tenyson.core.control import list_live_runs, request_stop


def _resolve_db_url(args: argparse.Namespace) -> str:
    db_url = str(args.db_url or os.getenv("TENYSON_DB_URL") or "").strip()
    if not db_url:
        print(
            "Error: set TENYSON_DB_URL to wandb://<entity>/<project> or pass --db-url.",
            file=sys.stderr,
        )
        sys.exit(1)
    return db_url


def _resolve_experiment_id(args: argparse.Namespace) -> str:
    experiment_id = str(args.experiment_id or os.getenv("TENYSON_EXPERIMENT_ID") or "").strip()
    if not experiment_id:
        print(
            "Error: set TENYSON_EXPERIMENT_ID or pass --experiment-id.",
            file=sys.stderr,
        )
        sys.exit(1)
    return experiment_id


def _format_last_seen(updated_at: datetime | None) -> str:
    if updated_at is None:
        return "unknown"
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    age_seconds = max(
        0,
        int((datetime.now(timezone.utc) - updated_at.astimezone(timezone.utc)).total_seconds()),
    )
    return f"{age_seconds}s ago"


def _choose_live_run_id(db_url: str, experiment_id: str, max_age_seconds: int) -> str:
    candidates = list_live_runs(
        db_url=db_url,
        experiment_id=experiment_id,
        max_age_seconds=max_age_seconds,
    )
    if not candidates:
        print(
            f"[tenyson.ctl] No live runs found for experiment_id={experiment_id}.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"[tenyson.ctl] Live runs for experiment_id={experiment_id}:",
        flush=True,
    )
    for index, candidate in enumerate(candidates, start=1):
        provider = f" provider={candidate.provider}" if candidate.provider else ""
        print(
            f"  {index}. run_id={candidate.run_id} phase={candidate.phase}{provider} "
            f"last_seen={_format_last_seen(candidate.updated_at)}",
            flush=True,
        )

    while True:
        try:
            choice = input(
                f"Choose run to stop (1-{len(candidates)}) or 'q' to cancel: "
            ).strip()
        except EOFError:
            print(
                "[tenyson.ctl] Interactive selection cancelled.",
                file=sys.stderr,
            )
            sys.exit(1)
        if choice.lower() in {"q", "quit", "exit"}:
            print("[tenyson.ctl] Cancelled.", file=sys.stderr)
            sys.exit(1)
        if choice.isdigit():
            selected_index = int(choice)
            if 1 <= selected_index <= len(candidates):
                return candidates[selected_index - 1].run_id
        print("Invalid choice. Please enter a listed number or 'q'.", file=sys.stderr)


def _cmd_stop(args: argparse.Namespace) -> None:
    """Mark a run as stop_requested in the active telemetry backend."""
    db_url = _resolve_db_url(args)
    experiment_id = _resolve_experiment_id(args)
    run_id = str(args.run_id or "").strip()
    if not run_id:
        run_id = _choose_live_run_id(
            db_url=db_url,
            experiment_id=experiment_id,
            max_age_seconds=int(args.max_age_seconds),
        )

    stopped = request_stop(
        db_url=db_url,
        run_id=run_id,
        experiment_id=experiment_id,
        create_if_missing=False,
    )
    if not stopped:
        print(
            (
                f"[tenyson.ctl] No active run control row found for run_id={run_id} "
                f"experiment_id={experiment_id}."
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        (
            f"[tenyson.ctl] Marked run_id={run_id}"
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
        default=None,
        help="Run identifier (typically training.run_name from the config).",
    )
    stop_parser.add_argument(
        "--db-url",
        default=None,
        help="Telemetry backend ref (wandb://<entity>/<project>).",
    )
    stop_parser.add_argument(
        "--experiment-id",
        default=os.getenv("TENYSON_EXPERIMENT_ID"),
        help=(
            "Experiment identifier for the target run. Required unless "
            "TENYSON_EXPERIMENT_ID is set."
        ),
    )
    stop_parser.add_argument(
        "--max-age-seconds",
        type=int,
        default=90,
        help=(
            "When --run-id is omitted, only show runs whose heartbeat was seen "
            "within this many seconds."
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
