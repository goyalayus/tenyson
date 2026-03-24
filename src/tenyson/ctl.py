import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any, Dict, List

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


def _safe_controller_name(value: str) -> str:
    text = str(value or "").strip().lower()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in text)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "controller"


def _controller_paths(
    *,
    controller_dir: str,
    name: str,
) -> tuple[Path, Path, Path]:
    root = Path(controller_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_controller_name(name)
    return (
        root / f"{safe_name}.pid",
        root / f"{safe_name}.log",
        root / f"{safe_name}.json",
    )


def _read_pidfile(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    raw = pid_path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _load_controller_metadata(metadata_path: Path) -> Dict[str, Any]:
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _normalize_launch_command(command: List[str]) -> List[str]:
    normalized = list(command or [])
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    return normalized


def _cmd_launch(args: argparse.Namespace) -> None:
    command = _normalize_launch_command(list(args.command or []))
    if not command:
        print(
            "[tenyson.ctl] launch requires a command after '--'.",
            file=sys.stderr,
        )
        sys.exit(1)

    pid_path, log_path, metadata_path = _controller_paths(
        controller_dir=str(args.controller_dir),
        name=str(args.name),
    )
    existing_pid = _read_pidfile(pid_path)
    if existing_pid is not None and _is_process_alive(existing_pid):
        print(
            (
                f"[tenyson.ctl] Controller '{args.name}' is already running "
                f"(pid={existing_pid})."
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    env = os.environ.copy()
    for assignment in args.env or []:
        if "=" not in assignment:
            print(
                f"[tenyson.ctl] Invalid --env value '{assignment}'. Use KEY=VALUE.",
                file=sys.stderr,
            )
            sys.exit(1)
        key, value = assignment.split("=", 1)
        env[str(key).strip()] = value

    cwd = os.path.abspath(str(args.cwd or "."))
    launch_time = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a", encoding="utf-8") as log_handle:
        log_handle.write(
            f"\n[tenyson.ctl] Launching controller '{args.name}' at {launch_time}\n"
        )
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    metadata = {
        "name": str(args.name),
        "pid": int(process.pid),
        "command": command,
        "cwd": cwd,
        "log_path": str(log_path),
        "launched_at": launch_time,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        (
            f"[tenyson.ctl] Detached controller started name={args.name} "
            f"pid={process.pid} log={log_path}"
        ),
        flush=True,
    )


def _cmd_status(args: argparse.Namespace) -> None:
    pid_path, log_path, metadata_path = _controller_paths(
        controller_dir=str(args.controller_dir),
        name=str(args.name),
    )
    metadata = _load_controller_metadata(metadata_path)
    pid = _read_pidfile(pid_path)
    alive = pid is not None and _is_process_alive(pid)
    status = "running" if alive else "stopped"

    print(f"[tenyson.ctl] Controller '{args.name}' status: {status}", flush=True)
    if pid is not None:
        print(f"  pid: {pid}", flush=True)
    if metadata.get("launched_at"):
        print(f"  launched_at: {metadata['launched_at']}", flush=True)
    if metadata.get("cwd"):
        print(f"  cwd: {metadata['cwd']}", flush=True)
    if metadata.get("command"):
        print(f"  command: {' '.join(metadata['command'])}", flush=True)
    print(f"  log: {log_path}", flush=True)

    if not alive:
        sys.exit(1)


def _cmd_stop_controller(args: argparse.Namespace) -> None:
    pid_path, log_path, metadata_path = _controller_paths(
        controller_dir=str(args.controller_dir),
        name=str(args.name),
    )
    del log_path, metadata_path
    pid = _read_pidfile(pid_path)
    if pid is None or not _is_process_alive(pid):
        print(
            f"[tenyson.ctl] No live controller found for '{args.name}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.kill(pid, signal.SIGTERM)
    print(
        f"[tenyson.ctl] Sent SIGTERM to controller '{args.name}' (pid={pid}).",
        flush=True,
    )


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

    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch a detached local controller command.",
    )
    launch_parser.add_argument(
        "--name",
        required=True,
        help="Logical controller name used for pid/log metadata files.",
    )
    launch_parser.add_argument(
        "--controller-dir",
        default=".tenyson_runs/controllers",
        help="Directory for controller pid/log/metadata files.",
    )
    launch_parser.add_argument(
        "--cwd",
        default=".",
        help="Working directory for the detached controller command.",
    )
    launch_parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment variable override. Repeatable.",
    )
    launch_parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after '--'.",
    )
    launch_parser.set_defaults(func=_cmd_launch)

    status_parser = subparsers.add_parser(
        "status",
        help="Show whether a detached controller is still running.",
    )
    status_parser.add_argument(
        "--name",
        required=True,
        help="Logical controller name used during launch.",
    )
    status_parser.add_argument(
        "--controller-dir",
        default=".tenyson_runs/controllers",
        help="Directory for controller pid/log/metadata files.",
    )
    status_parser.set_defaults(func=_cmd_status)

    stop_controller_parser = subparsers.add_parser(
        "stop-controller",
        help="Send SIGTERM to a detached local controller process.",
    )
    stop_controller_parser.add_argument(
        "--name",
        required=True,
        help="Logical controller name used during launch.",
    )
    stop_controller_parser.add_argument(
        "--controller-dir",
        default=".tenyson_runs/controllers",
        help="Directory for controller pid/log/metadata files.",
    )
    stop_controller_parser.set_defaults(func=_cmd_stop_controller)

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
