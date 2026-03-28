from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import List, Optional

from . import wandb_store
from .telemetry import LiveRunInfo, TelemetryClient, list_live_run_heartbeats


def request_stop(
    db_url: str,
    run_id: str,
    experiment_id: Optional[str] = None,
    phase: Optional[str] = None,
    create_if_missing: bool = True,
    attempt_token: Optional[str] = None,
) -> bool:
    """
    Set stop_requested = True for the given run_id in the active telemetry backend.
    """
    experiment_id = str(experiment_id or "").strip()
    if not experiment_id:
        raise ValueError(
            "experiment_id is required to request stop. "
            "Provide --experiment-id or set TENYSON_EXPERIMENT_ID."
        )

    client = TelemetryClient(db_url=db_url)
    explicit_phase = str(phase or "").strip().lower() or None
    now_iso = datetime.now(timezone.utc).isoformat()
    explicit_attempt = str(attempt_token or "").strip() or None
    if explicit_phase is not None and explicit_attempt is not None:
        return wandb_store.set_stop_requested(
            client.db_url,
            experiment_id=experiment_id,
            phase=explicit_phase,
            run_name=str(run_id),
            requested=True,
            when_iso=now_iso,
            create_if_missing=bool(create_if_missing),
            attempt_token=explicit_attempt,
        )
    if explicit_phase is not None:
        return wandb_store.set_stop_requested(
            client.db_url,
            experiment_id=experiment_id,
            phase=explicit_phase,
            run_name=str(run_id),
            requested=True,
            when_iso=now_iso,
            create_if_missing=bool(create_if_missing),
            attempt_token=None,
        )

    matched = list_live_run_heartbeats(
        client=client,
        experiment_id=experiment_id,
        max_age_seconds=365 * 24 * 60 * 60,
    )
    exact_live_matches = [
        row
        for row in matched
        if row.run_id == str(run_id)
        and (explicit_phase is None or row.phase == explicit_phase)
    ]
    exact_live_matches.sort(
        key=lambda row: row.updated_at or row.created_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    requested_any = False
    seen_live_attempts: set[tuple[str, Optional[str]]] = set()
    for row in exact_live_matches:
        key = (str(row.phase or "").strip().lower(), row.attempt_token)
        if key in seen_live_attempts:
            continue
        seen_live_attempts.add(key)
        if wandb_store.set_stop_requested(
            client.db_url,
            experiment_id=experiment_id,
            phase=key[0],
            run_name=str(run_id),
            requested=True,
            when_iso=now_iso,
            create_if_missing=True,
            attempt_token=row.attempt_token,
        ):
            requested_any = True
    if requested_any:
        return True

    detected_phase = None
    for row in matched:
        if row.run_id == str(run_id):
            detected_phase = str(row.phase or "").strip().lower() or None
            break
    candidate_phases: List[str] = []
    if explicit_phase is not None:
        candidate_phases.append(explicit_phase)
    if detected_phase is not None and detected_phase not in candidate_phases:
        candidate_phases.append(detected_phase)
    candidate_phases.extend(
        phase_name
        for phase_name in ["sft", "rl", "eval"]
        if phase_name not in candidate_phases
    )
    for phase_name in candidate_phases:
        if wandb_store.set_stop_requested(
            client.db_url,
            experiment_id=experiment_id,
            phase=phase_name,
            run_name=str(run_id),
            requested=True,
            when_iso=now_iso,
            create_if_missing=False,
            attempt_token=explicit_attempt,
        ):
            return True
    return False


def prime_stop_target(
    db_url: str,
    run_id: str,
    *,
    experiment_id: Optional[str] = None,
    phase: Optional[str] = None,
    attempt_token: Optional[str] = None,
) -> bool:
    """
    Ensure the canonical manual-stop control row exists before a worker has
    published a live heartbeat or W&B URL.
    """
    experiment_id = str(experiment_id or "").strip()
    if not experiment_id:
        raise ValueError(
            "experiment_id is required to prime stop control. "
            "Provide --experiment-id or set TENYSON_EXPERIMENT_ID."
        )

    phase_name = str(phase or "").strip().lower()
    if not phase_name:
        raise ValueError("phase is required to prime stop control.")

    return wandb_store.set_stop_requested(
        db_url,
        experiment_id=experiment_id,
        phase=phase_name,
        run_name=str(run_id),
        requested=False,
        when_iso=None,
        create_if_missing=True,
        attempt_token=str(attempt_token or "").strip() or None,
    )


def list_live_runs(
    db_url: str,
    experiment_id: str,
    *,
    max_age_seconds: int = 90,
) -> List[LiveRunInfo]:
    """
    Return currently-live runs for an experiment based on recent heartbeats.
    """
    experiment_id = str(experiment_id or "").strip()
    if not experiment_id:
        raise ValueError(
            "experiment_id is required to list live runs. "
            "Provide --experiment-id or set TENYSON_EXPERIMENT_ID."
        )
    client = TelemetryClient(db_url=db_url)
    return list_live_run_heartbeats(
        client=client,
        experiment_id=experiment_id,
        max_age_seconds=max_age_seconds,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Request a manual stop for a running tenyson job."
    )
    parser.add_argument(
        "--db-url",
        required=True,
        help="Telemetry backend ref (wandb://<entity>/<project>).",
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
            "Experiment identifier for the target run. Required unless "
            "TENYSON_EXPERIMENT_ID is set."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    experiment_id = str(args.experiment_id or "").strip()
    if not experiment_id:
        raise SystemExit(
            "Error: --experiment-id is required (or set TENYSON_EXPERIMENT_ID)."
        )
    request_stop(
        db_url=args.db_url,
        run_id=args.run_id,
        experiment_id=experiment_id,
        create_if_missing=True,
    )


if __name__ == "__main__":
    main()
