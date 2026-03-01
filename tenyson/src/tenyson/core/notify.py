"""
Failure notification: write to log dir, optional webhook, optional telemetry.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


def notify_failure(
    step_label: str,
    result: Any,
    failure_log_dir: Optional[str] = None,
    failure_webhook_url: Optional[str] = None,
    db_url: Optional[str] = None,
    experiment_id: Optional[str] = None,
    phase: Optional[str] = None,
) -> None:
    """
    On step failure: optionally write a JSON log file, POST to webhook, and/or
    write to the run_failures telemetry table.
    """
    run_id = getattr(result, "run_id", "unknown")
    failure_reason = getattr(result, "failure_reason", "unknown")
    instance_id = getattr(result, "instance_id", None)
    spot_interruption = getattr(result, "spot_interruption", None)

    payload = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "step_name": step_label,
        "failure_reason": failure_reason,
        "instance_id": instance_id,
        "spot_interruption": spot_interruption,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    if failure_log_dir:
        Path(failure_log_dir).mkdir(parents=True, exist_ok=True)
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in step_label)
        path = Path(failure_log_dir) / f"{safe_label}_{run_id}_{uuid4().hex[:8]}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    if failure_webhook_url:
        try:
            import urllib.request

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                failure_webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception:  # noqa: S110
            pass

    if db_url:
        try:
            from tenyson.core.telemetry import (
                record_run_summary,
                RunFailure,
                TelemetryClient,
            )

            client = TelemetryClient(db_url=db_url)
            session = client.Session()
            try:
                row = RunFailure(
                    id=str(uuid4()),
                    experiment_id=experiment_id,
                    run_id=run_id,
                    step_label=step_label,
                    failure_reason=failure_reason,
                    instance_id=instance_id,
                    spot_interruption=spot_interruption,
                )
                session.add(row)
                session.commit()
            finally:
                session.close()
            if experiment_id and phase:
                record_run_summary(
                    client=client,
                    experiment_id=experiment_id,
                    phase=phase,
                    result=result,
                )
        except Exception:  # noqa: S110
            pass
