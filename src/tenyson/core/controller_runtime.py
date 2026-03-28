from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Dict, Optional, Sequence


TENYSON_CONTROLLER_METADATA_PATH = "TENYSON_CONTROLLER_METADATA_PATH"

RUNTIME_STATE_KEY = "runtime_state"
RUNTIME_STATE_UPDATED_AT_KEY = "runtime_state_updated_at"
RUNTIME_ACTIVE_STAGE_IDS_KEY = "runtime_active_stage_ids"
RUNTIME_LAST_COMPLETED_STAGE_IDS_KEY = "runtime_last_completed_stage_ids"
STOP_AT_BOUNDARY_REQUESTED_KEY = "stop_at_boundary_requested"
STOP_AT_BOUNDARY_REQUESTED_AT_KEY = "stop_at_boundary_requested_at"

_CONTROLLER_METADATA_LOCK = threading.RLock()


def controller_metadata_path_from_env() -> Optional[Path]:
    raw = str(os.getenv(TENYSON_CONTROLLER_METADATA_PATH, "")).strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def load_controller_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
    os.replace(tmp_path, path)


def update_controller_metadata(
    path: Path,
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    with _CONTROLLER_METADATA_LOCK:
        payload = load_controller_metadata(path)
        payload.update(updates)
        _atomic_write_json(path, payload)
        return payload


def update_controller_runtime_state(
    *,
    state: str,
    active_stage_ids: Optional[Sequence[str]] = None,
    last_completed_stage_ids: Optional[Sequence[str]] = None,
) -> None:
    path = controller_metadata_path_from_env()
    if path is None:
        return
    update_controller_metadata(
        path,
        {
            RUNTIME_STATE_KEY: str(state),
            RUNTIME_STATE_UPDATED_AT_KEY: datetime.now(timezone.utc).isoformat(),
            RUNTIME_ACTIVE_STAGE_IDS_KEY: [str(value) for value in active_stage_ids or []],
            RUNTIME_LAST_COMPLETED_STAGE_IDS_KEY: [
                str(value) for value in last_completed_stage_ids or []
            ],
        },
    )


def request_stop_at_boundary(path: Path) -> Dict[str, Any]:
    return update_controller_metadata(
        path,
        {
            STOP_AT_BOUNDARY_REQUESTED_KEY: True,
            STOP_AT_BOUNDARY_REQUESTED_AT_KEY: datetime.now(timezone.utc).isoformat(),
        },
    )


def boundary_stop_requested(path: Optional[Path] = None) -> bool:
    resolved_path = path or controller_metadata_path_from_env()
    if resolved_path is None:
        return False
    payload = load_controller_metadata(resolved_path)
    return bool(payload.get(STOP_AT_BOUNDARY_REQUESTED_KEY))


def clear_stop_at_boundary_request(path: Optional[Path] = None) -> None:
    resolved_path = path or controller_metadata_path_from_env()
    if resolved_path is None:
        return
    update_controller_metadata(
        resolved_path,
        {
            STOP_AT_BOUNDARY_REQUESTED_KEY: False,
            STOP_AT_BOUNDARY_REQUESTED_AT_KEY: None,
        },
    )
