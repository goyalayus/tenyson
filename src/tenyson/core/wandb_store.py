from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple


SUMMARY_PREFIX = "tenyson/"
SUMMARY_ATTEMPT_TOKEN = f"{SUMMARY_PREFIX}attempt_token"
SUMMARY_STATUS = f"{SUMMARY_PREFIX}status"
SUMMARY_PHASE = f"{SUMMARY_PREFIX}phase"
SUMMARY_RUN_NAME = f"{SUMMARY_PREFIX}run_name"
SUMMARY_EXPERIMENT_ID = f"{SUMMARY_PREFIX}experiment_id"
SUMMARY_TOTAL_TIME = f"{SUMMARY_PREFIX}total_time_seconds"
SUMMARY_METRICS_JSON = f"{SUMMARY_PREFIX}metrics_json"
SUMMARY_HF_REPO_ID = f"{SUMMARY_PREFIX}hf_repo_id"
SUMMARY_HF_REVISION = f"{SUMMARY_PREFIX}hf_revision"
SUMMARY_WANDB_URL = f"{SUMMARY_PREFIX}wandb_url"
SUMMARY_FAILURE_REASON = f"{SUMMARY_PREFIX}failure_reason"
SUMMARY_INSTANCE_ID = f"{SUMMARY_PREFIX}instance_id"
SUMMARY_SPOT_INTERRUPTION = f"{SUMMARY_PREFIX}spot_interruption"
SUMMARY_IS_ACTIVE = f"{SUMMARY_PREFIX}is_active"
SUMMARY_PROVIDER = f"{SUMMARY_PREFIX}provider"
SUMMARY_HEARTBEAT_AT = f"{SUMMARY_PREFIX}heartbeat_at"
SUMMARY_STOP_REQUESTED = f"{SUMMARY_PREFIX}stop_requested"
SUMMARY_STOP_REQUESTED_AT = f"{SUMMARY_PREFIX}stop_requested_at"
SUMMARY_JOB_RESULT_JSON = f"{SUMMARY_PREFIX}job_result_json"
SUMMARY_RESULTS_JSON = f"{SUMMARY_PREFIX}results_json"
SUMMARY_RESULT_ARTIFACT = f"{SUMMARY_PREFIX}result_artifact"
SUMMARY_PROJECT_URL = f"{SUMMARY_PREFIX}project_url"


@dataclass(frozen=True)
class WandBTarget:
    entity: str
    project: str
    base_url: str = "https://wandb.ai"

    @property
    def project_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.entity}/{self.project}"

    def run_path(self, run_id: str) -> str:
        return f"{self.entity}/{self.project}/{run_id}"


def is_wandb_backend_ref(value: str) -> bool:
    return str(value or "").startswith("wandb://")


def parse_backend_ref(value: str) -> WandBTarget:
    raw = str(value or "").strip()
    if not raw.startswith("wandb://"):
        raise ValueError(f"Not a W&B backend ref: {value!r}")
    locator = raw[len("wandb://") :]
    parts = [part for part in locator.split("/") if part]
    if len(parts) != 2:
        raise ValueError(
            "W&B telemetry refs must look like wandb://<entity>/<project>."
        )
    return WandBTarget(entity=parts[0], project=parts[1])


def _normalize_attempt_token(value: Optional[str]) -> Optional[str]:
    normalized = str(value or "").strip()
    return normalized or None


def build_run_id(
    experiment_id: str,
    phase: str,
    run_name: str,
    *,
    attempt_token: Optional[str] = None,
) -> str:
    normalized_attempt = _normalize_attempt_token(attempt_token)
    raw = f"{experiment_id}:{phase}:{run_name}"
    slug_source = f"{experiment_id}-{phase}-{run_name}"
    if normalized_attempt:
        raw = f"{raw}:{normalized_attempt}"
        slug_source = f"{slug_source}-{normalized_attempt[:8]}"
    slug = _safe_component(slug_source, default="tenyson-run")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    if len(slug) > 95:
        slug = slug[:95].rstrip("-")
    return f"{slug}-{digest}"


def ensure_run(
    backend_ref: str,
    *,
    experiment_id: str,
    phase: str,
    run_name: str,
    config: Optional[Mapping[str, Any]] = None,
    attempt_token: Optional[str] = None,
) -> Any:
    target = parse_backend_ref(backend_ref)
    import wandb

    expected_id = build_run_id(
        experiment_id,
        phase,
        run_name,
        attempt_token=attempt_token,
    )
    current_run = getattr(wandb, "run", None)
    if current_run is not None and getattr(current_run, "id", None) != expected_id:
        try:
            wandb.finish()
        except Exception:  # noqa: BLE001
            pass
        current_run = None

    os.environ["WANDB_ENTITY"] = target.entity
    os.environ["WANDB_PROJECT"] = target.project
    os.environ["WANDB_RUN_ID"] = expected_id
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_NAME"] = run_name
    os.environ["WANDB_RUN_GROUP"] = experiment_id

    if current_run is None:
        current_run = wandb.init(
            entity=target.entity,
            project=target.project,
            id=expected_id,
            resume="allow",
            name=run_name,
            group=experiment_id,
            job_type=phase,
            config=dict(config or {}),
        )

    update_run_summary(
        current_run,
        {
            SUMMARY_EXPERIMENT_ID: experiment_id,
            SUMMARY_PHASE: phase,
            SUMMARY_RUN_NAME: run_name,
            SUMMARY_ATTEMPT_TOKEN: _normalize_attempt_token(attempt_token),
            SUMMARY_PROJECT_URL: target.project_url,
            SUMMARY_WANDB_URL: getattr(current_run, "url", None),
        },
    )
    return current_run


def active_run() -> Any:
    try:
        import wandb

        return getattr(wandb, "run", None)
    except Exception:  # noqa: BLE001
        return None


def update_run_summary(run: Any, values: Mapping[str, Any]) -> None:
    if run is None:
        return
    summary = getattr(run, "summary", None)
    if summary is None:
        return
    for key, value in values.items():
        summary[key] = value
    updater = getattr(summary, "update", None)
    if callable(updater):
        try:
            updater()
        except TypeError:
            updater(values)


def resolve_run(
    backend_ref: str,
    *,
    experiment_id: str,
    phase: str,
    run_name: str,
    create_if_missing: bool = False,
    attempt_token: Optional[str] = None,
) -> Any:
    expected_id = build_run_id(
        experiment_id,
        phase,
        run_name,
        attempt_token=attempt_token,
    )
    run = active_run()
    if run is not None and (
        getattr(run, "id", None) == expected_id
        or _match_run(
            run,
            experiment_id=experiment_id,
            phase=phase,
            run_name=run_name,
            attempt_token=attempt_token,
        )
    ):
        return run
    try:
        return fetch_run(
            backend_ref,
            experiment_id=experiment_id,
            phase=phase,
            run_name=run_name,
            attempt_token=attempt_token,
        )
    except Exception:  # noqa: BLE001
        if create_if_missing:
            return ensure_run(
                backend_ref,
                experiment_id=experiment_id,
                phase=phase,
                run_name=run_name,
                attempt_token=attempt_token,
            )
        raise


def log_result_payload(
    run: Any,
    *,
    experiment_id: str,
    phase: str,
    run_name: str,
    results_payload: Mapping[str, Any],
    job_result_payload: Mapping[str, Any],
    attempt_token: Optional[str] = None,
) -> Optional[str]:
    if run is None:
        return None
    import wandb

    payload = {
        "experiment_id": experiment_id,
        "phase": phase,
        "run_name": run_name,
        "results_payload": dict(results_payload),
        "job_result_payload": dict(job_result_payload),
    }
    normalized_attempt = _normalize_attempt_token(attempt_token)
    artifact_source = f"tenyson-{experiment_id}-{phase}-{run_name}-result"
    if normalized_attempt:
        artifact_source = f"{artifact_source}-{normalized_attempt[:8]}"
    artifact_name = _safe_component(artifact_source, default="tenyson-result")
    with tempfile.TemporaryDirectory() as tmpdir:
        payload_path = Path(tmpdir) / "run_result.json"
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        artifact = wandb.Artifact(
            name=artifact_name,
            type="tenyson-run-result",
            metadata={
                "experiment_id": experiment_id,
                "phase": phase,
                "run_name": run_name,
                "attempt_token": normalized_attempt,
            },
        )
        artifact.add_file(str(payload_path), name="run_result.json")
        run.log_artifact(artifact)
    return artifact_name


def fetch_run(
    backend_ref: str,
    *,
    experiment_id: str,
    phase: str,
    run_name: str,
    attempt_token: Optional[str] = None,
) -> Any:
    target = parse_backend_ref(backend_ref)
    import wandb

    api = wandb.Api()
    normalized_attempt = _normalize_attempt_token(attempt_token)
    if normalized_attempt:
        run_id = build_run_id(
            experiment_id,
            phase,
            run_name,
            attempt_token=normalized_attempt,
        )
        try:
            return api.run(target.run_path(run_id))
        except Exception:
            pass

    matched = _find_latest_matching_run(
        api.runs(path=f"{target.entity}/{target.project}"),
        experiment_id=experiment_id,
        phase=phase,
        run_name=run_name,
        attempt_token=normalized_attempt,
    )
    if matched is not None:
        return matched

    if normalized_attempt is None:
        run_id = build_run_id(experiment_id, phase, run_name)
        return api.run(target.run_path(run_id))

    raise LookupError(
        "No matching W&B run was found for "
        f"experiment_id={experiment_id}, phase={phase}, run_name={run_name}, "
        f"attempt_token={normalized_attempt}."
    )


def fetch_run_url(
    backend_ref: str,
    *,
    experiment_id: str,
    phase: str,
    run_name: str,
    attempt_token: Optional[str] = None,
) -> Optional[str]:
    try:
        run = fetch_run(
            backend_ref,
            experiment_id=experiment_id,
            phase=phase,
            run_name=run_name,
            attempt_token=attempt_token,
        )
    except Exception:  # noqa: BLE001
        return None
    if attempt_token:
        summary_token = _summary_get(run, SUMMARY_ATTEMPT_TOKEN)
        if summary_token and str(summary_token) != str(attempt_token):
            return None
    return getattr(run, "url", None)


def fetch_run_result(
    backend_ref: str,
    *,
    experiment_id: str,
    phase: str,
    run_name: str,
    attempt_token: Optional[str] = None,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    try:
        run = fetch_run(
            backend_ref,
            experiment_id=experiment_id,
            phase=phase,
            run_name=run_name,
            attempt_token=attempt_token,
        )
    except Exception:  # noqa: BLE001
        return None

    summary_token = _summary_get(run, SUMMARY_ATTEMPT_TOKEN)
    if attempt_token and str(summary_token or "") != str(attempt_token):
        return None

    job_result_json = _summary_get(run, SUMMARY_JOB_RESULT_JSON)
    if not job_result_json:
        return None

    job_result_payload = json.loads(job_result_json)
    results_json = _summary_get(run, SUMMARY_RESULTS_JSON)
    if results_json:
        results_payload = json.loads(results_json)
        return _as_mapping(results_payload), _as_mapping(job_result_payload)

    artifact_name = _summary_get(run, SUMMARY_RESULT_ARTIFACT)
    if artifact_name:
        results_payload = fetch_artifact_results(
            backend_ref,
            artifact_name=str(artifact_name),
        )
        if results_payload is not None:
            return _as_mapping(results_payload), _as_mapping(job_result_payload)

    return {}, _as_mapping(job_result_payload)


def fetch_stop_requested(
    backend_ref: str,
    *,
    experiment_id: str,
    phase: str,
    run_name: str,
) -> bool:
    try:
        run = fetch_run(
            backend_ref,
            experiment_id=experiment_id,
            phase=phase,
            run_name=run_name,
        )
    except Exception:  # noqa: BLE001
        return False
    return bool(_summary_get(run, SUMMARY_STOP_REQUESTED))


def set_stop_requested(
    backend_ref: str,
    *,
    experiment_id: str,
    phase: str,
    run_name: str,
    requested: bool,
    when_iso: Optional[str] = None,
    create_if_missing: bool = False,
) -> bool:
    try:
        run = resolve_run(
            backend_ref,
            experiment_id=experiment_id,
            phase=phase,
            run_name=run_name,
            create_if_missing=create_if_missing,
        )
    except Exception:  # noqa: BLE001
        return False
    update_run_summary(
        run,
        {
            SUMMARY_STOP_REQUESTED: bool(requested),
            SUMMARY_STOP_REQUESTED_AT: when_iso if requested else None,
        },
    )
    return True


def list_live_runs(
    backend_ref: str,
    *,
    experiment_id: str,
    max_age_seconds: int = 90,
) -> list[dict[str, Any]]:
    target = parse_backend_ref(backend_ref)
    import wandb

    api = wandb.Api()
    now = _utc_now()
    rows: list[dict[str, Any]] = []
    for run in api.runs(path=f"{target.entity}/{target.project}"):
        summary_experiment_id = str(
            _summary_get(run, SUMMARY_EXPERIMENT_ID) or getattr(run, "group", "") or ""
        ).strip()
        if summary_experiment_id != str(experiment_id):
            continue
        if not bool(_summary_get(run, SUMMARY_IS_ACTIVE)):
            continue
        heartbeat_at = _parse_datetime(_summary_get(run, SUMMARY_HEARTBEAT_AT))
        if heartbeat_at is None:
            continue
        age_seconds = (now - heartbeat_at).total_seconds()
        if age_seconds > max(1, int(max_age_seconds)):
            continue
        rows.append(
            {
                "run_id": str(_summary_get(run, SUMMARY_RUN_NAME) or getattr(run, "name", "")),
                "phase": str(_summary_get(run, SUMMARY_PHASE) or ""),
                "provider": _summary_get(run, SUMMARY_PROVIDER),
                "status": str(_summary_get(run, SUMMARY_STATUS) or "running"),
                "is_active": True,
                "created_at": _parse_datetime(getattr(run, "created_at", None)),
                "updated_at": heartbeat_at,
            }
        )
    rows.sort(
        key=lambda row: row.get("updated_at") or _utc_now(),
        reverse=True,
    )
    return rows


def fetch_artifact_results(
    backend_ref: str,
    *,
    artifact_name: str,
) -> Optional[Dict[str, Any]]:
    target = parse_backend_ref(backend_ref)
    import wandb

    api = wandb.Api()
    artifact = api.artifact(f"{target.entity}/{target.project}/{artifact_name}:latest")
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(artifact.download(root=tmpdir))
        payload_path = artifact_dir / "run_result.json"
        if not payload_path.exists():
            return None
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    return _as_mapping(payload.get("results_payload", {}))


def _summary_get(run: Any, key: str) -> Any:
    summary = getattr(run, "summary", None)
    if summary is None:
        return None
    try:
        return summary.get(key)
    except Exception:  # noqa: BLE001
        try:
            return summary[key]
        except Exception:  # noqa: BLE001
            return None


def _match_run(
    run: Any,
    *,
    experiment_id: str,
    phase: str,
    run_name: str,
    attempt_token: Optional[str] = None,
) -> bool:
    summary_experiment_id = str(
        _summary_get(run, SUMMARY_EXPERIMENT_ID) or getattr(run, "group", "") or ""
    ).strip()
    summary_phase = str(
        _summary_get(run, SUMMARY_PHASE) or getattr(run, "job_type", "") or ""
    ).strip()
    summary_run_name = str(
        _summary_get(run, SUMMARY_RUN_NAME) or getattr(run, "name", "") or ""
    ).strip()
    if (
        summary_experiment_id != str(experiment_id)
        or summary_phase != str(phase)
        or summary_run_name != str(run_name)
    ):
        return False
    normalized_attempt = _normalize_attempt_token(attempt_token)
    if normalized_attempt is None:
        return True
    return _normalize_attempt_token(_summary_get(run, SUMMARY_ATTEMPT_TOKEN)) == normalized_attempt


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _run_sort_key(run: Any) -> tuple[float, float]:
    heartbeat = _parse_datetime(_summary_get(run, SUMMARY_HEARTBEAT_AT))
    updated = _parse_datetime(getattr(run, "updated_at", None))
    created = _parse_datetime(getattr(run, "created_at", None))
    primary = heartbeat or updated or created or datetime.fromtimestamp(0, tz=timezone.utc)
    secondary = updated or created or primary
    return primary.timestamp(), secondary.timestamp()


def _find_latest_matching_run(
    runs: Any,
    *,
    experiment_id: str,
    phase: str,
    run_name: str,
    attempt_token: Optional[str] = None,
) -> Optional[Any]:
    candidates = [
        run
        for run in runs
        if _match_run(
            run,
            experiment_id=experiment_id,
            phase=phase,
            run_name=run_name,
            attempt_token=attempt_token,
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=_run_sort_key)


def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {"value": value}


def _safe_component(value: str, default: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return default
    output: list[str] = []
    for char in text:
        if char.isalnum():
            output.append(char)
        elif char in {"-", "_", "."}:
            output.append(char)
        else:
            output.append("-")
    cleaned = "".join(output).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned or default


def _utc_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def _parse_datetime(value: Any):
    from datetime import datetime, timezone

    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
