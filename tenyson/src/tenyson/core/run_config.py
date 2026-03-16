from datetime import datetime, timezone
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional
from uuid import uuid4

import yaml


def _safe_component(value: str, default: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[\s/\\]+", "-", text)
    text = re.sub(r"[^a-z0-9._-]", "", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or default


def materialize_run_config(
    *,
    config: Dict[str, Any],
    project_root: Path,
    job_type: str,
    run_name: str,
) -> Path:
    """
    Persist an immutable, per-run config snapshot to disk.

    The snapshot lives under:
      .tenyson_runs/<sanitized_run_name>/<job_type>-<timestamp>-<id>.yaml
    """
    safe_run = _safe_component(run_name, "run")
    safe_job = _safe_component(job_type, "job")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]

    runs_dir = project_root / ".tenyson_runs" / safe_run
    runs_dir.mkdir(parents=True, exist_ok=True)

    config_path = runs_dir / f"{safe_job}-{timestamp}-{suffix}.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path


def shared_overrides_from_env(
    *,
    hf_repo_base_env: str = "TENYSON_HF_REPO_BASE",
    experiment_id_env: str = "TENYSON_EXPERIMENT_ID",
) -> Optional[Dict[str, Any]]:
    hf_repo_base = str(os.getenv(hf_repo_base_env) or "").strip()
    experiment_id = str(os.getenv(experiment_id_env) or "").strip()

    shared_overrides: Dict[str, Any] = {}
    if hf_repo_base:
        shared_overrides.setdefault("training", {})["hf_repo_base"] = hf_repo_base
    if experiment_id:
        shared_overrides.setdefault("telemetry", {})["experiment_id"] = experiment_id
    return shared_overrides or None
