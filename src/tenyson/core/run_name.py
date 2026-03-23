from __future__ import annotations

from typing import Any, Dict, Tuple


_RUN_NAME_FIELDS: Dict[str, Tuple[str, str, str]] = {
    "sft": ("training", "run_name", "sft_job"),
    "rl": ("training", "run_name", "rl_job"),
    "eval": ("evaluation", "run_name", "eval_job"),
}

_JOB_CLASS_TO_TYPE = {
    "SFTJob": "sft",
    "RLJob": "rl",
    "EvalJob": "eval",
}


def infer_job_type(job_class: type, config: Dict[str, Any] | None = None) -> str:
    """
    Infer tenyson job type from class name with config-based fallback.
    """
    class_name = getattr(job_class, "__name__", "")
    mapped = _JOB_CLASS_TO_TYPE.get(class_name)
    if mapped:
        return mapped

    cfg = config or {}
    if isinstance(cfg.get("evaluation"), dict):
        return "eval"
    if isinstance(cfg.get("training"), dict):
        # Unknown trainer-like jobs default to training semantics.
        return "sft"
    return "sft"


def resolve_required_run_name(
    config: Dict[str, Any],
    job_type: str,
) -> str:
    """
    Return explicit non-default run_name for the given job type.

    The historical fallback defaults (sft_job / rl_job / eval_job) are now
    intentionally rejected to prevent artifact lineage collisions.
    """
    if job_type not in _RUN_NAME_FIELDS:
        raise ValueError(f"Unsupported job type for run_name resolution: {job_type}")

    section, key, disallowed_default = _RUN_NAME_FIELDS[job_type]
    section_cfg = config.get(section, {}) if isinstance(config, dict) else {}
    run_name = str(section_cfg.get(key, "")).strip()
    field_ref = f"{section}.{key}"

    if not run_name:
        raise ValueError(
            f"Missing required {field_ref} for {job_type.upper()} run. "
            "Set an explicit unique run_name."
        )
    if run_name == disallowed_default:
        raise ValueError(
            f"Invalid {field_ref}='{run_name}'. This default is not allowed. "
            "Set an explicit unique run_name."
        )
    return run_name


def resolve_required_run_name_for_job_class(
    config: Dict[str, Any],
    job_class: type,
) -> Tuple[str, str]:
    """
    Resolve (job_type, run_name) for a pipeline step/job class.
    """
    job_type = infer_job_type(job_class, config)
    return job_type, resolve_required_run_name(config, job_type)
