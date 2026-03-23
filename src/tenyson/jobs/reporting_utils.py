from __future__ import annotations

from typing import Any


def normalize_report_to(report_to: Any, *, telemetry_backend: str) -> Any:
    """
    Normalize Trainer/TRL reporting targets across runtimes.

    The trainer stack accepts disabled reporting as the string "none", but some
    call sites accidentally pass ["none"], which later becomes invalid once we
    append the telemetry-backed reporter. Strip no-op values from list forms and
    append W&B only when the telemetry backend requires it.
    """
    if telemetry_backend != "wandb":
        return report_to

    if report_to is None:
        return ["wandb"]

    if isinstance(report_to, str):
        normalized = report_to.strip()
        if not normalized or normalized.lower() == "none":
            return ["wandb"]
        if normalized == "wandb":
            return ["wandb"]
        return [normalized, "wandb"]

    if isinstance(report_to, list):
        normalized_list = []
        for item in report_to:
            normalized = str(item or "").strip()
            if not normalized or normalized.lower() == "none":
                continue
            if normalized not in normalized_list:
                normalized_list.append(normalized)
        if "wandb" not in normalized_list:
            normalized_list.append("wandb")
        return normalized_list

    normalized = str(report_to).strip()
    if not normalized or normalized.lower() == "none":
        return ["wandb"]
    if normalized == "wandb":
        return ["wandb"]
    return [normalized, "wandb"]
