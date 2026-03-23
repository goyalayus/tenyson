from __future__ import annotations

from typing import Any


_QWEN3_MARKERS: tuple[str, ...] = ("qwen3", "qwen-3", "qwen_3")


def require_qwen3_model_name(
    model_name: Any,
    *,
    field_name: str = "model.name",
) -> str:
    resolved = str(model_name or "").strip()
    if not resolved:
        raise ValueError(f"{field_name} is required.")

    normalized = resolved.lower()
    if not any(marker in normalized for marker in _QWEN3_MARKERS):
        raise ValueError(
            f"{field_name} must point to a Qwen 3 family model. "
            f"Got {resolved!r}."
        )
    return resolved
