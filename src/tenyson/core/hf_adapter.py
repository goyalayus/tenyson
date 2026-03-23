from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file as safetensors_load_file

from tenyson.core.hf_checkpoint import resolve_hf_repo_revision


@dataclass(frozen=True)
class HfLoraAdapter:
    repo_id: str
    requested_revision: str
    resolved_revision: str
    config_in_repo: str
    weights_in_repo: str
    config: Dict[str, Any]
    state_dict: Dict[str, Any]


def _coerce_target_modules(value: Any) -> list[str]:
    if value is None:
        raise ValueError("LoRA target_modules must be provided and non-empty.")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("LoRA target_modules must be non-empty.")
        return [text]
    if not isinstance(value, Sequence):
        raise ValueError("LoRA target_modules must be a string or sequence of strings.")

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    if not result:
        raise ValueError("LoRA target_modules must be non-empty.")
    return result


def _normalized_target_modules(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(sorted(_coerce_target_modules(value)))


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{field_name} must be an integer, got {value!r}.") from exc


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{field_name} must be numeric, got {value!r}.") from exc


def _resolve_adapter_artifact_paths(repo_files: Sequence[str]) -> tuple[str, str]:
    normalized_files = {str(path or "").strip("/") for path in repo_files}
    root_pair = ("adapter_config.json", "adapter_model.safetensors")
    if all(path in normalized_files for path in root_pair):
        return root_pair

    last_checkpoint_pair = (
        "last-checkpoint/adapter_config.json",
        "last-checkpoint/adapter_model.safetensors",
    )
    if all(path in normalized_files for path in last_checkpoint_pair):
        return last_checkpoint_pair

    checkpoint_steps: dict[int, set[str]] = {}
    for path in normalized_files:
        match = re.match(
            r"^checkpoint-(\d+)/(adapter_config\.json|adapter_model\.safetensors)$",
            path,
        )
        if not match:
            continue
        step = int(match.group(1))
        checkpoint_steps.setdefault(step, set()).add(match.group(2))

    completed_steps = [
        step
        for step, filenames in checkpoint_steps.items()
        if {"adapter_config.json", "adapter_model.safetensors"} <= filenames
    ]
    if completed_steps:
        latest_step = max(completed_steps)
        prefix = f"checkpoint-{latest_step}"
        return (
            f"{prefix}/adapter_config.json",
            f"{prefix}/adapter_model.safetensors",
        )

    raise ValueError(
        "No LoRA adapter artifacts found in Hugging Face repo revision. Expected "
        "root adapter files or checkpoint/last-checkpoint adapter files."
    )


def download_hf_lora_adapter(
    repo_id: str,
    revision: str = "main",
) -> HfLoraAdapter:
    repo_id = str(repo_id or "").strip()
    requested_revision = str(revision or "").strip() or "main"
    if not repo_id:
        raise ValueError("repo_id is required to download a Hugging Face adapter.")

    resolved_revision = resolve_hf_repo_revision(repo_id, requested_revision)
    repo_files = HfApi().list_repo_files(repo_id=repo_id, revision=resolved_revision)
    config_in_repo, weights_in_repo = _resolve_adapter_artifact_paths(repo_files)

    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=config_in_repo,
        revision=resolved_revision,
    )
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=weights_in_repo,
        revision=resolved_revision,
    )

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(
            f"adapter_config.json from repo '{repo_id}' did not parse as an object."
        )
    state_dict = dict(safetensors_load_file(weights_path))
    if not state_dict:
        raise ValueError(
            f"Adapter weights from repo '{repo_id}' revision '{resolved_revision}' are empty."
        )

    return HfLoraAdapter(
        repo_id=repo_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        config_in_repo=config_in_repo,
        weights_in_repo=weights_in_repo,
        config=config,
        state_dict=state_dict,
    )


def resolve_hf_lora_runtime_kwargs(
    adapter: HfLoraAdapter,
    *,
    expected_r: int | None = None,
    expected_alpha: float | None = None,
    expected_dropout: float | None = None,
    expected_bias: str | None = None,
    expected_target_modules: Sequence[str] | str | None = None,
) -> Dict[str, Any]:
    config = adapter.config
    peft_type = str(config.get("peft_type") or "").strip().upper()
    if peft_type and peft_type != "LORA":
        raise ValueError(
            f"Adapter repo '{adapter.repo_id}' revision '{adapter.resolved_revision}' "
            f"uses peft_type={peft_type!r}; only LORA adapters are supported."
        )

    r = _coerce_int(config.get("r"), "adapter_config.json field 'r'")
    lora_alpha = _coerce_float(
        config.get("lora_alpha"), "adapter_config.json field 'lora_alpha'"
    )
    lora_dropout = _coerce_float(
        config.get("lora_dropout", 0.0), "adapter_config.json field 'lora_dropout'"
    )
    bias = str(config.get("bias") or "none").strip() or "none"
    target_modules = _coerce_target_modules(config.get("target_modules"))

    errors: list[str] = []
    if expected_r is not None and r != int(expected_r):
        errors.append(f"r expected {int(expected_r)} but adapter stores {r}")
    if expected_alpha is not None and lora_alpha != float(expected_alpha):
        errors.append(
            "lora_alpha expected "
            f"{float(expected_alpha)} but adapter stores {lora_alpha}"
        )
    if expected_dropout is not None and lora_dropout != float(expected_dropout):
        errors.append(
            "lora_dropout expected "
            f"{float(expected_dropout)} but adapter stores {lora_dropout}"
        )
    if expected_bias is not None and bias != str(expected_bias).strip():
        errors.append(f"bias expected {expected_bias!r} but adapter stores {bias!r}")

    expected_modules = _normalized_target_modules(expected_target_modules)
    actual_modules = tuple(sorted(target_modules))
    if expected_modules is not None and actual_modules != expected_modules:
        errors.append(
            f"target_modules expected {list(expected_modules)!r} but adapter stores {list(actual_modules)!r}"
        )

    if errors:
        joined = "; ".join(errors)
        raise ValueError(
            f"Adapter repo '{adapter.repo_id}' revision '{adapter.resolved_revision}' "
            f"does not match the configured LoRA layout: {joined}."
        )

    return {
        "r": r,
        "target_modules": target_modules,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": bias,
    }


def _is_adapter_state_key(key: str) -> bool:
    return "lora_" in key or ".modules_to_save." in key


def _normalize_adapter_state_key(key: str) -> str:
    normalized = str(key)
    markers = (
        ".lora_A",
        ".lora_B",
        ".lora_embedding_A",
        ".lora_embedding_B",
        ".lora_magnitude_vector",
    )
    for marker in markers:
        default_marker = f"{marker}.default"
        if marker in normalized and default_marker not in normalized:
            normalized = normalized.replace(marker, default_marker)
    return normalized


def strict_load_hf_lora_adapter_weights(model: Any, adapter: HfLoraAdapter) -> int:
    model_state = model.state_dict()
    model_adapter_keys = {key for key in model_state if _is_adapter_state_key(key)}
    if not model_adapter_keys:
        raise ValueError("Model does not contain LoRA parameters to load into.")

    normalized_state: Dict[str, Any] = {}
    collisions: list[str] = []
    for raw_key, tensor in adapter.state_dict.items():
        normalized_key = _normalize_adapter_state_key(raw_key)
        if normalized_key in normalized_state and normalized_key != raw_key:
            collisions.append(f"{raw_key!r} -> {normalized_key!r}")
            continue
        normalized_state[normalized_key] = tensor
    if collisions:
        raise ValueError(
            "Adapter weights contain duplicate keys after normalization: "
            + ", ".join(collisions[:5])
        )

    unexpected_keys = sorted(key for key in normalized_state if key not in model_state)
    missing_keys = sorted(
        key for key in model_adapter_keys if key not in normalized_state
    )
    mismatched_shapes: list[str] = []
    for key, tensor in normalized_state.items():
        if key not in model_state:
            continue
        if tuple(tensor.shape) != tuple(model_state[key].shape):
            mismatched_shapes.append(
                f"{key}: adapter{tuple(tensor.shape)} != model{tuple(model_state[key].shape)}"
            )

    if unexpected_keys or missing_keys or mismatched_shapes:
        details: list[str] = []
        if unexpected_keys:
            details.append(f"unexpected keys {unexpected_keys[:5]!r}")
        if missing_keys:
            details.append(f"missing keys {missing_keys[:5]!r}")
        if mismatched_shapes:
            details.append(f"shape mismatches {mismatched_shapes[:5]!r}")
        raise ValueError(
            f"Adapter repo '{adapter.repo_id}' revision '{adapter.resolved_revision}' "
            "does not match the current LoRA parameter set: " + "; ".join(details)
        )

    load_result = model.load_state_dict(normalized_state, strict=False)
    remaining_missing = [
        key for key in load_result.missing_keys if key in model_adapter_keys
    ]
    remaining_unexpected = [
        key for key in load_result.unexpected_keys if _is_adapter_state_key(key)
    ]
    if remaining_missing or remaining_unexpected:
        raise ValueError(
            f"Adapter repo '{adapter.repo_id}' revision '{adapter.resolved_revision}' "
            "failed strict load validation after applying weights: "
            f"missing={remaining_missing[:5]!r}, unexpected={remaining_unexpected[:5]!r}"
        )

    return len(normalized_state)
