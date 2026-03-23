from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import modal

from tenyson.cloud.modal import ModalManager, _build_clone_repo_command
from tenyson.cloud.runtime_deps import runtime_pip_install_command
from tenyson.jobs.sft import _resolve_sft_special_tokens_kwargs


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANAGER = ModalManager.from_env()
_GIT_SOURCE = _MANAGER._resolve_git_source(str(_REPO_ROOT))
_PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"

_BUILD_SECRET_ENV: Dict[str, str] = {}
_GIT_AUTH_TOKEN = str(
    os.getenv("TENYSON_GIT_AUTH_TOKEN") or os.getenv("GITHUB_TOKEN") or ""
).strip()
if _GIT_AUTH_TOKEN:
    _BUILD_SECRET_ENV["TENYSON_GIT_AUTH_TOKEN"] = _GIT_AUTH_TOKEN
_BUILD_SECRETS = (
    [modal.Secret.from_dict(_BUILD_SECRET_ENV)] if _BUILD_SECRET_ENV else None
)

_RUNTIME_SECRET_ENV: Dict[str, str] = {}
_HF_TOKEN = str(os.getenv("HF_TOKEN") or "").strip()
if _HF_TOKEN:
    _RUNTIME_SECRET_ENV["HF_TOKEN"] = _HF_TOKEN
_RUNTIME_SECRETS = (
    [modal.Secret.from_dict(_RUNTIME_SECRET_ENV)] if _RUNTIME_SECRET_ENV else []
)

app = modal.App("tenyson-packed-attention-probe")
image = (
    modal.Image.debian_slim(python_version=_PYTHON_VERSION)
    .run_commands("apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*")
    .run_commands(runtime_pip_install_command())
    .run_commands(
        _build_clone_repo_command(),
        env={
            "TENYSON_GIT_REPO_URL": _GIT_SOURCE.clone_url,
            "TENYSON_GIT_COMMIT": _GIT_SOURCE.commit,
        },
        secrets=_BUILD_SECRETS,
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": "/workspace/src",
        }
    )
)


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    width = len(needle)
    for index in range(len(haystack) - width + 1):
        if haystack[index : index + width] == needle:
            return index
    raise ValueError("Needle subsequence not found in packed input_ids.")


@app.function(
    image=image,
    gpu=_MANAGER._resolve_modal_gpu_request(),
    timeout=_MANAGER.timeout,
    secrets=_RUNTIME_SECRETS,
    serialized=True,
)
def run_probe() -> Dict[str, Any]:
    import torch
    from datasets import Dataset, load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    from tenyson.jobs.tokenizer_utils import (
        ensure_assistant_mask_chat_template,
        normalize_tokenizer_special_tokens,
    )

    os.chdir("/workspace")

    model_name = "Qwen/Qwen3-0.6B"
    dataset_name = "goyalayus/wordle-reasoning-sft-prefix-keep-think"
    max_length = 2048

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        load_in_4bit=True,
        load_in_8bit=False,
        fast_inference=False,
        trust_remote_code=True,
    )
    normalize_tokenizer_special_tokens(tokenizer)
    ensure_assistant_mask_chat_template(tokenizer)
    model.eval()

    raw = load_dataset(dataset_name, split="train[:2]")
    raw_rows = [
        {"messages": list(raw[0]["messages"])},
        {"messages": list(raw[1]["messages"])},
    ]

    def tokenize_messages(row: Dict[str, Any]) -> Dict[str, Any]:
        processed = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
        )
        return {
            "input_ids": list(processed["input_ids"]),
            "assistant_masks": list(processed["assistant_masks"]),
        }

    ex1 = tokenize_messages(raw_rows[0])
    ex2 = tokenize_messages(raw_rows[1])

    replacement_tokens = tokenizer.encode(" the", add_special_tokens=False)
    replacement_id = (
        int(replacement_tokens[0])
        if replacement_tokens
        else int(tokenizer.eos_token_id)
    )
    ex1_mut = {
        "input_ids": [replacement_id] * len(ex1["input_ids"]),
        "assistant_masks": list(ex1["assistant_masks"]),
    }

    config_kwargs = {
        "output_dir": "/tmp/packed_attention_probe",
        "max_length": max_length,
        "packing": True,
        "assistant_only_loss": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_steps": 1,
        "learning_rate": 1.0e-5,
        "warmup_steps": 0,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "dataset_num_proc": 1,
        "shuffle_dataset": False,
    }
    accepted = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    config_kwargs.update(
        _resolve_sft_special_tokens_kwargs(
            tokenizer,
            accepted_fields=accepted,
        )
    )
    args = SFTConfig(**{key: value for key, value in config_kwargs.items() if key in accepted})

    def build_batch(rows: list[Dict[str, Any]]):
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=Dataset.from_list(rows),
            processing_class=tokenizer,
        )
        batch = next(iter(trainer.get_train_dataloader()))
        prepared_row = trainer.train_dataset[0]
        return batch, prepared_row

    batch_a, prepared_a = build_batch(raw_rows)

    batch_b: Dict[str, Any] = {}
    for key, value in batch_a.items():
        batch_b[key] = value.clone() if hasattr(value, "clone") else value

    def to_device(batch: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            moved[key] = value.to(model.device) if hasattr(value, "to") else value
        return moved

    input_ids_a = batch_a["input_ids"][0].tolist()
    ex1_start = _find_subsequence(input_ids_a, ex1["input_ids"])
    ex2_start = _find_subsequence(input_ids_a, ex2["input_ids"])
    batch_b["input_ids"][0, ex1_start : ex1_start + len(ex1["input_ids"])] = replacement_id

    with torch.inference_mode():
        moved_a = to_device(batch_a)
        moved_b = to_device(batch_b)
        outputs_a = model(
            input_ids=moved_a["input_ids"],
            position_ids=moved_a.get("position_ids"),
            use_cache=False,
        )
        outputs_b = model(
            input_ids=moved_b["input_ids"],
            position_ids=moved_b.get("position_ids"),
            use_cache=False,
        )

    ex2_len = len(ex2["input_ids"])

    logits_a = outputs_a.logits[0, ex2_start : ex2_start + ex2_len].float().cpu()
    logits_b = outputs_b.logits[0, ex2_start : ex2_start + ex2_len].float().cpu()
    logit_diff = (logits_a - logits_b).abs()

    assistant_mask_a = list(prepared_a["assistant_masks"])
    supervised_mask_a = [
        0 if token == -100 else 1 for token in batch_a["labels"][0].tolist()
    ]

    result = {
        "model_name": model_name,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "batch_a_keys": sorted(batch_a.keys()),
        "prepared_seq_lengths": list(prepared_a["seq_lengths"]),
        "example1_start": int(ex1_start),
        "assistant_token_count_a": int(sum(assistant_mask_a)),
        "supervised_label_count_a": int(sum(supervised_mask_a)),
        "assistant_mask_matches_labels_a": assistant_mask_a == supervised_mask_a,
        "batch_b_keys": sorted(batch_b.keys()),
        "ex2_start": int(ex2_start),
        "position_id_at_ex2_start": int(batch_a["position_ids"][0, ex2_start].item()),
        "example1_mutation_token_id": int(replacement_id),
        "max_abs_logit_diff_on_example2": float(logit_diff.max().item()),
        "mean_abs_logit_diff_on_example2": float(logit_diff.mean().item()),
    }
    result["assistant_only_loss_verified"] = bool(result["assistant_mask_matches_labels_a"])
    result["attention_isolated_for_example2"] = bool(
        result["max_abs_logit_diff_on_example2"] < 1e-5
    )
    return result


@app.local_entrypoint()
def main() -> None:
    result = run_probe.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("assistant_only_loss_verified"):
        raise SystemExit("assistant-only loss verification failed")
    if not result.get("attention_isolated_for_example2"):
        raise SystemExit("cross-example attention leakage detected")
