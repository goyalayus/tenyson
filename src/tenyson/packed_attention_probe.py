from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import modal

from tenyson.cloud.modal import ModalManager, _build_clone_repo_command
from tenyson.cloud.runtime_deps import runtime_pip_install_command


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
    import inspect
    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer
    from trl.data_utils import pack_dataset
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    from tenyson.jobs.sft import (
        _prepare_manual_assistant_only_dataset,
    )
    from tenyson.jobs.sft_dataset import normalize_builtin_sft_dataset
    from tenyson.jobs.tokenizer_utils import (
        ensure_assistant_mask_chat_template,
        normalize_tokenizer_special_tokens,
    )

    os.chdir("/workspace")

    model_name = "Qwen/Qwen3-0.6B"
    max_length = 2048

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        load_in_4bit=True,
        load_in_8bit=False,
        fast_inference=False,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    normalize_tokenizer_special_tokens(tokenizer)
    ensure_assistant_mask_chat_template(tokenizer)
    model.eval()

    raw = Dataset.from_list(
        [
            {
                "system": "You are a careful sorting assistant.",
                "prompt": "Sort these letters into alphabetical order: c b a",
                "answer": "abc",
            },
            {
                "system": "You are a careful sorting assistant.",
                "prompt": "Sort these letters into alphabetical order: f d e",
                "answer": "def",
            },
        ]
    )
    normalized = normalize_builtin_sft_dataset(
        raw,
        config={},
        dataset_name="probe",
    )

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

    ex1 = tokenize_messages(normalized[0])
    ex2 = tokenize_messages(normalized[1])

    replacement_tokens = tokenizer.encode(" the", add_special_tokens=False)
    replacement_id = (
        int(replacement_tokens[0])
        if replacement_tokens
        else int(tokenizer.eos_token_id)
    )
    packed = pack_dataset(
        Dataset.from_list([ex1, ex2]),
        seq_length=max_length,
        strategy="bfd",
    )
    prepared_a = packed[0]
    collator = DataCollatorForLanguageModeling(
        pad_token_id=int(tokenizer.pad_token_id),
        completion_only_loss=True,
        padding_free=True,
    )
    batch_a = collator([prepared_a])

    batch_b: Dict[str, Any] = {}
    for key, value in batch_a.items():
        batch_b[key] = value.clone() if hasattr(value, "clone") else value

    def _forward_kwargs(batch: Dict[str, Any]) -> Dict[str, Any]:
        accepted = set(inspect.signature(model.forward).parameters.keys())
        kwargs: Dict[str, Any] = {
            "input_ids": batch["input_ids"].to(model.device),
            "use_cache": False,
        }
        if "position_ids" in batch and "position_ids" in accepted:
            kwargs["position_ids"] = batch["position_ids"].to(model.device)
        if "packed_seq_lengths" in batch and "packed_seq_lengths" in accepted:
            kwargs["packed_seq_lengths"] = batch["packed_seq_lengths"].to(model.device)
        return kwargs

    input_ids_a = batch_a["input_ids"][0].tolist()
    ex1_start = _find_subsequence(input_ids_a, ex1["input_ids"])
    ex2_start = _find_subsequence(input_ids_a, ex2["input_ids"])
    batch_b["input_ids"][0, ex1_start : ex1_start + len(ex1["input_ids"])] = replacement_id

    with torch.inference_mode():
        outputs_a = model(**_forward_kwargs(batch_a))
        outputs_b = model(**_forward_kwargs(batch_b))

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
        "raw_dataset_columns": list(raw.column_names),
        "normalized_dataset_columns": list(normalized.column_names),
        "normalized_messages_example_0": normalized[0]["messages"],
        "model_forward_accepts_packed_seq_lengths": (
            "packed_seq_lengths" in inspect.signature(model.forward).parameters
        ),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "batch_a_keys": sorted(batch_a.keys()),
        "packed_row_count": int(len(packed)),
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

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_length,
    )
    model.eval()

    formatting_func_calls = {"count": 0}

    def trainer_formatting_func(_example: Dict[str, Any]) -> list[str]:
        formatting_func_calls["count"] += 1
        return [""]

    trainer_dataset = _prepare_manual_assistant_only_dataset(
        normalized,
        tokenizer=tokenizer,
        max_length=max_length,
        packing=True,
        packing_strategy="bfd",
        shuffle=False,
        seed=3407,
        dataset_name="probe",
    )
    trainer_args = SFTConfig(
        output_dir="/tmp/tenyson-packed-attention-probe",
        report_to="none",
        disable_tqdm=True,
        max_length=max_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=1,
        learning_rate=1.0e-5,
        logging_steps=1,
        save_strategy="no",
        packing=False,
        padding_free=True,
        assistant_only_loss=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=trainer_dataset,
        processing_class=tokenizer,
        formatting_func=trainer_formatting_func,
    )
    trainer_batch = next(iter(trainer.get_train_dataloader()))

    trainer_input_ids = trainer_batch["input_ids"][0].tolist()
    trainer_ex1_start = _find_subsequence(trainer_input_ids, ex1["input_ids"])
    trainer_ex2_start = _find_subsequence(trainer_input_ids, ex2["input_ids"])

    trainer_batch_b: Dict[str, Any] = {}
    for key, value in trainer_batch.items():
        trainer_batch_b[key] = value.clone() if hasattr(value, "clone") else value
    trainer_batch_b["input_ids"][
        0, trainer_ex1_start : trainer_ex1_start + len(ex1["input_ids"])
    ] = replacement_id

    with torch.inference_mode():
        trainer_outputs_a = model(**_forward_kwargs(trainer_batch))
        trainer_outputs_b = model(**_forward_kwargs(trainer_batch_b))

    trainer_row = trainer_dataset[0]
    trainer_labels_mask = [
        0 if token == -100 else 1 for token in trainer_batch["labels"][0].tolist()
    ]
    trainer_logit_diff = (
        trainer_outputs_a.logits[
            0, trainer_ex2_start : trainer_ex2_start + len(ex2["input_ids"])
        ].float().cpu()
        - trainer_outputs_b.logits[
            0, trainer_ex2_start : trainer_ex2_start + len(ex2["input_ids"])
        ].float().cpu()
    ).abs()

    result.update(
        {
            "trainer_formatting_func_calls": int(formatting_func_calls["count"]),
            "trainer_dataset_keys": sorted(trainer_row.keys()),
            "trainer_prepared_seq_lengths": list(trainer_row["seq_lengths"]),
            "trainer_batch_keys": sorted(trainer_batch.keys()),
            "trainer_example1_start": int(trainer_ex1_start),
            "trainer_ex2_start": int(trainer_ex2_start),
            "trainer_position_id_at_ex2_start": int(
                trainer_batch["position_ids"][0, trainer_ex2_start].item()
            ),
            "trainer_packed_seq_lengths": trainer_batch.get(
                "packed_seq_lengths"
            ).tolist()
            if hasattr(trainer_batch.get("packed_seq_lengths"), "tolist")
            else trainer_batch.get("packed_seq_lengths"),
            "trainer_assistant_mask_matches_labels": list(
                trainer_row["assistant_masks"]
            )
            == trainer_labels_mask,
            "trainer_max_abs_logit_diff_on_example2": float(
                trainer_logit_diff.max().item()
            ),
            "trainer_mean_abs_logit_diff_on_example2": float(
                trainer_logit_diff.mean().item()
            ),
        }
    )
    result["trainer_attention_isolated_for_example2"] = bool(
        result["trainer_max_abs_logit_diff_on_example2"] < 1e-5
    )
    trainer_train_result = trainer.train()
    result["trainer_global_step"] = int(trainer.state.global_step)
    result["trainer_train_runtime"] = float(
        trainer_train_result.metrics.get("train_runtime", 0.0)
    )
    result["trainer_train_completed"] = bool(result["trainer_global_step"] >= 1)
    return result


@app.local_entrypoint()
def main() -> None:
    result = run_probe.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("assistant_only_loss_verified"):
        raise SystemExit("assistant-only loss verification failed")
    if result.get("trainer_formatting_func_calls") != 0:
        raise SystemExit("trainer unexpectedly invoked formatting_func")
    if not result.get("trainer_assistant_mask_matches_labels"):
        raise SystemExit("trainer assistant-mask verification failed")
    if not result.get("trainer_attention_isolated_for_example2"):
        raise SystemExit("trainer cross-example attention leakage detected")
    if not result.get("trainer_train_completed"):
        raise SystemExit("trainer failed to complete a training step")
