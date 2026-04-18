from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any

import modal

from tenyson.cloud.modal import _resolve_modal_python_version
from tenyson.cloud.runtime_deps import (
    resolve_runtime_dependency_profile,
    runtime_pip_install_command,
)
from tenyson.core.hub_push import resolve_hf_token


REPO_ROOT = Path("/home/ayush/Desktop/code/tenyson")
GPU = "A100"
APP_NAME = "wordle-rl-vllm-adapter-probe"
ADAPTER_REPO = "goyalayus/wordle-lora-20260324-163252-sft_turn5"
ADAPTER_REVISION = "2f92897b5cd3f760da3bdc526aa3fd2842e9bd82"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _ignore_cache(path: Path) -> bool:
    return any(part == "__pycache__" for part in path.parts)


image = (
    modal.Image.debian_slim(python_version=_resolve_modal_python_version("3.12"))
    .run_commands("apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*")
    .run_commands(
        runtime_pip_install_command(
            profile=resolve_runtime_dependency_profile(gpu=GPU)
        )
    )
    .run_commands("python3 -m pip install datasets")
    .env(
        {
            "PYTHONPATH": "/workspace:/workspace/src",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "WANDB_DISABLED": "true",
        }
    )
    .add_local_dir(REPO_ROOT / "src", remote_path="/workspace/src", ignore=_ignore_cache)
    .add_local_dir(REPO_ROOT / "examples", remote_path="/workspace/examples", ignore=_ignore_cache)
)


secret_env: dict[str, str] = {}
hf_token = resolve_hf_token()
if hf_token:
    secret_env["HF_TOKEN"] = hf_token

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60,
    secrets=[modal.Secret.from_dict(secret_env)] if secret_env else [],
)
def probe_rl_vllm_adapter() -> dict[str, Any]:
    import sys

    os.chdir("/workspace")
    for path in ("/workspace", "/workspace/src"):
        if path not in sys.path:
            sys.path.insert(0, path)

    from datasets import Dataset
    from examples.wordle.functional import build_turn5_problem_rows
    from tenyson.core.chat_generation import resolve_generation_stop_strings
    from tenyson.core.hf_adapter import (
        download_hf_lora_adapter,
        resolve_hf_lora_runtime_kwargs,
        strict_load_hf_lora_adapter_weights,
    )
    from tenyson.jobs.rl import (
        _build_grpo_vllm_overrides,
        _configure_rl_unsloth_runtime_env,
        _ensure_trl_vllm_guided_decoding_compat,
        _ensure_trl_vllm_sampling_params_compat,
        _prepare_rl_generation_dataset,
    )
    from tenyson.jobs.tokenizer_utils import normalize_tokenizer_special_tokens

    vllm_cfg = {
        "enabled": True,
        "disable_flashinfer": True,
        "gpu_memory_utilization": 0.9,
        "temperature": 1.0,
        "min_p": 0.1,
        "top_p": 1.0,
        "top_k": -1,
        "standby_mode": True,
    }
    model_cfg = {
        "name": "Qwen/Qwen3-4B",
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "fast_inference": True,
    }
    config = {
        "chat_template": {
            "enabled": True,
            "stop_strings": ["</guess>"],
        },
    }

    _configure_rl_unsloth_runtime_env(vllm_cfg)
    from unsloth import FastLanguageModel

    adapter = download_hf_lora_adapter(
        repo_id=ADAPTER_REPO,
        revision=ADAPTER_REVISION,
    )
    lora_runtime_kwargs = resolve_hf_lora_runtime_kwargs(
        adapter,
        expected_target_modules=TARGET_MODULES,
    )
    base_model_name = str(adapter.config.get("base_model_name_or_path") or "").strip()
    if not base_model_name:
        raise RuntimeError("Adapter config does not record base_model_name_or_path.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=int(model_cfg["max_seq_length"]),
        load_in_4bit=bool(model_cfg["load_in_4bit"]),
        max_lora_rank=int(lora_runtime_kwargs["r"]),
        fast_inference=bool(model_cfg["fast_inference"]),
        gpu_memory_utilization=float(vllm_cfg["gpu_memory_utilization"]),
    )
    normalize_tokenizer_special_tokens(tokenizer, padding_side="left")
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_runtime_kwargs["r"]),
        target_modules=lora_runtime_kwargs["target_modules"],
        lora_alpha=lora_runtime_kwargs["lora_alpha"],
        lora_dropout=lora_runtime_kwargs["lora_dropout"],
        bias=lora_runtime_kwargs["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=456,
    )
    loaded_tensors = strict_load_hf_lora_adapter_weights(model, adapter)

    rows = build_turn5_problem_rows(sample_count=1, seed=456)
    dataset = Dataset.from_list(rows)
    dataset, _prompt_lookup = _prepare_rl_generation_dataset(
        dataset,
        tokenizer,
        config,
    )

    probe: dict[str, Any] = {
        "adapter_repo": ADAPTER_REPO,
        "adapter_revision": adapter.resolved_revision,
        "adapter_base_model": base_model_name,
        "loaded_tensors": loaded_tensors,
        "generate_called": False,
    }

    from vllm import LLM

    original_generate = LLM.generate

    def wrapped_generate(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not probe["generate_called"]:
            probe["generate_called"] = True
            lora_request = kwargs.get("lora_request")
            probe["lora_request_present"] = lora_request is not None
            probe["lora_request_type"] = (
                type(lora_request).__name__ if lora_request is not None else None
            )
            probe["lora_request_fields"] = {}
            if lora_request is not None:
                for name in (
                    "lora_name",
                    "lora_int_id",
                    "adapter_id",
                    "rank",
                    "path",
                    "lora_path",
                    "local_path",
                    "lora_local_path",
                ):
                    if hasattr(lora_request, name):
                        probe["lora_request_fields"][name] = getattr(
                            lora_request, name
                        )
                probe["lora_request_tensor_count"] = len(
                    _extract_lora_tensor_map(lora_request)
                )
                probe["lora_request_tensor_samples"] = _sample_tensor_summaries(
                    _extract_lora_tensor_map(lora_request)
                )

            model_runner = _find_first_attr_chain(
                self,
                [
                    "llm_engine.model_executor.driver_worker.model_runner",
                    "llm_engine.model_executor.worker.model_runner",
                    "llm_engine.model_executor.driver_worker.worker.model_runner",
                    "llm_engine.model_executor.workers[0].model_runner",
                ],
            )
            probe["model_runner_found"] = model_runner is not None
            lora_manager = None
            if model_runner is not None:
                lora_manager = getattr(model_runner, "lora_manager", None)
                if lora_manager is None:
                    model_obj = getattr(model_runner, "model", None)
                    lora_manager = getattr(model_obj, "lora_manager", None)
            probe["lora_manager_found"] = lora_manager is not None
            adapter_manager = None
            if lora_manager is not None:
                adapter_manager = getattr(lora_manager, "_adapter_manager", None)
                if adapter_manager is None:
                    adapter_manager = getattr(lora_manager, "adapter_manager", None)
            probe["adapter_manager_found"] = adapter_manager is not None
            if adapter_manager is not None:
                registered = getattr(adapter_manager, "_registered_adapters", None)
                active = getattr(adapter_manager, "_active_adapters", None)
                if registered is not None:
                    probe["registered_adapter_keys"] = list(registered.keys())
                if active is not None:
                    probe["active_adapter_keys"] = list(active.keys())
                punica_mapping = getattr(adapter_manager, "punica_wrapper_mapping", None)
                if isinstance(punica_mapping, dict):
                    probe["punica_wrapper_keys"] = list(punica_mapping.keys())
                    token_indices = {}
                    for key, wrapper in punica_mapping.items():
                        tensor = getattr(wrapper, "_token_lora_indices", None)
                        if tensor is not None:
                            token_indices[key] = _tensor_summary(tensor)
                    probe["punica_token_lora_indices"] = token_indices
        return original_generate(self, *args, **kwargs)

    LLM.generate = wrapped_generate

    _ensure_trl_vllm_guided_decoding_compat()
    _ensure_trl_vllm_sampling_params_compat()
    from trl import GRPOConfig, GRPOTrainer

    cfg_kwargs: dict[str, Any] = dict(
        output_dir="/tmp/trl_probe",
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5.0e-6,
        logging_steps=1,
        report_to="none",
        run_name="modal_rl_vllm_adapter_probe",
        seed=456,
        remove_unused_columns=False,
        num_generations=1,
        max_prompt_length=512,
        max_completion_length=32,
        temperature=1.0,
        bf16=False,
        fp16=False,
    )
    accepted = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    cfg_kwargs.update(
        _build_grpo_vllm_overrides(
            tokenizer,
            vllm_cfg,
            seed=456,
            prefer_explicit_sampling_params="vllm_sampling_params" in accepted,
            stop_strings=resolve_generation_stop_strings(config),
        )
    )
    if "save_only_model" in accepted:
        cfg_kwargs["save_only_model"] = False
    grpo_args = GRPOConfig(**{k: v for k, v in cfg_kwargs.items() if k in accepted})

    def zero_reward(
        prompts: list[Any],
        completions: list[Any],
        **_kwargs: Any,
    ) -> list[float]:
        return [0.0 for _ in completions]

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=dataset,
        reward_funcs=[zero_reward],
        processing_class=tokenizer,
        callbacks=[],
    )

    train_error = None
    try:
        trainer.train()
    except Exception as exc:  # noqa: BLE001
        train_error = repr(exc)

    probe["train_error"] = train_error
    return probe


def _find_first_attr_chain(root: Any, chains: list[str]) -> Any:
    for chain in chains:
        try:
            value = _resolve_attr_chain(root, chain)
        except Exception:  # noqa: BLE001
            continue
        if value is not None:
            return value
    return None


def _resolve_attr_chain(root: Any, chain: str) -> Any:
    current = root
    for piece in chain.split("."):
        if piece.endswith("]") and "[" in piece:
            name, raw_index = piece[:-1].split("[", 1)
            current = getattr(current, name)
            current = current[int(raw_index)]
        else:
            current = getattr(current, piece)
    return current


def _extract_lora_tensor_map(lora_request: Any) -> dict[str, Any]:
    for name in ("lora_tensors", "tensors", "adapter_tensors"):
        value = getattr(lora_request, name, None)
        if isinstance(value, dict):
            return value
    return {}


def _sample_tensor_summaries(tensor_map: dict[str, Any]) -> dict[str, str]:
    sampled: dict[str, str] = {}
    for key in sorted(tensor_map.keys())[:3]:
        sampled[key] = _tensor_summary(tensor_map[key])
    return sampled


def _tensor_summary(value: Any) -> str:
    try:
        shape = tuple(value.shape)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        shape = None
    try:
        norm = float(value.norm().item())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        norm = None
    try:
        max_abs = float(value.abs().max().item())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        max_abs = None
    return f"shape={shape} norm={norm} max_abs={max_abs}"


@app.local_entrypoint()
def main() -> None:
    result = probe_rl_vllm_adapter.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
