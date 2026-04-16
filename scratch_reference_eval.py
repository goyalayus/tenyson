import gc
import json
import os
from types import SimpleNamespace

import modal

from tenyson.cloud.runtime_deps import runtime_pip_install_command


APP_NAME = "arith-reference-eval"
ROOT = os.path.abspath(os.path.dirname(__file__))
HF_TOKEN = os.environ["HF_TOKEN"]

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        "apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*"
    )
    .run_commands(runtime_pip_install_command(profile="default"))
    .add_local_dir(os.path.join(ROOT, "src"), remote_path="/workspace/src", copy=True)
    .add_local_dir(
        os.path.join(ROOT, "examples"),
        remote_path="/workspace/examples",
        copy=True,
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": "/workspace:/workspace/src",
        }
    )
)
secret = modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN})


@app.function(image=image, gpu="A100", timeout=3600, secrets=[secret])
def run_reference_eval():
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    from unsloth import FastLanguageModel

    from examples.arithmetic.functional import (
        build_addition_dataset,
        compute_addition_metrics,
    )
    from tenyson.core.chat_generation import render_generation_prompt
    from tenyson.core.hf_adapter import (
        download_hf_lora_adapter,
        resolve_hf_lora_runtime_kwargs,
        strict_load_hf_lora_adapter_weights,
    )
    from tenyson.jobs.tokenizer_utils import normalize_tokenizer_special_tokens

    base_config = {
        "chat_template": {"enable_thinking": False, "stop_strings": ["</answer>"]},
        "model": {
            "name": "Qwen/Qwen3-0.6B",
            "load_in_4bit": False,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_target_modules": ["up_proj", "gate_proj", "down_proj"],
        },
        "vllm": {
            "enabled": True,
            "disable_flashinfer": True,
            "gpu_memory_utilization": 0.9,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 64,
        },
        "evaluation": {"run_name": "reference_eval", "batch_size": 8},
    }

    dataset = build_addition_dataset(digits=2, sample_count=100, seed=7)
    rows = [dataset[i] for i in range(len(dataset))]

    class StopOnSuffix(StoppingCriteria):
        def __init__(self, suffix_ids):
            self.suffix_ids = suffix_ids
            self.suffix_len = len(suffix_ids)

        def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001, D401
            if input_ids.shape[1] < self.suffix_len:
                return False
            tail = input_ids[:, -self.suffix_len :]
            suffix = torch.tensor(
                self.suffix_ids,
                device=tail.device,
                dtype=tail.dtype,
            )
            matches = torch.all(tail == suffix.unsqueeze(0), dim=1)
            return bool(torch.all(matches).item())

    def cleanup(*objs):
        for obj in objs:
            try:
                del obj
            except Exception:
                pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def run_mode(mode_name: str, adapter_ref: tuple[str, str] | None):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="Qwen/Qwen3-0.6B",
            max_seq_length=4096,
            load_in_4bit=False,
            fast_inference=False,
        )
        normalize_tokenizer_special_tokens(tokenizer, padding_side="left")

        if adapter_ref is not None:
            repo_id, revision = adapter_ref
            adapter = download_hf_lora_adapter(repo_id=repo_id, revision=revision)
            lora_kwargs = resolve_hf_lora_runtime_kwargs(
                adapter,
                expected_r=base_config["model"]["lora_r"],
                expected_alpha=base_config["model"]["lora_alpha"],
                expected_target_modules=base_config["model"]["lora_target_modules"],
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=int(lora_kwargs["r"]),
                target_modules=lora_kwargs["target_modules"],
                lora_alpha=lora_kwargs["lora_alpha"],
                lora_dropout=lora_kwargs["lora_dropout"],
                bias=lora_kwargs["bias"],
                use_gradient_checkpointing="unsloth",
            )
            strict_load_hf_lora_adapter_weights(model, adapter)

        model.train(False)
        model = FastLanguageModel.for_inference(model)

        suffix_ids = tokenizer.encode("</answer>", add_special_tokens=False)
        stopping = StoppingCriteriaList([StopOnSuffix(suffix_ids)])

        prompts = [
            render_generation_prompt(
                row=row,
                tokenizer=tokenizer,
                config=base_config,
            )
            for row in rows
        ]
        completions = []
        batch_size = int(base_config["evaluation"]["batch_size"])

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
            )
            device = getattr(model, "device", None)
            if device is None:
                device = next(model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(base_config["vllm"]["max_tokens"]),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping,
            )
            for batch_index in range(outputs.shape[0]):
                prompt_len = int(inputs["attention_mask"][batch_index].sum().item())
                new_tokens = outputs[batch_index][prompt_len:]
                completions.append(
                    tokenizer.decode(new_tokens, skip_special_tokens=False)
                )

        metrics = compute_addition_metrics(
            SimpleNamespace(
                completions=completions,
                dataset_rows=rows,
            )
        )
        result = {
            "mode": mode_name,
            "metrics": metrics["metrics"],
            "detailed_results": metrics["detailed_results"],
        }
        cleanup(model, tokenizer, inputs, outputs)
        return result

    base = run_mode("base", None)
    sft = run_mode(
        "sft",
        (
            "goyalayus/arithmetic-2digit-sft_2digit_06b",
            "ac6d760fbf5d28586a29fc4c7517670c99bc4dc2",
        ),
    )
    rl = run_mode(
        "rl",
        (
            "goyalayus/arithmetic-2digit-rl_2digit_06b",
            "507736f79a938fcb172dda6e75553ce4ef3558d2",
        ),
    )

    def diff_summary(label: str, other: dict):
        base_details = base["detailed_results"]
        other_details = other["detailed_results"]
        diffs = []
        identical = 0
        for base_row, other_row in zip(base_details, other_details):
            if base_row["completion"] == other_row["completion"]:
                identical += 1
                continue
            if len(diffs) < 10:
                diffs.append(
                    {
                        "id": base_row["id"],
                        "problem": f'{base_row["left"]} + {base_row["right"]}',
                        "expected": base_row["expected_answer"],
                        "base_completion": base_row["completion"],
                        f"{label}_completion": other_row["completion"],
                    }
                )
        return {
            "identical_count": identical,
            "different_count": len(base_details) - identical,
            "sample_diffs": diffs,
        }

    payload = {
        "base": {"metrics": base["metrics"]},
        "sft": {
            "metrics": sft["metrics"],
            "vs_base": diff_summary("sft", sft),
        },
        "rl": {
            "metrics": rl["metrics"],
            "vs_base": diff_summary("rl", rl),
        },
    }
    cleanup(base, sft, rl)
    return json.dumps(payload)


@app.local_entrypoint()
def main():
    print(run_reference_eval.remote(), flush=True)
