import os

import modal

from tenyson.cloud.runtime_deps import runtime_pip_install_command


APP_NAME = "arith-adapter-debug-probe"
ROOT = os.path.abspath(os.path.dirname(__file__))
HF_TOKEN = os.environ["HF_TOKEN"]

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands("apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*")
    .run_commands(runtime_pip_install_command(profile="default"))
    .add_local_dir(os.path.join(ROOT, "src"), remote_path="/workspace/src", copy=True)
    .add_local_dir(
        os.path.join(ROOT, "examples"),
        remote_path="/workspace/examples",
        copy=True,
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/workspace:/workspace/src"})
)
secret = modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN})


@app.function(image=image, gpu="A100", timeout=3600, secrets=[secret])
def probe(mode: str):
    import gc
    import json

    os.environ["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    from examples.arithmetic.functional import build_addition_messages, build_addition_prompt
    from tenyson.core.chat_generation import render_generation_prompt
    from tenyson.core.hf_adapter import (
        download_hf_lora_adapter,
        resolve_hf_lora_runtime_kwargs,
        strict_load_hf_lora_adapter_weights,
    )
    from tenyson.jobs.eval import EvalJob, _configure_eval_unsloth_runtime_env
    from tenyson.jobs.tokenizer_utils import normalize_tokenizer_special_tokens
    from unsloth import FastLanguageModel
    import torch

    prompt_text = build_addition_prompt(2, 51, 29)
    row = {"prompt": prompt_text, "messages": build_addition_messages(prompt_text)}
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
            "max_tokens": 16,
        },
        "evaluation": {"run_name": "probe", "batch_size": 1},
    }

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

    if mode == "base_fast":
        config = json.loads(json.dumps(base_config))
        config["model"]["fast_inference"] = True
        job = EvalJob(config=config, task=object())
        model, tokenizer = job._build_model_and_tokenizer()
        sampling_params = job._build_sampling_params(tokenizer)
        generation_prompt = render_generation_prompt(row=row, tokenizer=tokenizer, config=config)
        completion = job._generate_batch(model, tokenizer, [generation_prompt], sampling_params)[0]
        result = {
            "mode": mode,
            "prompt": prompt_text,
            "generation_prompt": generation_prompt,
            "completion": completion,
        }
        cleanup(model, tokenizer, job)
        return json.dumps(result)

    if mode == "sft_fast":
        config = json.loads(json.dumps(base_config))
        config["model"]["fast_inference"] = True
        config["model"]["init_adapter_repo"] = "goyalayus/arithmetic-2digit-sft_2digit_06b"
        config["model"]["init_adapter_revision"] = "ac6d760fbf5d28586a29fc4c7517670c99bc4dc2"
        job = EvalJob(config=config, task=object())
        model, tokenizer = job._build_model_and_tokenizer()
        sampling_params = job._build_sampling_params(tokenizer)
        generation_prompt = render_generation_prompt(row=row, tokenizer=tokenizer, config=config)
        completion = job._generate_batch(model, tokenizer, [generation_prompt], sampling_params)[0]
        result = {
            "mode": mode,
            "prompt": prompt_text,
            "generation_prompt": generation_prompt,
            "completion": completion,
        }
        cleanup(model, tokenizer, job)
        return json.dumps(result)

    if mode == "sft_slow":
        config = json.loads(json.dumps(base_config))
        config["model"]["fast_inference"] = False
        _configure_eval_unsloth_runtime_env(config["vllm"])
        adapter = download_hf_lora_adapter(
            repo_id="goyalayus/arithmetic-2digit-sft_2digit_06b",
            revision="ac6d760fbf5d28586a29fc4c7517670c99bc4dc2",
        )
        lora_kwargs = resolve_hf_lora_runtime_kwargs(
            adapter,
            expected_r=config["model"]["lora_r"],
            expected_alpha=config["model"]["lora_alpha"],
            expected_target_modules=config["model"]["lora_target_modules"],
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="Qwen/Qwen3-0.6B",
            max_seq_length=4096,
            load_in_4bit=False,
            revision=None,
            fast_inference=False,
        )
        normalize_tokenizer_special_tokens(tokenizer, padding_side="left")
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
        generation_prompt = render_generation_prompt(row=row, tokenizer=tokenizer, config=config)
        inputs = tokenizer(generation_prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is None:
            device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)
        result = {
            "mode": mode,
            "prompt": prompt_text,
            "generation_prompt": generation_prompt,
            "completion": decoded,
        }
        cleanup(model, tokenizer, inputs, outputs, new_tokens)
        return json.dumps(result)

    raise ValueError(mode)


@app.local_entrypoint()
def main():
    for mode in ["base_fast", "sft_fast", "sft_slow"]:
        print(f"RUNNING {mode}", flush=True)
        result = probe.remote(mode)
        print(f"RESULT {result}", flush=True)
