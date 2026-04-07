import modal


GPU = "T4"


def install_command_for_gpu(gpu: str) -> str:
    normalized_gpu = str(gpu).strip().upper()
    if normalized_gpu == "T4":
        return " && ".join(
            [
                "python3 -m pip install --upgrade -qqq uv",
                (
                    "python3 -m pip uninstall -y "
                    "unsloth vllm triton torchvision bitsandbytes xformers "
                    "transformers trl || true"
                ),
                (
                    "python3 -m uv pip install --system -qqq --upgrade "
                    "vllm==0.9.2 numpy pillow torchvision bitsandbytes "
                    "xformers unsloth"
                ),
                "python3 -m uv pip install --system -qqq triton==3.2.0",
                "python3 -m uv pip install --system transformers==4.56.2",
                "python3 -m uv pip install --system --no-deps trl==0.22.2",
                "python3 -m pip install huggingface_hub pyyaml wandb",
            ]
        )
    return "python3 -m pip install unsloth vllm huggingface_hub pyyaml wandb"


install_cmd = install_command_for_gpu(GPU)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(install_cmd)
)
app = modal.App("unsloth-fastinf-fullft-probe-t4-same-script-20260331")


@app.function(gpu=GPU, timeout=3600, image=image)
def probe():
    import importlib.metadata as m
    from unsloth import FastLanguageModel

    versions = {}
    for pkg in ("unsloth", "vllm", "transformers", "trl", "torch"):
        try:
            versions[pkg] = m.version(pkg)
        except Exception as exc:
            versions[pkg] = f"MISSING:{exc.__class__.__name__}"
    print("VERSIONS", versions, flush=True)
    print("ABOUT_TO_LOAD", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-4B",
        max_seq_length=1024,
        load_in_4bit=False,
        load_in_8bit=False,
        fast_inference=True,
        full_finetuning=True,
        gpu_memory_utilization=0.5,
        trust_remote_code=True,
    )
    print("LOAD_OK", type(model).__name__, type(tokenizer).__name__, flush=True)
    return {
        "versions": versions,
        "model_type": type(model).__name__,
        "tokenizer_type": type(tokenizer).__name__,
    }


if __name__ == "__main__":
    with app.run():
        result = probe.remote()
        print("RESULT", result)
