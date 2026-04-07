import modal


GPU = "A100-80GB"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands("python3 -m pip install --upgrade vllm unsloth")
)
app = modal.App("unsloth-fastinf-fullft-a100-plain-install-20260401")


@app.function(gpu=GPU, timeout=3600, image=image)
def probe():
    import importlib.metadata as metadata
    from unsloth import FastLanguageModel

    versions = {}
    for package_name in ("unsloth", "vllm", "transformers", "trl", "torch"):
        try:
            versions[package_name] = metadata.version(package_name)
        except Exception as exc:
            versions[package_name] = f"MISSING:{exc.__class__.__name__}"
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
    return {"versions": versions}


if __name__ == "__main__":
    with app.run():
        print(probe.remote())
