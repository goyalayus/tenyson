import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "wordle-lora-sft"
WORKDIR = "/workspace/wordle-prime-rl-reproduction/qwen4b-lora-wordle"
OUTPUT_ROOT = f"{WORKDIR}/outputs"
OUTPUT_VOLUME_NAME = "wordle-lora-outputs"
SECRETS_NAME = "wordle-train-secrets"
REPO_ROOT = str(Path(__file__).resolve().parents[1])

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential")
    .pip_install(
        "unsloth",
        "wandb",
        "torch>=2.1.0",
        "transformers>=4.56.2",
        "trl>=0.22.2",
        "datasets>=2.18.0",
        "peft",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
        "python-dotenv>=1.0.0",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/workspace/wordle-prime-rl-reproduction",
    )
)

output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name(SECRETS_NAME)],
    volumes={OUTPUT_ROOT: output_volume},
)
def run_training(
    hf_repo_id: str,
    preset: str = "qwen4b",
    max_steps: int = 3000,
    wandb: bool = True,
    wandb_name: str = "qwen4b-modal-retrain",
    extra_args: str = "",
):
    os.chdir(WORKDIR)

    cmd = [
        sys.executable,
        "train_sft_lora_qwen4b.py",
        "--preset",
        preset,
        "--hf-repo-id",
        hf_repo_id,
        "--max-steps",
        str(max_steps),
        "--output-root",
        OUTPUT_ROOT,
    ]

    if wandb:
        cmd.extend(["--wandb", "--wandb-name", wandb_name])

    if extra_args.strip():
        cmd.extend(shlex.split(extra_args))

    print("Starting command:", " ".join(shlex.quote(part) for part in cmd), flush=True)
    print(f"Using output root: {OUTPUT_ROOT}", flush=True)
    print(f"Using Modal volume: {OUTPUT_VOLUME_NAME}", flush=True)
    print(f"Using secret: {SECRETS_NAME}", flush=True)

    subprocess.run(cmd, check=True)

    output_volume.commit()
    return {"status": "ok", "output_root": OUTPUT_ROOT}


@app.local_entrypoint()
def main(
    hf_repo_id: str = "goyalayus/wordle-full-qwen4b",
    preset: str = "qwen4b",
    max_steps: int = 3000,
    wandb: bool = True,
    wandb_name: str = "qwen4b-modal-retrain",
    extra_args: str = "",
):
    result = run_training.remote(
        hf_repo_id=hf_repo_id,
        preset=preset,
        max_steps=max_steps,
        wandb=wandb,
        wandb_name=wandb_name,
        extra_args=extra_args,
    )
    print(result)
