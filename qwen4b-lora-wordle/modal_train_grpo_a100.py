import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import modal

APP_NAME = "wordle-grpo-a100"
REPO_ROOT = str(Path(__file__).resolve().parents[1])
WORKDIR = "/workspace/wordle-prime-rl-reproduction/qwen4b-lora-wordle"
SCRIPT_PATH = "RL/unsloth/train_grpo_mixed_qwen4b_unsloth.py"
OUTPUT_VOLUME_NAME = "wordle-rl-outputs"
OUTPUT_ROOT = f"{WORKDIR}/outputs/RL"
SECRETS_NAME = "wordle-train-secrets"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential")
    .pip_install(
        "unsloth",
        "vllm",
        "wandb",
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
    gpu="A100",
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name(SECRETS_NAME)],
    volumes={OUTPUT_ROOT: output_volume},
)
def run_training(
    max_steps: int = 1000,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-6,
    num_generations: int = 4,
    seq_len: int = 4096,
    max_prompt_length: int = 2048,
    max_completion_length: int = 2048,
    max_output_tokens: int = 2048,
    wandb: bool = True,
    wandb_project: str = "wordle-rl-grpo",
    wandb_name: str = "",
    fast_inference: bool = True,
    extra_args: str = "",
):
    os.chdir(WORKDIR)

    run_name = wandb_name.strip() or f"grpo_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_modal_a100"
    cmd = [
        sys.executable,
        SCRIPT_PATH,
        "--max-steps",
        str(max_steps),
        "--per-device-batch-size",
        str(per_device_batch_size),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--learning-rate",
        str(learning_rate),
        "--num-generations",
        str(num_generations),
        "--seq-len",
        str(seq_len),
        "--max-prompt-length",
        str(max_prompt_length),
        "--max-completion-length",
        str(max_completion_length),
        "--max-output-tokens",
        str(max_output_tokens),
        "--run-name",
        run_name,
        "--output-root",
        OUTPUT_ROOT,
    ]

    if fast_inference:
        cmd.append("--fast-inference")
    if wandb:
        cmd.extend(
            [
                "--wandb",
                "--wandb-project",
                wandb_project,
                "--wandb-name",
                run_name,
            ]
        )
    if extra_args.strip():
        cmd.extend(shlex.split(extra_args))

    print("Starting command:", " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)
    output_volume.commit()
    return {"status": "ok", "run_name": run_name, "output_root": OUTPUT_ROOT}


@app.local_entrypoint()
def main(
    max_steps: int = 1000,
    wandb: bool = True,
    wandb_project: str = "wordle-rl-grpo",
    wandb_name: str = "",
    fast_inference: bool = True,
    extra_args: str = "",
):
    result = run_training.remote(
        max_steps=max_steps,
        wandb=wandb,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        fast_inference=fast_inference,
        extra_args=extra_args,
    )
    print(result)
