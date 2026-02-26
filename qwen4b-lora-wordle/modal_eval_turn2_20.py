import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "wordle-eval-turn2-20"
REPO_ROOT = str(Path(__file__).resolve().parents[1])
WORKDIR = "/workspace/wordle-prime-rl-reproduction/qwen4b-lora-wordle"
SCRIPT_PATH = "inference/eval_constraint_accuracy_turn2_20.py"
SECRETS_NAME = "wordle-train-secrets"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "huggingface_hub",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/workspace/wordle-prime-rl-reproduction",
    )
)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name(SECRETS_NAME)],
)
def run_eval(
    adapter_repo: str = "goyalayus/wordle-qwen3-4b-rl-turn2",
    adapter_revision: str = "rl-final",
    max_new_tokens: int = 180,
    num_samples: int = 1,
    do_sample: bool = False,
):
    os.chdir(WORKDIR)
    out_path = f"/tmp/turn2_eval_{adapter_revision}.json"
    cmd = [
        sys.executable,
        SCRIPT_PATH,
        "--adapter-repo",
        adapter_repo,
        "--adapter-revision",
        adapter_revision,
        "--max-new-tokens",
        str(max_new_tokens),
        "--num-samples",
        str(num_samples),
        "--out",
        out_path,
    ]
    if do_sample:
        cmd.append("--do-sample")

    print("Running:", " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)
    with open(out_path, "r", encoding="utf-8") as f:
        return f.read()


@app.local_entrypoint()
def main(
    adapter_repo: str = "goyalayus/wordle-qwen3-4b-rl-turn2",
    adapter_revision: str = "rl-final",
    max_new_tokens: int = 180,
    num_samples: int = 1,
    do_sample: bool = False,
):
    result_json = run_eval.remote(
        adapter_repo=adapter_repo,
        adapter_revision=adapter_revision,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
        do_sample=do_sample,
    )
    print(result_json)
