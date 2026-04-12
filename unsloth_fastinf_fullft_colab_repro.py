#!/usr/bin/env python3
"""
Standalone repro for Unsloth + vLLM + full finetuning.

Usage in Colab or any GPU box:

    python unsloth_fastinf_fullft_colab_repro.py

The script:
1. Detects the GPU.
2. Installs a T4-specific stack or, on non-T4 GPUs, only `vllm` and `unsloth`.
3. Re-execs itself once after install.
4. Runs FastLanguageModel.from_pretrained(...) with:
   - fast_inference=True
   - full_finetuning=True

This has no Tenyson dependency.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import os
from pathlib import Path
import subprocess
import sys
import traceback


SCRIPT_PATH = Path(__file__).resolve()
REEXEC_MARKER = "--skip-install"
MODEL_NAME = "Qwen/Qwen3-4B"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def detect_gpu_name() -> str:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        first_line = output.splitlines()[0].strip()
        return first_line or "UNKNOWN"
    except Exception as exc:
        return f"UNKNOWN ({exc.__class__.__name__})"


def install_for_gpu(gpu_name: str) -> None:
    is_t4 = "T4" in gpu_name.upper()
    print(f"Detected GPU: {gpu_name}", flush=True)

    if is_t4:
        run([sys.executable, "-m", "pip", "install", "--upgrade", "-qqq", "uv"])
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "unsloth",
                "vllm",
                "triton",
                "torchvision",
                "bitsandbytes",
                "xformers",
                "transformers",
                "trl",
            ]
        )
        run(
            [
                sys.executable,
                "-m",
                "uv",
                "pip",
                "install",
                "--system",
                "-qqq",
                "--upgrade",
                "vllm==0.9.2",
                "numpy",
                "pillow",
                "torchvision",
                "bitsandbytes",
                "xformers",
                "unsloth",
            ]
        )
        run(
            [
                sys.executable,
                "-m",
                "uv",
                "pip",
                "install",
                "--system",
                "-qqq",
                "triton==3.2.0",
            ]
        )
        run(
            [
                sys.executable,
                "-m",
                "uv",
                "pip",
                "install",
                "--system",
                "transformers==4.56.2",
            ]
        )
        run(
            [
                sys.executable,
                "-m",
                "uv",
                "pip",
                "install",
                "--system",
                "--no-deps",
                "trl==0.22.2",
            ]
        )
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "huggingface_hub",
            ]
        )
        return

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "vllm",
            "unsloth",
        ]
    )


def reexec_after_install() -> None:
    os.execv(
        sys.executable,
        [sys.executable, str(SCRIPT_PATH), REEXEC_MARKER],
    )


def installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in ("unsloth", "vllm", "transformers", "trl", "torch"):
        try:
            versions[package_name] = metadata.version(package_name)
        except Exception as exc:
            versions[package_name] = f"MISSING:{exc.__class__.__name__}"
    return versions


def run_repro() -> None:
    gpu_name = detect_gpu_name()
    versions = installed_versions()

    print("GPU", gpu_name, flush=True)
    print("VERSIONS", versions, flush=True)
    print("ABOUT_TO_IMPORT_UNSLOTH", flush=True)

    importlib.invalidate_caches()
    from unsloth import FastLanguageModel

    print("ABOUT_TO_LOAD", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=1024,
        load_in_4bit=False,
        load_in_8bit=False,
        fast_inference=True,
        full_finetuning=True,
        gpu_memory_utilization=0.5,
        trust_remote_code=True,
    )
    print("LOAD_OK", type(model).__name__, type(tokenizer).__name__, flush=True)


def main() -> int:
    if REEXEC_MARKER not in sys.argv:
        install_for_gpu(detect_gpu_name())
        reexec_after_install()

    try:
        run_repro()
        return 0
    except Exception:
        print("REPRO_EXCEPTION_START", flush=True)
        traceback.print_exc()
        print("REPRO_EXCEPTION_END", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
