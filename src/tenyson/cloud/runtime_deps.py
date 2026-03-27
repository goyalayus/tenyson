from __future__ import annotations

import shlex

# Keep this list in sync across cloud providers so AWS and Modal workers run
# with the same dependency surface.
REMOTE_RUNTIME_PACKAGES: tuple[str, ...] = (
    "unsloth",
    "vllm",
    "huggingface_hub",
    "pyyaml",
    "wandb",
)


def _render_python_pip_install(packages: tuple[str, ...]) -> str:
    return "python3 -m pip install " + " ".join(
        shlex.quote(pkg) for pkg in packages
    )


def _render_modal_t4_colab_compat_install() -> str:
    # Mirrors the T4 branch from the Unsloth notebook install flow.
    # We keep this path isolated to T4 Modal workers because it pins older
    # vLLM/triton versions.
    cleanup_packages = (
        "unsloth",
        "vllm",
        "triton",
        "torchvision",
        "bitsandbytes",
        "xformers",
        "transformers",
        "trl",
    )
    cleanup_cmd = (
        "python3 -m pip uninstall -y "
        + " ".join(shlex.quote(pkg) for pkg in cleanup_packages)
        + " || true"
    )
    return " && ".join(
        [
            "python3 -m pip install --upgrade -qqq uv",
            cleanup_cmd,
            (
                "uv pip install --system -qqq --upgrade "
                "vllm==0.9.2 numpy pillow torchvision bitsandbytes xformers unsloth"
            ),
            "uv pip install --system -qqq triton==3.2.0",
            "uv pip install --system transformers==4.56.2",
            "uv pip install --system --no-deps trl==0.22.2",
            "python3 -m pip install huggingface_hub pyyaml wandb",
        ]
    )


def runtime_pip_install_command(*, profile: str = "default") -> str:
    normalized = str(profile or "default").strip().lower()
    if normalized == "modal_t4_colab_compat":
        return _render_modal_t4_colab_compat_install()
    if normalized != "default":
        raise ValueError(f"Unknown runtime dependency profile: {profile!r}")
    return _render_python_pip_install(REMOTE_RUNTIME_PACKAGES)
