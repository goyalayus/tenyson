from __future__ import annotations

import shlex

# Keep this list in sync across cloud providers so AWS and Modal workers run
# with the same dependency surface.
REMOTE_RUNTIME_PACKAGES: tuple[str, ...] = (
    "unsloth",
    "vllm",
    "huggingface_hub",
    "pyyaml",
    "sqlalchemy",
    "psycopg[binary]",
    "wandb",
)


def runtime_pip_install_command() -> str:
    return "python3 -m pip install " + " ".join(
        shlex.quote(pkg) for pkg in REMOTE_RUNTIME_PACKAGES
    )
