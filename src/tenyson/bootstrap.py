from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping, Optional, Sequence


DEFAULT_LOCAL_BOOTSTRAP_PACKAGES: dict[str, str] = {
    "boto3": "boto3",
    "datasets": "datasets",
    "huggingface_hub": "huggingface_hub",
    "modal": "modal",
    "wandb": "wandb",
    "yaml": "pyyaml",
}

_ENV_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def is_truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_project_root(anchor_file: str | Path) -> Path:
    anchor = Path(anchor_file).resolve()
    for candidate in [anchor.parent, *anchor.parents]:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "tenyson"
        ).is_dir():
            return candidate
    raise RuntimeError(
        f"Could not resolve project root from anchor {anchor_file!r}. "
        "Expected a parent containing pyproject.toml and src/tenyson/."
    )


def ensure_src_on_path(*, project_root: str | Path) -> Path:
    src_dir = Path(project_root).resolve() / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir


def missing_controller_packages(
    packages: Mapping[str, str] | None = None,
) -> list[str]:
    package_map = packages or DEFAULT_LOCAL_BOOTSTRAP_PACKAGES
    missing = [
        package
        for module_name, package in package_map.items()
        if importlib.util.find_spec(module_name) is None
    ]
    return sorted(set(missing))


def load_env_file(
    path: str | Path,
    *,
    override: bool = False,
) -> dict[str, str]:
    """
    Load simple dotenv-style KEY=VALUE assignments into os.environ.

    Blank lines and `#` comments are ignored.
    Existing environment variables are preserved unless override=True.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        match = _ENV_ASSIGNMENT.match(line)
        if match is None:
            continue

        key, value = match.groups()
        parsed_value = value.strip()
        if (
            len(parsed_value) >= 2
            and parsed_value[0] in {"'", '"'}
            and parsed_value[-1] == parsed_value[0]
        ):
            parsed_value = parsed_value[1:-1]
        elif " #" in parsed_value:
            parsed_value = parsed_value.split(" #", 1)[0].rstrip()

        if not override and key in os.environ:
            continue
        os.environ[key] = parsed_value
        loaded[key] = parsed_value
    return loaded


def ensure_local_controller_environment(
    *,
    anchor_file: str | Path,
    packages: Mapping[str, str] | None = None,
    skip_env_var: str = "TENYSON_SKIP_LOCAL_BOOTSTRAP",
    python_executable: Optional[str] = None,
) -> Sequence[str]:
    """
    Prepare local controller runtime for example/experiment scripts.

    - Ensures `src/` is on `sys.path`.
    - Optionally installs missing lightweight controller-side dependencies.
    """
    project_root = resolve_project_root(anchor_file)
    ensure_src_on_path(project_root=project_root)

    if is_truthy(os.getenv(skip_env_var)):
        return []

    missing = missing_controller_packages(packages=packages)
    if not missing:
        return []

    print(
        "[TENYSON] Installing missing local dependencies: "
        + ", ".join(missing),
        flush=True,
    )
    install_cmd = [
        python_executable or sys.executable,
        "-m",
        "pip",
        "install",
        *missing,
    ]
    result = subprocess.run(install_cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Automatic local dependency bootstrap failed. "
            "Re-run after installing the missing packages manually."
        )
    return missing
