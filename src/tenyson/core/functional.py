from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Mapping

from tenyson.experiment import AdapterRef
from tenyson.loader import load_module_from_path, load_task_from_module


@dataclass(frozen=True)
class FunctionalManifest:
    task: Any
    seeds: Mapping[str, AdapterRef] = field(default_factory=dict)
    run_aliases: Mapping[str, str] = field(default_factory=dict)
    source_path: Path | None = None

    def resolve_seed(self, alias: str) -> AdapterRef:
        key = str(alias or "").strip()
        if not key:
            raise KeyError("Seed alias cannot be empty.")
        seed = self.seeds.get(key)
        if seed is None:
            available = sorted(self.seeds)
            raise KeyError(
                f'Unknown seed alias "{key}". Available seeds: {available}.'
            )
        return seed

    def resolve_run(self, run_name: str) -> str:
        key = str(run_name or "").strip()
        if not key:
            raise KeyError("Run name cannot be empty.")
        return str(self.run_aliases.get(key, key))


def load_functional_manifest(path: str | Path) -> FunctionalManifest:
    source_path = Path(path).resolve()
    source_dir = str(source_path.parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    module = load_module_from_path(str(source_path))
    task = load_task_from_module(module, source=str(source_path))
    run_aliases = _default_run_aliases(task)
    explicit_aliases = _read_run_aliases(module)
    run_aliases.update(explicit_aliases)
    return FunctionalManifest(
        task=task,
        seeds=_read_seeds(module),
        run_aliases=run_aliases,
        source_path=source_path,
    )


def _default_run_aliases(task: Any) -> dict[str, str]:
    aliases: dict[str, str] = {}
    environment_name = str(getattr(task, "get_environment_name", lambda: None)() or "").strip()
    prefix = f"{environment_name}_" if environment_name else ""
    for run_name in getattr(task, "list_named_runs", lambda _run_type=None: [])():
        canonical = str(run_name or "").strip()
        if not canonical:
            continue
        aliases.setdefault(canonical, canonical)
        short_name = canonical
        if prefix and canonical.startswith(prefix):
            short_name = canonical[len(prefix) :]
        aliases.setdefault(short_name, canonical)
    return aliases


def _read_run_aliases(module: Any) -> dict[str, str]:
    raw = getattr(module, "RUN_ALIASES", None)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("RUN_ALIASES must be a mapping of alias -> run name.")
    aliases: dict[str, str] = {}
    for alias, run_name in raw.items():
        alias_key = str(alias or "").strip()
        canonical = str(run_name or "").strip()
        if not alias_key or not canonical:
            raise ValueError("RUN_ALIASES entries must use non-empty strings.")
        aliases[alias_key] = canonical
    return aliases


def _read_seeds(module: Any) -> dict[str, AdapterRef]:
    raw = getattr(module, "SEEDS", None)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("SEEDS must be a mapping of alias -> adapter ref.")
    seeds: dict[str, AdapterRef] = {}
    for alias, value in raw.items():
        key = str(alias or "").strip()
        if not key:
            raise ValueError("SEEDS entries must use non-empty aliases.")
        seeds[key] = _coerce_adapter_ref(value, alias=key)
    return seeds


def _coerce_adapter_ref(value: Any, *, alias: str) -> AdapterRef:
    if isinstance(value, AdapterRef):
        return value
    if isinstance(value, Mapping):
        repo_id = str(value.get("repo_id") or "").strip()
        revision = str(value.get("revision") or "").strip()
        if repo_id and revision:
            artifact_type = value.get("artifact_type")
            return AdapterRef(
                repo_id=repo_id,
                revision=revision,
                artifact_type=(
                    str(artifact_type).strip()
                    if artifact_type is not None
                    else None
                ),
            )
    raise TypeError(
        f'Seed "{alias}" must be an AdapterRef or a mapping with repo_id and revision.'
    )
