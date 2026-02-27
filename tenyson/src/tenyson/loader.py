"""
Shared loaders for config (YAML/JSON) and TaskPlugin (from file path or module:Class spec).
"""

import importlib
import importlib.util
import json
import os
from typing import Any, Dict

import yaml

from tenyson.core.plugin import TaskPlugin


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a config from a YAML or JSON file. Format is inferred from the path extension.
    """
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def load_task(path: str) -> TaskPlugin:
    """
    Load a TaskPlugin from a Python file path. The file must define exactly one
    concrete subclass of TaskPlugin; that class is instantiated and returned.
    """
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Task file not found: {path}")

    module_name = os.path.splitext(os.path.basename(abs_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load task module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    candidates = []
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, TaskPlugin)
            and obj is not TaskPlugin
        ):
            candidates.append(obj)

    if len(candidates) == 0:
        raise ValueError(
            f"Expected exactly one TaskPlugin subclass in {path}, found none."
        )
    if len(candidates) > 1:
        names = [c.__name__ for c in candidates]
        raise ValueError(
            f"Expected exactly one TaskPlugin subclass in {path}, found: {names}"
        )
    return candidates[0]()


def load_task_from_spec(spec: str) -> TaskPlugin:
    """
    Load a TaskPlugin from a module:ClassName spec (e.g. for remote runner use).
    """
    if ":" not in spec:
        raise ValueError(
            f"task spec must be of form 'module.path:ClassName', got: {spec}"
        )
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    task_cls = getattr(module, class_name)
    return task_cls()
