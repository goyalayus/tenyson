"""
Shared loaders for config (YAML/JSON) and TaskPlugin (from file path or module:Class spec).
"""

import importlib
import importlib.util
import hashlib
import json
import os
import sys
from types import ModuleType
from typing import Any, Dict, Optional

import yaml

from tenyson.core.environment import EnvironmentDefinition, EnvironmentTaskAdapter
from tenyson.core.plugin import TaskPlugin, TemplateTaskPlugin


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a config from a YAML or JSON file. Format is inferred from the path extension.
    """
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def load_module_from_path(path: str) -> ModuleType:
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Task file not found: {path}")

    module_hash = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()[:12]
    module_stem = os.path.splitext(os.path.basename(abs_path))[0]
    module_name = f"tenyson_task_{module_stem}_{module_hash}"
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load task module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_environment_definition_from_module(module: ModuleType) -> EnvironmentDefinition | None:
    environment = getattr(module, "ENVIRONMENT", None)
    if isinstance(environment, EnvironmentDefinition):
        return environment

    factory = getattr(module, "load_environment_definition", None)
    if callable(factory):
        loaded = factory()
        if not isinstance(loaded, EnvironmentDefinition):
            raise TypeError(
                "load_environment_definition() must return an EnvironmentDefinition."
            )
        return loaded
    return None


def _load_task_instance_from_module(module: ModuleType) -> TaskPlugin | None:
    task = getattr(module, "TASK", None)
    if task is None:
        return _maybe_build_template_task_from_module_path(module)
    if not isinstance(task, TaskPlugin):
        raise TypeError("TASK must be a TaskPlugin instance.")
    return task


def _maybe_build_template_task_from_module_path(module: ModuleType) -> TaskPlugin | None:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return None
    module_path = os.path.abspath(str(module_file))
    module_stem = os.path.splitext(os.path.basename(module_path))[0].strip().lower()
    if module_stem != "functional":
        return None
    environment_name = os.path.basename(os.path.dirname(module_path)).strip()
    if not environment_name:
        return None
    return TemplateTaskPlugin(environment_name=environment_name)


def load_task_from_module(
    module: ModuleType,
    *,
    source: Optional[str] = None,
) -> TaskPlugin:
    environment = _load_environment_definition_from_module(module)
    if environment is not None:
        adapter = EnvironmentTaskAdapter(environment)
        setattr(adapter, "__tenyson_source_path__", getattr(module, "__file__", None))
        setattr(adapter, "__tenyson_source_module__", module.__name__)
        return adapter

    task_instance = _load_task_instance_from_module(module)
    if task_instance is not None:
        setattr(task_instance, "__tenyson_source_path__", getattr(module, "__file__", None))
        setattr(task_instance, "__tenyson_source_module__", module.__name__)
        return task_instance

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
            f"Expected exactly one TaskPlugin subclass in {source or module.__name__}, found none."
        )
    if len(candidates) > 1:
        names = [c.__name__ for c in candidates]
        raise ValueError(
            f"Expected exactly one TaskPlugin subclass in {source or module.__name__}, found: {names}"
        )
    task = candidates[0]()
    setattr(task, "__tenyson_source_path__", getattr(module, "__file__", None))
    setattr(task, "__tenyson_source_module__", module.__name__)
    return task


def load_environment_definition(path: str) -> EnvironmentDefinition:
    """
    Load an EnvironmentDefinition from a Python file path.
    """
    module = load_module_from_path(path)
    environment = _load_environment_definition_from_module(module)
    if environment is None:
        raise ValueError(f"Expected ENVIRONMENT in {path}, found none.")
    return environment


def load_task(path: str) -> TaskPlugin:
    """
    Load either:
    - an EnvironmentDefinition (wrapped in an EnvironmentTaskAdapter), or
    - exactly one concrete TaskPlugin subclass.
    """
    module = load_module_from_path(path)
    return load_task_from_module(module, source=path)


def load_task_from_spec(spec: str) -> TaskPlugin:
    """
    Load a TaskPlugin from a module:ClassName spec (e.g. for remote runner use).
    """
    if ":" not in spec:
        module = importlib.import_module(spec)
        return load_task_from_module(module, source=spec)
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    task_cls = getattr(module, class_name)
    return task_cls()
