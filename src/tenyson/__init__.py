from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "SFTJob",
    "RLJob",
    "EvalJob",
    "JobResult",
    "AWSManager",
    "ModalManager",
    "CloudManager",
    "ensure_local_controller_environment",
    "AdapterRef",
    "ExperimentAborted",
    "ExperimentBranch",
    "ConfigTemplates",
    "ExperimentSession",
    "StageSpec",
    "ExperimentReport",
    "ReportBuilder",
    "load_config",
    "load_task",
    "load_task_from_spec",
    "run_pipeline",
    "run_experiment",
    "shared_overrides_from_env",
    "hub_chat_sft_dataset",
    "merge_config_overrides",
    "SFTDatasetTemplate",
    "RLDatasetTemplate",
    "RLRewardTemplate",
    "EvalDatasetTemplate",
    "EvalMetricsTemplate",
]

_EXPORTS = {
    "SFTJob": ("tenyson.jobs.sft", "SFTJob"),
    "RLJob": ("tenyson.jobs.rl", "RLJob"),
    "EvalJob": ("tenyson.jobs.eval", "EvalJob"),
    "JobResult": ("tenyson.jobs.result", "JobResult"),
    "AWSManager": ("tenyson.cloud.aws", "AWSManager"),
    "ModalManager": ("tenyson.cloud.modal", "ModalManager"),
    "CloudManager": ("tenyson.cloud.manager", "CloudManager"),
    "ensure_local_controller_environment": (
        "tenyson.bootstrap",
        "ensure_local_controller_environment",
    ),
    "AdapterRef": ("tenyson.experiment", "AdapterRef"),
    "ExperimentAborted": ("tenyson.experiment", "ExperimentAborted"),
    "ExperimentBranch": ("tenyson.experiment", "ExperimentBranch"),
    "ConfigTemplates": ("tenyson.experiment", "ConfigTemplates"),
    "ExperimentSession": ("tenyson.experiment", "ExperimentSession"),
    "StageSpec": ("tenyson.experiment", "StageSpec"),
    "ExperimentReport": ("tenyson.reporting.fixed", "ExperimentReport"),
    "ReportBuilder": ("tenyson.reporting.builder", "ReportBuilder"),
    "load_config": ("tenyson.loader", "load_config"),
    "load_task": ("tenyson.loader", "load_task"),
    "load_task_from_spec": ("tenyson.loader", "load_task_from_spec"),
    "run_pipeline": ("tenyson.pipeline", "run_pipeline"),
    "run_experiment": ("tenyson.core.experiment_runner", "run_experiment"),
    "shared_overrides_from_env": ("tenyson.core.run_config", "shared_overrides_from_env"),
    "hub_chat_sft_dataset": ("tenyson.core.chat_sft", "hub_chat_sft_dataset"),
    "merge_config_overrides": ("tenyson.core.environment", "merge_config_overrides"),
    "SFTDatasetTemplate": ("tenyson.core.stage_templates", "SFTDatasetTemplate"),
    "RLDatasetTemplate": ("tenyson.core.stage_templates", "RLDatasetTemplate"),
    "RLRewardTemplate": ("tenyson.core.stage_templates", "RLRewardTemplate"),
    "EvalDatasetTemplate": ("tenyson.core.stage_templates", "EvalDatasetTemplate"),
    "EvalMetricsTemplate": ("tenyson.core.stage_templates", "EvalMetricsTemplate"),
}

if TYPE_CHECKING:
    from tenyson.bootstrap import ensure_local_controller_environment
    from tenyson.cloud.aws import AWSManager
    from tenyson.cloud.manager import CloudManager
    from tenyson.cloud.modal import ModalManager
    from tenyson.experiment import (
        AdapterRef,
        ConfigTemplates,
        ExperimentAborted,
        ExperimentBranch,
        ExperimentSession,
        StageSpec,
    )
    from tenyson.jobs.eval import EvalJob
    from tenyson.jobs.result import JobResult
    from tenyson.jobs.rl import RLJob
    from tenyson.jobs.sft import SFTJob
    from tenyson.loader import load_config, load_task, load_task_from_spec
    from tenyson.pipeline import run_pipeline
    from tenyson.core.experiment_runner import run_experiment
    from tenyson.core.chat_sft import hub_chat_sft_dataset
    from tenyson.core.run_config import shared_overrides_from_env
    from tenyson.core.environment import merge_config_overrides
    from tenyson.core.stage_templates import (
        EvalDatasetTemplate,
        EvalMetricsTemplate,
        RLRewardTemplate,
        RLDatasetTemplate,
        SFTDatasetTemplate,
    )
    from tenyson.reporting.builder import ReportBuilder
    from tenyson.reporting.fixed import ExperimentReport


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'tenyson' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
