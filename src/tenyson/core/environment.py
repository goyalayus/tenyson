from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from datasets import Dataset

from tenyson.core.plugin import TaskPlugin
from tenyson.core.stage_templates import EvalMetricsBuilder, call_eval_metrics_builder


DatasetFactory = Callable[[Dict[str, Any], Any], Optional[Dataset]]
FormattingFactory = Callable[[Dict[str, Any], Any], Optional[Callable[..., Any]]]
CollatorFactory = Callable[[Dict[str, Any], Any], Optional[Any]]
RewardFactory = Callable[[Dict[str, Any], Any], List[Callable[..., Any]]]
MetricFactory = EvalMetricsBuilder
RunFamilyConfigFactory = Callable[[Any], Mapping[str, Any]]

_ENV_META_KEY = "_tenyson"
_ENV_RUN_NAME_KEY = "environment_run"


@dataclass(frozen=True)
class DatasetHooks:
    primary: Optional[DatasetFactory] = None
    evaluation: Optional[DatasetFactory] = None
    formatting: Optional[FormattingFactory] = None
    collator: Optional[CollatorFactory] = None


@dataclass(frozen=True)
class RubricHooks:
    reward_funcs: Optional[RewardFactory] = None
    compute_metrics: Optional[MetricFactory] = None


@dataclass(frozen=True)
class EnvironmentRunSpec:
    run_type: str
    datasets: DatasetHooks = field(default_factory=DatasetHooks)
    rubric: Optional[RubricHooks] = None
    env: Optional[Any] = None
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    variants: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def merged_config_overrides(self, variant: Optional[str] = None) -> Dict[str, Any]:
        merged = deepcopy(self.config_overrides)
        if variant is None:
            return merged
        if variant not in self.variants:
            raise KeyError(
                f'Unknown variant "{variant}" for run type "{self.run_type}".'
            )
        for key, value in self.variants[variant].items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                _deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged


@dataclass(frozen=True)
class EnvironmentDefinition:
    name: str
    runs: Mapping[str, EnvironmentRunSpec]

    def get_run_spec(self, run_name: str) -> EnvironmentRunSpec:
        key = str(run_name or "").strip()
        if key in self.runs:
            return self.runs[key]

        lowered = key.lower()
        for candidate_name, candidate_spec in self.runs.items():
            if str(candidate_name).strip().lower() == lowered:
                return candidate_spec
        raise KeyError(
            f'Environment "{self.name}" does not define run "{run_name}".'
        )

    def list_run_names(self, run_type: Optional[str] = None) -> Sequence[str]:
        if run_type is None:
            return sorted(self.runs.keys())
        selected = [
            run_name
            for run_name, spec in self.runs.items()
            if _normalize_run_type(spec.run_type) == _normalize_run_type(run_type)
        ]
        return sorted(selected)

    def get_named_run_type(self, run_name: str) -> str:
        return _normalize_run_type(self.get_run_spec(run_name).run_type)

    def resolve_run_spec(
        self,
        run_type: str,
        *,
        run_name: Optional[str] = None,
    ) -> EnvironmentRunSpec:
        expected_type = _normalize_run_type(run_type)
        if run_name:
            spec = self.get_run_spec(run_name)
            actual_type = _normalize_run_type(spec.run_type)
            if actual_type != expected_type:
                raise ValueError(
                    f'Environment run "{run_name}" has run_type "{actual_type}", '
                    f'expected "{expected_type}".'
                )
            return spec

        candidates = [
            spec
            for spec in self.runs.values()
            if _normalize_run_type(spec.run_type) == expected_type
        ]
        if not candidates:
            raise KeyError(
                f'Environment "{self.name}" does not define any run for "{expected_type}".'
            )
        if len(candidates) > 1:
            names = self.list_run_names(expected_type)
            raise ValueError(
                f'Environment "{self.name}" has multiple "{expected_type}" runs {names}. '
                "Select one by name in experiments.py."
            )
        return candidates[0]


def merge_config_overrides(*chunks: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for chunk in chunks:
        if not chunk:
            continue
        _deep_merge(merged, chunk)
    return merged


def build_run_family(
    *,
    prefix: str,
    run_type: str,
    values: Sequence[Any],
    datasets: DatasetHooks = DatasetHooks(),
    rubric: Optional[RubricHooks] = None,
    env: Optional[Any] = None,
    base_config_overrides: Optional[Mapping[str, Any]] = None,
    config_for_value: Optional[RunFamilyConfigFactory] = None,
    name_for_value: Optional[Callable[[Any], str]] = None,
    variants: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> Dict[str, EnvironmentRunSpec]:
    runs: Dict[str, EnvironmentRunSpec] = {}
    for value in values:
        value_name = (
            name_for_value(value)
            if name_for_value is not None
            else str(value)
        )
        run_name = f"{prefix}{value_name}"
        value_overrides = (
            config_for_value(value)
            if config_for_value is not None
            else {}
        )
        runs[run_name] = EnvironmentRunSpec(
            run_type=run_type,
            datasets=datasets,
            rubric=rubric,
            env=env,
            config_overrides=merge_config_overrides(
                base_config_overrides,
                value_overrides,
            ),
            variants=deepcopy(dict(variants or {})),
        )
    return runs


def bind_environment_run(config: Dict[str, Any], run_name: str) -> None:
    meta = config.setdefault(_ENV_META_KEY, {})
    meta[_ENV_RUN_NAME_KEY] = str(run_name)


def resolve_bound_environment_run(config: Mapping[str, Any]) -> Optional[str]:
    meta = config.get(_ENV_META_KEY, {}) if isinstance(config, Mapping) else {}
    if isinstance(meta, Mapping):
        selected = str(meta.get(_ENV_RUN_NAME_KEY) or "").strip()
        if selected:
            return selected

    # Legacy/fallback slot if task-level metadata is ever used.
    task_cfg = config.get("task", {}) if isinstance(config, Mapping) else {}
    if isinstance(task_cfg, Mapping):
        selected = str(task_cfg.get("_environment_run") or "").strip()
        if selected:
            return selected
    return None


def _normalize_run_type(value: str) -> str:
    return str(value or "").strip().lower()


def _deep_merge(base: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


class EnvironmentTaskAdapter(TaskPlugin):
    def __init__(self, environment: EnvironmentDefinition):
        self.environment = environment

    def get_environment_name(self) -> Optional[str]:
        return self.environment.name

    def list_named_runs(self, run_type: Optional[str] = None) -> Sequence[str]:
        return self.environment.list_run_names(run_type)

    def get_named_run_type(self, run_name: str) -> Optional[str]:
        return self.environment.get_named_run_type(run_name)

    def get_named_run_config_overrides(self, run_name: str) -> Optional[Mapping[str, Any]]:
        spec = self.environment.get_run_spec(run_name)
        return spec.merged_config_overrides()

    def get_run_config_overrides(
        self,
        run_type: str,
        *,
        variant: Optional[str] = None,
    ) -> Optional[Mapping[str, Any]]:
        spec = self.environment.resolve_run_spec(run_type)
        return spec.merged_config_overrides(variant=variant)

    def list_run_variants(self, run_type: str) -> Sequence[str]:
        variants: set[str] = set()
        for run_name in self.environment.list_run_names(run_type):
            spec = self.environment.get_run_spec(run_name)
            variants.update(spec.variants.keys())
        return sorted(variants)

    def _runtime_spec(self, run_type: str, config: Dict[str, Any]) -> EnvironmentRunSpec:
        selected_run = resolve_bound_environment_run(config)
        return self.environment.resolve_run_spec(run_type, run_name=selected_run)

    def get_sft_dataset(self, config: Dict[str, Any], tokenizer: Any) -> Dataset:
        spec = self._runtime_spec("sft", config)
        if spec.datasets.primary is None:
            raise ValueError(
                f'Environment "{self.environment.name}" does not define an SFT dataset.'
            )
        dataset = spec.datasets.primary(config, tokenizer)
        if dataset is None:
            raise ValueError(
                f'Environment "{self.environment.name}" returned no SFT dataset.'
            )
        return dataset

    def get_sft_eval_dataset(
        self,
        config: Dict[str, Any],
        tokenizer: Any,
    ) -> Optional[Dataset]:
        spec = self._runtime_spec("sft", config)
        if spec.datasets.evaluation is None:
            return None
        return spec.datasets.evaluation(config, tokenizer)

    def get_sft_formatting_func(self, config: Dict[str, Any], tokenizer: Any):
        spec = self._runtime_spec("sft", config)
        if spec.datasets.formatting is None:
            return None
        return spec.datasets.formatting(config, tokenizer)

    def get_sft_data_collator(self, config: Dict[str, Any], tokenizer: Any) -> Optional[Any]:
        spec = self._runtime_spec("sft", config)
        if spec.datasets.collator is None:
            return None
        return spec.datasets.collator(config, tokenizer)

    def get_rl_dataset(self, config: Dict[str, Any]) -> Dataset:
        spec = self._runtime_spec("rl", config)
        if spec.datasets.primary is None:
            raise ValueError(
                f'Environment "{self.environment.name}" does not define an RL dataset.'
            )
        dataset = spec.datasets.primary(config, None)
        if dataset is None:
            raise ValueError(
                f'Environment "{self.environment.name}" returned no RL dataset.'
            )
        return dataset

    def get_reward_funcs(self, config: Dict[str, Any], tokenizer: Any) -> List[Callable]:
        spec = self._runtime_spec("rl", config)
        if spec.rubric is None or spec.rubric.reward_funcs is None:
            raise ValueError(
                f'Environment "{self.environment.name}" does not define RL reward functions.'
            )
        return spec.rubric.reward_funcs(config, tokenizer)

    def get_eval_dataset(self, config: Dict[str, Any]) -> Dataset:
        spec = self._runtime_spec("eval", config)
        if spec.datasets.primary is None:
            raise ValueError(
                f'Environment "{self.environment.name}" does not define an eval dataset.'
            )
        dataset = spec.datasets.primary(config, None)
        if dataset is None:
            raise ValueError(
                f'Environment "{self.environment.name}" returned no eval dataset.'
            )
        return dataset

    def compute_metrics(
        self,
        prompts: List[str],
        completions: List[str],
        dataset_rows: Dataset,
        config: Dict[str, Any],
        tokenizer: Any,
    ) -> Dict[str, Any]:
        spec = self._runtime_spec("eval", config)
        if spec.rubric is None or spec.rubric.compute_metrics is None:
            raise ValueError(
                f'Environment "{self.environment.name}" does not define eval metrics.'
            )
        return call_eval_metrics_builder(
            spec.rubric.compute_metrics,
            prompts,
            completions,
            dataset_rows,
            config,
            tokenizer,
        )
