from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from datasets import Dataset

from tenyson.core.plugin import TaskPlugin


SFTTrainDatasetBuilder = Callable[[Dict[str, Any], Any], Dataset]
SFTEvalDatasetBuilder = Callable[[Dict[str, Any], Any], Optional[Dataset]]
SFTFormattingBuilder = Callable[[Dict[str, Any], Any], Optional[Callable[..., Any]]]
SFTCollatorBuilder = Callable[[Dict[str, Any], Any], Optional[Any]]
RLDatasetBuilder = Callable[[Dict[str, Any]], Dataset]
RLRewardBuilder = Callable[[Dict[str, Any], Any], List[Callable[..., Any]]]
EvalDatasetBuilder = Callable[[Dict[str, Any]], Dataset]
EvalMetricsBuilder = Callable[
    [List[str], List[str], Dataset, Dict[str, Any], Any],
    Dict[str, Any],
]


@dataclass(frozen=True)
class SFTDatasetTemplate:
    train: SFTTrainDatasetBuilder
    evaluation: Optional[SFTEvalDatasetBuilder] = None
    formatting: Optional[SFTFormattingBuilder] = None
    collator: Optional[SFTCollatorBuilder] = None


@dataclass(frozen=True)
class RLDatasetTemplate:
    build: RLDatasetBuilder


@dataclass(frozen=True)
class RLRewardTemplate:
    build: RLRewardBuilder


@dataclass(frozen=True)
class EvalDatasetTemplate:
    build: EvalDatasetBuilder


@dataclass(frozen=True)
class EvalMetricsTemplate:
    compute: EvalMetricsBuilder


def has_explicit_stage_templates(
    *,
    sft_dataset: Optional[SFTDatasetTemplate] = None,
    rl_dataset: Optional[RLDatasetTemplate] = None,
    rl_reward: Optional[RLRewardTemplate] = None,
    eval_dataset: Optional[EvalDatasetTemplate] = None,
    eval_metrics: Optional[EvalMetricsTemplate] = None,
) -> bool:
    return any(
        template is not None
        for template in (
            sft_dataset,
            rl_dataset,
            rl_reward,
            eval_dataset,
            eval_metrics,
        )
    )


def bind_stage_templates(
    base_task: TaskPlugin,
    *,
    sft_dataset: Optional[SFTDatasetTemplate] = None,
    rl_dataset: Optional[RLDatasetTemplate] = None,
    rl_reward: Optional[RLRewardTemplate] = None,
    eval_dataset: Optional[EvalDatasetTemplate] = None,
    eval_metrics: Optional[EvalMetricsTemplate] = None,
) -> TaskPlugin:
    if not has_explicit_stage_templates(
        sft_dataset=sft_dataset,
        rl_dataset=rl_dataset,
        rl_reward=rl_reward,
        eval_dataset=eval_dataset,
        eval_metrics=eval_metrics,
    ):
        return base_task

    return _StageTemplateTaskAdapter(
        base_task=base_task,
        sft_dataset=sft_dataset,
        rl_dataset=rl_dataset,
        rl_reward=rl_reward,
        eval_dataset=eval_dataset,
        eval_metrics=eval_metrics,
    )


class _StageTemplateTaskAdapter(TaskPlugin):
    def __init__(
        self,
        *,
        base_task: TaskPlugin,
        sft_dataset: Optional[SFTDatasetTemplate],
        rl_dataset: Optional[RLDatasetTemplate],
        rl_reward: Optional[RLRewardTemplate],
        eval_dataset: Optional[EvalDatasetTemplate],
        eval_metrics: Optional[EvalMetricsTemplate],
    ) -> None:
        self._base_task = base_task
        self._sft_dataset = _require_instance(
            sft_dataset,
            SFTDatasetTemplate,
            name="sft dataset template",
            allow_none=True,
        )
        self._rl_dataset = _require_instance(
            rl_dataset,
            RLDatasetTemplate,
            name="rl dataset template",
            allow_none=True,
        )
        self._rl_reward = _require_instance(
            rl_reward,
            RLRewardTemplate,
            name="rl reward template",
            allow_none=True,
        )
        self._eval_dataset = _require_instance(
            eval_dataset,
            EvalDatasetTemplate,
            name="eval dataset template",
            allow_none=True,
        )
        self._eval_metrics = _require_instance(
            eval_metrics,
            EvalMetricsTemplate,
            name="eval metrics template",
            allow_none=True,
        )

    def get_environment_name(self) -> Optional[str]:
        return self._base_task.get_environment_name()

    def list_named_runs(self, run_type: Optional[str] = None) -> Sequence[str]:
        return self._base_task.list_named_runs(run_type)

    def get_named_run_type(self, run_name: str) -> Optional[str]:
        return self._base_task.get_named_run_type(run_name)

    def get_named_run_config_overrides(
        self, run_name: str
    ) -> Optional[Mapping[str, Any]]:
        return self._base_task.get_named_run_config_overrides(run_name)

    def get_run_config_overrides(
        self,
        run_type: str,
        *,
        variant: Optional[str] = None,
    ) -> Optional[Mapping[str, Any]]:
        return self._base_task.get_run_config_overrides(run_type, variant=variant)

    def list_run_variants(self, run_type: str) -> Sequence[str]:
        return self._base_task.list_run_variants(run_type)

    def get_sft_dataset(self, config: Dict[str, Any], tokenizer: Any) -> Dataset:
        if self._sft_dataset is None:
            return self._base_task.get_sft_dataset(config, tokenizer)
        dataset = self._sft_dataset.train(config, tokenizer)
        return _require_dataset(dataset, label="SFT train dataset")

    def get_sft_eval_dataset(
        self,
        config: Dict[str, Any],
        tokenizer: Any,
    ) -> Optional[Dataset]:
        if self._sft_dataset is None or self._sft_dataset.evaluation is None:
            return self._base_task.get_sft_eval_dataset(config, tokenizer)
        dataset = self._sft_dataset.evaluation(config, tokenizer)
        return _require_optional_dataset(dataset, label="SFT eval dataset")

    def get_sft_formatting_func(self, config: Dict[str, Any], tokenizer: Any) -> Optional[Callable]:
        if self._sft_dataset is None or self._sft_dataset.formatting is None:
            return self._base_task.get_sft_formatting_func(config, tokenizer)
        formatting = self._sft_dataset.formatting(config, tokenizer)
        if formatting is not None and not callable(formatting):
            raise TypeError(
                "SFT formatting template must return a callable or None."
            )
        return formatting

    def get_sft_data_collator(self, config: Dict[str, Any], tokenizer: Any) -> Optional[Any]:
        if self._sft_dataset is None or self._sft_dataset.collator is None:
            return self._base_task.get_sft_data_collator(config, tokenizer)
        return self._sft_dataset.collator(config, tokenizer)

    def get_rl_dataset(self, config: Dict[str, Any]) -> Dataset:
        if self._rl_dataset is None:
            return self._base_task.get_rl_dataset(config)
        dataset = self._rl_dataset.build(config)
        return _require_dataset(dataset, label="RL dataset")

    def get_reward_funcs(self, config: Dict[str, Any], tokenizer: Any) -> List[Callable]:
        if self._rl_reward is None:
            return self._base_task.get_reward_funcs(config, tokenizer)
        reward_funcs = self._rl_reward.build(config, tokenizer)
        if not isinstance(reward_funcs, list) or not reward_funcs:
            raise TypeError(
                "RL reward template must return a non-empty list of callables."
            )
        for index, reward_func in enumerate(reward_funcs):
            if not callable(reward_func):
                raise TypeError(
                    f"RL reward template returned a non-callable at index {index}."
                )
        return reward_funcs

    def get_rl_callbacks(self, config: Dict[str, Any], tokenizer: Any, output_dir: str) -> List[Any]:
        return self._base_task.get_rl_callbacks(config, tokenizer, output_dir)

    def get_eval_dataset(self, config: Dict[str, Any]) -> Dataset:
        if self._eval_dataset is None:
            return self._base_task.get_eval_dataset(config)
        dataset = self._eval_dataset.build(config)
        return _require_dataset(dataset, label="eval dataset")

    def compute_metrics(
        self,
        prompts: List[str],
        completions: List[str],
        dataset_rows: Dataset,
        config: Dict[str, Any],
        tokenizer: Any,
    ) -> Dict[str, Any]:
        if self._eval_metrics is None:
            return self._base_task.compute_metrics(
                prompts,
                completions,
                dataset_rows,
                config,
                tokenizer,
            )
        metrics = self._eval_metrics.compute(
            prompts,
            completions,
            dataset_rows,
            config,
            tokenizer,
        )
        if not isinstance(metrics, dict):
            raise TypeError(
                "Eval metrics template must return a dict payload."
            )
        return metrics


def _require_instance(
    value: Any,
    expected_type: type,
    *,
    name: str,
    allow_none: bool,
) -> Any:
    if value is None and allow_none:
        return None
    if isinstance(value, expected_type):
        return value
    expected_name = expected_type.__name__
    actual_name = type(value).__name__
    raise TypeError(f"{name} must be a {expected_name}, got {actual_name}.")


def _require_dataset(dataset: Any, *, label: str) -> Dataset:
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"{label} template must return datasets.Dataset, got {type(dataset).__name__}."
        )
    return dataset


def _require_optional_dataset(dataset: Any, *, label: str) -> Optional[Dataset]:
    if dataset is None:
        return None
    return _require_dataset(dataset, label=label)
