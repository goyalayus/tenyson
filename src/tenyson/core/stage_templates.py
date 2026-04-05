from __future__ import annotations

from dataclasses import dataclass, field, replace
import functools
import importlib
import inspect
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
STAGE_TEMPLATE_CONFIG_KEY = "_tenyson_stage_templates"


@dataclass(frozen=True)
class TemplateFactoryRef:
    module: str
    factory: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def as_payload(self) -> Dict[str, Any]:
        return {
            "module": str(self.module).strip(),
            "factory": str(self.factory).strip(),
            "kwargs": dict(self.kwargs),
        }


@dataclass(frozen=True)
class SFTDatasetTemplate:
    train: SFTTrainDatasetBuilder
    evaluation: Optional[SFTEvalDatasetBuilder] = None
    formatting: Optional[SFTFormattingBuilder] = None
    collator: Optional[SFTCollatorBuilder] = None
    factory_ref: Optional[TemplateFactoryRef] = None


@dataclass(frozen=True)
class RLDatasetTemplate:
    build: RLDatasetBuilder
    factory_ref: Optional[TemplateFactoryRef] = None


@dataclass(frozen=True)
class RLRewardTemplate:
    build: RLRewardBuilder
    factory_ref: Optional[TemplateFactoryRef] = None


@dataclass(frozen=True)
class EvalDatasetTemplate:
    build: EvalDatasetBuilder
    factory_ref: Optional[TemplateFactoryRef] = None


@dataclass(frozen=True)
class EvalMetricsTemplate:
    compute: EvalMetricsBuilder
    factory_ref: Optional[TemplateFactoryRef] = None


def template_factory_ref(
    module: str,
    factory: str,
    **kwargs: Any,
) -> TemplateFactoryRef:
    return TemplateFactoryRef(
        module=str(module).strip(),
        factory=str(factory).strip(),
        kwargs=dict(kwargs),
    )


def sft_dataset_template(
    factory: Callable[..., SFTDatasetTemplate],
) -> Callable[..., SFTDatasetTemplate]:
    return _decorate_template_factory(
        factory,
        expected_type=SFTDatasetTemplate,
        label="SFT dataset template",
    )


def rl_dataset_template(
    factory: Callable[..., RLDatasetTemplate],
) -> Callable[..., RLDatasetTemplate]:
    return _decorate_template_factory(
        factory,
        expected_type=RLDatasetTemplate,
        label="RL dataset template",
    )


def rl_reward_template(
    factory: Callable[..., RLRewardTemplate],
) -> Callable[..., RLRewardTemplate]:
    return _decorate_template_factory(
        factory,
        expected_type=RLRewardTemplate,
        label="RL reward template",
    )


def eval_dataset_template(
    factory: Callable[..., EvalDatasetTemplate],
) -> Callable[..., EvalDatasetTemplate]:
    return _decorate_template_factory(
        factory,
        expected_type=EvalDatasetTemplate,
        label="Eval dataset template",
    )


def eval_metrics_template(
    factory: Callable[..., EvalMetricsTemplate],
) -> Callable[..., EvalMetricsTemplate]:
    return _decorate_template_factory(
        factory,
        expected_type=EvalMetricsTemplate,
        label="Eval metrics template",
    )


def _decorate_template_factory(
    factory: Callable[..., Any],
    *,
    expected_type: type,
    label: str,
) -> Callable[..., Any]:
    signature = inspect.signature(factory)
    _validate_template_factory_signature(
        signature,
        factory=factory,
        label=label,
    )

    @functools.wraps(factory)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        template = factory(*args, **kwargs)
        validated_template = _require_instance(
            template,
            expected_type,
            name=label,
            allow_none=False,
        )
        if validated_template.factory_ref is not None:
            return validated_template

        bound_call = signature.bind(*args, **kwargs)
        call_kwargs = dict(bound_call.arguments)
        return replace(
            validated_template,
            factory_ref=template_factory_ref(
                factory.__module__,
                factory.__name__,
                **call_kwargs,
            ),
        )

    return wrapped


def _validate_template_factory_signature(
    signature: inspect.Signature,
    *,
    factory: Callable[..., Any],
    label: str,
) -> None:
    disallowed_kinds = {
        inspect.Parameter.POSITIONAL_ONLY: "positional-only parameters",
        inspect.Parameter.VAR_POSITIONAL: "*args",
        inspect.Parameter.VAR_KEYWORD: "**kwargs",
    }
    for parameter in signature.parameters.values():
        problem = disallowed_kinds.get(parameter.kind)
        if problem is None:
            continue
        raise TypeError(
            f"{label} factory {factory.__module__}.{factory.__name__} cannot use "
            f"{problem} because remote template rebuild calls it back with named kwargs."
        )


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

    adapter = _StageTemplateTaskAdapter(
        base_task=base_task,
        sft_dataset=sft_dataset,
        rl_dataset=rl_dataset,
        rl_reward=rl_reward,
        eval_dataset=eval_dataset,
        eval_metrics=eval_metrics,
    )
    _copy_task_source_attrs(base_task=base_task, target_task=adapter)
    return adapter


def serialize_stage_templates(
    *,
    sft_dataset: Optional[SFTDatasetTemplate] = None,
    rl_dataset: Optional[RLDatasetTemplate] = None,
    rl_reward: Optional[RLRewardTemplate] = None,
    eval_dataset: Optional[EvalDatasetTemplate] = None,
    eval_metrics: Optional[EvalMetricsTemplate] = None,
) -> Optional[Dict[str, Dict[str, Any]]]:
    payload: Dict[str, Dict[str, Any]] = {}
    _maybe_add_serialized_template(
        payload,
        key="sft_dataset",
        template=sft_dataset,
        expected_type=SFTDatasetTemplate,
        label="SFT dataset template",
    )
    _maybe_add_serialized_template(
        payload,
        key="rl_dataset",
        template=rl_dataset,
        expected_type=RLDatasetTemplate,
        label="RL dataset template",
    )
    _maybe_add_serialized_template(
        payload,
        key="rl_reward",
        template=rl_reward,
        expected_type=RLRewardTemplate,
        label="RL reward template",
    )
    _maybe_add_serialized_template(
        payload,
        key="eval_dataset",
        template=eval_dataset,
        expected_type=EvalDatasetTemplate,
        label="Eval dataset template",
    )
    _maybe_add_serialized_template(
        payload,
        key="eval_metrics",
        template=eval_metrics,
        expected_type=EvalMetricsTemplate,
        label="Eval metrics template",
    )
    return payload or None


def bind_stage_templates_from_config(
    base_task: TaskPlugin,
    config: Dict[str, Any],
) -> TaskPlugin:
    raw_payload = config.get(STAGE_TEMPLATE_CONFIG_KEY)
    if raw_payload is None:
        return base_task
    if not isinstance(raw_payload, Mapping):
        raise TypeError(
            f"{STAGE_TEMPLATE_CONFIG_KEY} must be a mapping payload."
        )
    return bind_stage_templates(
        base_task,
        sft_dataset=_load_template_from_payload(
            raw_payload.get("sft_dataset"),
            expected_type=SFTDatasetTemplate,
            label="SFT dataset template",
        ),
        rl_dataset=_load_template_from_payload(
            raw_payload.get("rl_dataset"),
            expected_type=RLDatasetTemplate,
            label="RL dataset template",
        ),
        rl_reward=_load_template_from_payload(
            raw_payload.get("rl_reward"),
            expected_type=RLRewardTemplate,
            label="RL reward template",
        ),
        eval_dataset=_load_template_from_payload(
            raw_payload.get("eval_dataset"),
            expected_type=EvalDatasetTemplate,
            label="Eval dataset template",
        ),
        eval_metrics=_load_template_from_payload(
            raw_payload.get("eval_metrics"),
            expected_type=EvalMetricsTemplate,
            label="Eval metrics template",
        ),
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


def _copy_task_source_attrs(
    *,
    base_task: TaskPlugin,
    target_task: TaskPlugin,
) -> None:
    for attr in ("__tenyson_source_path__", "__tenyson_source_module__"):
        value = getattr(base_task, attr, None)
        if value is not None:
            setattr(target_task, attr, value)


def _maybe_add_serialized_template(
    payload: Dict[str, Dict[str, Any]],
    *,
    key: str,
    template: Any,
    expected_type: type,
    label: str,
) -> None:
    if template is None:
        return
    validated_template = _require_instance(
        template,
        expected_type,
        name=label,
        allow_none=False,
    )
    factory_ref = validated_template.factory_ref
    if factory_ref is None:
        raise ValueError(
            f"{label} must define a factory_ref so cloud jobs can rebuild it remotely."
        )
    payload[key] = _serialize_factory_ref(factory_ref, label=label)


def _serialize_factory_ref(
    factory_ref: TemplateFactoryRef,
    *,
    label: str,
) -> Dict[str, Any]:
    module_name = str(factory_ref.module).strip()
    factory_name = str(factory_ref.factory).strip()
    if not module_name or not factory_name:
        raise ValueError(
            f"{label} factory_ref must include non-empty module and factory names."
        )
    return factory_ref.as_payload()


def _load_template_from_payload(
    payload: Any,
    *,
    expected_type: type,
    label: str,
) -> Any:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} payload must be a mapping.")

    module_name = str(payload.get("module") or "").strip()
    factory_name = str(payload.get("factory") or "").strip()
    kwargs = payload.get("kwargs", {})
    if not module_name or not factory_name:
        raise ValueError(
            f"{label} payload must include non-empty module and factory."
        )
    if not isinstance(kwargs, Mapping):
        raise TypeError(f"{label} payload kwargs must be a mapping.")

    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    if not callable(factory):
        raise TypeError(
            f"{label} factory {module_name}.{factory_name} is not callable."
        )
    template = factory(**dict(kwargs))
    return _require_instance(
        template,
        expected_type,
        name=label,
        allow_none=False,
    )


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
