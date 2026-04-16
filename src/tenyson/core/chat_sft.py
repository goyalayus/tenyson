from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Mapping, Optional

from datasets import Dataset, load_dataset

from tenyson.core.environment import DatasetHooks
from tenyson.core.stage_templates import SFTDatasetTemplate, template_factory_ref


def chat_sft_dataset_fn(
    function: Callable[..., Dataset],
) -> Callable[..., Dataset]:
    """Mark a plain function as a Tenyson chat-SFT train-dataset hook.

    This decorator exists for readability first. A function like
    `build_addition_sft_train_dataset(...)` looks like an ordinary helper in a
    task file, but Tenyson can bind it into an `SFTDatasetTemplate` with
    `bind_chat_sft_dataset(...)`.

    The decorator makes that role visible at the definition site and validates
    the small contract that chat-SFT bound builders must follow: they must be a
    module-level named function and they must use a callback-safe signature.
    """

    _require_module_level_named_function(
        function,
        label="chat SFT train builder",
    )
    _validate_chat_sft_train_builder_signature(
        function,
        label="chat SFT train builder",
    )
    setattr(function, "__tenyson_chat_sft_dataset_fn__", True)
    return function


def build_hub_chat_sft_dataset_hooks(
    *,
    default_dataset: str | None = None,
    dataset_key: str = "sft_dataset",
    messages_column: str = "messages",
    split: str = "train",
) -> DatasetHooks:
    template = hub_chat_sft_dataset(
        default_dataset=default_dataset,
        dataset_key=dataset_key,
        messages_column=messages_column,
        split=split,
    )
    return DatasetHooks(
        primary=template.train,
        evaluation=template.evaluation,
        formatting=template.formatting,
        collator=template.collator,
    )


def bind_chat_sft_dataset(
    train_builder: Callable[..., Dataset],
    /,
    *,
    messages_column: str = "messages",
    **bound_kwargs: Any,
) -> SFTDatasetTemplate:
    """Bind a plain chat-messages dataset builder into an SFT template.

    The builder must be a module-level named function, and every required
    parameter must be bound here because the SFT job will call it later without
    passing task-specific kwargs.
    """

    module_name, function_name = _require_module_level_named_function(
        train_builder,
        label="chat SFT train builder",
    )
    _validate_chat_sft_bound_builder_kwargs(
        train_builder,
        bound_kwargs=bound_kwargs,
    )
    return _build_bound_chat_sft_dataset_template(
        train_builder=train_builder,
        messages_column=messages_column,
        bound_kwargs=bound_kwargs,
        factory_ref=template_factory_ref(
            "tenyson.core.chat_sft",
            "_bound_chat_sft_dataset_from_callable_ref",
            module_name=module_name,
            function_name=function_name,
            messages_column=messages_column,
            **({"bound_kwargs": dict(bound_kwargs)} if bound_kwargs else {}),
        ),
    )


def hub_chat_sft_dataset(
    *,
    default_dataset: str | None = None,
    dataset_key: str = "sft_dataset",
    messages_column: str = "messages",
    split: str = "train",
) -> SFTDatasetTemplate:
    def _train_dataset(config: dict[str, Any], _tokenizer: Any) -> Dataset:
        train_dataset, _ = load_hub_chat_sft_train_eval_split(
            config,
            default_dataset=default_dataset,
            dataset_key=dataset_key,
            messages_column=messages_column,
            split=split,
        )
        return train_dataset

    def _eval_dataset(config: dict[str, Any], _tokenizer: Any) -> Optional[Dataset]:
        _, eval_dataset = load_hub_chat_sft_train_eval_split(
            config,
            default_dataset=default_dataset,
            dataset_key=dataset_key,
            messages_column=messages_column,
            split=split,
        )
        return eval_dataset

    def _formatting(config: dict[str, Any], tokenizer: Any):
        return build_chat_messages_formatting_func(
            tokenizer=tokenizer,
            messages_column=messages_column,
        )

    return SFTDatasetTemplate(
        train=_train_dataset,
        evaluation=_eval_dataset,
        formatting=_formatting,
        factory_ref=template_factory_ref(
            "tenyson.core.chat_sft",
            "hub_chat_sft_dataset",
            default_dataset=default_dataset,
            dataset_key=dataset_key,
            messages_column=messages_column,
            split=split,
        ),
    )


def _build_bound_chat_sft_dataset_template(
    *,
    train_builder: Callable[..., Dataset],
    messages_column: str,
    bound_kwargs: Mapping[str, Any],
    factory_ref: Any,
) -> SFTDatasetTemplate:
    resolved_bound_kwargs = dict(bound_kwargs)

    def _train_dataset(_config: dict[str, Any], _tokenizer: Any) -> Dataset:
        dataset = train_builder(**resolved_bound_kwargs)
        if not isinstance(dataset, Dataset):
            raise TypeError(
                "chat SFT train builder must return datasets.Dataset, "
                f"got {type(dataset).__name__}."
            )
        validate_chat_messages_dataset(
            dataset,
            messages_column=messages_column,
            dataset_name=f"{train_builder.__module__}.{train_builder.__name__}",
            split_name="train",
        )
        return dataset

    def _formatting(_config: dict[str, Any], tokenizer: Any):
        return build_chat_messages_formatting_func(
            tokenizer=tokenizer,
            messages_column=messages_column,
        )

    return SFTDatasetTemplate(
        train=_train_dataset,
        formatting=_formatting,
        factory_ref=factory_ref,
    )


def _bound_chat_sft_dataset_from_callable_ref(
    *,
    module_name: str,
    function_name: str,
    messages_column: str = "messages",
    bound_kwargs: Mapping[str, Any] | None = None,
) -> SFTDatasetTemplate:
    module = importlib.import_module(str(module_name).strip())
    train_builder = getattr(module, str(function_name).strip())
    if not callable(train_builder):
        raise TypeError(
            f"chat SFT train builder {module_name}.{function_name} is not callable."
        )

    return _build_bound_chat_sft_dataset_template(
        train_builder=train_builder,
        messages_column=messages_column,
        bound_kwargs=dict(bound_kwargs or {}),
        factory_ref=template_factory_ref(
            "tenyson.core.chat_sft",
            "_bound_chat_sft_dataset_from_callable_ref",
            module_name=module_name,
            function_name=function_name,
            messages_column=messages_column,
            **({"bound_kwargs": dict(bound_kwargs)} if bound_kwargs else {}),
        ),
    )


def load_hub_chat_sft_train_eval_split(
    config: dict[str, Any],
    *,
    default_dataset: str | None = None,
    dataset_key: str = "sft_dataset",
    messages_column: str = "messages",
    split: str = "train",
) -> tuple[Dataset, Optional[Dataset]]:
    task_cfg = config.get("task", {})
    dataset_name = str(task_cfg.get(dataset_key, default_dataset) or "").strip()
    if not dataset_name:
        raise ValueError(
            f'task.{dataset_key} must be set to a Hugging Face dataset repo id.'
        )

    dataset = load_dataset(dataset_name, split=split)
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f'Expected datasets.Dataset for "{dataset_name}" split "{split}", '
            f"got {type(dataset).__name__}."
        )

    train_sample_limit_raw = task_cfg.get("sft_train_samples")
    train_sample_limit = (
        max(1, int(train_sample_limit_raw))
        if train_sample_limit_raw is not None
        else None
    )

    val_size = int(config.get("training", {}).get("val_size", 0) or 0)
    if train_sample_limit is not None and len(dataset) > train_sample_limit + max(0, val_size):
        dataset = dataset.select(range(train_sample_limit + max(0, val_size)))

    if val_size <= 0 or len(dataset) <= 1:
        train_dataset = dataset
        if train_sample_limit is not None and len(train_dataset) > train_sample_limit:
            train_dataset = train_dataset.select(range(train_sample_limit))
        validate_chat_messages_dataset(
            train_dataset,
            messages_column=messages_column,
            dataset_name=dataset_name,
            split_name=split,
        )
        return train_dataset, None

    val_size = min(val_size, max(1, len(dataset) - 1))
    split_seed = int(config.get("training", {}).get("seed", 3407))
    split_result = dataset.train_test_split(test_size=val_size, seed=split_seed)
    train_dataset = split_result["train"]
    if train_sample_limit is not None and len(train_dataset) > train_sample_limit:
        train_dataset = train_dataset.select(range(train_sample_limit))
    eval_dataset = split_result["test"]

    validate_chat_messages_dataset(
        train_dataset,
        messages_column=messages_column,
        dataset_name=dataset_name,
        split_name=f"{split}:train",
    )
    validate_chat_messages_dataset(
        eval_dataset,
        messages_column=messages_column,
        dataset_name=dataset_name,
        split_name=f"{split}:eval",
    )
    return train_dataset, eval_dataset


def build_chat_messages_formatting_func(
    *,
    tokenizer: Any,
    messages_column: str = "messages",
):
    def _format_example(example: Mapping[str, Any]) -> list[str]:
        messages = example[messages_column]
        if messages and isinstance(messages[0], list):
            return [
                tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for conversation in messages
            ]
        return [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        ]

    return _format_example


def _require_module_level_named_function(
    value: Callable[..., Any],
    *,
    label: str,
) -> tuple[str, str]:
    module_name = str(getattr(value, "__module__", "") or "").strip()
    function_name = str(getattr(value, "__name__", "") or "").strip()
    qualname = str(getattr(value, "__qualname__", "") or "").strip()

    if not module_name or not function_name:
        raise TypeError(f"{label} must be a module-level importable function.")
    if "<locals>" in qualname or function_name == "<lambda>":
        raise TypeError(
            f"{label} must be a module-level named function, not a lambda or nested function."
        )
    return module_name, function_name


def _validate_chat_sft_train_builder_signature(
    function: Callable[..., Dataset],
    *,
    label: str,
) -> None:
    signature = inspect.signature(function)

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"{label} cannot use positional-only parameters."
            )
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(
                f"{label} cannot use *args."
            )
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            raise TypeError(
                f"{label} cannot use **kwargs."
            )


def _validate_chat_sft_bound_builder_kwargs(
    train_builder: Callable[..., Dataset],
    *,
    bound_kwargs: Mapping[str, Any],
) -> None:
    _validate_chat_sft_train_builder_signature(
        train_builder,
        label="chat SFT train builder",
    )
    signature = inspect.signature(train_builder)

    try:
        signature.bind_partial(**dict(bound_kwargs))
    except TypeError as exc:
        raise TypeError(
            "chat SFT train builder received invalid bound kwargs for "
            f"{train_builder.__module__}.{train_builder.__name__}: {exc}"
        ) from exc

    for parameter in signature.parameters.values():
        if parameter.name in bound_kwargs:
            continue
        if parameter.default is not inspect.Signature.empty:
            continue
        raise TypeError(
            "chat SFT train builder must bind every required parameter. "
            f'Missing bound kwarg "{parameter.name}" for '
            f"{train_builder.__module__}.{train_builder.__name__}."
        )


def validate_chat_messages_dataset(
    dataset: Dataset,
    *,
    messages_column: str = "messages",
    dataset_name: str = "dataset",
    split_name: str = "train",
) -> None:
    if messages_column not in dataset.column_names:
        raise ValueError(
            f'{dataset_name} split "{split_name}" must contain a '
            f'"{messages_column}" column.'
        )

    messages_rows = dataset[messages_column]
    for row_index, messages in enumerate(messages_rows):
        _validate_chat_messages_row(
            messages,
            dataset_name=dataset_name,
            split_name=split_name,
            row_index=row_index,
            messages_column=messages_column,
        )


def _validate_chat_messages_row(
    messages: Any,
    *,
    dataset_name: str,
    split_name: str,
    row_index: int,
    messages_column: str,
) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError(
            f'{dataset_name} split "{split_name}" row {row_index} must have a '
            f'non-empty list in "{messages_column}".'
        )

    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(
                f'{dataset_name} split "{split_name}" row {row_index} message '
                f"{message_index} must be an object with role/content fields."
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(
                f'{dataset_name} split "{split_name}" row {row_index} message '
                f'{message_index} must have a non-empty string "role".'
            )
        if not isinstance(content, str):
            raise ValueError(
                f'{dataset_name} split "{split_name}" row {row_index} message '
                f'{message_index} must have string "content".'
            )
