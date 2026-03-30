from __future__ import annotations

from typing import Any, Mapping, Optional

from datasets import Dataset, load_dataset

from tenyson.core.environment import DatasetHooks
from tenyson.core.stage_templates import SFTDatasetTemplate, template_factory_ref


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
