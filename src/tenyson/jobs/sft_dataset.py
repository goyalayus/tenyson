from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


_ROW_SYSTEM_KEYS: tuple[str, ...] = ("system", "system_prompt")
_PROMPT_ANSWER_KEYS: tuple[tuple[str, str], ...] = (
    ("prompt", "answer"),
    ("prompt", "completion"),
    ("question", "answer"),
)


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _coerce_text_content(value: Any, *, field_name: str) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts: list[str] = []
        for index, block in enumerate(value):
            if not isinstance(block, dict):
                raise ValueError(
                    f"{field_name} block at index {index} must be a dict with text content."
                )
            block_type = str(block.get("type") or "text").strip().lower()
            text_value = block.get("text")
            if block_type != "text" or not isinstance(text_value, str):
                raise ValueError(
                    f"{field_name} block at index {index} must be a text block."
                )
            parts.append(text_value)
        return "".join(parts)

    raise ValueError(
        f"{field_name} must be a string or a list of text blocks, got "
        f"{type(value).__name__}."
    )


def _is_message_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def _normalize_messages(messages: Any, *, field_name: str) -> list[Dict[str, str]]:
    if not _is_message_list(messages):
        raise ValueError(f"{field_name} must be a non-empty list of messages.")

    normalized: list[Dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"{field_name}[{index}] must be a mapping.")
        role = str(message.get("role") or "").strip()
        if not role:
            raise ValueError(f"{field_name}[{index}] is missing a non-empty role.")
        content = _coerce_text_content(
            message.get("content"),
            field_name=f"{field_name}[{index}].content",
        )
        normalized.append({"role": role, "content": content})
    return normalized


def _resolve_default_system_prompt(config: Mapping[str, Any] | None) -> Optional[str]:
    if not isinstance(config, Mapping):
        return None
    task_cfg = config.get("task", {})
    if not isinstance(task_cfg, Mapping):
        return None
    prompt = str(task_cfg.get("sft_system_prompt") or "").strip()
    return prompt or None


def _resolve_system_prompt(
    row: Mapping[str, Any],
    *,
    default_system_prompt: Optional[str],
) -> Optional[str]:
    for key in _ROW_SYSTEM_KEYS:
        value = row.get(key)
        if _is_nonempty_string(value):
            return str(value)
    return default_system_prompt


def _build_instruction_user_text(row: Mapping[str, Any]) -> Optional[str]:
    instruction = row.get("instruction")
    output = row.get("output")
    if not _is_nonempty_string(instruction) or not _is_nonempty_string(output):
        return None

    input_text = row.get("input")
    if _is_nonempty_string(input_text):
        return (
            "Instruction:\n"
            + str(instruction).strip()
            + "\n\nInput:\n"
            + str(input_text).strip()
        )
    return str(instruction)


def supports_builtin_sft_schema(row: Any) -> bool:
    if not isinstance(row, Mapping):
        return False

    messages = row.get("messages")
    if _is_message_list(messages):
        return True

    prompt = row.get("prompt")
    completion = row.get("completion")
    if _is_message_list(prompt) and _is_message_list(completion):
        return True

    for prompt_key, answer_key in _PROMPT_ANSWER_KEYS:
        if _is_nonempty_string(row.get(prompt_key)) and _is_nonempty_string(
            row.get(answer_key)
        ):
            return True

    return _build_instruction_user_text(row) is not None


def build_builtin_sft_messages(
    row: Mapping[str, Any],
    *,
    default_system_prompt: Optional[str] = None,
) -> list[Dict[str, str]]:
    if not isinstance(row, Mapping):
        raise ValueError(
            "SFT rows must be mappings. Supported built-in schemas are "
            "`messages`, conversational `prompt`/`completion`, string "
            "`prompt`/`answer`, string `prompt`/`completion`, string "
            "`question`/`answer`, and `instruction`/`output` with optional `input`."
        )

    if _is_message_list(row.get("messages")):
        return _normalize_messages(row.get("messages"), field_name="messages")

    prompt = row.get("prompt")
    completion = row.get("completion")
    if _is_message_list(prompt) and _is_message_list(completion):
        merged = list(prompt) + list(completion)
        return _normalize_messages(
            merged,
            field_name="prompt+completion",
        )

    system_prompt = _resolve_system_prompt(
        row,
        default_system_prompt=default_system_prompt,
    )

    for prompt_key, answer_key in _PROMPT_ANSWER_KEYS:
        prompt_text = row.get(prompt_key)
        answer_text = row.get(answer_key)
        if _is_nonempty_string(prompt_text) and _is_nonempty_string(answer_text):
            messages: list[Dict[str, str]] = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": str(prompt_text)})
            messages.append({"role": "assistant", "content": str(answer_text)})
            return messages

    instruction_text = _build_instruction_user_text(row)
    if instruction_text is not None:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": instruction_text})
        messages.append({"role": "assistant", "content": str(row["output"])})
        return messages

    raise ValueError(
        "Unsupported SFT row schema. Tenyson accepts one of these built-in "
        "schemas:\n"
        "- `messages`: list of `{role, content}` rows\n"
        "- conversational `prompt` + `completion` lists of messages\n"
        "- string `prompt` + `answer`\n"
        "- string `prompt` + `completion`\n"
        "- string `question` + `answer`\n"
        "- `instruction` + `output`, with optional `input`\n"
        "You can also provide optional `system` or `system_prompt` string fields "
        "on the string-based schemas."
    )


def normalize_builtin_sft_dataset(
    dataset: Any,
    *,
    config: Mapping[str, Any] | None,
    dataset_name: str,
) -> Any:
    if dataset is None:
        return None

    default_system_prompt = _resolve_default_system_prompt(config)
    first_row = next(iter(dataset))
    if not supports_builtin_sft_schema(first_row):
        return dataset

    map_kwargs: Dict[str, Any] = {}
    column_names = getattr(dataset, "column_names", None)
    if isinstance(column_names, list):
        map_kwargs["desc"] = f"Normalizing {dataset_name} dataset to Tenyson SFT messages"

    def _normalize_row(row: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "messages": build_builtin_sft_messages(
                row,
                default_system_prompt=default_system_prompt,
            )
        }

    normalized = dataset.map(
        _normalize_row,
        **map_kwargs,
    )

    return normalized


def build_builtin_sft_formatting_func(tokenizer: Any):
    def _format_conversation(example: Mapping[str, Any]) -> list[str]:
        messages = build_builtin_sft_messages(example)
        return [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        ]

    return _format_conversation
