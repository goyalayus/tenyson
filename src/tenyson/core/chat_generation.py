from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreparedGenerationPrompt:
    raw_prompt_text: str
    # raw_prompt_text example:
    # "Solve 314 + 592."
    generation_prompt: str
    # generation_prompt example:
    # "<|im_start|>user\nSolve 314 + 592.<|im_end|>\n<|im_start|>assistant\n"


def extract_raw_prompt_texts(rows: Sequence[Any]) -> list[str]:
    raw_prompt_texts: list[str] = []
    for row_index, row in enumerate(rows):
        raw_prompt_texts.append(
            extract_raw_prompt_text(
                row,
                row_index=row_index,
            )
        )
    return raw_prompt_texts


def extract_raw_prompt_text(
    row: Any,
    *,
    row_index: int | None = None,
) -> str:
    """Return the human-readable prompt text for metrics, rewards, and logs."""

    row_mapping = _coerce_row_mapping(
        row,
        row_index=row_index,
    )
    prompt_text = row_mapping.get("prompt")
    if isinstance(prompt_text, str):
        return prompt_text

    messages_value = row_mapping.get("messages")
    if messages_value is None:
        raise ValueError(
            f"{_row_label(row_index)} must define either string `prompt` or list `messages`."
        )

    messages = _normalize_messages(
        messages_value,
        row_index=row_index,
    )
    return _messages_to_transcript(messages)


def render_generation_prompt(
    *,
    row: Any,
    tokenizer: Any,
    config: Mapping[str, Any],
    row_index: int | None = None,
) -> str:
    """Render the exact string sent into generation for one dataset row."""

    row_mapping = _coerce_row_mapping(
        row,
        row_index=row_index,
    )
    messages = _resolve_generation_messages(
        row_mapping,
        config=config,
        row_index=row_index,
    )
    if messages is None:
        prompt_text = row_mapping.get("prompt")
        if not isinstance(prompt_text, str):
            raise ValueError(
                f"{_row_label(row_index)} must provide a string `prompt` when chat templating is disabled."
            )
        return prompt_text

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError(
            "This tokenizer does not expose apply_chat_template(...), so "
            "chat-template generation cannot be used."
        )

    kwargs = _build_chat_template_kwargs(
        messages,
    )
    enable_thinking = _resolve_enable_thinking(config)
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    try:
        rendered_prompt = apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        _raise_chat_template_keyword_error(exc, kwargs)

    if not isinstance(rendered_prompt, str):
        raise ValueError(
            "tokenizer.apply_chat_template(..., tokenize=False) must return a string."
        )
    return rendered_prompt


def prepare_generation_prompts(
    rows: Sequence[Any],
    tokenizer: Any,
    config: Mapping[str, Any],
) -> list[PreparedGenerationPrompt]:
    """Build both the raw prompt text and the generation prompt for each row.

    Each prepared prompt looks like:
    PreparedGenerationPrompt(
        raw_prompt_text="Solve 314 + 592.",
        generation_prompt="<|im_start|>user\\nSolve 314 + 592.<|im_end|>\\n<|im_start|>assistant\\n",
    )
    """

    prepared_prompts: list[PreparedGenerationPrompt] = []
    for row_index, row in enumerate(rows):
        prepared_prompts.append(
            PreparedGenerationPrompt(
                raw_prompt_text=extract_raw_prompt_text(
                    row,
                    row_index=row_index,
                ),
                generation_prompt=render_generation_prompt(
                    row=row,
                    tokenizer=tokenizer,
                    config=config,
                    row_index=row_index,
                ),
            )
        )
    return prepared_prompts


def resolve_generation_stop_strings(config: Mapping[str, Any]) -> list[str]:
    chat_cfg = _resolve_chat_template_config(config)
    stop_strings_value = chat_cfg.get("stop_strings")
    if stop_strings_value is None:
        return []

    if isinstance(stop_strings_value, str) or not isinstance(stop_strings_value, Sequence):
        raise TypeError("chat_template.stop_strings must be a sequence of strings.")

    stop_strings: list[str] = []
    # stop_strings example:
    # ["</answer>", "</final>"]
    for index, stop_string in enumerate(stop_strings_value):
        if not isinstance(stop_string, str) or not stop_string:
            raise ValueError(
                "chat_template.stop_strings entries must be non-empty strings. "
                f"Invalid value at index {index}: {stop_string!r}"
            )
        stop_strings.append(stop_string)

    return merge_stop_strings(stop_strings)


def merge_stop_strings(*stop_string_groups: Sequence[str]) -> list[str]:
    merged_stop_strings: list[str] = []
    seen_stop_strings: set[str] = set()

    for stop_string_group in stop_string_groups:
        for stop_string in stop_string_group:
            if stop_string in seen_stop_strings:
                continue
            seen_stop_strings.add(stop_string)
            merged_stop_strings.append(stop_string)

    return merged_stop_strings


def _resolve_generation_messages(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    row_index: int | None,
) -> list[dict[str, str]] | None:
    messages_value = row.get("messages")
    if messages_value is not None:
        return _normalize_messages(
            messages_value,
            row_index=row_index,
        )

    if not _chat_template_enabled(config):
        return None

    prompt_text = row.get("prompt")
    if not isinstance(prompt_text, str):
        raise ValueError(
            f"{_row_label(row_index)} must define string `prompt` when no `messages` are present."
        )
    return [{"role": "user", "content": prompt_text}]


def _chat_template_enabled(config: Mapping[str, Any]) -> bool:
    chat_cfg = _resolve_chat_template_config(config)
    return bool(chat_cfg.get("enabled", True))


def _resolve_enable_thinking(config: Mapping[str, Any]) -> bool | None:
    chat_cfg = _resolve_chat_template_config(config)
    if "enable_thinking" not in chat_cfg:
        return None

    value = chat_cfg.get("enable_thinking")
    if value is None:
        return None
    return bool(value)


def _resolve_chat_template_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    chat_cfg = config.get("chat_template", {})
    if chat_cfg is None:
        return {}
    if not isinstance(chat_cfg, Mapping):
        raise TypeError("chat_template must be a mapping when provided.")
    return chat_cfg


def _build_chat_template_kwargs(
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if _continues_final_assistant_message(messages):
        kwargs["add_generation_prompt"] = False
        kwargs["continue_final_message"] = True

    return kwargs


def _continues_final_assistant_message(messages: list[dict[str, str]]) -> bool:
    return messages[-1]["role"].strip().lower() == "assistant"


def _raise_chat_template_keyword_error(
    exc: TypeError,
    kwargs: Mapping[str, Any],
) -> None:
    error_text = str(exc)
    if "continue_final_message" in error_text or kwargs.get("continue_final_message"):
        raise ValueError(
            "This tokenizer's apply_chat_template(...) does not accept "
            "continue_final_message=True, so assistant-prefill generation "
            "cannot be used."
        ) from exc
    if "enable_thinking" in error_text or "enable_thinking" in kwargs:
        raise ValueError(
            "chat_template.enable_thinking was set, but this tokenizer's "
            "apply_chat_template(...) does not accept that keyword."
        ) from exc
    raise exc


def _coerce_row_mapping(
    row: Any,
    *,
    row_index: int | None,
) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError(f"{_row_label(row_index)} must be mapping-like.")
    return row


def _normalize_messages(
    messages_value: Any,
    *,
    row_index: int | None,
) -> list[dict[str, str]]:
    if not isinstance(messages_value, list) or not messages_value:
        raise ValueError(
            f"{_row_label(row_index)} must provide a non-empty `messages` list."
        )

    normalized_messages: list[dict[str, str]] = []
    # normalized_messages example:
    # [{"role": "system", "content": "You are helpful."},
    #  {"role": "user", "content": "What is 2+2?"}]
    for message_index, message in enumerate(messages_value):
        if not isinstance(message, Mapping):
            raise ValueError(
                f"{_row_label(row_index)} message {message_index} must be a mapping."
            )

        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(
                f"{_row_label(row_index)} message {message_index} must include a non-empty string `role`."
            )
        if not isinstance(content, str):
            raise ValueError(
                f"{_row_label(row_index)} message {message_index} must include a string `content`."
            )

        normalized_messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    return normalized_messages


def _messages_to_transcript(messages: list[dict[str, str]]) -> str:
    # messages example:
    # [{"role": "system", "content": "Be concise."},
    #  {"role": "user", "content": "Solve 314 + 592."}]
    lines = [
        f'{message["role"]}: {message["content"]}'
        for message in messages
    ]
    return "\n\n".join(lines)


def _row_label(row_index: int | None) -> str:
    if row_index is None:
        return "row"
    return f"row {row_index}"
