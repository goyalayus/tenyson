from __future__ import annotations

import re
from typing import Any


_FALLBACK_EOS_TOKENS = (
    "<|im_end|>",
    "<|endoftext|>",
    "</s>",
    "<|eot_id|>",
)

_GENERATION_BLOCK_RE = re.compile(r"\{%-?\s*generation\s*-?%\}")
_ENDGENERATION_BLOCK_RE = re.compile(r"\{%-?\s*endgeneration\s*-?%\}")

_QWEN_ASSISTANT_MASK_CHAT_TEMPLATE = """
{% for message in messages %}
{% if message.role == 'assistant' %}
{{ '<|im_start|>assistant\n' }}{% generation %}{{ message.content }}{% endgeneration %}{{ '<|im_end|>\n' }}
{% else %}
{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
""".strip()


def _is_valid_token(tokenizer: Any, token: Any) -> bool:
    if not isinstance(token, str) or not token:
        return False
    vocab = None
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocab = get_vocab()
        except Exception:  # noqa: BLE001
            vocab = None
    if isinstance(vocab, dict) and token not in vocab:
        return False
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:  # noqa: BLE001
        return False
    if token_id is None:
        return False
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    unk_token = getattr(tokenizer, "unk_token", None)
    if unk_token_id is not None and token_id == unk_token_id and token != unk_token:
        return False
    return True


def _resolve_eos_token(tokenizer: Any) -> str:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        try:
            token_from_id = tokenizer.convert_ids_to_tokens(eos_token_id)
        except Exception:  # noqa: BLE001
            token_from_id = None
        if _is_valid_token(tokenizer, token_from_id):
            return token_from_id

    eos_token = getattr(tokenizer, "eos_token", None)
    if _is_valid_token(tokenizer, eos_token):
        return eos_token

    for candidate in _FALLBACK_EOS_TOKENS:
        if _is_valid_token(tokenizer, candidate):
            return candidate

    raise ValueError(
        "Could not resolve a valid EOS token for tokenizer "
        f"{tokenizer.__class__.__name__}."
    )


def _chat_template_supports_assistant_masks(template: Any) -> bool:
    if not isinstance(template, str) or not template:
        return False
    return bool(_GENERATION_BLOCK_RE.search(template)) and bool(
        _ENDGENERATION_BLOCK_RE.search(template)
    )


def ensure_assistant_mask_chat_template(tokenizer: Any) -> Any:
    template = getattr(tokenizer, "chat_template", None)
    if _chat_template_supports_assistant_masks(template):
        return tokenizer

    tokenizer.chat_template = _QWEN_ASSISTANT_MASK_CHAT_TEMPLATE
    print(
        "[Tokenizer] Installed assistant-mask-compatible Qwen chat template "
        "for conversational SFT.",
        flush=True,
    )
    return tokenizer


def normalize_tokenizer_special_tokens(
    tokenizer: Any,
    *,
    padding_side: str | None = None,
) -> Any:
    resolved_eos = _resolve_eos_token(tokenizer)
    if getattr(tokenizer, "eos_token", None) != resolved_eos:
        print(
            "[Tokenizer] Replacing invalid EOS token "
            f"{getattr(tokenizer, 'eos_token', None)!r} with {resolved_eos!r}.",
            flush=True,
        )
        tokenizer.eos_token = resolved_eos

    if not _is_valid_token(tokenizer, getattr(tokenizer, "pad_token", None)):
        tokenizer.pad_token = tokenizer.eos_token

    if padding_side is not None:
        tokenizer.padding_side = padding_side

    eos_token_id = None
    pad_token_id = None
    try:
        eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    except Exception:  # noqa: BLE001
        eos_token_id = None
    try:
        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    except Exception:  # noqa: BLE001
        pad_token_id = None
    print(
        "[Tokenizer] Using "
        f"eos_token={tokenizer.eos_token!r} (id={eos_token_id}), "
        f"pad_token={tokenizer.pad_token!r} (id={pad_token_id}), "
        f"padding_side={getattr(tokenizer, 'padding_side', None)!r}.",
        flush=True,
    )

    return tokenizer
