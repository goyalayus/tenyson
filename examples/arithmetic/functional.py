from __future__ import annotations

import re
from random import Random
from typing import Any

from datasets import Dataset
from tenyson import eval_dataset_fn, eval_metrics_fn

_ANSWER_PREFIX = "<answer>"
_ANSWER_SUFFIX = "</answer>"

_STRICT_ANSWER_RE = re.compile(
    rf"^\s*([0-9]+)\s*{re.escape(_ANSWER_SUFFIX)}\s*$",
    re.IGNORECASE | re.DOTALL,
)  # full-match regex for the intended tail-only form like "6859</answer>"

_ANSWER_TEXT_RE = re.compile(
    rf"(?:{re.escape(_ANSWER_PREFIX)}\s*)?([0-9]+)\s*{re.escape(_ANSWER_SUFFIX)}",
    re.IGNORECASE | re.DOTALL,
)  # extraction regex for text like "906</answer>" or "reasoning... <answer>906</answer>"


def build_addition_prompt(left: int, right: int) -> str:
    """Build the human-readable prompt shown to metrics, logs, and reviewers."""

    # left example:
    # 4312
    # right example:
    # 2547
    return (
        "You are solving a 4-digit addition problem.\n"
        "Do not show your working.\n"
        f"Reply only with digits inside {_ANSWER_PREFIX}...{_ANSWER_SUFFIX}.\n\n"
        f"Problem: {left} + {right}"
    )


def build_addition_messages(prompt_text: str) -> list[dict[str, str]]:
    """Build the generation messages with the answer tag already opened."""

    # prompt_text example:
    # "You are solving a 4-digit addition problem...\nProblem: 4312 + 2547"
    return [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": _ANSWER_PREFIX},
    ]


def parse_answer(completion_text: str) -> str | None:
    """Extract the answer digits from either the tail-only or full-tag reply."""

    # completion_text example:
    # "6859</answer>" or "reasoning... <answer>6859</answer>"
    match = _ANSWER_TEXT_RE.search(completion_text)
    # match example:
    # regex match for "6859</answer>" or "<answer>6859</answer>", or None
    if match is None:
        return None

    return str(int(match.group(1)))


def has_strict_answer_format(completion_text: str) -> bool:
    """Accept only the tail-only form because `<answer>` is prefilled."""

    # completion_text example:
    # "6859</answer>"
    return _STRICT_ANSWER_RE.match(completion_text) is not None


def score_addition_completion(
    completion_text: str,
    # completion_text example:
    # "6859</answer>" or "<answer>6859</answer>"
    row: dict[str, Any],
    # row example:
    # {"id": 0, "left": 4312, "right": 2547, "expected_answer": "6859",
    #  "prompt": "...", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<answer>"}]}
) -> dict[str, Any]:
    """Score one model completion against one addition problem.

    Returns:
    {"id": 0, "left": 4312, "right": 2547, "expected_answer": "6859",
     "completion": "6859</answer>", "parsed_answer": "6859",
     "format_ok": True, "exact_match": True, "absolute_error": 0}
    """

    expected_answer = str(row["expected_answer"])  # e.g. "6859"
    parsed_answer = parse_answer(completion_text)  # e.g. "6859" or None
    format_ok = has_strict_answer_format(completion_text)  # e.g. True
    absolute_error: int | None = None
    # absolute_error example:
    # 0 when parsed_answer is "906", or None when parsing fails
    exact_match = False  # e.g. True when parsed_answer == expected_answer

    if parsed_answer is not None:
        absolute_error = abs(int(parsed_answer) - int(expected_answer))
        exact_match = parsed_answer == expected_answer

    return {
        "id": row["id"],
        "left": row["left"],
        "right": row["right"],
        "expected_answer": expected_answer,
        "completion": completion_text,
        "parsed_answer": parsed_answer,
        "format_ok": format_ok,
        "exact_match": exact_match,
        "absolute_error": absolute_error,
    }


def summarize_addition_results(
    row_results: list[dict[str, Any]],
    # row_results example:
    # [{"id": 0, "left": 4312, "right": 2547, "expected_answer": "6859",
    #   "completion": "6859</answer>", "parsed_answer": "6859",
    #   "format_ok": True, "exact_match": True, "absolute_error": 0}]
    *,
    total_samples: int,
    # total_samples example:
    # 100
) -> dict[str, float | int]:
    """Roll per-row eval results up into aggregate metrics.

    Returns:
    {"format_accuracy": 0.91, "exact_match_accuracy": 0.84,
     "avg_abs_error": 3.6, "parsed_answer_rate": 0.95, "total_samples": 100}
    """

    format_matches = 0  # e.g. 91
    exact_matches = 0  # e.g. 84
    parsed_answers = 0  # e.g. 95
    total_absolute_error = 0  # e.g. 360

    for row in row_results:
        # row example:
        # {"id": 0, "left": 4312, "right": 2547, "expected_answer": "6859",
        #  "completion": "6859</answer>", "parsed_answer": "6859",
        #  "format_ok": True, "exact_match": True, "absolute_error": 0}
        if bool(row["format_ok"]):
            format_matches += 1
        if bool(row["exact_match"]):
            exact_matches += 1
        if row["parsed_answer"] is not None:
            parsed_answers += 1
        if row["absolute_error"] is not None:
            total_absolute_error += int(row["absolute_error"])

    return {
        "format_accuracy": format_matches / max(total_samples, 1),
        "exact_match_accuracy": exact_matches / max(total_samples, 1),
        "avg_abs_error": total_absolute_error / max(parsed_answers, 1),
        "parsed_answer_rate": parsed_answers / max(total_samples, 1),
        "total_samples": total_samples,
    }


@eval_dataset_fn
def build_four_digit_addition_dataset(
    *,
    sample_count: int,
    # sample_count example:
    # 100
    seed: int,
    # seed example:
    # 7
) -> Dataset:
    """Build a synthetic eval set of unique 4-digit addition problems.

    `@eval_dataset_fn` makes the Tenyson hook contract visible in this file.
    The eval runner can call this function and fill its named kwargs from
    `config["evaluation"]`.

    Each row looks like:
    {"id": 0, "left": 4312, "right": 2547, "expected_answer": "6859",
     "prompt": "...\nProblem: 4312 + 2547",
     "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<answer>"}]}
    """

    rng = Random(seed)  # e.g. Random(7)
    rows: list[dict[str, Any]] = []
    # rows example:
    # [{"id": 0, "left": 4312, "right": 2547, "expected_answer": "6859",
    #   "prompt": "...\nProblem: 4312 + 2547",
    #   "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<answer>"}]}]
    seen_problems: set[tuple[int, int]] = set()  # {(4312, 2547), (9081, 1776)}

    while len(rows) < sample_count:
        left = rng.randint(1000, 9999)  # e.g. 4312
        right = rng.randint(1000, 9999)  # e.g. 2547
        problem = (left, right)  # e.g. (4312, 2547)

        if problem in seen_problems:
            continue

        seen_problems.add(problem)
        prompt_text = build_addition_prompt(left, right)
        # prompt_text example:
        # "You are solving a 4-digit addition problem...\nProblem: 4312 + 2547"
        messages = build_addition_messages(prompt_text)
        # messages example:
        # [{"role": "user", "content": "You are solving a 4-digit addition problem...\nProblem: 4312 + 2547"},
        #  {"role": "assistant", "content": "<answer>"}]
        rows.append(
            {
                "id": len(rows),
                "left": left,
                "right": right,
                "expected_answer": str(left + right),
                "prompt": prompt_text,
                "messages": messages,
            }
        )

    return Dataset.from_list(rows)


@eval_metrics_fn
def compute_addition_metrics(
    _prompts: list[str],
    # _prompts example:
    # ["You are solving a 4-digit addition problem...\nProblem: 4312 + 2547"]
    completions: list[str],
    # completions example:
    # ["6859</answer>", "1111</answer>"]
    dataset_rows: Dataset,
    # each row looks like:
    # {"id": 0, "left": 4312, "right": 2547, "expected_answer": "6859",
    #  "prompt": "...\nProblem: 4312 + 2547",
    #  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<answer>"}]}
    _config: dict[str, Any],
    # _config example:
    # {"evaluation": {"sample_count": 100, "seed": 7}}
    _tokenizer: Any,
    # _tokenizer example:
    # a Hugging Face tokenizer object for the eval model, e.g. Qwen tokenizer
) -> dict[str, Any]:
    """Compute eval metrics for 4-digit addition.

    `@eval_metrics_fn` makes the Tenyson hook contract visible in this file.
    The eval runner calls this with
    `(prompts, completions, dataset_rows, config, tokenizer)`, and the
    decorator validates that signature early.

    config example:
    {"evaluation": {"sample_count": 100, "seed": 7}}

    Returns:
    {
        "metrics": {
            "format_accuracy": 0.91,
            "exact_match_accuracy": 0.84,
            "avg_abs_error": 3.6,
            "parsed_answer_rate": 0.95,
            "total_samples": 100,
        },
        "detailed_results": [
            {
                "id": 0,
                "left": 4312,
                "right": 2547,
                "expected_answer": "6859",
                "completion": "6859</answer>",
                "parsed_answer": "6859",
                "format_ok": True,
                "exact_match": True,
                "absolute_error": 0,
            }
        ],
    }
    """

    detailed_results: list[dict[str, Any]] = []
    # detailed_results example:
    # [{"id": 0, "left": 4312, "right": 2547, "expected_answer": "6859",
    #   "completion": "6859</answer>", "parsed_answer": "6859",
    #   "format_ok": True, "exact_match": True, "absolute_error": 0}]

    for completion_text, row in zip(completions, dataset_rows):
        # completion_text example:
        # "6859</answer>"
        # row example:
        # {"id": 0, "left": 4312, "right": 2547, "expected_answer": "6859",
        #  "prompt": "...\nProblem: 4312 + 2547",
        #  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<answer>"}]}
        detailed_results.append(score_addition_completion(completion_text, row))

    metrics = summarize_addition_results(
        detailed_results,
        total_samples=len(completions),
    )
    # metrics example:
    # {"format_accuracy": 0.91, "exact_match_accuracy": 0.84,
    #  "avg_abs_error": 3.6, "parsed_answer_rate": 0.95, "total_samples": 100}

    return {
        "metrics": metrics,
        "detailed_results": detailed_results,
    }
