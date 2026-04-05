from __future__ import annotations

import re
from random import Random
from typing import Any

from datasets import Dataset

from tenyson import (
    EvalDatasetTemplate,
    EvalMetricsTemplate,
    eval_dataset_template,
    eval_metrics_template,
)

_DEFAULT_EVAL_SAMPLES = 100  # e.g. 100 eval rows
_DEFAULT_EVAL_SEED = 7  # e.g. 7

_STRICT_ANSWER_RE = re.compile(
    r"^\s*<answer>\s*([0-9]+)\s*</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)  # full-match regex for replies like "<answer>906</answer>"

_ANSWER_TAG_RE = re.compile(
    r"<answer>\s*([0-9]+)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)  # tag-extraction regex for text like "reasoning... <answer>906</answer>"


def build_three_digit_addition_dataset(
    *,
    sample_count: int,
    # sample_count example:
    # 100
    seed: int,
    # seed example:
    # 7
) -> Dataset:
    """Build a synthetic eval set of unique 3-digit addition problems.

    Each row looks like:
    {"id": 0, "left": 314, "right": 592, "expected_answer": "906", "prompt": "..."}
    """
    rng = Random(seed)  # e.g. Random(7)
    rows: list[dict[str, int | str]] = []
    # rows example:
    # [{"id": 0, "left": 314, "right": 592, "expected_answer": "906", "prompt": "..."}]
    seen_problems: set[tuple[int, int]] = set()  # {(314, 592), (408, 177)}

    while len(rows) < sample_count:
        left = rng.randint(100, 999)  # e.g. 314
        right = rng.randint(100, 999)  # e.g. 592
        problem = (left, right)  # e.g. (314, 592)
        if problem in seen_problems:
            continue

        seen_problems.add(problem)
        rows.append(
            {
                "id": len(rows),
                "left": left,
                "right": right,
                "expected_answer": str(left + right),
                "prompt": (
                    "You are solving a 3-digit addition problem.\n"
                    "Work it out carefully.\n"
                    "Reply with the final sum inside <answer>...</answer>.\n"
                    "Do not return anything else.\n\n"
                    f"Problem: {left} + {right}"
                ),
            }
        )

    return Dataset.from_list(rows)


def parse_answer(completion_text: str) -> str | None:
    # completion_text example:
    # "<answer>906</answer>"
    match = _ANSWER_TAG_RE.search(completion_text)
    # match example:
    # regex match for "<answer>906</answer>", or None
    if match is None:
        return None
    return str(int(match.group(1)))


def score_addition_completion(
    completion_text: str,
    # completion_text example:
    # "<answer>906</answer>"
    row: dict[str, int | str],
    # row example:
    # {"id": 0, "left": 314, "right": 592, "expected_answer": "906", "prompt": "..."}
) -> dict[str, Any]:
    """Score one model completion against one addition problem.

    Returns:
    {"id": 0, "left": 314, "right": 592, "expected_answer": "906",
     "completion": "<answer>906</answer>", "parsed_answer": "906",
     "format_ok": True, "exact_match": True, "absolute_error": 0}
    """
    expected_answer = str(row["expected_answer"])  # e.g. "906"
    parsed_answer = parse_answer(completion_text)  # e.g. "906" or None
    format_ok = _STRICT_ANSWER_RE.match(completion_text) is not None  # e.g. True
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
    # [{"id": 0, "left": 314, "right": 592, "expected_answer": "906",
    #   "completion": "<answer>906</answer>", "parsed_answer": "906",
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
        # {"id": 0, "left": 314, "right": 592, "expected_answer": "906",
        #  "completion": "<answer>906</answer>", "parsed_answer": "906",
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


def compute_addition_metrics(
    prompts: list[str],
    # prompts example:
    # ["You are solving a 3-digit addition problem...\n\nProblem: 314 + 592"]
    completions: list[str],
    # completions example:
    # ["<answer>906</answer>", "<answer>111</answer>"]
    dataset_rows: Dataset,
    # each row looks like:
    # {"id": 0, "left": 314, "right": 592, "expected_answer": "906", "prompt": "..."}
    config: dict[str, Any],
    # config example:
    # {"task": {"eval_samples": 100, "eval_seed": 7}}
    tokenizer: Any,
    # tokenizer example:
    # a Hugging Face tokenizer object for the eval model, e.g. Qwen tokenizer
) -> dict[str, Any]:
    """Compute eval metrics for 3-digit addition.

    config example:
    {"task": {"eval_samples": 100, "eval_seed": 7}}

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
                "left": 314,
                "right": 592,
                "expected_answer": "906",
                "completion": "<answer>906</answer>",
                "parsed_answer": "906",
                "format_ok": True,
                "exact_match": True,
                "absolute_error": 0,
            }
        ],
    }
    """
    del prompts, config, tokenizer

    detailed_results: list[dict[str, Any]] = []
    # detailed_results example:
    # [{"id": 0, "left": 314, "right": 592, "expected_answer": "906",
    #   "completion": "<answer>906</answer>", "parsed_answer": "906",
    #   "format_ok": True, "exact_match": True, "absolute_error": 0}]
    for completion_text, row in zip(completions, dataset_rows):
        # completion_text example:
        # "<answer>906</answer>"
        # row example:
        # {"id": 0, "left": 314, "right": 592, "expected_answer": "906", "prompt": "..."}
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


@eval_dataset_template
def three_digit_addition_eval_dataset() -> EvalDatasetTemplate:
    return EvalDatasetTemplate(
        # config example: {"task": {"eval_samples": 100, "eval_seed": 7}}
        build=lambda config: build_three_digit_addition_dataset(
            sample_count=int(
                config.get("task", {}).get("eval_samples", _DEFAULT_EVAL_SAMPLES)
            ),  # e.g. 100
            seed=int(config.get("task", {}).get("eval_seed", _DEFAULT_EVAL_SEED)),  # e.g. 7
        ),
    )


@eval_metrics_template
def three_digit_addition_metrics() -> EvalMetricsTemplate:
    return EvalMetricsTemplate(
        compute=compute_addition_metrics,
    )
