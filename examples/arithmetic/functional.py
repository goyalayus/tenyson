from __future__ import annotations

import re
from random import Random
from typing import Any, Sequence

from datasets import Dataset
from tenyson import (
    EvalMetricsContext,
    RLDatasetTemplate,
    RLRewardTemplate,
    chat_sft_dataset_fn,
    eval_dataset_fn,
    eval_metrics_fn,
    rl_dataset_template,
    rl_reward_template,
)

_ANSWER_PREFIX = "<answer>"
_ANSWER_SUFFIX = "</answer>"
_CORRECT_ANSWER_REWARD = 1.0
_STRICT_FORMAT_REWARD = 0.1

_STRICT_ANSWER_RE = re.compile(
    rf"^\s*([0-9]+)\s*{re.escape(_ANSWER_SUFFIX)}\s*$",
    re.IGNORECASE | re.DOTALL,
)  # full-match regex for the intended tail-only form like "6859</answer>"

_ANSWER_TEXT_RE = re.compile(
    rf"(?:{re.escape(_ANSWER_PREFIX)}\s*)?([0-9]+)\s*{re.escape(_ANSWER_SUFFIX)}",
    re.IGNORECASE | re.DOTALL,
)  # extraction regex for text like "906</answer>" or "reasoning... <answer>906</answer>"


def build_addition_prompt(
    digits: int,
    left: int,
    right: int,
) -> str:
    """Build the human-readable prompt shown to metrics, logs, and reviewers."""

    # digits example:
    # 4
    # left example:
    # 4312
    # right example:
    # 2547
    return (
        f"You are solving a {digits}-digit addition problem.\n"
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


def build_sft_addition_messages(
    prompt_text: str,
    expected_answer: str,
) -> list[dict[str, str]]:
    """Build supervised chat messages with the full tagged answer."""

    # prompt_text example:
    # "You are solving a 2-digit addition problem...\nProblem: 31 + 47"
    # expected_answer example:
    # "78"
    return [
        {"role": "user", "content": prompt_text},
        {
            "role": "assistant",
            "content": f"{_ANSWER_PREFIX}{expected_answer}{_ANSWER_SUFFIX}",
        },
    ]


def build_addition_problem_rows(
    *,
    digits: int,
    # digits example:
    # 2
    sample_count: int,
    # sample_count example:
    # 100
    seed: int,
    # seed example:
    # 7
    excluded_problems: set[tuple[int, int]] | None = None,
    # excluded_problems example:
    # {(31, 47), (52, 18)} or None
) -> list[dict[str, Any]]:
    """Build unique addition problems before task-specific message formatting."""

    min_value = 10 ** max(digits - 1, 0)  # e.g. 10 for 2-digit
    max_value = (10**digits) - 1  # e.g. 99 for 2-digit
    rng = Random(seed)  # e.g. Random(7)
    rows: list[dict[str, Any]] = []
    # rows example:
    # [{"id": 0, "left": 31, "right": 47, "expected_answer": "78",
    #   "prompt": "...\nProblem: 31 + 47"}]
    seen_problems: set[tuple[int, int]] = set(excluded_problems or set())
    # seen_problems example:
    # {(31, 47), (52, 18)}

    while len(rows) < sample_count:
        left = rng.randint(min_value, max_value)  # e.g. 31
        right = rng.randint(min_value, max_value)  # e.g. 47
        problem = (left, right)  # e.g. (31, 47)

        if problem in seen_problems:
            continue

        seen_problems.add(problem)
        expected_answer = str(left + right)  # e.g. "78"
        prompt_text = build_addition_prompt(digits, left, right)
        # prompt_text example:
        # "You are solving a 2-digit addition problem...\nProblem: 31 + 47"
        rows.append(
            {
                "id": len(rows),
                "left": left,
                "right": right,
                "expected_answer": expected_answer,
                "prompt": prompt_text,
            }
        )

    return rows


def build_heldout_addition_train_rows(
    *,
    digits: int,
    # digits example:
    # 2
    sample_count: int,
    # sample_count example:
    # 4096
    seed: int,
    # seed example:
    # 123
    benchmark_sample_count: int,
    # benchmark_sample_count example:
    # 100
    benchmark_seed: int,
    # benchmark_seed example:
    # 7
) -> list[dict[str, Any]]:
    """Build training rows while holding the fixed eval benchmark out."""

    heldout_eval_rows = build_addition_problem_rows(
        digits=digits,
        sample_count=benchmark_sample_count,
        seed=benchmark_seed,
    )
    # heldout_eval_rows example:
    # [{"id": 0, "left": 31, "right": 47, "expected_answer": "78",
    #   "prompt": "...\nProblem: 31 + 47"}]
    heldout_eval_problems = {
        (int(row["left"]), int(row["right"]))
        for row in heldout_eval_rows
    }
    # heldout_eval_problems example:
    # {(31, 47), (52, 18)}
    train_rows = build_addition_problem_rows(
        digits=digits,
        sample_count=sample_count,
        seed=seed,
        excluded_problems=heldout_eval_problems,
    )
    # train_rows example:
    # [{"id": 0, "left": 64, "right": 22, "expected_answer": "86",
    #   "prompt": "...\nProblem: 64 + 22"}]
    return train_rows


def build_addition_chat_rows(
    problem_rows: Sequence[dict[str, Any]],
    # problem_rows example:
    # [{"id": 0, "left": 4312, "right": 2547, "expected_answer": "6859",
    #   "prompt": "...\nProblem: 4312 + 2547"}]
) -> list[dict[str, Any]]:
    """Attach the prefilled assistant answer span used by eval and RL."""

    rows: list[dict[str, Any]] = []
    # rows example:
    # [{"id": 0, "left": 4312, "right": 2547, "expected_answer": "6859",
    #   "prompt": "...\nProblem: 4312 + 2547",
    #   "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<answer>"}]}]

    for row in problem_rows:
        messages = build_addition_messages(str(row["prompt"]))
        # messages example:
        # [{"role": "user", "content": "You are solving a 4-digit addition problem...\nProblem: 4312 + 2547"},
        #  {"role": "assistant", "content": "<answer>"}]
        rows.append(
            {
                **row,
                "messages": messages,
            }
        )

    return rows


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


def extract_generation_text(value: Any) -> str:
    """Normalize a generation payload into plain text for scoring/rewards."""

    # value example:
    # "78</answer>" or {"content": "78</answer>"}
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content

    return str(value)


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


def get_addition_reward_funcs() -> list[Any]:
    """Build the tiny arithmetic RL reward: exact answer + strict format."""

    def reward_correct_answer(
        _prompts: Sequence[Any],
        completions: Sequence[Any],
        expected_answer: Sequence[Any],
    ) -> list[float]:
        rewards: list[float] = []
        # rewards example:
        # [1.0, 0.0, 0.0]

        for completion_obj, expected in zip(completions, expected_answer):
            completion_text = extract_generation_text(completion_obj)
            # completion_text example:
            # "78</answer>"
            parsed_answer = parse_answer(completion_text)
            # parsed_answer example:
            # "78" or None
            rewards.append(
                _CORRECT_ANSWER_REWARD
                if parsed_answer is not None and parsed_answer == str(expected)
                else 0.0
            )

        return rewards

    def reward_strict_format(
        _prompts: Sequence[Any],
        completions: Sequence[Any],
    ) -> list[float]:
        rewards: list[float] = []
        # rewards example:
        # [0.1, 0.0, 0.1]

        for completion_obj in completions:
            completion_text = extract_generation_text(completion_obj)
            # completion_text example:
            # "78</answer>"
            rewards.append(
                _STRICT_FORMAT_REWARD
                if has_strict_answer_format(completion_text)
                else 0.0
            )

        return rewards

    setattr(reward_correct_answer, "tenyson_reward_name", "correct_answer")
    setattr(reward_strict_format, "tenyson_reward_name", "strict_format")
    return [reward_correct_answer, reward_strict_format]


@eval_dataset_fn
def build_addition_dataset(
    *,
    digits: int,
    # digits example:
    # 4
    sample_count: int,
    # sample_count example:
    # 100
    seed: int,
    # seed example:
    # 7
) -> Dataset:
    """Build a synthetic eval set of unique fixed-width addition problems.

    `@eval_dataset_fn` makes the Tenyson hook contract visible in this file.
    This builder stays explicit about the values it really needs. The arithmetic
    experiment binds `digits=`, `sample_count=`, and `seed=` directly at the
    call site with `bind_eval_dataset(...)`, so the task file does not depend on
    hidden config injection for those inputs.

    Each row looks like:
    {"id": 0, "left": 4312, "right": 2547, "expected_answer": "6859",
     "prompt": "...\nProblem: 4312 + 2547",
     "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<answer>"}]}
    """

    problem_rows = build_addition_problem_rows(
        digits=digits,
        sample_count=sample_count,
        seed=seed,
    )
    # problem_rows example:
    # [{"id": 0, "left": 4312, "right": 2547, "expected_answer": "6859",
    #   "prompt": "...\nProblem: 4312 + 2547"}]
    rows = build_addition_chat_rows(problem_rows)
    # rows example:
    # [{"id": 0, "left": 4312, "right": 2547, "expected_answer": "6859",
    #   "prompt": "...\nProblem: 4312 + 2547",
    #   "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<answer>"}]}]
    return Dataset.from_list(rows)


@eval_metrics_fn
def compute_addition_metrics(
    ctx: EvalMetricsContext,
    # ctx example:
    # EvalMetricsContext(
    #   prompts=["You are solving a 4-digit addition problem...\nProblem: 4312 + 2547"],
    #   completions=["6859</answer>", "1111</answer>"],
    #   dataset_rows=Dataset([...]),
    #   config={"evaluation": {"batch_size": 32},
    #           "chat_template": {"enable_thinking": False, "stop_strings": ["</answer>"]}},
    #   tokenizer=<Qwen tokenizer>,
    # )
) -> dict[str, Any]:
    """Compute eval metrics for addition.

    `@eval_metrics_fn` makes the Tenyson hook contract visible in this file.
    The eval runner passes one `EvalMetricsContext`, so simple tasks do not
    need to expose every framework-plumbing input in their function signature.

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

    for completion_text, row in zip(ctx.completions, ctx.dataset_rows):
        # completion_text example:
        # "6859</answer>"
        # row example:
        # {"id": 0, "left": 4312, "right": 2547, "expected_answer": "6859",
        #  "prompt": "...\nProblem: 4312 + 2547",
        #  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<answer>"}]}
        detailed_results.append(score_addition_completion(completion_text, row))

    metrics = summarize_addition_results(
        detailed_results,
        total_samples=len(ctx.completions),
    )
    # metrics example:
    # {"format_accuracy": 0.91, "exact_match_accuracy": 0.84,
    #  "avg_abs_error": 3.6, "parsed_answer_rate": 0.95, "total_samples": 100}

    return {
        "metrics": metrics,
        "detailed_results": detailed_results,
    }


@chat_sft_dataset_fn
def build_addition_sft_train_dataset(
    *,
    digits: int,
    # digits example:
    # 2
    train_sample_count: int,
    # train_sample_count example:
    # 4096
    train_seed: int,
    # train_seed example:
    # 123
    benchmark_sample_count: int,
    # benchmark_sample_count example:
    # 100
    benchmark_seed: int,
    # benchmark_seed example:
    # 7
) -> Dataset:
    """Build SFT chat data while holding the eval benchmark problems out."""

    train_rows = build_heldout_addition_train_rows(
        digits=digits,
        sample_count=train_sample_count,
        seed=train_seed,
        benchmark_sample_count=benchmark_sample_count,
        benchmark_seed=benchmark_seed,
    )
    # train_rows example:
    # [{"id": 0, "left": 64, "right": 22, "expected_answer": "86",
    #   "prompt": "...\nProblem: 64 + 22"}]
    sft_rows: list[dict[str, Any]] = []
    # sft_rows example:
    # [{"id": 0, "left": 64, "right": 22, "expected_answer": "86",
    #   "prompt": "...\nProblem: 64 + 22",
    #   "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<answer>86</answer>"}]}]

    for row in train_rows:
        messages = build_sft_addition_messages(
            str(row["prompt"]),
            str(row["expected_answer"]),
        )
        # messages example:
        # [{"role": "user", "content": "You are solving a 2-digit addition problem...\nProblem: 64 + 22"},
        #  {"role": "assistant", "content": "<answer>86</answer>"}]
        sft_rows.append(
            {
                **row,
                "messages": messages,
            }
        )

    return Dataset.from_list(sft_rows)


@rl_dataset_template
def addition_rl_dataset(
    *,
    digits: int,
    # digits example:
    # 2
    sample_count: int,
    # sample_count example:
    # 4096
    seed: int,
    # seed example:
    # 456
    benchmark_sample_count: int,
    # benchmark_sample_count example:
    # 100
    benchmark_seed: int,
    # benchmark_seed example:
    # 7
) -> RLDatasetTemplate:
    """Build an RL train set while holding the eval benchmark problems out."""

    def _build(_config: dict[str, Any]) -> Dataset:
        train_rows = build_heldout_addition_train_rows(
            digits=digits,
            sample_count=sample_count,
            seed=seed,
            benchmark_sample_count=benchmark_sample_count,
            benchmark_seed=benchmark_seed,
        )
        # train_rows example:
        # [{"id": 0, "left": 64, "right": 22, "expected_answer": "86",
        #   "prompt": "...\nProblem: 64 + 22"}]
        rows = build_addition_chat_rows(train_rows)
        # rows example:
        # [{"id": 0, "left": 64, "right": 22, "expected_answer": "86",
        #   "prompt": "...\nProblem: 64 + 22",
        #   "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<answer>"}]}]
        return Dataset.from_list(rows)

    return RLDatasetTemplate(build=_build)


@rl_reward_template
def addition_reward() -> RLRewardTemplate:
    """Reward exact arithmetic heavily and strict answer formatting lightly."""

    def _build(_config: dict[str, Any], _tokenizer: Any) -> list[Any]:
        return get_addition_reward_funcs()

    return RLRewardTemplate(build=_build)
