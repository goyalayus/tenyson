from __future__ import annotations

from random import Random
from typing import Any

from datasets import Dataset
from tenyson import eval_dataset_fn

from functional import build_addition_messages


def build_three_digit_prompt(left: int, right: int) -> str:
    return (
        "You are solving a 3-digit addition problem.\n"
        "Do not show your working.\n"
        "Reply only with digits inside <answer>...</answer>.\n\n"
        f"Problem: {left} + {right}"
    )


@eval_dataset_fn
def build_three_digit_addition_dataset(
    *,
    sample_count: int,
    seed: int,
) -> Dataset:
    """Build a synthetic eval set of unique 3-digit addition problems."""

    rng = Random(seed)
    rows: list[dict[str, Any]] = []
    seen_problems: set[tuple[int, int]] = set()

    while len(rows) < sample_count:
        left = rng.randint(100, 999)
        right = rng.randint(100, 999)
        problem = (left, right)

        if problem in seen_problems:
            continue

        seen_problems.add(problem)
        prompt_text = build_three_digit_prompt(left, right)
        rows.append(
            {
                "id": len(rows),
                "left": left,
                "right": right,
                "expected_answer": str(left + right),
                "prompt": prompt_text,
                "messages": build_addition_messages(prompt_text),
            }
        )

    return Dataset.from_list(rows)
