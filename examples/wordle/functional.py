from __future__ import annotations

import re
from functools import lru_cache
from random import Random
from typing import Any, Sequence
from urllib.request import urlopen

from datasets import Dataset, load_dataset
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
from tenyson.core.stage_templates import template_factory_ref

_DEFAULT_SFT_DATASET = "goyalayus/wordle-reasoning-sft-prefix-keep-think"
_WORDLE_ANSWER_LIST_URL = (
    "https://gist.githubusercontent.com/cfreshman/"
    "a03ef2cba789d8cf00c08f767e0fad7b/raw/"
    "a9e55d7e0c08100ce62133a1fa0d9c4f0f542f2c/"
    "wordle-answers-alphabetical.txt"
)
_FUNCTIONAL_MODULE = "examples.wordle.functional"

_TURN_INDEX = 5
_FORMAT_REWARD = 0.1
_YELLOW_REWARD = 0.1
_GREEN_REWARD = 0.15
_PERFECT_BONUS = 0.2

_STRICT_GUESS_RE = re.compile(
    r"<guess>\s*\[([a-zA-Z]{5})\]\s*</guess>",
    re.IGNORECASE | re.DOTALL,
)

_SYSTEM_PROMPT = (
    "You are a competitive game player. Make sure you read the game instructions "
    "carefully, and always follow the required format.\n\n"
    "In each turn, think step-by-step inside <think>...</think> tags, then "
    "follow the instructions inside <guess>...</guess> tags."
)


def _filter_five_letter_words(lines: Sequence[str]) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()

    for raw_line in lines:
        word = str(raw_line).strip().lower()
        if len(word) != 5 or not word.isalpha() or word in seen:
            continue
        seen.add(word)
        words.append(word)

    return words


@lru_cache(maxsize=8)
def load_wordle_answer_list() -> list[str]:
    with urlopen(_WORDLE_ANSWER_LIST_URL, timeout=30) as response:
        lines = response.read().decode("utf-8").splitlines()
    return _filter_five_letter_words(lines)


def compute_feedback(secret: str, guess: str) -> str:
    result = ["X"] * 5
    remaining_secret = list(secret)

    for index, letter in enumerate(guess):
        if letter == remaining_secret[index]:
            result[index] = "G"
            remaining_secret[index] = "*"

    for index, letter in enumerate(guess):
        if result[index] == "G":
            continue
        if letter in remaining_secret:
            result[index] = "Y"
            remaining_secret[remaining_secret.index(letter)] = "*"

    return "".join(result)


def parse_guess(completion_text: str) -> str | None:
    match = _STRICT_GUESS_RE.search(str(completion_text))
    if match is None:
        return None
    return match.group(1).lower()


def build_turn5_prompt(history_rows: Sequence[tuple[str, str]]) -> str:
    if len(history_rows) != 4:
        raise ValueError(
            f"Turn-5 Wordle prompts need exactly 4 history rows, got {len(history_rows)}."
        )

    history_block = "\n".join(
        f"Turn {turn_index}: [{guess}] -> {feedback}"
        for turn_index, (guess, feedback) in enumerate(history_rows, start=1)
    )
    return (
        "You are Player 0 in Wordle.\n"
        "A secret 5-letter word has been chosen.\n"
        "This is turn 5, your final attempt.\n"
        "Feedback uses G for green, Y for yellow, and X for absent.\n\n"
        "Previous turns:\n"
        f"{history_block}\n\n"
        "Think through the constraints, then give your final guess."
    )


def build_turn5_messages(history_rows: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": build_turn5_prompt(history_rows)},
    ]


def sample_turn5_history(
    *,
    valid_words: Sequence[str],
    secret: str,
    rng: Random,
) -> list[tuple[str, str]]:
    history_rows: list[tuple[str, str]] = []
    used_guesses: set[str] = set()

    while len(history_rows) < 4:
        guess = rng.choice(valid_words)
        if guess == secret or guess in used_guesses:
            continue

        feedback = compute_feedback(secret, guess)
        if feedback == "GGGGG":
            continue

        used_guesses.add(guess)
        history_rows.append((guess, feedback))

    return history_rows


def build_turn5_problem_rows(
    *,
    sample_count: int,
    seed: int,
    excluded_secrets: set[str] | None = None,
) -> list[dict[str, Any]]:
    valid_words = load_wordle_answer_list()
    available_words = [word for word in valid_words if word not in (excluded_secrets or set())]
    if not available_words:
        raise ValueError("No words left after excluding eval secrets.")

    rng = Random(seed)
    rows: list[dict[str, Any]] = []

    while len(rows) < sample_count:
        secret = rng.choice(available_words)
        history_rows = sample_turn5_history(
            valid_words=available_words,
            secret=secret,
            rng=rng,
        )
        prompt_text = build_turn5_prompt(history_rows)
        rows.append(
            {
                "id": len(rows),
                "secret": secret,
                "answer": secret,
                "turn_index": _TURN_INDEX,
                "history_rows": history_rows,
                "prompt": prompt_text,
                "messages": build_turn5_messages(history_rows),
            }
        )

    return rows


def score_turn5_completion(
    completion_text: str,
    secret: str,
    *,
    format_reward: float,
    yellow_reward: float,
    green_reward: float,
    perfect_bonus: float,
) -> dict[str, Any]:
    guess = parse_guess(completion_text)
    format_ok = guess is not None

    if not format_ok:
        return {
            "parsed_guess": None,
            "format_ok": False,
            "feedback": None,
            "num_greens": 0,
            "num_yellows": 0,
            "is_correct": False,
            "reward_format": 0.0,
            "reward_feedback": 0.0,
            "reward_total": 0.0,
        }

    feedback = compute_feedback(secret, guess)
    num_greens = feedback.count("G")
    num_yellows = feedback.count("Y")
    is_correct = guess == secret
    reward_feedback = (
        (num_greens * green_reward)
        + (num_yellows * yellow_reward)
        + (perfect_bonus if is_correct else 0.0)
    )

    return {
        "parsed_guess": guess,
        "format_ok": True,
        "feedback": feedback,
        "num_greens": num_greens,
        "num_yellows": num_yellows,
        "is_correct": is_correct,
        "reward_format": format_reward,
        "reward_feedback": reward_feedback,
        "reward_total": format_reward + reward_feedback,
    }


def reward_settings() -> dict[str, float]:
    return {
        "format_reward": _FORMAT_REWARD,
        "yellow_reward": _YELLOW_REWARD,
        "green_reward": _GREEN_REWARD,
        "perfect_bonus": _PERFECT_BONUS,
    }


@chat_sft_dataset_fn
def build_turn5_sft_train_dataset(
    *,
    dataset_name: str = _DEFAULT_SFT_DATASET,
) -> Dataset:
    dataset = load_dataset(dataset_name, split="train")
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f'Expected datasets.Dataset for "{dataset_name}", got {type(dataset).__name__}.'
        )

    turn5_dataset = dataset.filter(lambda row: int(row["turn_index"]) == _TURN_INDEX)
    if len(turn5_dataset) == 0:
        raise ValueError(f'No turn-5 rows found in "{dataset_name}".')

    return turn5_dataset


@eval_dataset_fn
def build_turn5_eval_dataset(
    *,
    sample_count: int,
    seed: int,
) -> Dataset:
    rows = build_turn5_problem_rows(
        sample_count=sample_count,
        seed=seed,
    )
    return Dataset.from_list(rows)


@eval_metrics_fn
def compute_turn5_wordle_metrics(ctx: EvalMetricsContext) -> dict[str, Any]:
    settings = reward_settings()
    detailed_results: list[dict[str, Any]] = []

    total_correct = 0
    total_format_ok = 0
    total_reward = 0.0
    total_feedback_reward = 0.0
    total_greens = 0
    total_yellows = 0

    for completion_text, row in zip(ctx.completions, ctx.dataset_rows):
        scored = score_turn5_completion(
            completion_text,
            str(row["secret"]),
            format_reward=settings["format_reward"],
            yellow_reward=settings["yellow_reward"],
            green_reward=settings["green_reward"],
            perfect_bonus=settings["perfect_bonus"],
        )
        detailed_results.append(
            {
                "id": row["id"],
                "secret": row["secret"],
                "prompt": row["prompt"],
                "completion": completion_text,
                **scored,
            }
        )

        total_correct += int(scored["is_correct"])
        total_format_ok += int(scored["format_ok"])
        total_reward += float(scored["reward_total"])
        total_feedback_reward += float(scored["reward_feedback"])
        total_greens += int(scored["num_greens"])
        total_yellows += int(scored["num_yellows"])

    total_samples = len(ctx.completions)
    return {
        "metrics": {
            "exact_match_accuracy": total_correct / max(total_samples, 1),
            "format_accuracy": total_format_ok / max(total_samples, 1),
            "avg_total_reward": total_reward / max(total_samples, 1),
            "avg_feedback_reward": total_feedback_reward / max(total_samples, 1),
            "avg_greens": total_greens / max(total_samples, 1),
            "avg_yellows": total_yellows / max(total_samples, 1),
            "total_samples": total_samples,
        },
        "detailed_results": detailed_results,
    }


@rl_dataset_template
def turn5_wordle_rl_dataset(
    *,
    sample_count: int,
    seed: int,
    benchmark_sample_count: int,
    benchmark_seed: int,
) -> RLDatasetTemplate:
    def _build(_config: dict[str, Any]) -> Dataset:
        eval_rows = build_turn5_problem_rows(
            sample_count=benchmark_sample_count,
            seed=benchmark_seed,
        )
        excluded_secrets = {str(row["secret"]) for row in eval_rows}
        train_rows = build_turn5_problem_rows(
            sample_count=sample_count,
            seed=seed,
            excluded_secrets=excluded_secrets,
        )
        return Dataset.from_list(train_rows)

    return RLDatasetTemplate(build=_build)


@rl_reward_template
def turn5_wordle_reward() -> RLRewardTemplate:
    def _build(config: dict[str, Any], _tokenizer: Any) -> list[Any]:
        del config
        settings = reward_settings()

        def reward_format(
            _prompts: Sequence[Any],
            completions: Sequence[Any],
            **_kwargs: Any,
        ) -> list[float]:
            rewards: list[float] = []
            for completion in completions:
                guess = parse_guess(_extract_completion_text(completion))
                rewards.append(settings["format_reward"] if guess is not None else 0.0)
            return rewards

        def reward_turn5_feedback(
            _prompts: Sequence[Any],
            completions: Sequence[Any],
            secret: Sequence[Any],
            **_kwargs: Any,
        ) -> list[float]:
            rewards: list[float] = []
            for completion, target in zip(completions, secret):
                scored = score_turn5_completion(
                    _extract_completion_text(completion),
                    str(target),
                    format_reward=settings["format_reward"],
                    yellow_reward=settings["yellow_reward"],
                    green_reward=settings["green_reward"],
                    perfect_bonus=settings["perfect_bonus"],
                )
                rewards.append(float(scored["reward_feedback"]))
            return rewards

        setattr(reward_format, "tenyson_reward_name", "format")
        setattr(reward_turn5_feedback, "tenyson_reward_name", "turn5_feedback")
        return [reward_format, reward_turn5_feedback]

    return RLRewardTemplate(
        build=_build,
        factory_ref=template_factory_ref(
            _FUNCTIONAL_MODULE,
            "turn5_wordle_reward",
        ),
    )


def _extract_completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return str(value)
