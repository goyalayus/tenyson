#!/usr/bin/env python3
"""Build a 60-case eval prompt set with 20 cases each for turns 3/4/5."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a competitive game player. Make sure you read the game instructions carefully, "
    "and always follow the required format.\n\n"
    "In each turn, think step-by-step inside <think>...</think> tags, then follow the "
    "instructions inside <guess>...</guess> tags."
)

USER_HEADER = (
    "You are Player 0 in Wordle.\n"
    "A secret 5-letter word has been chosen. You have 6 attempts to guess it.\n"
    "For each guess, wrap your word in square brackets (e.g., [apple]).\n"
    "Feedback for each letter will be given as follows:\n"
    "  - G (green): correct letter in the correct position\n"
    "  - Y (yellow): letter exists in the word but in the wrong position\n"
    "  - X (wrong): letter is not in the word\n"
    "Enter your guess to begin.\n\n"
)

USER_FOOTER = (
    "\nEnter your next guess (think step-by-step in <think> tags, then output <guess>[word]</guess>)."
)


def load_words(path: Path) -> list[str]:
    words = [w.strip().lower() for w in path.read_text().splitlines() if w.strip()]
    return [w for w in words if len(w) == 5 and w.isalpha()]


def wordle_feedback(guess: str, secret: str) -> str:
    result = ["X"] * 5
    remaining = Counter()
    for i, (g, s) in enumerate(zip(guess, secret)):
        if g == s:
            result[i] = "G"
        else:
            remaining[s] += 1
    for i, g in enumerate(guess):
        if result[i] == "G":
            continue
        if remaining[g] > 0:
            result[i] = "Y"
            remaining[g] -= 1
    return " ".join(result)


def build_case(case_id: str, turn: int, history: list[tuple[str, str]]) -> dict:
    attempts_left = 6 - turn
    history_lines = "\n".join(
        f"Turn {i}: [{guess}] -> {feedback}" for i, (guess, feedback) in enumerate(history, start=1)
    )
    user = (
        USER_HEADER
        + f"This is turn {turn} of the game. You have {attempts_left} attempts left.\n\n"
        + "Prior turns and feedback:\n"
        + history_lines
        + USER_FOOTER
    )
    return {
        "case_id": case_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build turn3/4/5 prompt file with 20 cases each.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--per-turn", type=int, default=20)
    parser.add_argument(
        "--solutions",
        default="qwen4b-lora-wordle/RL/wordlists/wordle_solutions.txt",
    )
    parser.add_argument(
        "--guesses",
        default="qwen4b-lora-wordle/RL/wordlists/wordle_allowed_guesses.txt",
    )
    parser.add_argument(
        "--out",
        default="qwen4b-lora-wordle/inference/wordle_eval_prompts_turn345_20_each.json",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    solutions = load_words(Path(args.solutions))
    guesses = load_words(Path(args.guesses))
    if len(solutions) < args.per_turn:
        raise ValueError("Not enough solution words to sample requested cases.")

    # Use distinct secrets across all generated cases.
    total_cases = args.per_turn * 3
    picked_secrets = rng.sample(solutions, total_cases)

    cases = []
    idx = 0
    for turn in (3, 4, 5):
        for j in range(args.per_turn):
            secret = picked_secrets[idx]
            idx += 1
            history: list[tuple[str, str]] = []
            used = {secret}
            needed_history_turns = turn - 1
            while len(history) < needed_history_turns:
                guess = guesses[rng.randrange(len(guesses))]
                if guess in used:
                    continue
                used.add(guess)
                feedback = wordle_feedback(guess, secret)
                # Skip solved rows; keep tasks focused on continuation turns.
                if feedback == "G G G G G":
                    continue
                history.append((guess, feedback))

            case_id = f"t{turn}_{j + 1:02d}"
            cases.append(build_case(case_id, turn, history))

    output = {
        "name": "wordle_eval_prompts_turn345_20_each",
        "version": "v1",
        "notes": "Auto-generated eval prompts with 20 cases each for turns 3, 4, and 5.",
        "cases": cases,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {len(cases)} cases to {out_path}")


if __name__ == "__main__":
    main()
