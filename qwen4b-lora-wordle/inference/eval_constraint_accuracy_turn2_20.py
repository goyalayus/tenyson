#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "You are a competitive game player. Make sure you read the game instructions carefully, "
    "and always follow the required format.\n\n"
    "In each turn, think step-by-step inside <think>...</think> tags, then follow the "
    "instructions inside <guess>...</guess> tags."
)

USER_PROMPT_TEMPLATE = (
    "You are Player 0 in Wordle.\n"
    "A secret 5-letter word has been chosen. You have 6 attempts to guess it.\n"
    "For each guess, wrap your word in square brackets (e.g., [apple]).\n"
    "Feedback for each letter will be given as follows:\n"
    "  - G (green): correct letter in the correct position\n"
    "  - Y (yellow): letter exists in the word but in the wrong position\n"
    "  - X (wrong): letter is not in the word\n"
    "Enter your guess to begin.\n\n"
    "This is turn 2 of the game. You have 4 attempts left.\n\n"
    "Prior turns and feedback:\n"
    "Turn 1: [{turn1_guess}] -> {feedback}\n\n"
    "Enter your next guess (think step-by-step in <think> tags, then output <guess>[word]</guess>)."
)

STRICT_GUESS_RE = re.compile(
    r"<guess>\s*\[([a-zA-Z]{5})\]\s*</guess>", re.IGNORECASE | re.DOTALL
)
FALLBACK_GUESS_RE = re.compile(r"\[([a-zA-Z]{5})\]")


@dataclass
class ConstraintSet:
    green_positions: dict
    yellow_banned_positions: dict
    yellow_min_counts: dict
    letter_min_counts: dict
    letter_max_counts: dict
    gray_only_letters: set


def parse_guess(text: str):
    strict = STRICT_GUESS_RE.search(text)
    if strict:
        return strict.group(1).lower(), True
    fallback = FALLBACK_GUESS_RE.search(text)
    if fallback:
        return fallback.group(1).lower(), False
    return None, False


def build_constraints(turn1_guess: str, feedback: str) -> ConstraintSet:
    turn1_guess = turn1_guess.lower()
    feedback = "".join(ch for ch in feedback.upper() if ch in "GYX")

    green_positions = {}
    yellow_banned_positions = defaultdict(set)
    yellow_counts = defaultdict(int)
    letter_rows = defaultdict(list)

    for idx, (letter, signal) in enumerate(zip(turn1_guess, feedback)):
        letter_rows[letter].append(signal)
        if signal == "G":
            green_positions[idx] = letter
        elif signal == "Y":
            yellow_banned_positions[letter].add(idx)
            yellow_counts[letter] += 1

    letter_min_counts = {}
    letter_max_counts = {}
    gray_only_letters = set()

    for letter, rows in letter_rows.items():
        non_gray_count = sum(1 for signal in rows if signal in {"G", "Y"})
        has_gray = any(signal == "X" for signal in rows)
        if non_gray_count > 0:
            letter_min_counts[letter] = non_gray_count
        letter_max_counts[letter] = non_gray_count if has_gray else 5
        if non_gray_count == 0 and has_gray:
            gray_only_letters.add(letter)

    return ConstraintSet(
        green_positions=green_positions,
        yellow_banned_positions=dict(yellow_banned_positions),
        yellow_min_counts=dict(yellow_counts),
        letter_min_counts=letter_min_counts,
        letter_max_counts=letter_max_counts,
        gray_only_letters=gray_only_letters,
    )


def is_consistent(guess: str, cs: ConstraintSet):
    if not re.fullmatch(r"[a-z]{5}", guess):
        return False, "non_alpha_or_len"

    counts = Counter(guess)

    for idx, letter in cs.green_positions.items():
        if guess[idx] != letter:
            return False, "green_violation"

    for letter, banned_positions in cs.yellow_banned_positions.items():
        for idx in banned_positions:
            if guess[idx] == letter:
                return False, "yellow_pos_violation"

    for letter, min_count in cs.yellow_min_counts.items():
        if counts[letter] < min_count:
            return False, "yellow_count_violation"

    for letter, min_count in cs.letter_min_counts.items():
        if counts[letter] < min_count:
            return False, "min_count_violation"

    for letter, max_count in cs.letter_max_counts.items():
        if counts[letter] > max_count:
            return False, "max_count_violation"

    for letter in cs.gray_only_letters:
        if counts[letter] != 0:
            return False, "gray_only_violation"

    return True, "ok"


def run_case(
    model,
    tokenizer,
    case: dict,
    num_samples: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
):
    prompt = USER_PROMPT_TEMPLATE.format(
        turn1_guess=case["turn1_guess"], feedback=case["turn1_feedback"]
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    constraints = build_constraints(case["turn1_guess"], case["turn1_feedback"])
    samples = []
    num_valid = 0
    num_strict = 0
    parse_fail = 0

    for sample_idx in range(num_samples):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        guess, strict_ok = parse_guess(response)
        if strict_ok:
            num_strict += 1

        if guess is None:
            parse_fail += 1
            is_ok = False
            reason = "parse_fail"
        else:
            is_ok, reason = is_consistent(guess, constraints)

        num_valid += int(is_ok)
        samples.append(
            {
                "sample_idx": sample_idx,
                "guess": guess,
                "strict_format_ok": strict_ok,
                "constraints_ok": is_ok,
                "status": reason,
                "response": response[:1200],
            }
        )

    return {
        "id": case["id"],
        "turn1_guess": case["turn1_guess"],
        "turn1_feedback": case["turn1_feedback"],
        "num_samples": num_samples,
        "num_constraint_valid": num_valid,
        "constraint_accuracy": num_valid / num_samples,
        "num_strict_format_ok": num_strict,
        "strict_format_rate": num_strict / num_samples,
        "num_parse_fail": parse_fail,
        "samples": samples,
    }


def load_cases(path: Path):
    cases = []
    with path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            cases.append(
                {
                    "id": row["id"],
                    "turn1_guess": row["turn1_guess"].lower(),
                    "turn1_feedback": row["turn1_feedback"],
                }
            )
    return cases


def main():
    parser = argparse.ArgumentParser(
        description="Constraint-following accuracy for 20 turn-2 single-history prompts."
    )
    parser.add_argument("--base-model-fallback", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--adapter-repo", default="goyalayus/wordle-qwen3-4b-rl-turn2"
    )
    parser.add_argument("--adapter-revision", default="rl-step-00640")
    parser.add_argument(
        "--cases-path",
        default=str(
            Path(__file__).resolve().parent / "turn2_single_history_20.jsonl"
        ),
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parent
            / "turn2_constraint_accuracy_20_results.json"
        ),
    )
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.num_samples > 1 and not args.do_sample:
        raise ValueError("--num-samples > 1 requires --do-sample")

    cases = load_cases(Path(args.cases_path))
    if len(cases) != 20:
        print(
            f"Warning: expected 20 cases for this benchmark, found {len(cases)} at {args.cases_path}."
        )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    peft_cfg = PeftConfig.from_pretrained(
        args.adapter_repo, revision=args.adapter_revision
    )
    base_model = peft_cfg.base_model_name_or_path or args.base_model_fallback

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb,
        torch_dtype=torch.float16,
    )
    print(f"Loading adapter: {args.adapter_repo}@{args.adapter_revision}")
    model = PeftModel.from_pretrained(
        base, args.adapter_repo, revision=args.adapter_revision
    )
    model.eval()

    per_case = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] Running {case['id']} ...", flush=True)
        per_case.append(
            run_case(
                model=model,
                tokenizer=tokenizer,
                case=case,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        )

    total_predictions = sum(case["num_samples"] for case in per_case)
    total_valid = sum(case["num_constraint_valid"] for case in per_case)
    total_strict = sum(case["num_strict_format_ok"] for case in per_case)
    total_parse_fail = sum(case["num_parse_fail"] for case in per_case)
    case_any_success = sum(1 for case in per_case if case["num_constraint_valid"] > 0)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "constraint_accuracy_turn2_single_history_20",
        "base_model": base_model,
        "adapter_repo": args.adapter_repo,
        "adapter_revision": args.adapter_revision,
        "num_cases": len(per_case),
        "num_samples_per_case": args.num_samples,
        "do_sample": args.do_sample,
        "constraint_accuracy_overall": total_valid / total_predictions,
        "strict_format_rate_overall": total_strict / total_predictions,
        "parse_fail_rate_overall": total_parse_fail / total_predictions,
        "case_any_success_rate": case_any_success / len(per_case),
        "totals": {
            "num_predictions": total_predictions,
            "num_constraint_valid": total_valid,
            "num_strict_format_ok": total_strict,
            "num_parse_fail": total_parse_fail,
            "num_cases_with_any_success": case_any_success,
        },
        "cases": per_case,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"DONE {out_path}")
    print("constraint_accuracy_overall:", summary["constraint_accuracy_overall"])
    print("strict_format_rate_overall:", summary["strict_format_rate_overall"])
    print("parse_fail_rate_overall:", summary["parse_fail_rate_overall"])
    print(
        "num_cases_with_any_success:",
        summary["totals"]["num_cases_with_any_success"],
        "/",
        summary["num_cases"],
    )


if __name__ == "__main__":
    main()
