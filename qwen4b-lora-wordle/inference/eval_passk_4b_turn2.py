#!/usr/bin/env python3
"""Compute pass@k for the 4B Wordle LoRA model on turn-2/single-history prompts.

Default scoring mode is `constraints`, which marks a sampled guess as correct if it is
consistent with all constraints implied by prompt history feedback.

If the prompt case contains explicit gold fields (`expected_guess`, `target_word`,
`expected`, or `answer`), mode `exact_guess` can be used to require exact match.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_eval_two_models import resolve_revision


TURN_RE = re.compile(r"Turn\s+\d+:\s*\[([a-zA-Z]{5})\]\s*->\s*([GYX\s]+)")
GAME_TURN_RE = re.compile(r"This is turn\s+(\d+)\s+of the game", re.IGNORECASE)
GUESS_TAG_RE = re.compile(r"<guess>\s*\[([a-zA-Z]{5})\]\s*</guess>")

DEFAULT_SPEC = {
    "key": "qwen3_4b",
    "label": "Qwen3-4B LoRA",
    "base_model": "Qwen/Qwen3-4B",
    "adapter_repo": "goyalayus/wordle-lora-qwen3-4b",
}


@dataclass
class Constraints:
    fixed_positions: Dict[int, str]
    banned_positions: Dict[str, set]
    min_count: Dict[str, int]
    max_count: Dict[str, int]


@dataclass
class GuessCheck:
    is_valid_word: bool
    is_success: bool
    reasons: List[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pass@k for 4B model on turn-2 prompts.")
    p.add_argument("--prompts-json", default="wordle_eval_prompts_10.json", help="Prompt JSON path")
    p.add_argument("--output-json", default="passk_turn2_qwen3_4b.json", help="Output report JSON")
    p.add_argument("--k", type=int, default=64, help="pass@k target")
    p.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of sampled generations per prompt (must be >= k for exact pass@k)",
    )
    p.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens per sample")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=0.95, help="Sampling top-p")
    p.add_argument("--seed", type=int, default=1234, help="Base random seed")
    p.add_argument(
        "--revision-strategy",
        choices=["latest_step", "final", "main"],
        default="latest_step",
        help="How adapter revision is selected",
    )
    p.add_argument(
        "--scoring",
        choices=["constraints", "exact_guess"],
        default="constraints",
        help="Success predicate for a sampled guess",
    )
    p.add_argument(
        "--require-single-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only prompts with exactly one prior Turn history entry",
    )
    p.add_argument(
        "--require-turn",
        type=int,
        default=2,
        help="If set, keep only prompts explicitly marked as this game turn",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional cap on number of filtered cases (0 = all)",
    )
    p.add_argument("--base-model", default=DEFAULT_SPEC["base_model"], help="HF base model id")
    p.add_argument("--adapter-repo", default=DEFAULT_SPEC["adapter_repo"], help="HF adapter repo id")
    return p.parse_args()


def load_prompt_cases(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    cases = data.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No prompt cases found in {path}")
    for i, case in enumerate(cases):
        if not isinstance(case, dict) or "case_id" not in case or "messages" not in case:
            raise ValueError(f"Case index {i} missing required keys")
    return cases


def get_user_message(case: Dict[str, Any]) -> str:
    for m in case.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def parse_history(user_text: str) -> List[Tuple[str, str]]:
    history: List[Tuple[str, str]] = []
    for guess, pattern_raw in TURN_RE.findall(user_text):
        pattern = "".join(ch for ch in pattern_raw if ch in "GYX")
        if len(guess) == 5 and len(pattern) == 5:
            history.append((guess.lower(), pattern))
    return history


def parse_turn_number(user_text: str) -> Optional[int]:
    m = GAME_TURN_RE.search(user_text)
    if not m:
        return None
    return int(m.group(1))


def filter_cases(cases: Sequence[Dict[str, Any]], require_single_history: bool, require_turn: int | None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for case in cases:
        user = get_user_message(case)
        history = parse_history(user)
        turn_no = parse_turn_number(user)

        if require_single_history and len(history) != 1:
            continue
        if require_turn is not None and turn_no != require_turn:
            continue
        out.append(case)
    return out


def build_constraints(history: Sequence[Tuple[str, str]]) -> Constraints:
    fixed_positions: Dict[int, str] = {}
    banned_positions: Dict[str, set] = defaultdict(set)
    min_count: Dict[str, int] = defaultdict(int)
    max_count: Dict[str, int] = defaultdict(lambda: 5)

    for guess, pattern in history:
        by_letter = defaultdict(list)
        for i, (ch, fb) in enumerate(zip(guess, pattern)):
            by_letter[ch].append((i, fb))
            if fb == "G":
                fixed_positions[i] = ch
            if fb in {"Y", "X"}:
                banned_positions[ch].add(i)

        for ch, entries in by_letter.items():
            gy = sum(1 for _, fb in entries if fb in {"G", "Y"})
            x = sum(1 for _, fb in entries if fb == "X")
            if gy > min_count[ch]:
                min_count[ch] = gy
            if x > 0:
                # At least one gray for this letter in this guess means no extra copies beyond gy.
                max_count[ch] = min(max_count[ch], gy)

    return Constraints(
        fixed_positions=fixed_positions,
        banned_positions=banned_positions,
        min_count=dict(min_count),
        max_count=dict(max_count),
    )


def extract_gold_guess(case: Dict[str, Any]) -> Optional[str]:
    for key in ("expected_guess", "target_word", "expected", "answer"):
        val = case.get(key)
        if isinstance(val, str) and re.fullmatch(r"[a-zA-Z]{5}", val):
            return val.lower()
    return None


def parse_guess(response_text: str) -> Optional[str]:
    m = GUESS_TAG_RE.search(response_text)
    return m.group(1).lower() if m else None


def check_guess_constraints(guess: Optional[str], constraints: Constraints) -> GuessCheck:
    if guess is None or not re.fullmatch(r"[a-z]{5}", guess):
        return GuessCheck(False, False, ["no_valid_5_letter_guess"])

    reasons: List[str] = []
    counts = Counter(guess)

    for idx, ch in constraints.fixed_positions.items():
        if guess[idx] != ch:
            reasons.append(f"green_mismatch_pos{idx + 1}")

    for ch, bad_positions in constraints.banned_positions.items():
        for idx in sorted(bad_positions):
            if guess[idx] == ch:
                reasons.append(f"forbidden_pos_{ch}_{idx + 1}")

    for ch, nmin in constraints.min_count.items():
        if counts[ch] < nmin:
            reasons.append(f"min_count_violation_{ch}_{nmin}")

    for ch, nmax in constraints.max_count.items():
        if counts[ch] > nmax:
            reasons.append(f"max_count_violation_{ch}_{nmax}")

    return GuessCheck(True, len(reasons) == 0, reasons)


def check_guess_exact(guess: Optional[str], gold: Optional[str]) -> GuessCheck:
    if guess is None or not re.fullmatch(r"[a-z]{5}", guess):
        return GuessCheck(False, False, ["no_valid_5_letter_guess"])
    if gold is None:
        return GuessCheck(True, False, ["no_gold_guess_in_case"])
    if guess == gold:
        return GuessCheck(True, True, [])
    return GuessCheck(True, False, ["guess_not_equal_gold"])


def pass_at_k(n: int, c: int, k: int) -> float:
    """Standard pass@k estimator used in code-generation evals."""
    if c <= 0:
        return 0.0
    if n < k:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def build_model_and_tokenizer(base_model: str, adapter_repo: str, revision: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_repo, revision=revision)
    model.eval()
    device = str(next(model.parameters()).device)
    return model, tokenizer, device


@torch.inference_mode()
def generate_one_sample(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        generator=gen,
    )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=False).strip()


def main() -> None:
    args = parse_args()

    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")
    if args.num_samples < args.k:
        raise ValueError("--num-samples must be >= --k for exact pass@k")

    prompts_path = Path(args.prompts_json)
    if not prompts_path.exists():
        candidate = Path(__file__).resolve().parent / args.prompts_json
        if candidate.exists():
            prompts_path = candidate
        else:
            raise FileNotFoundError(
                f"Prompts file not found: {args.prompts_json} (also checked {candidate})"
            )
    output_path = Path(args.output_json)

    all_cases = load_prompt_cases(prompts_path)
    cases = filter_cases(
        all_cases,
        require_single_history=args.require_single_history,
        require_turn=args.require_turn,
    )

    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]

    if not cases:
        raise ValueError("No cases remained after filtering. Adjust filter arguments.")

    print(f"Loaded {len(all_cases)} cases; evaluating {len(cases)} filtered cases", flush=True)

    revision, revision_source = resolve_revision(args.adapter_repo, args.revision_strategy)
    print(f"Using adapter revision: {revision} ({revision_source})", flush=True)

    model, tokenizer, device = build_model_and_tokenizer(
        base_model=args.base_model,
        adapter_repo=args.adapter_repo,
        revision=revision,
    )

    per_case_reports: List[Dict[str, Any]] = []
    case_binary_hits = []

    for case_idx, case in enumerate(cases):
        case_id = case["case_id"]
        user = get_user_message(case)
        history = parse_history(user)
        constraints = build_constraints(history)
        gold_guess = extract_gold_guess(case)

        print(
            f"[{case_idx + 1}/{len(cases)}] case={case_id} samples={args.num_samples}",
            flush=True,
        )

        sample_rows: List[Dict[str, Any]] = []
        success_count = 0

        for sample_i in range(args.num_samples):
            sample_seed = args.seed + case_idx * 100_000 + sample_i
            response_text = generate_one_sample(
                model=model,
                tokenizer=tokenizer,
                messages=case["messages"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=sample_seed,
            )
            guess = parse_guess(response_text)

            if args.scoring == "constraints":
                check = check_guess_constraints(guess, constraints)
            else:
                check = check_guess_exact(guess, gold_guess)

            if check.is_success:
                success_count += 1

            sample_rows.append(
                {
                    "sample_index": sample_i,
                    "seed": sample_seed,
                    "parsed_guess": guess,
                    "is_valid_word": check.is_valid_word,
                    "is_success": check.is_success,
                    "failure_reasons": check.reasons,
                    "response_text": response_text,
                }
            )

        case_pass_k = pass_at_k(args.num_samples, success_count, args.k)
        case_hit = int(success_count > 0)
        case_binary_hits.append(case_hit)

        per_case_reports.append(
            {
                "case_id": case_id,
                "turn_number": parse_turn_number(user),
                "history_turn_count": len(history),
                "history": [{"guess": g, "feedback": fb} for g, fb in history],
                "gold_guess": gold_guess,
                "constraints": {
                    "fixed_positions": {str(k + 1): v for k, v in constraints.fixed_positions.items()},
                    "banned_positions": {
                        ch: [idx + 1 for idx in sorted(pos)] for ch, pos in constraints.banned_positions.items()
                    },
                    "min_count": constraints.min_count,
                    "max_count": constraints.max_count,
                },
                "num_samples": args.num_samples,
                "num_success": success_count,
                "binary_hit": case_hit,
                "pass_at_k": case_pass_k,
                "samples": sample_rows,
            }
        )

    aggregate_binary = sum(case_binary_hits) / len(case_binary_hits)
    aggregate_estimator = sum(x["pass_at_k"] for x in per_case_reports) / len(per_case_reports)

    report = {
        "generated_at_utc": utc_now(),
        "model_spec": {
            "key": DEFAULT_SPEC["key"],
            "label": DEFAULT_SPEC["label"],
            "base_model": args.base_model,
            "adapter_repo": args.adapter_repo,
            "adapter_revision_used": revision,
            "adapter_revision_source": revision_source,
            "device": device,
        },
        "config": {
            "prompts_json": str(prompts_path),
            "output_json": str(output_path),
            "k": args.k,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "revision_strategy": args.revision_strategy,
            "scoring": args.scoring,
            "require_single_history": args.require_single_history,
            "require_turn": args.require_turn,
            "max_cases": args.max_cases,
        },
        "dataset": {
            "total_cases_in_file": len(all_cases),
            "evaluated_cases": len(cases),
        },
        "metrics": {
            "binary_hit_rate_any_success": aggregate_binary,
            "mean_pass_at_k_estimator": aggregate_estimator,
        },
        "per_case": per_case_reports,
    }

    output_path.write_text(json.dumps(report, indent=2))
    print(f"Saved report to: {output_path}", flush=True)
    print(f"Aggregate binary hit rate (any success): {aggregate_binary:.4f}", flush=True)
    print(f"Aggregate mean pass@{args.k} estimator: {aggregate_estimator:.4f}", flush=True)


if __name__ == "__main__":
    random.seed(0)
    main()
