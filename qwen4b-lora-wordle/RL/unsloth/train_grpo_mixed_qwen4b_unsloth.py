#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset
from safetensors.torch import load_file as safetensors_load_file


MAX_MODEL_OUTPUT_TOKENS = 2048  # Hard cap to avoid runaway generations / OOMs

SYSTEM_PROMPT = (
    "You are an expert AI playing Wordle.\n"
    "GOAL: Guess the secret 5-letter word in 6 tries.\n\n"
    "GAME RULES:\n"
    "1. You must input a valid 5-letter English word.\n"
    "2. Feedback is given for each letter:\n"
    "   - G (Green): The letter is in the word and in the CORRECT position.\n"
    "   - Y (Yellow): The letter is in the word but in the WRONG position.\n"
    "   - X (Gray): The letter is NOT in the word (or no extra copies exist).\n\n"
    "FORMATTING:\n"
    "First, think step-by-step inside <think>...</think> tags.\n"
    "Then, output your guess inside <guess>[word]</guess> tags.\n"
)

USER_PROMPT_PREFIX = (
    "You are Player 0 in Wordle.\n"
    "A secret 5-letter word has been chosen. You have 6 attempts to guess it.\n"
    "For each guess, wrap your word in square brackets (e.g., [apple]).\n"
    "Feedback for each letter will be given as follows:\n"
    "  - G (green): correct letter in the correct position\n"
    "  - Y (yellow): letter exists in the word but in the wrong position\n"
    "  - X (wrong): letter is not in the word\n"
)

STRICT_FORMAT_RE = re.compile(
    # We append "<think>" to the prompt (see build_prompt_text), so the model's completion
    # starts inside the thinking block. Require ONLY the closing </think> and the guess tags.
    r"</think>.*?<guess>\s*\[([a-zA-Z]{5})\]\s*</guess>",
    re.IGNORECASE | re.DOTALL,
)
APPROX_BRACKET_GUESS_RE = re.compile(r"\[([a-zA-Z]{5})\]")
TURN_LINE_RE = re.compile(r"Turn\s*([0-9]+):\s*\[([a-zA-Z]{5})\]\s*->\s*([GYX\s]+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_word_list(path: Path) -> List[str]:
    words: List[str] = []
    for line in path.read_text().splitlines():
        w = line.strip().lower()
        if re.fullmatch(r"[a-z]{5}", w):
            words.append(w)
    if not words:
        raise ValueError(f"No valid 5-letter words found in {path}")
    return sorted(set(words))


def compute_feedback(secret: str, guess: str) -> str:
    result = ["X"] * 5
    secret_chars = list(secret)

    for i, ch in enumerate(guess):
        if ch == secret_chars[i]:
            result[i] = "G"
            secret_chars[i] = "*"

    for i, ch in enumerate(guess):
        if result[i] == "G":
            continue
        if ch in secret_chars:
            result[i] = "Y"
            secret_chars[secret_chars.index(ch)] = "*"

    return " ".join(result)


def render_user_prompt(history_rows: Sequence[Tuple[str, str]]) -> str:
    turn_idx = len(history_rows) + 1
    attempts_left = max(0, 6 - turn_idx)
    lines = [f"Turn {i}: [{guess}] -> {feedback}" for i, (guess, feedback) in enumerate(history_rows, start=1)]
    history_block = "\n".join(lines)
    return (
        USER_PROMPT_PREFIX
        + f"\nThis is turn {turn_idx} of the game. You have {attempts_left} attempts left.\n\n"
        + "Prior turns and feedback:\n"
        + history_block
        + "\n\n"
        + "Enter your next guess."
    )


def build_prompt_text(history_rows: Sequence[Tuple[str, str]]) -> str:
    # Force the model to begin its completion inside a <think> block.
    return SYSTEM_PROMPT + "\n\n" + render_user_prompt(history_rows=history_rows) + "\n\n<think>"


def _extract_prompt_text(prompt_obj: Any) -> str:
    if isinstance(prompt_obj, str):
        return prompt_obj
    if isinstance(prompt_obj, dict):
        c = prompt_obj.get("content")
        if isinstance(c, str):
            return c
        return json.dumps(prompt_obj, ensure_ascii=True)
    if isinstance(prompt_obj, list):
        texts: List[str] = []
        for item in prompt_obj:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                texts.append(item["content"])
            else:
                texts.append(str(item))
        return "\n".join(texts)
    return str(prompt_obj)


def _extract_completion_text(completion_obj: Any) -> str:
    if isinstance(completion_obj, str):
        return completion_obj
    if isinstance(completion_obj, dict):
        c = completion_obj.get("content")
        if isinstance(c, str):
            return c
        return json.dumps(completion_obj, ensure_ascii=True)
    if isinstance(completion_obj, list):
        chunks: List[str] = []
        for item in completion_obj:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                chunks.append(item["content"])
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(completion_obj)


def parse_strict_guess(response_text: str) -> Optional[str]:
    m = STRICT_FORMAT_RE.search(response_text)
    if not m:
        return None
    return m.group(1).lower()


def parse_bracket_guess(response_text: str) -> Optional[str]:
    m = APPROX_BRACKET_GUESS_RE.search(response_text)
    if not m:
        return None
    return m.group(1).lower()


def _parse_history_from_prompt(prompt_text: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for m in TURN_LINE_RE.finditer(prompt_text):
        guess = m.group(2).lower()
        feedback = "".join(ch for ch in m.group(3).upper() if ch in "GYX")
        if len(guess) != 5 or len(feedback) != 5:
            continue
        rows.append((guess, feedback))
    return rows


@dataclass
class ConstraintSet:
    green_positions: Dict[int, str]
    yellow_banned_positions: Dict[str, set]
    letter_min_counts: Dict[str, int]
    letter_max_counts: Dict[str, int]


def build_constraints(turn_guess: str, feedback: str) -> ConstraintSet:
    green_positions: Dict[int, str] = {}
    yellow_banned_positions: Dict[str, set] = defaultdict(set)

    letter_rows: Dict[str, List[str]] = defaultdict(list)
    for i, (ch, fb) in enumerate(zip(turn_guess, feedback)):
        letter_rows[ch].append(fb)
        if fb == "G":
            green_positions[i] = ch
        elif fb == "Y":
            yellow_banned_positions[ch].add(i)

    letter_min_counts: Dict[str, int] = {}
    letter_max_counts: Dict[str, int] = {}
    for ch, fbs in letter_rows.items():
        non_x = sum(1 for fb in fbs if fb in {"G", "Y"})
        has_x = any(fb == "X" for fb in fbs)
        if non_x > 0:
            letter_min_counts[ch] = non_x
        letter_max_counts[ch] = non_x if has_x else 5

    return ConstraintSet(
        green_positions=green_positions,
        yellow_banned_positions=dict(yellow_banned_positions),
        letter_min_counts=letter_min_counts,
        letter_max_counts=letter_max_counts,
    )


@dataclass
class AggregatedConstraints:
    green_positions: Dict[int, str]
    yellow_banned_positions: Dict[str, set]
    yellow_letters: set
    min_counts: Dict[str, int]
    max_counts: Dict[str, int]


def aggregate_constraints(history_rows: Sequence[Tuple[str, str]]) -> AggregatedConstraints:
    green_positions: Dict[int, str] = {}
    yellow_banned_positions: Dict[str, set] = defaultdict(set)
    yellow_letters: set = set()
    min_counts: Dict[str, int] = defaultdict(int)
    max_counts: Dict[str, int] = defaultdict(lambda: 5)

    for turn_guess, turn_feedback in history_rows:
        cs = build_constraints(turn_guess=turn_guess, feedback=turn_feedback)

        for i, ch in cs.green_positions.items():
            prev = green_positions.get(i)
            if prev is not None and prev != ch:
                raise ValueError(f"Contradictory green constraints at pos {i}: {prev} vs {ch}")
            green_positions[i] = ch

        for ch, posset in cs.yellow_banned_positions.items():
            yellow_letters.add(ch)
            yellow_banned_positions[ch] |= set(posset)

        for ch, mn in cs.letter_min_counts.items():
            if mn > min_counts[ch]:
                min_counts[ch] = mn

        for ch, mx in cs.letter_max_counts.items():
            if mx < max_counts[ch]:
                max_counts[ch] = mx

    return AggregatedConstraints(
        green_positions=green_positions,
        yellow_banned_positions=dict(yellow_banned_positions),
        yellow_letters=yellow_letters,
        min_counts=dict(min_counts),
        max_counts=dict(max_counts),
    )


def compute_sat_count(guess: str, ac: AggregatedConstraints) -> Tuple[int, Dict[str, int], Dict[str, int]]:
    gcount = Counter(guess)
    green_guaranteed = Counter(ac.green_positions.values())

    totals = {"green": 0, "yellow": 0, "absent": 0, "maxcap": 0}
    sats = {"green": 0, "yellow": 0, "absent": 0, "maxcap": 0}
    sat = 0

    for i, ch in ac.green_positions.items():
        totals["green"] += 1
        if guess[i] == ch:
            sat += 1
            sats["green"] += 1

    for ch in sorted(ac.yellow_letters):
        totals["yellow"] += 1
        banned = ac.yellow_banned_positions.get(ch, set())
        banned_ok = all(guess[i] != ch for i in banned)
        need = ac.min_counts.get(ch, 0)
        guaranteed = green_guaranteed.get(ch, 0)
        # If count is already implied by green constraints, don't score it separately.
        count_ok = True if need <= guaranteed else (gcount.get(ch, 0) >= need)
        if banned_ok and count_ok:
            sat += 1
            sats["yellow"] += 1

    for ch, mx in sorted(ac.max_counts.items()):
        if mx == 0 and ac.min_counts.get(ch, 0) == 0:
            totals["absent"] += 1
            if gcount.get(ch, 0) == 0:
                sat += 1
                sats["absent"] += 1

    for ch, mx in sorted(ac.max_counts.items()):
        if 0 < mx < 5:
            totals["maxcap"] += 1
            if gcount.get(ch, 0) <= mx:
                sat += 1
                sats["maxcap"] += 1

    return sat, totals, sats


def sample_history_rows(valid_words: Sequence[str], secret: str, history_len: int, rng: random.Random) -> List[Tuple[str, str]]:
    history: List[Tuple[str, str]] = []
    used = set()
    for _ in range(history_len):
        selected: Optional[Tuple[str, str]] = None
        for _attempt in range(128):
            guess = rng.choice(valid_words)
            if guess == secret or guess in used:
                continue
            fb = compute_feedback(secret=secret, guess=guess)
            if fb == "G G G G G":
                continue
            selected = (guess, fb)
            break
        if selected is None:
            break
        used.add(selected[0])
        history.append(selected)
    return history


def generate_synthetic_dataset(
    solutions: Sequence[str],
    valid_words: Sequence[str],
    n_samples: int,
    seed: int,
    min_history_turns: int,
    max_history_turns: int,
) -> Dataset:
    if not (1 <= min_history_turns <= max_history_turns <= 4):
        raise ValueError("Require 1 <= min_history_turns <= max_history_turns <= 4 (turn 2-5 only).")
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    i = 0
    while len(rows) < n_samples:
        secret = rng.choice(solutions)
        history_len = rng.randint(min_history_turns, max_history_turns)
        history_rows = sample_history_rows(valid_words=valid_words, secret=secret, history_len=history_len, rng=rng)
        if len(history_rows) != history_len:
            continue
        rows.append(
            {
                "id": i,
                "secret": secret,
                "history_len": history_len,
                "history_rows": history_rows,
                "prompt": build_prompt_text(history_rows=history_rows),
            }
        )
        i += 1
    return Dataset.from_list(rows)


def _count_completion_tokens(completion_text: str, tokenizer: Any) -> int:
    return int(len(tokenizer.encode(completion_text, add_special_tokens=False)))


def score_completion(prompt_text: str, completion_text: str, valid_set: set, args: Any, tokenizer: Any) -> Dict[str, Any]:
    history_rows = _parse_history_from_prompt(prompt_text)
    history_guesses = {g for g, _ in history_rows}
    history_len = max(1, len(history_rows))

    # Overlength penalty applies even if strict format fails (strict-only reward still holds, but negatives are allowed).
    completion_tokens = _count_completion_tokens(completion_text, tokenizer=tokenizer)
    is_overlength = int(completion_tokens > int(args.max_output_tokens))
    reward_overlength = float(args.overlength_penalty) if is_overlength else 0.0

    guess = parse_strict_guess(completion_text)
    strict_ok = bool(guess and re.fullmatch(r"[a-z]{5}", guess))
    if not strict_ok:
        return {
            "strict_ok": False,
            "parsed_guess": guess,
            "is_wordle_valid": 0,
            "is_repeat": 0,
            "is_overlength": is_overlength,
            "completion_tokens": completion_tokens,
            "sat_count": 0,
            "totals": {"green": 0, "yellow": 0, "absent": 0, "maxcap": 0},
            "satisfied": {"green": 0, "yellow": 0, "absent": 0, "maxcap": 0},
            "reward_format": 0.0,
            "reward_dict": 0.0,
            "reward_repeat": 0.0,
            "reward_constraints": 0.0,
            "reward_overlength": reward_overlength,
            "reward_total": reward_overlength,
        }

    ac = aggregate_constraints(history_rows)
    sat_count, totals, sats = compute_sat_count(guess, ac)
    # Normalize by number of prior turns to reduce reward variance across prompts.
    reward_constraints = float(args.constraint_reward) * float(sat_count) / float(history_len)
    reward_dict = float(args.dict_reward) if (guess in valid_set) else 0.0
    is_repeat = int(guess in history_guesses)
    reward_repeat = float(args.repeat_penalty) if is_repeat else 0.0
    reward_total = float(args.format_reward) + reward_dict + reward_repeat + reward_constraints + reward_overlength

    return {
        "strict_ok": True,
        "parsed_guess": guess,
        "is_wordle_valid": int(guess in valid_set),
        "is_repeat": is_repeat,
        "is_overlength": is_overlength,
        "completion_tokens": completion_tokens,
        "sat_count": sat_count,
        "totals": totals,
        "satisfied": sats,
        "reward_format": float(args.format_reward),
        "reward_dict": reward_dict,
        "reward_repeat": reward_repeat,
        "reward_constraints": reward_constraints,
        "reward_overlength": reward_overlength,
        "reward_total": reward_total,
    }


def run_reward_tests() -> None:
    class _Args:
        format_reward = 0.2
        dict_reward = 0.2
        repeat_penalty = -0.5
        constraint_reward = 0.1
        max_output_tokens = 2048
        overlength_penalty = -0.5

    args = _Args()
    valid_set = {"crane", "slate", "roast", "trace"}
    prompt = build_prompt_text([("crane", "X X G X X"), ("slate", "X X G X X")])
    class _Tok:
        def encode(self, text: str, add_special_tokens: bool = False):
            return text.split()
    tok = _Tok()

    assert parse_strict_guess("no") is None
    assert parse_strict_guess("<think>x</think><guess>[crane]</guess>") == "crane"

    # Strict gating: missing closing </think> should yield strict parse fail => 0 reward.
    out_no_think = "<guess>[crane]</guess>"
    scored = score_completion(prompt, out_no_think, valid_set=valid_set, args=args, tokenizer=tok)
    assert scored["strict_ok"] is False
    assert scored["reward_total"] == 0.0

    # Dict reward: strict + in dict should add dict component.
    out_good = "<think>x</think><guess>[crane]</guess>"
    scored2 = score_completion(prompt, out_good, valid_set=valid_set, args=args, tokenizer=tok)
    assert scored2["reward_dict"] == args.dict_reward

    # Repeat penalty: repeating a history guess applies -0.5.
    out_repeat = "<think>x</think><guess>[slate]</guess>"
    scored3 = score_completion(prompt, out_repeat, valid_set=valid_set, args=args, tokenizer=tok)
    assert scored3["is_repeat"] == 1
    assert scored3["reward_repeat"] == args.repeat_penalty

    # Overlength penalty: apply even if strict format fails.
    args.max_output_tokens = 2
    out_long = "<guess>[crane]</guess>" + ("x " * 100)
    scored4 = score_completion(prompt, out_long, valid_set=valid_set, args=args, tokenizer=tok)
    assert scored4["strict_ok"] is False
    assert scored4["is_overlength"] == 1
    assert scored4["reward_overlength"] == args.overlength_penalty


def build_model_and_tokenizer(args: Any):
    from unsloth import FastLanguageModel
    # Use Unsloth loader; prefer 4bit on T4.
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.seq_len,
        load_in_4bit=bool(args.load_in_4bit),
        fast_inference=bool(args.fast_inference),
        max_lora_rank=args.lora_r,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
    )

    # Unsloth provides a tokenizer; but keep behavior predictable for plain-text prompts.
    try:
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    except Exception:
        pass

    # Notebook-faithful: GRPO + vLLM expects the Unsloth LoRA model (it provides load_lora()).
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=list(args.target_modules),
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    if args.init_adapter_repo:
        # Warm-start from an existing LoRA by loading its tensors into this Unsloth LoRA model.
        # Do NOT wrap with PeftModel (that breaks Unsloth GRPO+vLLM which calls model.load_lora()).
        from huggingface_hub import hf_hub_download

        adapter_cfg_path = hf_hub_download(
            repo_id=args.init_adapter_repo,
            filename="adapter_config.json",
            revision=args.init_adapter_revision,
        )
        adapter_cfg = json.loads(Path(adapter_cfg_path).read_text())
        expected_r = int(adapter_cfg.get("r", args.lora_r))
        expected_alpha = int(adapter_cfg.get("lora_alpha", args.lora_alpha))
        expected_targets = list(adapter_cfg.get("target_modules", list(args.target_modules)))

        if int(args.lora_r) != expected_r or int(args.lora_alpha) != expected_alpha or list(args.target_modules) != expected_targets:
            raise ValueError(
                "Init adapter LoRA config mismatch.\n"
                f"- adapter expects r={expected_r}, lora_alpha={expected_alpha}, target_modules={expected_targets}\n"
                f"- script args are    r={args.lora_r}, lora_alpha={args.lora_alpha}, target_modules={list(args.target_modules)}\n"
                "Fix by passing matching --lora-r/--lora-alpha/--target-modules, or use a different init adapter."
            )

        adapter_weights_path = hf_hub_download(
            repo_id=args.init_adapter_repo,
            filename="adapter_model.safetensors",
            revision=args.init_adapter_revision,
        )
        adapter_state = safetensors_load_file(adapter_weights_path)
        model_state = model.state_dict()

        # Key format differences across PEFT versions:
        # - adapter: ...lora_A.weight / ...lora_B.weight
        # - model:   ...lora_A.default.weight / ...lora_B.default.weight
        def _maybe_default(k: str) -> str:
            if ".lora_A.weight" in k:
                return k.replace(".lora_A.weight", ".lora_A.default.weight")
            if ".lora_B.weight" in k:
                return k.replace(".lora_B.weight", ".lora_B.default.weight")
            return k

        direct_hits = sum(1 for k in adapter_state.keys() if k in model_state)
        if direct_hits >= int(0.5 * len(adapter_state)):
            mapped_state = {k: v for k, v in adapter_state.items() if k in model_state}
        else:
            mapped = {(_maybe_default(k)): v for k, v in adapter_state.items()}
            mapped_state = {k: v for k, v in mapped.items() if k in model_state}

        if len(mapped_state) < int(0.8 * len(adapter_state)):
            raise RuntimeError(
                "Failed to load most init adapter tensors into Unsloth LoRA model.\n"
                f"- matched_keys={len(mapped_state)}/{len(adapter_state)}\n"
                "- This usually means base model mismatch or target_modules/r mismatch."
            )

        model.load_state_dict(mapped_state, strict=False)

    if bool(args.fast_inference) and not hasattr(model, "load_lora"):
        raise RuntimeError("Unsloth LoRA model is missing load_lora(); GRPO+vLLM fast inference will break.")

    model.train()
    return model, tok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unsloth GRPO Wordle RL (Qwen3-4B + LoRA, single-GPU)")
    # Default to the SFT LoRA adapter we already trained in this repo.
    # NOTE: The SFT adapter repo's adapter_config.json lists this as its base model.
    p.add_argument("--model-name", default="unsloth/qwen3-4b-unsloth-bnb-4bit")
    p.add_argument("--init-adapter-repo", default="goyalayus/wordle-lora-qwen3-4b")
    p.add_argument("--init-adapter-revision", default="step-1320")

    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-completion-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--synthetic-samples", type=int, default=4096)
    p.add_argument("--min-history-turns", type=int, default=1)
    p.add_argument("--max-history-turns", type=int, default=4)

    # Unsloth knobs
    p.add_argument("--load-in-4bit", action="store_true", default=False)
    p.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    p.add_argument("--fast-inference", action="store_true", default=False, help="Enable Unsloth fast inference (requires vLLM).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--vllm-min-p", type=float, default=0.1)
    p.add_argument("--vllm-top-p", type=float, default=1.0)
    p.add_argument("--vllm-top-k", type=int, default=-1)

    # Match the SFT adapter config by default.
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument(
        "--target-modules",
        nargs="+",
        default=["up_proj", "gate_proj", "down_proj"],
        help="LoRA target modules. Must match init adapter config if --init-adapter-repo is set.",
    )

    # Reward knobs (keep defaults as agreed)
    p.add_argument("--format-reward", type=float, default=0.2)
    p.add_argument("--dict-reward", type=float, default=0.2)
    p.add_argument("--repeat-penalty", type=float, default=-0.5)
    p.add_argument("--constraint-reward", type=float, default=0.1)
    p.add_argument("--max-output-tokens", type=int, default=2048)
    p.add_argument("--overlength-penalty", type=float, default=-0.5)

    p.add_argument("--solutions-list", default=None)
    p.add_argument("--allowed-list", default=None)

    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", default="wordle-rl-grpo")
    p.add_argument("--wandb-name", default=None)

    p.add_argument("--output-root", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--preflight-only", action="store_true", default=False)
    p.add_argument("--run-reward-tests", action="store_true", default=False)
    p.add_argument("--scale-rewards", default="none")
    p.add_argument("--dump-completions-path", default=None, help="Optional JSONL path to dump raw completions for debugging (default: /tmp/<run>.completions.jsonl).")
    p.add_argument("--dump-completions-n", type=int, default=32, help="Dump only the first N completions to --dump-completions-path.")
    return p.parse_args()


def build_vllm_sampling_params(tokenizer: Any, args: Any):
    if not bool(args.fast_inference):
        return None
    from vllm import SamplingParams

    # Keep vLLM's cap aligned with TRL's max_completion_length and our global limit.
    max_tokens = int(min(int(args.max_completion_length), int(args.max_output_tokens), MAX_MODEL_OUTPUT_TOKENS))
    return SamplingParams(
        temperature=float(args.temperature),
        min_p=float(args.vllm_min_p),
        top_p=float(args.vllm_top_p),
        top_k=int(args.vllm_top_k),
        seed=int(args.seed),
        max_tokens=max_tokens,
        stop=[tokenizer.eos_token] if getattr(tokenizer, "eos_token", None) is not None else None,
        include_stop_str_in_output=True,
    )


def main() -> None:
    args = parse_args()

    if args.run_reward_tests:
        run_reward_tests()
        print("reward tests passed", flush=True)
        return

    script_dir = Path(__file__).resolve().parents[1]
    solutions_path = Path(args.solutions_list) if args.solutions_list else (script_dir / "wordlists" / "wordle_solutions.txt")
    allowed_path = Path(args.allowed_list) if args.allowed_list else (script_dir / "wordlists" / "wordle_allowed_guesses.txt")
    solutions = load_word_list(solutions_path)
    allowed = load_word_list(allowed_path)
    valid_words = sorted(set(solutions) | set(allowed))
    valid_set = set(valid_words)

    ds = generate_synthetic_dataset(
        solutions=solutions,
        valid_words=valid_words,
        n_samples=args.synthetic_samples,
        seed=args.seed,
        min_history_turns=args.min_history_turns,
        max_history_turns=args.max_history_turns,
    )

    if args.preflight_only:
        # Just validate reward wiring + dataset shape without touching GPU training.
        class _Tok:
            def encode(self, text: str, add_special_tokens: bool = False):
                return text.split()

        row0 = ds[0]
        prompt_text = _extract_prompt_text(row0["prompt"])
        completion = "<think>preflight</think><guess>[crane]</guess>"
        scored = score_completion(prompt_text, completion, valid_set=valid_set, args=args, tokenizer=_Tok())
        if not scored["strict_ok"]:
            raise RuntimeError("Preflight failed: strict format parsing unexpectedly failed.")
        print("preflight passed", flush=True)
        return

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"unsloth_grpo_mixed_2_5_{now}"
    output_root = Path(args.output_root) if args.output_root else (Path(__file__).resolve().parents[2] / "outputs" / "RL")
    run_dir = output_root / run_name
    ensure_dir(run_dir)

    # Import Unsloth *before* importing TRL GRPO so Unsloth can patch GRPO/vLLM integration.
    # This mirrors the notebook import order in /home/ayush/Downloads/qwen3_(4b)_grpo(3).py.
    from unsloth import FastLanguageModel  # noqa: F401

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "generated_at_utc": utc_now(),
                "run_name": run_name,
                "args": vars(args),
                "resolved": {
                    "solutions_path": str(solutions_path),
                    "allowed_path": str(allowed_path),
                    "valid_word_count": len(valid_words),
                    "solution_count": len(solutions),
                    "effective_max_completion_length": int(
                        max(1, min(int(args.max_completion_length), int(args.seq_len) - int(args.max_prompt_length)))
                    ),
                },
            },
            indent=2,
        )
    )

    # Import Unsloth *before* importing TRL GRPO so Unsloth can patch GRPO/vLLM integration.
    # This mirrors the import order in the original Unsloth GRPO notebook.
    from unsloth import FastLanguageModel  # noqa: F401
    from trl import GRPOConfig, GRPOTrainer

    model, tokenizer = build_model_and_tokenizer(args)
    vllm_sampling_params = build_vllm_sampling_params(tokenizer=tokenizer, args=args)

    # Clamp completion length by:
    # - sequence budget (seq_len - max_prompt_length)
    # - user-configured max_completion_length
    # - reward's max_output_tokens
    # - a hard global cap (requested: 2048)
    effective_max_completion_length = max(
        1,
        min(
            int(args.max_completion_length),
            int(args.seq_len) - int(args.max_prompt_length),
            int(args.max_output_tokens),
            MAX_MODEL_OUTPUT_TOKENS,
        ),
    )

    metrics_path = run_dir / "metrics.jsonl"
    dump_path = (
        Path(args.dump_completions_path)
        if args.dump_completions_path
        else Path("/tmp") / f"{run_name}.completions.jsonl"
    )
    dumped = {"n": 0}

    def reward_format_exact(prompts: Sequence[Any], completions: Sequence[Any], **kwargs) -> List[float]:
        rewards: List[float] = []
        for completion_obj in completions:
            completion_text = _extract_completion_text(completion_obj)
            rewards.append(float(args.format_reward) if parse_strict_guess(completion_text) is not None else 0.0)
        return rewards

    reward_format_exact.__name__ = "reward_format_exact"

    def reward_wordle_strict(prompts: Sequence[Any], completions: Sequence[Any], **kwargs) -> List[float]:
        rewards: List[float] = []
        for prompt_obj, completion_obj in zip(prompts, completions):
            prompt_text = _extract_prompt_text(prompt_obj)
            completion_text = _extract_completion_text(completion_obj)
            scored = score_completion(prompt_text, completion_text, valid_set=valid_set, args=args, tokenizer=tokenizer)
            # score_completion() includes `reward_format` inside `reward_total` for strict matches.
            # We pay the format reward ONLY via reward_format_exact to keep it at 0.2, once.
            rewards.append(float(scored["reward_total"] - scored["reward_format"]))
            if dumped["n"] < int(args.dump_completions_n):
                dumped_row = {
                    "timestamp_utc": utc_now(),
                    "strict_ok": bool(scored["strict_ok"]),
                    "parsed_guess": scored.get("parsed_guess"),
                    "completion_tokens": scored["completion_tokens"],
                    "completion_text": completion_text,
                    "prompt_text": prompt_text,
                }
                with dump_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(dumped_row, ensure_ascii=True) + "\n")
                dumped["n"] += 1
            row = {
                "timestamp_utc": utc_now(),
                "strict_ok": bool(scored["strict_ok"]),
                "parsed_guess": scored.get("parsed_guess"),
                "is_wordle_valid": scored["is_wordle_valid"],
                "is_repeat": scored["is_repeat"],
                "is_overlength": scored["is_overlength"],
                "completion_tokens": scored["completion_tokens"],
                "sat_count": scored["sat_count"],
                "totals": scored["totals"],
                "satisfied": scored["satisfied"],
                "reward_format": scored["reward_format"],
                "reward_dict": scored["reward_dict"],
                "reward_repeat": scored["reward_repeat"],
                "reward_constraints": scored["reward_constraints"],
                "reward_overlength": scored["reward_overlength"],
                "reward_total": scored["reward_total"],
                "prompt_excerpt": prompt_text[-220:],
                "completion_excerpt": completion_text[:240],
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return rewards

    reward_wordle_strict.__name__ = "reward_wordle_strict"

    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_name:
            os.environ["WANDB_NAME"] = args.wandb_name

    report_to = ["wandb"] if args.wandb else "none"

    # This is intentionally close to the Unsloth notebook structure:
    # use GRPOConfig + GRPOTrainer; no vLLM path is required here.
    cfg_kwargs: Dict[str, Any] = {
        "output_dir": str(run_dir / "trainer_out"),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "temperature": float(args.temperature),
        # Enable TRL's vLLM generation path when Unsloth fast inference is requested.
        "use_vllm": bool(args.fast_inference),
        "vllm_sampling_params": vllm_sampling_params,
        "optim": "adamw_8bit",
        "logging_steps": 1,
        "report_to": report_to,
        "run_name": run_name,
        "seed": args.seed,
        "remove_unused_columns": False,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": effective_max_completion_length,
        # TRL versions differ; filter unsupported kwargs by signature below.
        "scale_rewards": args.scale_rewards,
        "bf16": False,
        "fp16": False,
    }

    # Make this script tolerant to TRL API drift (eg `scale_rewards` not present).
    accepted = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    cfg = GRPOConfig(**{k: v for k, v in cfg_kwargs.items() if k in accepted})

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        reward_funcs=[reward_format_exact, reward_wordle_strict],
        processing_class=tokenizer,
    )
    trainable_count = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    if trainable_count <= 0:
        raise RuntimeError("No trainable parameters detected after trainer initialization.")
    trainer.train()


if __name__ == "__main__":
    main()
