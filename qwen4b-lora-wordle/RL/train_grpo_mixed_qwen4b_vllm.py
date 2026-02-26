#!/usr/bin/env python3
import argparse
import importlib.util
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
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from trl import GRPOConfig, GRPOTrainer

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
    r"<think>.*?</think>.*?<guess>\s*\[([a-zA-Z]{5})\]\s*</guess>",
    re.IGNORECASE | re.DOTALL,
)

TURN_LINE_RE = re.compile(r"Turn\s*([0-9]+):\s*\[([a-zA-Z]{5})\]\s*->\s*([GYX\s]+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_main_process() -> bool:
    return os.environ.get("RANK", "0") == "0"


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


def build_prompt(history_rows: Sequence[Tuple[str, str]]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(history_rows=history_rows)},
    ]


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


def sample_history_rows(
    valid_words: Sequence[str],
    secret: str,
    history_len: int,
    rng: random.Random,
) -> List[Tuple[str, str]]:
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
        raise ValueError("Require 1 <= min_history_turns <= max_history_turns <= 4 (turn 2–5 only).")

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
                "prompt": build_prompt(history_rows=history_rows),
            }
        )
        i += 1
    return Dataset.from_list(rows)


class RewardTracker:
    def __init__(self) -> None:
        self.n = 0
        self.sum_total = 0.0
        self.sum_dict = 0.0
        self.sum_repeat = 0.0
        self.sum_constraints = 0.0
        self.sum_sat_count = 0.0
        self.sum_overlength = 0.0
        self.strict_fail = 0
        self.repeat_count = 0
        self.overlength_count = 0

    def add(self, row: Dict[str, Any]) -> None:
        self.n += 1
        self.sum_total += float(row["reward_total"])
        self.sum_dict += float(row["reward_dict"])
        self.sum_repeat += float(row["reward_repeat"])
        self.sum_constraints += float(row["reward_constraints"])
        self.sum_sat_count += float(row["sat_count"])
        self.sum_overlength += float(row["reward_overlength"])
        self.strict_fail += int(not row["strict_ok"])
        self.repeat_count += int(row["is_repeat"])
        self.overlength_count += int(row["is_overlength"])

    def means(self) -> Dict[str, float]:
        d = max(self.n, 1)
        return {
            "reward_total_mean": self.sum_total / d,
            "reward_dict_mean": self.sum_dict / d,
            "reward_repeat_mean": self.sum_repeat / d,
            "reward_constraints_mean": self.sum_constraints / d,
            "reward_overlength_mean": self.sum_overlength / d,
            "sat_count_mean": self.sum_sat_count / d,
            "strict_fail_rate": self.strict_fail / d,
            "repeat_rate": self.repeat_count / d,
            "overlength_rate": self.overlength_count / d,
        }


class MetricsCallback(TrainerCallback):
    def __init__(self, tracker: RewardTracker) -> None:
        self.tracker = tracker

    def on_log(self, args, state, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            logs = {}
        if not is_main_process():
            return
        logs.update({f"custom/{k}": v for k, v in self.tracker.means().items()})
        logs["custom/step"] = int(state.global_step)


class CheckpointPushCallback(TrainerCallback):
    def __init__(self, run_dir: Path, repo_id: str, push_every_steps: int, tokenizer) -> None:
        self.run_dir = run_dir
        self.repo_id = repo_id
        self.push_every_steps = push_every_steps
        self.api = HfApi()
        self.tokenizer = tokenizer
        self.checkpoints_dir = run_dir / "checkpoints"
        ensure_dir(self.checkpoints_dir)
        if is_main_process():
            self.api.create_repo(repo_id=self.repo_id, exist_ok=True)

    def _save_and_push(self, model, step: int, final: bool = False) -> None:
        if not is_main_process():
            return
        rev = "rl-final" if final else f"rl-step-{step:05d}"
        local_dir = self.checkpoints_dir / ("final" if final else f"step_{step:05d}")
        ensure_dir(local_dir)
        model.save_pretrained(local_dir)
        try:
            self.tokenizer.save_pretrained(local_dir)
        except Exception:
            pass
        self.api.upload_folder(
            repo_id=self.repo_id,
            folder_path=str(local_dir),
            path_in_repo=".",
            revision=rev,
            commit_message=rev,
        )

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.push_every_steps <= 0:
            return
        step = int(state.global_step)
        if step > 0 and step % self.push_every_steps == 0:
            model = kwargs.get("model")
            if model is not None:
                self._save_and_push(model, step=step, final=False)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model")
        if model is not None:
            self._save_and_push(model, step=int(state.global_step), final=True)


def build_model_and_tokenizer(args) -> Tuple[torch.nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
    )

    if args.init_adapter_repo:
        model = PeftModel.from_pretrained(model, args.init_adapter_repo, revision=args.init_adapter_revision)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
    else:
        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

    model.train()
    return model, tokenizer


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
    class _Tok:
        def encode(self, text: str, add_special_tokens: bool = False):
            return text.split()
    tok = _Tok()

    assert parse_strict_guess("no") is None
    assert parse_strict_guess("<think>x</think><guess>[crane]</guess>") == "crane"

    prompt = build_prompt([("crane", "X X G X X"), ("slate", "X X G X X")])
    prompt_text = _extract_prompt_text(prompt)

    # Green position should dedupe (same pos green across turns counts once).
    history = _parse_history_from_prompt(prompt_text)
    ac = aggregate_constraints(history)
    _, totals, _ = compute_sat_count("aaaaa", ac)
    assert totals["green"] == 1

    # Strict gating: missing <think> should yield strict parse fail => 0 reward.
    out_no_think = "<guess>[crane]</guess>"
    scored = score_completion(prompt_text, out_no_think, valid_set=valid_set, args=args, tokenizer=tok)
    assert scored["strict_ok"] is False
    assert scored["reward_total"] == 0.0

    # Dict reward: strict + in dict should add dict component.
    out_good = "<think>x</think><guess>[crane]</guess>"
    scored2 = score_completion(prompt_text, out_good, valid_set=valid_set, args=args, tokenizer=tok)
    assert scored2["strict_ok"] is True
    assert scored2["reward_dict"] == args.dict_reward

    # Repeat penalty: repeating a history guess applies -0.5.
    out_repeat = "<think>x</think><guess>[slate]</guess>"
    scored3 = score_completion(prompt_text, out_repeat, valid_set=valid_set, args=args, tokenizer=tok)
    assert scored3["is_repeat"] == 1
    assert scored3["reward_repeat"] == args.repeat_penalty

    # Overlength penalty: apply even if strict format fails.
    args.max_output_tokens = 2
    out_long = "<guess>[crane]</guess>" + ("x " * 100)
    scored4 = score_completion(prompt_text, out_long, valid_set=valid_set, args=args, tokenizer=tok)
    assert scored4["strict_ok"] is False
    assert scored4["is_overlength"] == 1
    assert scored4["reward_overlength"] == args.overlength_penalty


def run_preflight_checks(args: Any, solutions: Sequence[str], valid_words: Sequence[str], valid_set: set) -> None:
    if args.use_vllm and importlib.util.find_spec("vllm") is None:
        raise RuntimeError("Preflight failed: vLLM is not installed but --use-vllm was requested.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    ds = generate_synthetic_dataset(
        solutions=solutions,
        valid_words=valid_words,
        n_samples=min(args.synthetic_samples, 16),
        seed=args.seed,
        min_history_turns=args.min_history_turns,
        max_history_turns=args.max_history_turns,
    )
    if len(ds) == 0:
        raise RuntimeError("Preflight failed: synthetic dataset generation returned no samples.")

    row0 = ds[0]
    prompt_obj = row0["prompt"]
    prompt_text = _extract_prompt_text(prompt_obj)
    # strict parse should score > 0 for a valid known word
    strict_completion = "<think>preflight</think><guess>[crane]</guess>"
    scored = score_completion(prompt_text, strict_completion, valid_set=valid_set, args=args, tokenizer=tokenizer)
    if not scored["strict_ok"]:
        raise RuntimeError("Preflight failed: strict format parsing unexpectedly failed.")

    bad_completion = "<guess>[crane]</guess>"
    scored_bad = score_completion(prompt_text, bad_completion, valid_set=valid_set, args=args, tokenizer=tokenizer)
    if scored_bad["reward_total"] != 0.0:
        raise RuntimeError("Preflight failed: non-strict format should have zero reward.")

    cfg_kwargs = dict(
        output_dir="/tmp/grpo_preflight",
        max_steps=1,
        per_device_train_batch_size=max(1, args.per_device_batch_size),
        gradient_accumulation_steps=max(1, args.gradient_accumulation_steps),
        learning_rate=args.learning_rate,
        report_to="none",
        remove_unused_columns=False,
        num_generations=max(1, args.num_generations),
        max_prompt_length=args.max_prompt_length,
        max_completion_length=max(1, min(int(args.max_completion_length), int(args.seq_len) - int(args.max_prompt_length))),
        use_vllm=False,   # config validation path only
        use_cpu=True,     # must be CPU-safe for preflight on non-GPU hosts
        bf16=False,
        fp16=False,
        scale_rewards=args.scale_rewards,
    )
    _ = GRPOConfig(**cfg_kwargs)
    print("preflight passed", flush=True)


def _count_completion_tokens(completion_text: str, tokenizer: Any) -> int:
    # Token-based length check (not chars). Tokenizer is expected to have .encode().
    # We intentionally exclude special tokens to measure only the raw completion.
    return int(len(tokenizer.encode(completion_text, add_special_tokens=False)))


def score_completion(prompt_text: str, completion_text: str, valid_set: set, args: Any, tokenizer: Any) -> Dict[str, Any]:
    history_rows = _parse_history_from_prompt(prompt_text)
    history_guesses = {g for g, _ in history_rows}

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
            "completion_tokens": completion_tokens,
            "sat_count": 0,
            "totals": {"green": 0, "yellow": 0, "absent": 0, "maxcap": 0},
            "satisfied": {"green": 0, "yellow": 0, "absent": 0, "maxcap": 0},
            "reward_format": 0.0,
            "reward_dict": 0.0,
            "reward_repeat": 0.0,
            "reward_constraints": 0.0,
            "reward_overlength": reward_overlength,
            "is_overlength": is_overlength,
            "reward_total": reward_overlength,
        }

    ac = aggregate_constraints(history_rows)
    sat_count, totals, sats = compute_sat_count(guess, ac)
    reward_constraints = float(args.constraint_reward) * float(sat_count)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO Wordle RL (Qwen3-4B + LoRA, TRL + vLLM)")
    p.add_argument("--base-model", default="Qwen/Qwen3-4B")
    p.add_argument("--init-adapter-repo", default=None)
    p.add_argument("--init-adapter-revision", default="main")
    p.add_argument("--hf-repo-id", default=None)

    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-completion-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--synthetic-samples", type=int, default=8192)

    p.add_argument("--min-history-turns", type=int, default=1)
    p.add_argument("--max-history-turns", type=int, default=4)

    p.add_argument("--format-reward", type=float, default=0.2)
    p.add_argument("--dict-reward", type=float, default=0.2)
    p.add_argument("--repeat-penalty", type=float, default=-0.5)
    p.add_argument("--constraint-reward", type=float, default=0.1)
    p.add_argument("--max-output-tokens", type=int, default=2048, help="If completion exceeds this many tokens, apply overlength penalty.")
    p.add_argument("--overlength-penalty", type=float, default=-0.5, help="Penalty added when output exceeds max-output-tokens.")

    p.add_argument("--solutions-list", default=None)
    p.add_argument("--allowed-list", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--push-every-steps", type=int, default=20)

    p.add_argument("--use-vllm", action="store_true")
    p.add_argument("--vllm-mode", default="colocate", choices=["colocate", "server"])
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.30)
    p.add_argument("--scale-rewards", default="none", choices=["none", "group", "batch", "rewards"])

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "wordle-rl-grpo"))
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--run-reward-tests", action="store_true")
    p.add_argument("--preflight-only", action="store_true", help="CPU-safe validation path; no model training.")
    p.add_argument("--skip-hf-push", action="store_true", help="Disable checkpoint pushes to HF.")
    p.add_argument("--use-cpu", action="store_true", help="Force CPU mode in GRPOConfig for local smoke checks.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    wordlists_dir = Path(__file__).resolve().parent / "wordlists"
    solutions_path = Path(args.solutions_list) if args.solutions_list else (wordlists_dir / "wordle_solutions.txt")
    allowed_path = Path(args.allowed_list) if args.allowed_list else (wordlists_dir / "wordle_allowed_guesses.txt")

    solutions = load_word_list(solutions_path)
    allowed = load_word_list(allowed_path)
    valid_words = sorted(set(solutions) | set(allowed))
    valid_set = set(valid_words)

    if args.run_reward_tests:
        run_reward_tests()
        print("reward tests passed", flush=True)
        return

    if args.preflight_only:
        run_preflight_checks(args=args, solutions=solutions, valid_words=valid_words, valid_set=valid_set)
        return

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"grpo_mixed_2_5_qwen3_4b_vllm_{now}"
    output_root = Path(args.output_root) if args.output_root else (Path(__file__).resolve().parents[1] / "outputs" / "RL")
    run_dir = output_root / run_name
    ensure_dir(run_dir)

    if (not args.skip_hf_push) and (not args.hf_repo_id):
        raise ValueError("--hf-repo-id is required unless --skip-hf-push is set.")

    if is_main_process():
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

    dataset = generate_synthetic_dataset(
        solutions=solutions,
        valid_words=valid_words,
        n_samples=args.synthetic_samples,
        seed=args.seed,
        min_history_turns=args.min_history_turns,
        max_history_turns=args.max_history_turns,
    )

    model, tokenizer = build_model_and_tokenizer(args)

    # Generation can never exceed the model context window. Clamp to avoid runtime errors when users
    # set a very large --max-completion-length (must fit within seq_len after prompt truncation).
    effective_max_completion_length = max(
        1,
        min(int(args.max_completion_length), int(args.seq_len) - int(args.max_prompt_length)),
    )

    tracker = RewardTracker()
    metrics_path = run_dir / "metrics.jsonl"

    def reward_fn(prompts: Sequence[Any], completions: Sequence[Any], **kwargs) -> List[float]:
        rewards: List[float] = []
        for prompt_obj, completion_obj in zip(prompts, completions):
            prompt_text = _extract_prompt_text(prompt_obj)
            completion_text = _extract_completion_text(completion_obj)
            scored = score_completion(prompt_text, completion_text, valid_set=valid_set, args=args, tokenizer=tokenizer)
            rewards.append(float(scored["reward_total"]))

            if is_main_process():
                tracker.add(
                    {
                        "reward_total": scored["reward_total"],
                        "reward_dict": scored["reward_dict"],
                        "reward_repeat": scored["reward_repeat"],
                        "reward_constraints": scored["reward_constraints"],
                        "reward_overlength": scored["reward_overlength"],
                        "sat_count": scored["sat_count"],
                        "strict_ok": scored["strict_ok"],
                        "is_repeat": scored["is_repeat"],
                        "is_overlength": scored["is_overlength"],
                    }
                )
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

    reward_fn.__name__ = "wordle_strict_dedup_reward"

    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_name:
            os.environ["WANDB_NAME"] = args.wandb_name

    report_to = ["wandb"] if args.wandb else "none"
    cfg_kwargs = dict(
        output_dir=str(run_dir / "trainer_out"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        report_to=report_to,
        run_name=run_name,
        seed=args.seed,
        remove_unused_columns=False,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=effective_max_completion_length,
        use_vllm=bool(args.use_vllm),
        vllm_mode=args.vllm_mode,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.seq_len,
        scale_rewards=args.scale_rewards,
        use_cpu=bool(args.use_cpu),
    )
    if args.use_cpu:
        cfg_kwargs["bf16"] = False
        cfg_kwargs["fp16"] = False
    cfg = GRPOConfig(**cfg_kwargs)

    callbacks: List[TrainerCallback] = [MetricsCallback(tracker=tracker)]
    if not args.skip_hf_push:
        callbacks.append(
            CheckpointPushCallback(
                run_dir=run_dir,
                repo_id=args.hf_repo_id,
                push_every_steps=args.push_every_steps,
                tokenizer=tokenizer,
            )
        )

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train()


if __name__ == "__main__":
    main()
