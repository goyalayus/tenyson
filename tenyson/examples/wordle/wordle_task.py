import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datasets import Dataset
from tenyson.core.plugin import TaskPlugin

# ==============================================================================
# PROMPTS
# ==============================================================================

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
    r"</think>.*?<guess>\s*\[([a-zA-Z]{5})\]\s*</guess>",
    re.IGNORECASE | re.DOTALL,
)

TURN_LINE_RE = re.compile(r"Turn\s*([0-9]+):\s*\[([a-zA-Z]{5})\]\s*->\s*([GYX\s]+)")


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


# ==============================================================================
# WORDLE LOGIC (FEEDBACK & CONSTRAINTS)
# ==============================================================================

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


def parse_strict_guess(response_text: str) -> Optional[str]:
    m = STRICT_FORMAT_RE.search(response_text)
    if not m:
        return None
    return m.group(1).lower()


def _count_completion_tokens(completion_text: str, tokenizer: Any) -> int:
    return int(len(tokenizer.encode(completion_text, add_special_tokens=False)))


def score_completion(prompt_text: str, completion_text: str, valid_set: set, task_cfg: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    history_rows = _parse_history_from_prompt(prompt_text)
    history_guesses = {g for g, _ in history_rows}
    history_len = max(1, len(history_rows))

    rewards_cfg = task_cfg.get("rewards", {})
    format_reward = float(rewards_cfg.get("format", 0.2))
    dict_reward = float(rewards_cfg.get("dict", 0.2))
    repeat_penalty = float(rewards_cfg.get("repeat_penalty", -0.5))
    constraint_reward = float(rewards_cfg.get("constraint", 0.1))
    
    max_output_tokens = int(rewards_cfg.get("max_output_tokens", 2048))
    overlength_penalty = float(rewards_cfg.get("overlength_penalty", -0.5))

    completion_tokens = _count_completion_tokens(completion_text, tokenizer=tokenizer)
    is_overlength = int(completion_tokens > max_output_tokens)
    reward_overlength = overlength_penalty if is_overlength else 0.0

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
    reward_constraints = constraint_reward * float(sat_count) / float(history_len)
    reward_dict = dict_reward if (guess in valid_set) else 0.0
    is_repeat = int(guess in history_guesses)
    reward_repeat = repeat_penalty if is_repeat else 0.0
    
    reward_total = format_reward + reward_dict + reward_repeat + reward_constraints + reward_overlength

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
        "reward_format": format_reward,
        "reward_dict": reward_dict,
        "reward_repeat": reward_repeat,
        "reward_constraints": reward_constraints,
        "reward_overlength": reward_overlength,
        "reward_total": reward_total,
    }


# ==============================================================================
# DATASET GENERATION
# ==============================================================================

def get_wordlists(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    paths = config.get("task", {}).get("wordlists", {})
    base_dir = Path(__file__).parent / "wordlists"
    
    sol_path = paths.get("solutions", str(base_dir / "wordle_solutions.txt"))
    allow_path = paths.get("allowed", str(base_dir / "wordle_allowed_guesses.txt"))
    
    def load(p):
        with open(p) as f:
            return [w.strip().lower() for w in f if len(w.strip()) == 5 and w.strip().isalpha()]
    
    return load(sol_path), load(allow_path)


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


def generate_synthetic_wordle_dataset(config: Dict[str, Any], seed: int, n_samples: int) -> Dataset:
    solutions, allowed = get_wordlists(config)
    valid_words = sorted(list(set(solutions) | set(allowed)))
    
    rng = random.Random(seed)
    task_cfg = config.get("task", {})
    min_turns = task_cfg.get("min_history_turns", 1)
    max_turns = task_cfg.get("max_history_turns", 4)
    
    rows = []
    
    i = 0
    while len(rows) < n_samples:
        secret = rng.choice(solutions)
        history_len = rng.randint(min_turns, max_turns)
        history_rows = sample_history_rows(valid_words=valid_words, secret=secret, history_len=history_len, rng=rng)
        
        if len(history_rows) != history_len:
            continue
            
        prompt_text = build_prompt_text(history_rows=history_rows)
        
        rows.append({
            "id": i,
            "secret": secret,
            "history_len": history_len,
            "history_rows": history_rows,
            "prompt": prompt_text,
        })
        i += 1
        
    return Dataset.from_list(rows)


def _validate_eval_exact_turns(turns: Any) -> List[int]:
    if not isinstance(turns, list) or not turns:
        raise ValueError("task.eval_exact_turns must be a non-empty list of turns (1..5).")
    parsed: List[int] = []
    for turn in turns:
        if not isinstance(turn, int):
            raise ValueError(
                f"task.eval_exact_turns must contain integers only, got {type(turn).__name__}."
            )
        if turn < 1 or turn > 5:
            raise ValueError(f"task.eval_exact_turns values must be within 1..5, got {turn}.")
        parsed.append(turn)
    return sorted(list(set(parsed)))


# ==============================================================================
# SFT PLUGIN HOOKS
# ==============================================================================

def get_sft_dataset(config: Dict[str, Any], tokenizer: Any) -> Dataset:
    """Returns the training dataset for SFT."""
    from datasets import load_dataset
    dataset_name = config.get("task", {}).get("sft_dataset", "goyalayus/wordle-reasoning-sft-prefix-keep-think")
    
    # Example logic: load a dataset with a 'messages' column
    ds = load_dataset(dataset_name, split="train")
    
    # Val split if requested
    val_size = config.get("training", {}).get("val_size", 0)
    if val_size > 0:
        ds = ds.train_test_split(test_size=val_size)["train"]
        
    return ds


def get_sft_eval_dataset(config: Dict[str, Any], tokenizer: Any) -> Dataset:
    """Returns the evaluation dataset for SFT."""
    from datasets import load_dataset
    dataset_name = config.get("task", {}).get("sft_dataset", "goyalayus/wordle-reasoning-sft-prefix-keep-think")
    
    ds = load_dataset(dataset_name, split="train")
    val_size = config.get("training", {}).get("val_size", 0)
    if val_size > 0:
        return ds.train_test_split(test_size=val_size)["test"]
    return None


def get_sft_formatting_func(config: Dict[str, Any], tokenizer: Any):
    """Applies the chat template during SFT."""
    def format_conversation(example):
        messages = example["messages"]
        if messages and isinstance(messages[0], list):
            return [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in messages
            ]
        return [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        ]
    return format_conversation


# ==============================================================================
# RL PLUGIN HOOKS
# ==============================================================================

def get_rl_dataset(config: Dict[str, Any]) -> Dataset:
    """Returns the dataset for GRPO training (just prompts)."""
    task_cfg = config.get("task", {})
    n_samples = task_cfg.get("synthetic_samples", 1024)
    seed = config.get("training", {}).get("seed", 3407)
    
    return generate_synthetic_wordle_dataset(config, seed, n_samples)


def _extract_prompt_text(prompt_obj: Any) -> str:
    if isinstance(prompt_obj, str): return prompt_obj
    if isinstance(prompt_obj, dict):
        c = prompt_obj.get("content")
        return c if isinstance(c, str) else json.dumps(prompt_obj)
    return str(prompt_obj)


def _extract_completion_text(completion_obj: Any) -> str:
    if isinstance(completion_obj, str): return completion_obj
    if isinstance(completion_obj, dict):
        c = completion_obj.get("content")
        return c if isinstance(c, str) else json.dumps(completion_obj)
    return str(completion_obj)


def get_reward_funcs(config: Dict[str, Any], tokenizer: Any) -> List[Any]:
    """Returns a list of callable reward functions for GRPO."""
    
    task_cfg = config.get("task", {})
    rewards_cfg = task_cfg.get("rewards", {})
    format_reward = float(rewards_cfg.get("format", 0.2))
    
    solutions, allowed = get_wordlists(config)
    valid_set = set(solutions) | set(allowed)
    
    def reward_format_exact(prompts: Sequence[Any], completions: Sequence[Any], **kwargs) -> List[float]:
        rewards = []
        for comp in completions:
            text = _extract_completion_text(comp)
            rewards.append(format_reward if parse_strict_guess(text) is not None else 0.0)
        return rewards

    def reward_wordle_strict(prompts: Sequence[Any], completions: Sequence[Any], **kwargs) -> List[float]:
        rewards = []
        for prompt_obj, comp_obj in zip(prompts, completions):
            prompt_text = _extract_prompt_text(prompt_obj)
            completion_text = _extract_completion_text(comp_obj)
            
            scored = score_completion(prompt_text, completion_text, valid_set, task_cfg, tokenizer)
            
            # The format reward is paid out in reward_format_exact, so we subtract it here to avoid double-counting
            rewards.append(float(scored["reward_total"] - scored["reward_format"]))
        return rewards

    return [reward_format_exact, reward_wordle_strict]


# ==============================================================================
# EVALS PLUGIN HOOKS
# ==============================================================================

def get_eval_dataset(config: Dict[str, Any]) -> Dataset:
    """Returns the dataset for the Evals run."""
    task_cfg = config.get("task", {})
    n_samples = task_cfg.get("eval_samples", 100)
    seed = task_cfg.get("eval_seed", 42)
    exact_turns = task_cfg.get("eval_exact_turns")

    if exact_turns is None:
        return generate_synthetic_wordle_dataset(config, seed, n_samples)

    turns = _validate_eval_exact_turns(exact_turns)

    # Build a balanced exact-turn eval set by round-robin sampling each turn.
    per_turn = max(1, n_samples // len(turns))
    rows: List[Dict[str, Any]] = []
    row_id = 0
    for idx, turn in enumerate(turns):
        local_cfg = json.loads(json.dumps(config))
        local_cfg.setdefault("task", {})["min_history_turns"] = turn
        local_cfg.setdefault("task", {})["max_history_turns"] = turn
        local_seed = int(seed) + idx
        local_ds = generate_synthetic_wordle_dataset(local_cfg, local_seed, per_turn)
        for row in local_ds:
            copied = dict(row)
            copied["id"] = row_id
            row_id += 1
            rows.append(copied)

    # Fill any remainder from the first requested turn for deterministic size.
    while len(rows) < n_samples:
        turn = turns[0]
        local_cfg = json.loads(json.dumps(config))
        local_cfg.setdefault("task", {})["min_history_turns"] = turn
        local_cfg.setdefault("task", {})["max_history_turns"] = turn
        local_seed = int(seed) + len(rows)
        local_ds = generate_synthetic_wordle_dataset(local_cfg, local_seed, 1)
        copied = dict(local_ds[0])
        copied["id"] = row_id
        row_id += 1
        rows.append(copied)

    return Dataset.from_list(rows[:n_samples])


def compute_metrics(prompts: List[str], completions: List[str], dataset_rows: Dataset, config: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    """Scores completions and returns an aggregated metric dictionary."""
    solutions, allowed = get_wordlists(config)
    valid_set = set(solutions) | set(allowed)
    
    total = len(completions)
    format_ok = 0
    dict_ok = 0
    consistent = 0
    
    detailed_results = []
    
    for prompt, comp in zip(prompts, completions):
        guess = parse_strict_guess(comp)
        
        row_res = {
            "prompt": prompt,
            "completion": comp,
            "parsed_guess": guess,
            "format_ok": False,
            "dict_ok": False,
            "consistent": False
        }
        
        if guess:
            format_ok += 1
            row_res["format_ok"] = True
            
            if guess in valid_set:
                dict_ok += 1
                row_res["dict_ok"] = True
                
            # Check constraint consistency
            history_rows = _parse_history_from_prompt(prompt)
            if history_rows:
                ac = aggregate_constraints(history_rows)
                sat_count, totals, _ = compute_sat_count(guess, ac)
                
                # Is consistent if all constraints are fully satisfied
                is_consistent = (sat_count == sum(totals.values()))
                if is_consistent:
                    consistent += 1
                    row_res["consistent"] = True
            else:
                # If no history, it's consistent by definition
                consistent += 1
                row_res["consistent"] = True
                
        detailed_results.append(row_res)
            
    return {
        "metrics": {
            "format_accuracy": format_ok / max(total, 1),
            "dict_accuracy": dict_ok / max(total, 1),
            "constraint_accuracy": consistent / max(total, 1),
            "total_samples": total
        },
        "detailed_results": detailed_results
    }


class WordleTask(TaskPlugin):
    def get_sft_dataset(self, config: Dict[str, Any], tokenizer: Any) -> Dataset:
        return get_sft_dataset(config, tokenizer)

    def get_sft_eval_dataset(self, config: Dict[str, Any], tokenizer: Any) -> Dataset:
        return get_sft_eval_dataset(config, tokenizer)

    def get_sft_formatting_func(self, config: Dict[str, Any], tokenizer: Any):
        return get_sft_formatting_func(config, tokenizer)

    def get_rl_dataset(self, config: Dict[str, Any]) -> Dataset:
        return get_rl_dataset(config)

    def get_reward_funcs(self, config: Dict[str, Any], tokenizer: Any) -> List[Any]:
        return get_reward_funcs(config, tokenizer)

    def get_eval_dataset(self, config: Dict[str, Any]) -> Dataset:
        return get_eval_dataset(config)

    def compute_metrics(
        self,
        prompts: List[str],
        completions: List[str],
        dataset_rows: Dataset,
        config: Dict[str, Any],
        tokenizer: Any,
    ) -> Dict[str, Any]:
        return compute_metrics(prompts, completions, dataset_rows, config, tokenizer)
