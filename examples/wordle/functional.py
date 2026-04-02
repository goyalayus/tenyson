import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.request import urlopen

from datasets import Dataset
from tenyson.core.chat_sft import hub_chat_sft_dataset
from tenyson.core.environment import merge_config_overrides
from tenyson.core.plugin import TemplateTaskPlugin
from tenyson.core.stage_templates import (
    EvalDatasetTemplate,
    EvalMetricsTemplate,
    RLRewardTemplate,
    RLDatasetTemplate,
    SFTDatasetTemplate,
    template_factory_ref,
)
from tenyson.experiment import AdapterRef

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

TURN_LINE_RE = re.compile(
    r"Turn\s*([0-9]+):\s*\[([a-zA-Z]{5})\]\s*(?:->|→)\s*([GYX\s]+)"
)

_DEFAULT_WORD_SOURCE_URL = (
    "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
)
_DEFAULT_SFT_DATASET = "goyalayus/wordle-reasoning-sft-prefix-keep-think"
_FUNCTIONAL_MODULE = "examples.wordle.functional"

_DEFAULT_REWARD_CONFIG = {
    "format": 0.2,
    "dict": 0.2,
    "repeat_penalty": -0.5,
    "constraint": 0.5,
    "constraint_perfect_bonus": 0.2,
    "overlength_penalty": -0.5,
}

_DEFAULT_REWARD_PROFILE = "constraint"
_PRIME_NO_LENGTH_REWARD_PROFILE = "prime_no_length"
_PRIME_NO_LENGTH_REWARD_CONFIG = {
    "format": 0.2,
    "correct_answer": 1.0,
    "green": 0.2,
    "yellow": 0.1,
}


def _normalize_feedback_string(feedback: str) -> str:
    compact = "".join(ch for ch in str(feedback).upper() if ch in {"G", "Y", "X"})
    if len(compact) != 5:
        raise ValueError(
            "Wordle feedback must resolve to exactly 5 G/Y/X markers, "
            f"got {feedback!r}."
        )
    return compact


def render_user_prompt(history_rows: Sequence[Tuple[str, str]]) -> str:
    turn_idx = len(history_rows) + 1
    # Count remaining attempts including the current turn.
    attempts_remaining = max(0, 6 - len(history_rows))
    lines = [
        f"Turn {i}: [{guess}] -> {feedback}"
        for i, (guess, feedback) in enumerate(history_rows, start=1)
    ]
    history_block = "\n".join(lines)
    return (
        USER_PROMPT_PREFIX
        + f"\nThis is turn {turn_idx} of the game. You have {attempts_remaining} attempts left.\n\n"
        + "Prior turns and feedback:\n"
        + history_block
        + "\n\n"
        + "Enter your next guess."
    )


def build_prompt_text(history_rows: Sequence[Tuple[str, str]]) -> str:
    # Force the model to begin its completion inside a <think> block.
    return (
        SYSTEM_PROMPT
        + "\n\n"
        + render_user_prompt(history_rows=history_rows)
        + "\n\n<think>"
    )


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
    feedback = _normalize_feedback_string(feedback)
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


def aggregate_constraints(
    history_rows: Sequence[Tuple[str, str]],
) -> AggregatedConstraints:
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
                raise ValueError(
                    f"Contradictory green constraints at pos {i}: {prev} vs {ch}"
                )
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


def compute_sat_count(
    guess: str, ac: AggregatedConstraints
) -> Tuple[int, Dict[str, int], Dict[str, int]]:
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


def resolve_reward_max_output_tokens(
    config: Optional[Dict[str, Any]] = None,
    *,
    task_cfg: Optional[Dict[str, Any]] = None,
) -> int:
    local_task_cfg = task_cfg or {}
    rewards_cfg = local_task_cfg.get("rewards", {})
    explicit_limit = rewards_cfg.get("max_output_tokens")
    if explicit_limit is not None:
        return int(explicit_limit)

    local_config = config or {}
    training_limit = local_config.get("training", {}).get("max_completion_length")
    if training_limit is not None:
        return int(training_limit)

    vllm_limit = local_config.get("vllm", {}).get("max_tokens")
    if vllm_limit is not None:
        return int(vllm_limit)

    return 2048


def _resolve_reward_profile(task_cfg: Dict[str, Any]) -> str:
    profile = str(
        task_cfg.get("reward_profile", _DEFAULT_REWARD_PROFILE)
    ).strip() or _DEFAULT_REWARD_PROFILE
    return profile


def _score_constraint_completion(
    prompt_text: str,
    completion_text: str,
    valid_set: set,
    task_cfg: Dict[str, Any],
    tokenizer: Any,
    *,
    reward_max_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    history_rows = _parse_history_from_prompt(prompt_text)
    history_guesses = {g for g, _ in history_rows}

    rewards_cfg = task_cfg.get("rewards", {})
    format_reward = float(rewards_cfg.get("format", _DEFAULT_REWARD_CONFIG["format"]))
    dict_reward = float(rewards_cfg.get("dict", _DEFAULT_REWARD_CONFIG["dict"]))
    repeat_penalty = float(
        rewards_cfg.get("repeat_penalty", _DEFAULT_REWARD_CONFIG["repeat_penalty"])
    )
    constraint_reward = float(
        rewards_cfg.get("constraint", _DEFAULT_REWARD_CONFIG["constraint"])
    )
    constraint_perfect_bonus = float(
        rewards_cfg.get(
            "constraint_perfect_bonus",
            _DEFAULT_REWARD_CONFIG["constraint_perfect_bonus"],
        )
    )

    max_output_tokens = int(
        reward_max_output_tokens
        if reward_max_output_tokens is not None
        else rewards_cfg.get("max_output_tokens", 2048)
    )
    overlength_penalty = float(
        rewards_cfg.get(
            "overlength_penalty",
            _DEFAULT_REWARD_CONFIG["overlength_penalty"],
        )
    )

    completion_tokens = _count_completion_tokens(completion_text, tokenizer=tokenizer)
    is_overlength = int(completion_tokens > max_output_tokens)
    reward_overlength = overlength_penalty if is_overlength else 0.0

    guess = parse_strict_guess(completion_text)
    strict_ok = bool(guess and re.fullmatch(r"[a-z]{5}", guess))
    if not strict_ok:
        return {
            "reward_profile": _DEFAULT_REWARD_PROFILE,
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
    is_wordle_valid = int(guess in valid_set)
    total_constraints = sum(int(value or 0) for value in totals.values())

    constraint_ratio = (
        float(sat_count) / float(total_constraints) if total_constraints > 0 else 0.0
    )
    is_perfect_constraint_match = bool(
        is_wordle_valid and total_constraints > 0 and sat_count == total_constraints
    )
    reward_constraints = (
        (constraint_reward * constraint_ratio)
        + (constraint_perfect_bonus if is_perfect_constraint_match else 0.0)
        if is_wordle_valid and total_constraints > 0
        else 0.0
    )
    reward_dict = dict_reward if is_wordle_valid else 0.0
    is_repeat = int(guess in history_guesses)
    reward_repeat = repeat_penalty if is_repeat else 0.0

    reward_total = (
        format_reward
        + reward_dict
        + reward_repeat
        + reward_constraints
        + reward_overlength
    )

    return {
        "reward_profile": _DEFAULT_REWARD_PROFILE,
        "strict_ok": True,
        "parsed_guess": guess,
        "is_wordle_valid": is_wordle_valid,
        "is_repeat": is_repeat,
        "is_overlength": is_overlength,
        "completion_tokens": completion_tokens,
        "sat_count": sat_count,
        "constraint_ratio": constraint_ratio,
        "is_perfect_constraint_match": int(is_perfect_constraint_match),
        "totals": totals,
        "satisfied": sats,
        "reward_format": format_reward,
        "reward_dict": reward_dict,
        "reward_repeat": reward_repeat,
        "reward_constraints": reward_constraints,
        "reward_overlength": reward_overlength,
        "reward_total": reward_total,
    }


def _score_prime_no_length_completion(
    completion_text: str,
    valid_set: set,
    task_cfg: Dict[str, Any],
    tokenizer: Any,
    *,
    secret: Optional[str],
) -> Dict[str, Any]:
    rewards_cfg = task_cfg.get("rewards", {})
    format_reward = float(
        rewards_cfg.get("format", _PRIME_NO_LENGTH_REWARD_CONFIG["format"])
    )
    correct_answer_reward = float(
        rewards_cfg.get(
            "correct_answer",
            _PRIME_NO_LENGTH_REWARD_CONFIG["correct_answer"],
        )
    )
    green_reward = float(
        rewards_cfg.get("green", _PRIME_NO_LENGTH_REWARD_CONFIG["green"])
    )
    yellow_reward = float(
        rewards_cfg.get("yellow", _PRIME_NO_LENGTH_REWARD_CONFIG["yellow"])
    )

    target = str(secret or "").strip().lower()
    if len(target) != 5 or not target.isalpha():
        raise ValueError(
            "Prime-style Wordle rewards require a 5-letter secret target for each sample."
        )

    completion_tokens = _count_completion_tokens(completion_text, tokenizer=tokenizer)
    guess = parse_strict_guess(completion_text)
    strict_ok = bool(guess and re.fullmatch(r"[a-z]{5}", guess))
    if not strict_ok:
        return {
            "reward_profile": _PRIME_NO_LENGTH_REWARD_PROFILE,
            "strict_ok": False,
            "parsed_guess": guess,
            "completion_tokens": completion_tokens,
            "is_wordle_valid": 0,
            "is_exact_correct": 0,
            "feedback": None,
            "num_greens": 0,
            "num_yellows": 0,
            "reward_format": 0.0,
            "reward_correct_answer": 0.0,
            "reward_partial_answer": 0.0,
            "reward_total": 0.0,
        }

    feedback = _normalize_feedback_string(compute_feedback(secret=target, guess=guess))
    num_greens = feedback.count("G")
    num_yellows = feedback.count("Y")
    is_exact_correct = int(guess == target)
    reward_correct_answer = correct_answer_reward if is_exact_correct else 0.0
    reward_partial_answer = (
        0.0 if is_exact_correct else (green_reward * num_greens) + (yellow_reward * num_yellows)
    )
    reward_total = format_reward + reward_correct_answer + reward_partial_answer

    return {
        "reward_profile": _PRIME_NO_LENGTH_REWARD_PROFILE,
        "strict_ok": True,
        "parsed_guess": guess,
        "completion_tokens": completion_tokens,
        "is_wordle_valid": int(guess in valid_set),
        "is_exact_correct": is_exact_correct,
        "feedback": feedback,
        "num_greens": num_greens,
        "num_yellows": num_yellows,
        "reward_format": format_reward,
        "reward_correct_answer": reward_correct_answer,
        "reward_partial_answer": reward_partial_answer,
        "reward_total": reward_total,
    }



def score_completion(
    prompt_text: str,
    completion_text: str,
    valid_set: set,
    task_cfg: Dict[str, Any],
    tokenizer: Any,
    *,
    reward_max_output_tokens: Optional[int] = None,
    secret: Optional[str] = None,
) -> Dict[str, Any]:
    """Backward-compat dispatcher used by the dashboard (server.py)."""
    profile = _resolve_reward_profile(task_cfg)
    if profile == _DEFAULT_REWARD_PROFILE:
        return _score_constraint_completion(
            prompt_text,
            completion_text,
            valid_set,
            task_cfg,
            tokenizer,
            reward_max_output_tokens=reward_max_output_tokens,
        )
    if profile == _PRIME_NO_LENGTH_REWARD_PROFILE:
        return _score_prime_no_length_completion(
            completion_text,
            valid_set,
            task_cfg,
            tokenizer,
            secret=secret,
        )
    raise ValueError(f"Unsupported Wordle reward_profile: {profile!r}")


# ==============================================================================
# DATASET GENERATION
# ==============================================================================


def _resolve_wordlist_path(path_like: Any) -> Path:
    candidate = Path(str(path_like))
    if candidate.is_absolute():
        return candidate
    return Path(__file__).parent / candidate


def _filter_five_letter_words(lines: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    filtered: List[str] = []
    for raw_word in lines:
        word = str(raw_word).strip().lower()
        if len(word) != 5 or not word.isalpha() or word in seen:
            continue
        seen.add(word)
        filtered.append(word)
    return filtered


@lru_cache(maxsize=8)
def _load_word_source(source: str) -> List[str]:
    source_text = str(source).strip()
    if not source_text:
        raise ValueError("Word source cannot be empty.")

    if source_text.startswith(("http://", "https://", "file://")):
        with urlopen(source_text, timeout=30) as response:
            raw_lines = response.read().decode("utf-8").splitlines()
        return _filter_five_letter_words(raw_lines)

    with open(_resolve_wordlist_path(source_text), encoding="utf-8") as handle:
        return _filter_five_letter_words(handle.readlines())


def load_valid_word_set(
    config: Dict[str, Any],
    *,
    word_source: Optional[str] = None,
) -> set:
    """Load and return the set of valid 5-letter words for Wordle."""
    task_cfg = config.get("task", {})
    if task_cfg.get("wordlists") is not None:
        raise ValueError(
            "task.wordlists is no longer supported for Wordle. "
            "Pass word_source=... to the Wordle dataset/reward/metrics helper instead."
        )

    explicit_word_source = str(word_source or "").strip()
    resolved_source = explicit_word_source or _DEFAULT_WORD_SOURCE_URL
    return set(_load_word_source(resolved_source))


def get_wordlists(
    config: Dict[str, Any],
    *,
    word_source: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Backward-compat shim used by the dashboard (server.py)."""
    words = sorted(load_valid_word_set(config, word_source=word_source))
    return words, words


def sample_history_rows(
    valid_words: Sequence[str], secret: str, history_len: int, rng: random.Random
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


def generate_synthetic_wordle_dataset(
    config: Dict[str, Any],
    *,
    seed: Optional[int] = None,
    n_samples: Optional[int] = None,
    word_source: Optional[str] = None,
) -> Dataset:
    task_cfg = config.get("task", {})
    resolved_seed = seed if seed is not None else config.get("training", {}).get("seed", 3407)
    resolved_n_samples = n_samples if n_samples is not None else task_cfg.get("synthetic_samples", 1024)

    valid_words = sorted(load_valid_word_set(config, word_source=word_source))

    rng = random.Random(resolved_seed)
    min_turns = task_cfg.get("min_history_turns", 1)
    max_turns = task_cfg.get("max_history_turns", 4)

    rows = []

    i = 0
    while len(rows) < resolved_n_samples:
        secret = rng.choice(valid_words)
        history_len = rng.randint(min_turns, max_turns)
        history_rows = sample_history_rows(
            valid_words=valid_words, secret=secret, history_len=history_len, rng=rng
        )

        if len(history_rows) != history_len:
            continue

        prompt_text = build_prompt_text(history_rows=history_rows)

        rows.append(
            {
                "id": i,
                "secret": secret,
                "answer": secret,
                "history_len": history_len,
                "history_rows": history_rows,
                "prompt": prompt_text,
            }
        )
        i += 1

    return Dataset.from_list(rows)


def _validate_eval_exact_turns(turns: Any) -> List[int]:
    if not isinstance(turns, list) or not turns:
        raise ValueError(
            "task.eval_exact_turns must be a non-empty list of turns (1..6)."
        )
    parsed: List[int] = []
    for turn in turns:
        if not isinstance(turn, int):
            raise ValueError(
                f"task.eval_exact_turns must contain integers only, got {type(turn).__name__}."
            )
        if turn < 1 or turn > 6:
            raise ValueError(
                f"task.eval_exact_turns values must be within 1..6, got {turn}."
            )
        parsed.append(turn)
    return sorted(list(set(parsed)))


# ==============================================================================
# RL PLUGIN HOOKS
# ==============================================================================


def _extract_prompt_text(prompt_obj: Any) -> str:
    if isinstance(prompt_obj, str):
        return prompt_obj
    if isinstance(prompt_obj, dict):
        c = prompt_obj.get("content")
        return c if isinstance(c, str) else json.dumps(prompt_obj)
    return str(prompt_obj)


def _extract_completion_text(completion_obj: Any) -> str:
    if isinstance(completion_obj, str):
        return completion_obj
    if isinstance(completion_obj, dict):
        c = completion_obj.get("content")
        return c if isinstance(c, str) else json.dumps(completion_obj)
    return str(completion_obj)


def _coerce_batch_string_values(value: Any, batch_size: int) -> Optional[List[str]]:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return [value] * batch_size
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if len(value) != batch_size:
            return None
        return ["" if item is None else str(item) for item in value]
    return None


def _resolve_prime_reward_targets(kwargs: Dict[str, Any], batch_size: int) -> List[str]:
    for key in ("secret", "answer", "target", "secrets", "answers", "targets"):
        values = _coerce_batch_string_values(kwargs.get(key), batch_size)
        if values is not None:
            return [value.strip().lower() for value in values]
    raise ValueError(
        "Prime-style Wordle rewards require a dataset target column. "
        "Expected one of: secret, answer, target."
    )


def get_constraint_reward_funcs(
    config: Dict[str, Any],
    tokenizer: Any,
    *,
    word_source: Optional[str] = None,
) -> List[Any]:
    task_cfg = config.get("task", {})
    rewards_cfg = task_cfg.get("rewards", {})
    format_reward = float(rewards_cfg.get("format", 0.2))
    reward_max_output_tokens = resolve_reward_max_output_tokens(
        config,
        task_cfg=task_cfg,
    )

    valid_set = load_valid_word_set(config, word_source=word_source)

    def reward_format_exact(
        prompts: Sequence[Any], completions: Sequence[Any], **kwargs
    ) -> List[float]:
        rewards = []
        for comp in completions:
            text = _extract_completion_text(comp)
            rewards.append(
                format_reward if parse_strict_guess(text) is not None else 0.0
            )
        return rewards

    def reward_wordle_constraint(
        prompts: Sequence[Any], completions: Sequence[Any], **kwargs
    ) -> List[float]:
        rewards = []
        for prompt_obj, comp_obj in zip(prompts, completions):
            prompt_text = _extract_prompt_text(prompt_obj)
            completion_text = _extract_completion_text(comp_obj)

            scored = _score_constraint_completion(
                prompt_text,
                completion_text,
                valid_set,
                task_cfg,
                tokenizer,
                reward_max_output_tokens=reward_max_output_tokens,
            )

            # The format reward is paid out in reward_format_exact, so we subtract it here to avoid double-counting
            rewards.append(float(scored["reward_total"] - scored["reward_format"]))
        return rewards

    setattr(reward_format_exact, "tenyson_reward_name", "format_exact")
    setattr(reward_wordle_constraint, "tenyson_reward_name", "wordle_strict")
    return [reward_format_exact, reward_wordle_constraint]


def get_prime_reward_funcs(
    config: Dict[str, Any],
    tokenizer: Any,
    *,
    word_source: Optional[str] = None,
) -> List[Any]:
    task_cfg = config.get("task", {})
    rewards_cfg = task_cfg.get("rewards", {})
    format_reward = float(rewards_cfg.get("format", 0.2))

    valid_set = load_valid_word_set(config, word_source=word_source)

    def reward_format_exact(
        prompts: Sequence[Any], completions: Sequence[Any], **kwargs
    ) -> List[float]:
        rewards = []
        for comp in completions:
            text = _extract_completion_text(comp)
            rewards.append(
                format_reward if parse_strict_guess(text) is not None else 0.0
            )
        return rewards

    def reward_wordle_prime(
        prompts: Sequence[Any], completions: Sequence[Any], **kwargs
    ) -> List[float]:
        rewards = []
        targets = _resolve_prime_reward_targets(kwargs, len(completions))
        for _prompt_obj, comp_obj, target in zip(prompts, completions, targets):
            completion_text = _extract_completion_text(comp_obj)

            scored = _score_prime_no_length_completion(
                completion_text,
                valid_set,
                task_cfg,
                tokenizer,
                secret=target,
            )
            rewards.append(float(scored["reward_total"] - scored["reward_format"]))
        return rewards

    setattr(reward_format_exact, "tenyson_reward_name", "format_exact")
    setattr(reward_wordle_prime, "tenyson_reward_name", "wordle_prime")
    return [reward_format_exact, reward_wordle_prime]



# ==============================================================================
# EVALS PLUGIN HOOKS
# ==============================================================================


def get_eval_dataset(
    config: Dict[str, Any],
    *,
    word_source: Optional[str] = None,
) -> Dataset:
    """Returns the dataset for the Evals run."""
    task_cfg = config.get("task", {})
    n_samples = task_cfg.get("eval_samples", 100)
    seed = task_cfg.get("eval_seed", 42)
    exact_turns = task_cfg.get("eval_exact_turns")

    if exact_turns is None:
        return generate_synthetic_wordle_dataset(
            config,
            seed=seed,
            n_samples=n_samples,
            word_source=word_source,
        )

    turns = _validate_eval_exact_turns(exact_turns)

    # Build a balanced exact-turn eval set by round-robin sampling each turn.
    per_turn = max(1, n_samples // len(turns))
    rows: List[Dict[str, Any]] = []
    row_id = 0
    for idx, turn in enumerate(turns):
        history_len = max(0, int(turn) - 1)
        local_cfg = json.loads(json.dumps(config))
        local_cfg.setdefault("task", {})["min_history_turns"] = history_len
        local_cfg.setdefault("task", {})["max_history_turns"] = history_len
        local_seed = int(seed) + idx
        local_ds = generate_synthetic_wordle_dataset(
            local_cfg,
            seed=local_seed,
            n_samples=per_turn,
            word_source=word_source,
        )
        for row in local_ds:
            copied = dict(row)
            copied["id"] = row_id
            row_id += 1
            rows.append(copied)

    # Fill any remainder from the first requested turn for deterministic size.
    while len(rows) < n_samples:
        turn = turns[0]
        history_len = max(0, int(turn) - 1)
        local_cfg = json.loads(json.dumps(config))
        local_cfg.setdefault("task", {})["min_history_turns"] = history_len
        local_cfg.setdefault("task", {})["max_history_turns"] = history_len
        local_seed = int(seed) + len(rows)
        local_ds = generate_synthetic_wordle_dataset(
            local_cfg,
            seed=local_seed,
            n_samples=1,
            word_source=word_source,
        )
        copied = dict(local_ds[0])
        copied["id"] = row_id
        row_id += 1
        rows.append(copied)

    return Dataset.from_list(rows[:n_samples])


def _compute_constraint_metrics(
    prompts: List[str],
    completions: List[str],
    dataset_rows: Dataset,
    config: Dict[str, Any],
    tokenizer: Any,
    *,
    word_source: Optional[str] = None,
) -> Dict[str, Any]:
    del dataset_rows
    valid_set = load_valid_word_set(config, word_source=word_source)
    task_cfg = config.get("task", {})
    reward_max_output_tokens = resolve_reward_max_output_tokens(
        config,
        task_cfg=task_cfg,
    )

    total = len(completions)
    format_ok = 0
    dict_ok = 0
    consistent = 0
    total_constraint_reward = 0.0

    detailed_results = []

    for prompt, comp in zip(prompts, completions):
        scored = _score_constraint_completion(
            prompt_text=prompt,
            completion_text=comp,
            valid_set=valid_set,
            task_cfg=task_cfg,
            tokenizer=tokenizer,
            reward_max_output_tokens=reward_max_output_tokens,
        )
        guess = scored.get("parsed_guess")
        totals = scored.get("totals") if isinstance(scored.get("totals"), dict) else {}
        format_passed = bool(scored.get("strict_ok"))
        dict_passed = bool(scored.get("is_wordle_valid"))
        is_consistent = (
            format_passed
            and dict_passed
            and int(scored.get("sat_count") or 0)
            == sum(int(value or 0) for value in totals.values())
        )
        failure_reasons: List[str] = []
        if not format_passed:
            failure_reasons.append("format")
        else:
            if not dict_passed:
                failure_reasons.append("dictionary")
            if not is_consistent:
                failure_reasons.append("constraints")

        row_res = {
            "prompt": prompt,
            "completion": comp,
            "parsed_guess": guess,
            "format_ok": format_passed,
            "dict_ok": dict_passed,
            "consistent": is_consistent,
            "passed": len(failure_reasons) == 0,
            "failure_reasons": failure_reasons,
        }
        row_res.update(scored)

        if format_passed:
            format_ok += 1
        if dict_passed:
            dict_ok += 1
        if is_consistent:
            consistent += 1
        total_constraint_reward += float(scored.get("reward_constraints") or 0.0)

        detailed_results.append(row_res)

    return {
        "metrics": {
            "avg_constraint_reward": total_constraint_reward / max(total, 1),
            "format_accuracy": format_ok / max(total, 1),
            "dict_accuracy": dict_ok / max(total, 1),
            "constraint_accuracy": consistent / max(total, 1),
            "total_samples": total,
        },
        "detailed_results": detailed_results,
    }


def _compute_prime_no_length_metrics(
    prompts: List[str],
    completions: List[str],
    dataset_rows: Dataset,
    config: Dict[str, Any],
    tokenizer: Any,
    *,
    word_source: Optional[str] = None,
) -> Dict[str, Any]:
    valid_set = load_valid_word_set(config, word_source=word_source)
    task_cfg = config.get("task", {})

    total = len(completions)
    format_ok = 0
    dict_ok = 0
    correct_ok = 0
    total_correct_reward = 0.0
    total_partial_reward = 0.0
    total_format_reward = 0.0
    total_reward = 0.0

    detailed_results = []

    for prompt, comp, row in zip(prompts, completions, dataset_rows):
        secret = str(row.get("secret") or row.get("answer") or "").strip().lower()
        scored = _score_prime_no_length_completion(
            completion_text=comp,
            valid_set=valid_set,
            task_cfg=task_cfg,
            tokenizer=tokenizer,
            secret=secret,
        )

        format_passed = bool(scored.get("strict_ok"))
        dict_passed = bool(scored.get("is_wordle_valid"))
        is_exact_correct = bool(scored.get("is_exact_correct"))

        failure_reasons: List[str] = []
        if not format_passed:
            failure_reasons.append("format")
        if not is_exact_correct:
            failure_reasons.append("exact_answer")

        row_res = {
            "prompt": prompt,
            "completion": comp,
            "parsed_guess": scored.get("parsed_guess"),
            "format_ok": format_passed,
            "dict_ok": dict_passed,
            "exact_ok": is_exact_correct,
            "passed": format_passed and is_exact_correct,
            "failure_reasons": failure_reasons,
        }
        row_res.update(scored)

        if format_passed:
            format_ok += 1
        if dict_passed:
            dict_ok += 1
        if is_exact_correct:
            correct_ok += 1

        total_correct_reward += float(scored.get("reward_correct_answer") or 0.0)
        total_partial_reward += float(scored.get("reward_partial_answer") or 0.0)
        total_format_reward += float(scored.get("reward_format") or 0.0)
        total_reward += float(scored.get("reward_total") or 0.0)
        detailed_results.append(row_res)

    return {
        "metrics": {
            "avg_correct_answer_reward": total_correct_reward / max(total, 1),
            "avg_partial_answer_reward": total_partial_reward / max(total, 1),
            "avg_format_reward": total_format_reward / max(total, 1),
            "avg_total_reward": total_reward / max(total, 1),
            "correct_answer_accuracy": correct_ok / max(total, 1),
            "format_accuracy": format_ok / max(total, 1),
            "dict_accuracy": dict_ok / max(total, 1),
            "total_samples": total,
        },
        "detailed_results": detailed_results,
    }



def sft_chat_dataset() -> SFTDatasetTemplate:
    return hub_chat_sft_dataset()


def rl_mixed_dataset(*, word_source: Optional[str] = None) -> RLDatasetTemplate:
    return RLDatasetTemplate(
        build=lambda config: generate_synthetic_wordle_dataset(
            merge_config_overrides(config, {"task": {"min_history_turns": 1, "max_history_turns": 5}}),
            word_source=word_source,
        ),
        factory_ref=template_factory_ref(
            _FUNCTIONAL_MODULE,
            "rl_mixed_dataset",
            word_source=word_source,
        ),
    )


def rl_turn_dataset(
    turn: int,
    *,
    word_source: Optional[str] = None,
) -> RLDatasetTemplate:
    _validate_turn(turn)
    history_len = turn - 1
    return RLDatasetTemplate(
        build=lambda config: generate_synthetic_wordle_dataset(
            merge_config_overrides(config, {"task": {"min_history_turns": history_len, "max_history_turns": history_len}}),
            word_source=word_source,
        ),
        factory_ref=template_factory_ref(
            _FUNCTIONAL_MODULE,
            "rl_turn_dataset",
            turn=int(turn),
            word_source=word_source,
        ),
    )


def eval_mixed_dataset(*, word_source: Optional[str] = None) -> EvalDatasetTemplate:
    return EvalDatasetTemplate(
        build=lambda config: get_eval_dataset(
            merge_config_overrides(config, {"task": {"min_history_turns": 1, "max_history_turns": 5}}),
            word_source=word_source,
        ),
        factory_ref=template_factory_ref(
            _FUNCTIONAL_MODULE,
            "eval_mixed_dataset",
            word_source=word_source,
        ),
    )


def eval_turn_dataset(
    turn: int,
    *,
    word_source: Optional[str] = None,
) -> EvalDatasetTemplate:
    _validate_turn(turn)
    history_len = turn - 1
    return EvalDatasetTemplate(
        build=lambda config: get_eval_dataset(
            merge_config_overrides(config, {
                "task": {
                    "min_history_turns": history_len,
                    "max_history_turns": history_len,
                    "eval_exact_turns": [int(turn)],
                },
            }),
            word_source=word_source,
        ),
        factory_ref=template_factory_ref(
            _FUNCTIONAL_MODULE,
            "eval_turn_dataset",
            turn=int(turn),
            word_source=word_source,
        ),
    )


def constraint_reward(*, word_source: Optional[str] = None) -> RLRewardTemplate:
    return RLRewardTemplate(
        build=lambda config, tokenizer: get_constraint_reward_funcs(
            config,
            tokenizer,
            word_source=word_source,
        ),
        factory_ref=template_factory_ref(
            _FUNCTIONAL_MODULE,
            "constraint_reward",
            word_source=word_source,
        ),
    )


def prime_reward(*, word_source: Optional[str] = None) -> RLRewardTemplate:
    return RLRewardTemplate(
        build=lambda config, tokenizer: get_prime_reward_funcs(
            config,
            tokenizer,
            word_source=word_source,
        ),
        factory_ref=template_factory_ref(
            _FUNCTIONAL_MODULE,
            "prime_reward",
            word_source=word_source,
        ),
    )


def constraint_metrics(*, word_source: Optional[str] = None) -> EvalMetricsTemplate:
    return EvalMetricsTemplate(
        compute=lambda prompts, completions, dataset_rows, config, tokenizer: _compute_constraint_metrics(
            prompts,
            completions,
            dataset_rows,
            config,
            tokenizer,
            word_source=word_source,
        ),
        factory_ref=template_factory_ref(
            _FUNCTIONAL_MODULE,
            "constraint_metrics",
            word_source=word_source,
        ),
    )


def prime_metrics(*, word_source: Optional[str] = None) -> EvalMetricsTemplate:
    return EvalMetricsTemplate(
        compute=lambda prompts, completions, dataset_rows, config, tokenizer: _compute_prime_no_length_metrics(
            prompts,
            completions,
            dataset_rows,
            config,
            tokenizer,
            word_source=word_source,
        ),
        factory_ref=template_factory_ref(
            _FUNCTIONAL_MODULE,
            "prime_metrics",
            word_source=word_source,
        ),
    )


def _validate_turn(turn: int) -> None:
    if int(turn) < 1 or int(turn) > 6:
        raise ValueError(f"Wordle turn must be within 1..6, got {turn}.")


# Public API defaults — inline these in experiment.py overrides if preferred.
def sft_defaults() -> Dict[str, Any]:
    return {
        "training": {
            "loss_on_assistant_only": True,
            "response_template": "<|im_start|>assistant\n",
        },
        "task": {
            "sft_dataset": _DEFAULT_SFT_DATASET,
        },
    }


def rl_defaults() -> Dict[str, Any]:
    return {
        "task": {
            "synthetic_samples": 4096,
        }
    }


def eval_defaults() -> Dict[str, Any]:
    return {
        "task": {
            "eval_samples": 100,
            "eval_seed": 42,
        }
    }


TASK = TemplateTaskPlugin(environment_name="wordle")

SEEDS = {
    "experiment2_sft": AdapterRef(
        repo_id="goyalayus/wordle-lora-20260324-163252-sft_main",
        revision="30a33278640fcc5bcce216adce59984bfb8f7698",
    )
}
