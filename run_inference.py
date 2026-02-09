"""
Run inference on 10 LoRA checkpoints from Hugging Face using Unsloth (2x faster).
Plays full Wordle games for 3 words per checkpoint.
Saves results to inference_results.json for later HTML comparison.

Usage (on T4 GPU):
    pip install -r requirements-inference.txt
    python run_inference.py

Requires: HF_TOKEN or being logged in for goyalayus/wordle-lora-qwen06b
"""

import json
import re

import torch
from huggingface_hub import snapshot_download
from unsloth import FastLanguageModel

# --- Config ---
LORA_REPO = "goyalayus/wordle-lora-qwen06b"
BASE_MODEL = "Qwen/Qwen3-0.6B"
EVAL_WORDS = ["corny", "align", "sober"]  # 3 words for full games
CHECKPOINTS = [
    "step-40", "step-80", "step-120", "step-160", "step-200",
    "step-240", "step-280", "step-320", "step-360", "step-400",
]
OUTPUT_FILE = "inference_results.json"

# 2^13 tokens max to avoid OOM on T4 (15GB)
MAX_NEW_TOKENS = 8192


def log(msg: str) -> None:
    """Print with immediate flush for proper logging under nohup."""
    print(msg, flush=True)


SYSTEM_PROMPT = (
    "You are an expert AI playing Wordle.\n"
    "GOAL: Guess the secret 5-letter word in 6 tries.\n\n"
    "CRITICAL:\n"
    "- Each game is independent. Do NOT reuse any words, reasoning, or guesses from any example or previous game.\n"
    "- Only use the user's provided guesses and feedback from THIS game to decide the next guess.\n"
    "- Never repeat a previous guess from this game.\n"
    "- Always follow the required output format exactly.\n\n"
    "FEEDBACK MEANING (per letter):\n"
    "- G (Green): correct letter in the correct position.\n"
    "- Y (Yellow): letter is in the word but in the wrong position.\n"
    "- X (Gray): letter is not in the word.\n"
    "  Duplicate-letter rule: if a letter appears multiple times in a guess, and at least one instance is G or Y,\n"
    "  then any X for that same letter means 'no additional copies beyond the confirmed count'.\n\n"
    "LOGIC RULES:\n"
    "- Keep all G letters fixed in their positions.\n"
    "- For Y letters: keep the letter in the word but forbid that specific position.\n"
    "- For X letters: eliminate the letter unless duplicate-letter rule applies.\n"
    "- Your next guess must satisfy all constraints so far.\n"
    "- On the first turn (no feedback yet), choose a strong common 5-letter starting word with unique letters.\n\n"
    "INPUT YOU WILL RECEIVE:\n"
    "- First turn: 'NEW GAME STARTED...' with no feedback.\n"
    "- Later turns: lines like:\n"
    "  Assistant previous guess: [CRANE]\n"
    "  Feedback: X X Y G G\n"
    "5 guesses left.\n\n"
    "OUTPUT FORMAT (MUST MATCH EXACTLY):\n"
    "<guess>[word]</guess>\n\n"
    "OUTPUT CONSTRAINTS:\n"
    "- [word] must be exactly 5 lowercase letters a-z.\n"
    "- Do not output anything else.\n"
    "you must must follow this output format i gave you"
)

BASE_USER_PROMPT = "NEW GAME STARTED. A secret word has been chosen. Enter your first guess."


def get_wordle_feedback(guess: str, secret: str) -> str:
    """G/Y/X feedback for a 5-letter guess."""
    guess, secret = guess.lower(), secret.lower()
    feedback = ["X"] * 5
    secret_list = list(secret)
    guess_list = list(guess)

    for i in range(5):
        if guess_list[i] == secret_list[i]:
            feedback[i] = "G"
            secret_list[i] = None
            guess_list[i] = None

    for i in range(5):
        if guess_list[i] is not None and guess_list[i] in secret_list:
            feedback[i] = "Y"
            secret_list[secret_list.index(guess_list[i])] = None

    return " ".join(feedback)


def extract_guess(text: str) -> str | None:
    """Extract 5-letter guess from <guess>[word]</guess> or [word]."""
    if not text:
        return None
    text_wo_think = re.sub(r"<think>.*?(</think>|$)", " ", text, flags=re.DOTALL | re.IGNORECASE)
    m = re.search(r"<guess>\s*\[?([a-zA-Z]{5})\]?\s*</guess>", text_wo_think, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r"<\s*([a-zA-Z]{5})\s*>", text_wo_think)
    if m and m.group(1).lower() != "think":
        return m.group(1).lower()
    m = re.search(r"\[\s*([a-zA-Z]{5})\s*\]", text_wo_think)
    if m:
        return m.group(1).lower()
    return None


def run_one_game(
    model, tokenizer, target_word: str, device: str, checkpoint: str = ""
) -> dict:
    """Play one full Wordle game. Returns game record."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": BASE_USER_PROMPT},
    ]
    turns = []
    won = False
    for turn_num in range(6):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text += "<think>\n"
        inputs = tokenizer([text], return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        gen_kwargs = dict(
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    **gen_kwargs,
                )
        except torch.cuda.OutOfMemoryError as e:
            log(f"\n*** OOM ERROR ***")
            log(f"  Checkpoint: {checkpoint}, Word: {target_word}, Turn: {turn_num + 1}")
            log(f"  Prompt tokens: {input_len}")
            log(f"  Error: {e}")
            raise

        response_ids = out[0][input_len:]
        response = tokenizer.decode(response_ids, skip_special_tokens=False).strip()
        response = "<think>\n" + response

        guess = extract_guess(response)
        turn_data = {
            "turn": turn_num + 1,
            "assistant_response": response,
            "extracted_guess": guess,
        }

        if not guess:
            turn_data["error"] = "Could not extract guess"
            turns.append(turn_data)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "ERROR. Output ONLY: <guess>[word]</guess>"})
            continue

        feedback = get_wordle_feedback(guess, target_word)
        turn_data["feedback"] = feedback
        turns.append(turn_data)

        if guess == target_word:
            won = True
            break

        messages.append({"role": "assistant", "content": response})
        messages.append({
            "role": "user",
            "content": f"Previous: [{guess.upper()}]\nFeedback: {feedback}\n{5 - turn_num} guesses left.",
        })

    return {
        "target_word": target_word,
        "won": won,
        "turns": turns,
        "num_turns": len(turns),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    results = {ckpt: [] for ckpt in CHECKPOINTS}

    for ckpt in CHECKPOINTS:
        log(f"\n{'='*60}")
        log(f"Checkpoint: {ckpt}")
        log(f"{'='*60}")

        # Load adapter from HF (revision = branch) into temp dir, then Unsloth loads it
        adapter_path = snapshot_download(LORA_REPO, revision=ckpt)
        log(f"Loaded adapter from {adapter_path}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=8192,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
        log(f"Model ready (Unsloth fast inference enabled)")

        for word in EVAL_WORDS:
            log(f"  Word: {word}")
            game = run_one_game(model, tokenizer, word, device, checkpoint=ckpt)
            results[ckpt].append(game)
            log(f"    -> {'WIN' if game['won'] else 'LOSS'} in {game['num_turns']} turns")

        del model
        torch.cuda.empty_cache()

    log(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    log("Done.")


if __name__ == "__main__":
    main()
