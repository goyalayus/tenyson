"""
Run inference on 10 LoRA checkpoints from Hugging Face.
Plays full Wordle games for 3 words per checkpoint.
No max token cap on model output.
Saves results to inference_results.json for later HTML comparison.

Usage (on T4 GPU):
    pip install -r requirements-inference.txt
    python run_inference.py

Requires: HF_TOKEN or being logged in for goyalayus/wordle-lora-qwen06b
"""

import json
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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

        # Token-by-token generation to track exact OOM position
        gen_kwargs = dict(
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        current_ids = inputs.input_ids
        current_mask = inputs.get("attention_mask")
        num_new_tokens = 0
        last_out = None
        try:
            with torch.no_grad():
                for _ in range(MAX_NEW_TOKENS):
                    out = model.generate(
                        current_ids,
                        attention_mask=current_mask,
                        max_new_tokens=1,
                        **gen_kwargs,
                    )
                    last_out = out
                    new_token = out[0][-1].item()
                    if new_token == tokenizer.eos_token_id:
                        break
                    num_new_tokens += 1
                    current_ids = out
                    if current_mask is not None:
                        current_mask = torch.cat(
                            [current_mask, torch.ones((1, 1), device=current_mask.device, dtype=current_mask.dtype)],
                            dim=1,
                        )
            out = last_out
        except torch.cuda.OutOfMemoryError as e:
            total_at_failure = input_len + num_new_tokens
            print(f"\n*** OOM ERROR ***")
            print(f"  Checkpoint: {checkpoint}, Word: {target_word}, Turn: {turn_num + 1}")
            print(f"  Prompt tokens: {input_len}")
            print(f"  New tokens generated before OOM: {num_new_tokens}")
            print(f"  Total context length at OOM: {total_at_failure} tokens")
            print(f"  OOM occurred while generating token #{total_at_failure + 1}")
            print(f"  Error: {e}")
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
    print(f"Device: {device}")

    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {ckpt: [] for ckpt in CHECKPOINTS}

    for ckpt in CHECKPOINTS:
        print(f"\n{'='*60}\nCheckpoint: {ckpt}\n{'='*60}")
        # Load base + LoRA adapter for this checkpoint
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            base, LORA_REPO, revision=ckpt
        )
        model.eval()

        for word in EVAL_WORDS:
            print(f"  Word: {word}")
            game = run_one_game(model, tokenizer, word, device, checkpoint=ckpt)
            results[ckpt].append(game)
            print(f"    -> {'WIN' if game['won'] else 'LOSS'} in {game['num_turns']} turns")

        del model
        del base
        torch.cuda.empty_cache()

    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
