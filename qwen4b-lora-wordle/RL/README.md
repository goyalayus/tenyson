# TRL GRPO + vLLM Wordle RL (Qwen3-4B + LoRA)

Multi-GPU GRPO RL stack for Qwen3-4B LoRA on Wordle using TRL + vLLM.

## What We Train

- Synthetic Wordle games, **mixed turn 2–5 only** (history length 1–4; no turn-1 samples).
- Reward is **strict-format-only**: if output does not match strict `<think>...</think><guess>[abcde]</guess>`, reward is `0.0`.

## Reward (strict-only)

If strict output format is invalid: `R = 0.0`.

If strict output format is valid:
- `+0.2` format base reward
- `+0.2` dictionary bonus if guess is in Wordle-valid list (solutions ∪ allowed guesses)
- `-0.5` penalty if guess repeats a prior guess from history
- `+0.1 * sat_count` where `sat_count` is satisfied **deduped unique constraints** from history
- Overlength penalty: if completion exceeds `--max-output-tokens` (default `2048`), add `--overlength-penalty` (default `-0.5`). This applies even if strict format is invalid.

Reward is not clamped/normalized.

## Length Settings

- Completion output cap is token-based: `--max-completion-length 2048`.
- Overlength penalty threshold is also token-based: `--max-output-tokens 2048`.
- For a true 2048-token completion budget, use context budget like `--seq-len 4096 --max-prompt-length 2048`.

## Word Lists

Wordle dictionary = `wordle_solutions.txt ∪ wordle_allowed_guesses.txt`.

- Secrets are sampled from `wordle_solutions.txt`
- History guesses are sampled from the union set

## Files

- `train_grpo_mixed_qwen4b_vllm.py`: main training entrypoint
- `run_sft_to_rl_mixed_vllm.sh`: launch SFT LoRA -> RL (mixed turn 2–5)
- `run_rl_to_rl_mixed_vllm.sh`: launch RL LoRA -> RL continuation (mixed turn 2–5)
- `wordlists/wordle_solutions.txt`: Wordle solutions list
- `wordlists/wordle_allowed_guesses.txt`: Wordle allowed guesses list

## Outputs

All outputs go under `qwen4b-lora-wordle/outputs/RL/<run_id>/` including:
- `config.json`
- `metrics.jsonl` (rank-0 per-sample reward debug rows)
- `checkpoints/` (rank-0 LoRA checkpoints)

## Run (8x GPUs)

Use the run scripts; they call:
`torchrun --standalone --nproc_per_node 8 ...`

## Reward Tests

```bash
python qwen4b-lora-wordle/RL/train_grpo_mixed_qwen4b_vllm.py --run-reward-tests --hf-repo-id dummy
```

(`--hf-repo-id` is ignored in test mode.)
