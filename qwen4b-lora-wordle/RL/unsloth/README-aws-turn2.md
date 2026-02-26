# AWS turn2 runbook (unsloth GRPO)

This folder now includes a turn-2-only AWS launcher and a simple GPU monitor.

## Files

- `train_grpo_turn2_qwen4b_unsloth.py`: turn-2 wrapper over the mixed script (same reward logic)
- `run_turn2_h100_aws.sh`: stage/full launcher with optional auto-shutdown
- `monitor_gpu_h100.sh`: periodic `nvidia-smi` monitor

## Stage 1 (quick tune, 20-30 steps)

```bash
bash qwen4b-lora-wordle/RL/unsloth/run_turn2_h100_aws.sh --stage1 --max-steps 30
```

## Full run (1000 steps)

```bash
bash qwen4b-lora-wordle/RL/unsloth/run_turn2_h100_aws.sh --full
```

## Auto shutdown after run

```bash
bash qwen4b-lora-wordle/RL/unsloth/run_turn2_h100_aws.sh --full --auto-shutdown
```

## Live monitoring

```bash
bash qwen4b-lora-wordle/RL/unsloth/monitor_gpu_h100.sh
```

## Useful overrides

```bash
bash qwen4b-lora-wordle/RL/unsloth/run_turn2_h100_aws.sh \
  --stage1 \
  --max-steps 30 \
  --constraint-reward 0.3 \
  -- --per-device-batch-size 1 --gradient-accumulation-steps 4 --num-generations 4
```
