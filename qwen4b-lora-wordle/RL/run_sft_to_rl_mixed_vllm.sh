#!/usr/bin/env bash
set -euo pipefail

# 8x GPU launch: SFT LoRA -> mixed turn 2-5 RL (TRL GRPO + vLLM colocate).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

NOW="$(date +%Y%m%d_%H%M%S)"
RUN_ID="grpo_mixed_2_5_sft_init_vllm_${NOW}"

OUT_ROOT="${ROOT_DIR}/qwen4b-lora-wordle/outputs/RL"
mkdir -p "${OUT_ROOT}/${RUN_ID}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
INIT_ADAPTER_REPO="${INIT_ADAPTER_REPO:-goyalayus/wordle-lora-qwen3-4b}"
INIT_ADAPTER_REVISION="${INIT_ADAPTER_REVISION:-step-1320}"
HF_REPO_ID="${HF_REPO_ID:?set HF_REPO_ID (target hub repo id)}"

WANDB_PROJECT="${WANDB_PROJECT:-wordle-rl-grpo}"
WANDB_NAME="${WANDB_NAME:-${RUN_ID}}"
export WANDB_PROJECT WANDB_NAME

torchrun --standalone --nproc_per_node 8 \
  qwen4b-lora-wordle/RL/train_grpo_mixed_qwen4b_vllm.py \
  --base-model "${BASE_MODEL}" \
  --init-adapter-repo "${INIT_ADAPTER_REPO}" \
  --init-adapter-revision "${INIT_ADAPTER_REVISION}" \
  --hf-repo-id "${HF_REPO_ID}" \
  --run-name "${RUN_ID}" \
  --output-root "${OUT_ROOT}" \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-name "${WANDB_NAME}" \
  --max-steps 1000 \
  --per-device-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --num-generations 4 \
  --seq-len 4096 \
  --max-prompt-length 2048 \
  --max-completion-length 2048 \
  --min-history-turns 1 \
  --max-history-turns 4 \
  --use-vllm \
  --vllm-mode colocate \
  --vllm-tensor-parallel-size 1 \
  --vllm-gpu-memory-utilization 0.30 \
  --scale-rewards none \
  --push-every-steps 20
