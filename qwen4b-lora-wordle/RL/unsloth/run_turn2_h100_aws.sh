#!/usr/bin/env bash
set -euo pipefail

# Turn-2-only GRPO launcher for single-GPU AWS instances (H100/L40S/A10G).
# Reuses the turn2 wrapper script and keeps reward logic unchanged.
#
# Examples:
#   bash qwen4b-lora-wordle/RL/unsloth/run_turn2_h100_aws.sh --stage1
#   bash qwen4b-lora-wordle/RL/unsloth/run_turn2_h100_aws.sh --full --auto-shutdown
#   bash qwen4b-lora-wordle/RL/unsloth/run_turn2_h100_aws.sh --max-steps 250 --constraint-reward 0.3

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

MODE="stage1"            # stage1 -> 30 steps, full -> 1000 steps
MAX_STEPS_OVERRIDE=""
AUTO_SHUTDOWN=0
WANDB=1
FAST_INFERENCE=1
CONSTRAINT_REWARD="0.3"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage1)
      MODE="stage1"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --max-steps)
      MAX_STEPS_OVERRIDE="$2"
      shift 2
      ;;
    --constraint-reward)
      CONSTRAINT_REWARD="$2"
      shift 2
      ;;
    --no-wandb)
      WANDB=0
      shift
      ;;
    --no-fast-inference)
      FAST_INFERENCE=0
      shift
      ;;
    --auto-shutdown)
      AUTO_SHUTDOWN=1
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${MAX_STEPS_OVERRIDE}" ]]; then
  MAX_STEPS="${MAX_STEPS_OVERRIDE}"
elif [[ "${MODE}" == "full" ]]; then
  MAX_STEPS="1000"
else
  MAX_STEPS="30"
fi

RUN_TAG="${MODE}"
NOW="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="grpo_turn2_${RUN_TAG}_${NOW}"
LOG_PATH="/tmp/${RUN_NAME}.log"

echo "Run name: ${RUN_NAME}"
echo "Max steps: ${MAX_STEPS}"
echo "Constraint reward: ${CONSTRAINT_REWARD}"
echo "Fast inference: ${FAST_INFERENCE}"
echo "Wandb: ${WANDB}"
echo "Log: ${LOG_PATH}"

if [[ "${AUTO_SHUTDOWN}" -eq 1 ]]; then
  trap 'status=$?; echo "Training exited with status ${status}; shutting down instance..."; sudo shutdown -h now' EXIT INT TERM
fi

CMD=(
  python3 qwen4b-lora-wordle/RL/unsloth/train_grpo_turn2_qwen4b_unsloth.py
  --max-steps "${MAX_STEPS}"
  --run-name "${RUN_NAME}"
  --max-output-tokens 2048
  --constraint-reward "${CONSTRAINT_REWARD}"
)

if [[ "${WANDB}" -eq 1 ]]; then
  CMD+=(--wandb --wandb-project wordle-rl-grpo --wandb-name "${RUN_NAME}")
fi
if [[ "${FAST_INFERENCE}" -eq 1 ]]; then
  CMD+=(--fast-inference)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

set -x
"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
