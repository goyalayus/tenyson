#!/bin/bash
# Run inference on 10 LoRA checkpoints. Use on T4 GPU.
#
# Prerequisites:
#   - HF_TOKEN in .env or env (for goyalayus/wordle-lora-qwen06b)
#   - pip install -r requirements-inference.txt
#
# Usage:
#   bash run_inference.sh

set -euo pipefail

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Clear GPU: kill any process using CUDA
echo "Clearing GPU processes..."
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null || true
done
sleep 2

# Reduce memory fragmentation (PyTorch recommendation for OOM)
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# Unbuffered stdout so logs appear immediately under nohup
export PYTHONUNBUFFERED=1

python run_inference.py && bash push_inference_results.sh
