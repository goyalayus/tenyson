#!/usr/bin/env bash
set -euo pipefail

# Lightweight GPU monitor for single-node AWS training.
# Prints util, memory, power, and process snapshot once every 5 seconds.

while true; do
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits
  echo "--- processes ---"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
  echo
  sleep 5
done
