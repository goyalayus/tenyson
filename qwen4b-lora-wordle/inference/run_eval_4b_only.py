#!/usr/bin/env python3
"""Run eval for 4B model only, using latest-step adapter revision."""
import json
from pathlib import Path

from run_eval_two_models import load_prompt_cases, run_eval_for_model

spec = {
    "key": "qwen3_4b",
    "label": "Qwen3-4B LoRA",
    "base_model": "Qwen/Qwen3-4B",
    "adapter_repo": "goyalayus/wordle-full-qwen4b",
}

cases = load_prompt_cases(Path("wordle_eval_prompts_10.json"))
print(f"Running eval on {len(cases)} prompts for 4B...", flush=True)
result = run_eval_for_model(
    spec=spec, cases=cases, max_new_tokens=8192, revision_strategy="latest_step"
)
out = Path("results_qwen3_4b_latest.json")
out.write_text(json.dumps(result, indent=2))
print(f"DONE. Saved to {out}", flush=True)
print(f"Revision used: {result['adapter_revision_used']}", flush=True)
