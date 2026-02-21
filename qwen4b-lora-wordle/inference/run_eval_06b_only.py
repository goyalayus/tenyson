#!/usr/bin/env python3
"""Run eval for 0.6B model only, using latest-step adapter revision."""
import json
from pathlib import Path
from run_eval_two_models import load_prompt_cases, run_eval_for_model

spec = {
    "key": "qwen3_06b",
    "label": "Qwen3-0.6B LoRA",
    "base_model": "Qwen/Qwen3-0.6B",
    "adapter_repo": "goyalayus/wordle-full-qwen06b",
}

cases = load_prompt_cases(Path("wordle_eval_prompts_10.json"))
print(f"Running eval on {len(cases)} prompts for 0.6B...")
result = run_eval_for_model(
    spec=spec, cases=cases, max_new_tokens=8192, revision_strategy="latest_step"
)
out = Path("results_qwen3_06b_latest.json")
out.write_text(json.dumps(result, indent=2))
print(f"DONE. Saved to {out}")
print(f"Revision used: {result['adapter_revision_used']}")
