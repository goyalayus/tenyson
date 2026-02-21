#!/usr/bin/env python3
"""Run eval for 0.6B model only, forcing adapter revision step-1720."""
import json
from pathlib import Path

import run_eval_two_models as core
from run_eval_two_models import load_prompt_cases, run_eval_for_model


def _force_step_1720(repo_id: str, strategy: str):
    return "step-1720", "forced_step1720"


core.resolve_revision = _force_step_1720

spec = {
    "key": "qwen3_06b",
    "label": "Qwen3-0.6B LoRA",
    "base_model": "Qwen/Qwen3-0.6B",
    "adapter_repo": "goyalayus/wordle-full-qwen06b",
}

cases = load_prompt_cases(Path("wordle_eval_prompts_10.json"))
print(f"Running eval on {len(cases)} prompts for 0.6B with forced step-1720...")
result = run_eval_for_model(
    spec=spec,
    cases=cases,
    max_new_tokens=2048,
    revision_strategy="latest_step",
)
out = Path("results_qwen3_06b_step1720.json")
out.write_text(json.dumps(result, indent=2))
print(f"DONE. Saved to {out}")
print("Revision used:", result["adapter_revision_used"])
