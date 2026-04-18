from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys


REPO_ROOT = Path("/home/ayush/Desktop/code/tenyson")
SRC_DIR = REPO_ROOT / "src"

for path in (str(REPO_ROOT), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


from functional import debug_turn5_reward, probe_turn5_rl_dataset  # noqa: E402
from tenyson import run_experiment  # noqa: E402


_MODEL_NAME = "Qwen/Qwen3-4B"
_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build(exp):
    return exp.run_branches(
        {
            "main": [
                exp.rl_stage(
                    "rl_rollout_probe",
                    adapter=exp.seed("stopped_sft_turn5"),
                    dataset=probe_turn5_rl_dataset(
                        sample_count=1,
                        seed=456,
                    ),
                    reward=debug_turn5_reward(),
                    overrides={
                        "model": {
                            "name": _MODEL_NAME,
                        },
                        "lora": {
                            "target_modules": _LORA_TARGET_MODULES,
                        },
                        "chat_template": {
                            "stop_strings": ["</guess>"],
                        },
                        "training": {
                            "hf_repo_id": (
                                "goyalayus/"
                                f"wordle-lora-rollout-probe-{_TIMESTAMP}"
                            ),
                            "max_steps": 1,
                            "per_device_batch_size": 1,
                            "gradient_accumulation_steps": 1,
                            "num_generations": 1,
                            "max_prompt_length": 512,
                            "max_completion_length": 64,
                            "hf_push_every_steps": 1000,
                        },
                    },
                ),
            ]
        }
    )


if __name__ == "__main__":
    run_experiment(__file__, build)
