from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys


REPO_ROOT = Path("/home/ayush/Desktop/code/tenyson")
SRC_DIR = REPO_ROOT / "src"

for path in (str(REPO_ROOT), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


from functional import (  # noqa: E402
    build_turn5_eval_dataset,
    compute_turn5_wordle_metrics,
    turn5_wordle_reward,
    turn5_wordle_rl_dataset,
)
from tenyson import bind_eval_dataset, run_experiment  # noqa: E402


_MODEL_NAME = "Qwen/Qwen3-4B"
_RL_SAMPLE_COUNT = 4096
_RL_SEED = 456
_EVAL_SAMPLE_COUNT = 100
_EVAL_SEED = 7
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
                    "rl_turn5",
                    adapter=exp.seed("stopped_sft_turn5"),
                    dataset=turn5_wordle_rl_dataset(
                        sample_count=_RL_SAMPLE_COUNT,
                        seed=_RL_SEED,
                        benchmark_sample_count=_EVAL_SAMPLE_COUNT,
                        benchmark_seed=_EVAL_SEED,
                    ),
                    reward=turn5_wordle_reward(),
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
                                f"wordle-lora-20260324-163252-rl_turn5_from_saved_sft_{_TIMESTAMP}"
                            ),
                        },
                    },
                ),
                exp.eval_stage(
                    "eval_turn5_after_rl",
                    adapter=exp.adapter("rl_turn5"),
                    dataset=bind_eval_dataset(
                        build_turn5_eval_dataset,
                        sample_count=_EVAL_SAMPLE_COUNT,
                        seed=_EVAL_SEED,
                    ),
                    metrics=compute_turn5_wordle_metrics,
                    overrides={
                        "model": {
                            "name": _MODEL_NAME,
                            "lora_target_modules": _LORA_TARGET_MODULES,
                        },
                        "chat_template": {
                            "stop_strings": ["</guess>"],
                        },
                    },
                ),
            ]
        }
    )


if __name__ == "__main__":
    run_experiment(__file__, build)
