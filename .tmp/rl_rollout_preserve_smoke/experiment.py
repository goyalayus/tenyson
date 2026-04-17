from __future__ import annotations

import time

from functional import addition_reward, addition_rl_dataset
from tenyson import run_experiment


def build(exp):
    repo_suffix = str(int(time.time()))
    return exp.run_branches(
        {
            "rl_branch": [
                exp.rl_stage(
                    "rl_rollout_preserve_smoke",
                    dataset=addition_rl_dataset(
                        digits=2,
                        sample_count=256,
                        seed=456,
                        benchmark_sample_count=32,
                        benchmark_seed=7,
                    ),
                    reward=addition_reward(),
                    overrides={
                        "model": {
                            "name": "Qwen/Qwen3-0.6B",
                            "load_in_4bit": False,
                        },
                        "chat_template": {
                            "enable_thinking": False,
                            "stop_strings": ["</answer>"],
                        },
                        "training": {
                            "hf_repo_id": f"goyalayus/arithmetic-rollout-preserve-smoke-{repo_suffix}",
                            "max_steps": 8,
                            "per_device_batch_size": 1,
                            "gradient_accumulation_steps": 2,
                            "learning_rate": 5.0e-6,
                            "num_generations": 2,
                            "max_prompt_length": 128,
                            "max_completion_length": 32,
                            "logging_steps": 1,
                            "save_steps": 2,
                            "hf_push_every_steps": 2,
                        },
                    },
                ),
            ]
        }
    )


if __name__ == "__main__":
    run_experiment(__file__, build)
