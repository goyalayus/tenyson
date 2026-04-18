from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from functional import turn5_wordle_reward, turn5_wordle_rl_dataset
from tenyson import run_experiment


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


def build(exp):
    return exp.run_branches(
        {
            "main": [
                exp.rl_stage(
                    "rl_turn5_no_vllm_longrun",
                    adapter=exp.seed("stopped_sft_turn5"),
                    dataset=turn5_wordle_rl_dataset(
                        sample_count=256,
                        seed=456,
                        benchmark_sample_count=16,
                        benchmark_seed=7,
                    ),
                    reward=turn5_wordle_reward(),
                    overrides={
                        "model": {
                            "name": _MODEL_NAME,
                            "fast_inference": False,
                        },
                        "lora": {
                            "target_modules": _LORA_TARGET_MODULES,
                        },
                        "vllm": {
                            "enabled": False,
                        },
                        "chat_template": {
                            "stop_strings": ["</guess>"],
                        },
                        "training": {
                            "max_steps": 25,
                            "gradient_accumulation_steps": 1,
                            "num_generations": 2,
                            "max_prompt_length": 512,
                            "max_completion_length": 256,
                            "hf_push_every_steps": 25,
                            "hf_repo_base": "goyalayus/wordle-no-vllm-longrun",
                        },
                    },
                )
            ]
        }
    )


if __name__ == "__main__":
    run_experiment(__file__, build)
