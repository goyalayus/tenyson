from functional import (
    build_turn5_eval_dataset,
    build_turn5_sft_train_dataset,
    compute_turn5_wordle_metrics,
    turn5_wordle_reward,
    turn5_wordle_rl_dataset,
)
from tenyson import bind_chat_sft_dataset, bind_eval_dataset, run_experiment


_SFT_DATASET = "goyalayus/wordle-reasoning-sft-prefix-keep-think"
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


def build(exp):
    return exp.run_branches(
        {
            "main": [
                exp.sft_stage(
                    "sft_turn5",
                    dataset=bind_chat_sft_dataset(
                        build_turn5_sft_train_dataset,
                        dataset_name=_SFT_DATASET,
                    ),
                    overrides={
                        "model": {
                            "name": _MODEL_NAME,
                        },
                        "lora": {
                            "target_modules": _LORA_TARGET_MODULES,
                        },
                        "training": {
                            "loss_on_assistant_only": True,
                            "response_template": "<|im_start|>assistant\n",
                            "val_size": 0,
                            "early_stopping_patience": None,
                        },
                    },
                ),
                exp.eval_stage(
                    "eval_turn5_after_sft",
                    adapter=exp.adapter("sft_turn5"),
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
                exp.rl_stage(
                    "rl_turn5",
                    adapter=exp.adapter("sft_turn5"),
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
