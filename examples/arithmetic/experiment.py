from functional import (
    addition_reward,
    addition_rl_dataset,
    build_addition_dataset,
    build_addition_sft_train_dataset,
    compute_addition_metrics,
)
from tenyson import bind_chat_sft_dataset, bind_eval_dataset, run_experiment


_DIGITS = 2
_BENCHMARK_SAMPLE_COUNT = 8
_BENCHMARK_SEED = 7
_SFT_TRAIN_SAMPLE_COUNT = 64
_SFT_TRAIN_SEED = 123
_RL_TRAIN_SAMPLE_COUNT = 64
_RL_TRAIN_SEED = 456
_SFT_STAGE_ID = "sft_2digit_06b_smoke"
_RL_STAGE_ID = "rl_2digit_06b_smoke"
_BASELINE_EVAL_STAGE_ID = "eval_2digit_baseline_smoke"
_SFT_EVAL_STAGE_ID = "eval_2digit_after_sft_smoke"
_RL_EVAL_STAGE_ID = "eval_2digit_after_rl_smoke"
_SFT_HF_REPO_ID = "goyalayus/arithmetic-2digit-sft_2digit_06b_smoke"
_RL_HF_REPO_ID = "goyalayus/arithmetic-2digit-rl_2digit_06b_smoke"

_BASE_MODEL_OVERRIDES = {
    "name": "Qwen/Qwen3-0.6B",
    "load_in_4bit": False,
}

_EVAL_OVERRIDES = {
    "model": dict(_BASE_MODEL_OVERRIDES),
    "chat_template": {
        "enable_thinking": False,
        "stop_strings": ["</answer>"],
    },
    "vllm": {
        "max_tokens": 32,
    },
}


def _benchmark_dataset():
    return bind_eval_dataset(
        build_addition_dataset,
        digits=_DIGITS,
        sample_count=_BENCHMARK_SAMPLE_COUNT,
        seed=_BENCHMARK_SEED,
    )


def build(exp):
    exp.eval(
        _BASELINE_EVAL_STAGE_ID,
        dataset=_benchmark_dataset(),
        metrics=compute_addition_metrics,
        overrides=_EVAL_OVERRIDES,
    )

    return exp.run_branches(
        {
            "sft_branch": [
                exp.sft_stage(
                    _SFT_STAGE_ID,
                    dataset=bind_chat_sft_dataset(
                        build_addition_sft_train_dataset,
                        digits=_DIGITS,
                        train_sample_count=_SFT_TRAIN_SAMPLE_COUNT,
                        train_seed=_SFT_TRAIN_SEED,
                        benchmark_sample_count=_BENCHMARK_SAMPLE_COUNT,
                        benchmark_seed=_BENCHMARK_SEED,
                    ),
                    overrides={
                        "model": {
                            **_BASE_MODEL_OVERRIDES,
                            "max_seq_length": 1024,
                        },
                        "training": {
                            "hf_repo_id": _SFT_HF_REPO_ID,
                            "loss_on_assistant_only": True,
                            "response_template": "<|im_start|>assistant\n",
                            "val_size": 0,
                            "early_stopping_patience": None,
                            "max_steps": 2,
                            "per_device_train_batch_size": 4,
                            "gradient_accumulation_steps": 2,
                            "learning_rate": 5.0e-5,
                            "logging_steps": 1,
                            "save_steps": 1,
                            "hf_push_every_steps": 1,
                        },
                    },
                ),
                exp.eval_stage(
                    _SFT_EVAL_STAGE_ID,
                    adapter=exp.adapter(_SFT_STAGE_ID),
                    dataset=_benchmark_dataset(),
                    metrics=compute_addition_metrics,
                    overrides=_EVAL_OVERRIDES,
                ),
            ],
            "rl_branch": [
                exp.rl_stage(
                    _RL_STAGE_ID,
                    dataset=addition_rl_dataset(
                        digits=_DIGITS,
                        sample_count=_RL_TRAIN_SAMPLE_COUNT,
                        seed=_RL_TRAIN_SEED,
                        benchmark_sample_count=_BENCHMARK_SAMPLE_COUNT,
                        benchmark_seed=_BENCHMARK_SEED,
                    ),
                    reward=addition_reward(),
                    overrides={
                        "model": dict(_BASE_MODEL_OVERRIDES),
                        "chat_template": {
                            "enable_thinking": False,
                            "stop_strings": ["</answer>"],
                        },
                        "training": {
                            "hf_repo_id": _RL_HF_REPO_ID,
                            "max_steps": 1,
                            "per_device_batch_size": 1,
                            "gradient_accumulation_steps": 2,
                            "learning_rate": 5.0e-6,
                            "logging_steps": 1,
                            "num_generations": 2,
                            "max_prompt_length": 256,
                            "max_completion_length": 32,
                            "hf_push_every_steps": 1,
                        },
                    },
                ),
                exp.eval_stage(
                    _RL_EVAL_STAGE_ID,
                    adapter=exp.adapter(_RL_STAGE_ID),
                    dataset=_benchmark_dataset(),
                    metrics=compute_addition_metrics,
                    overrides=_EVAL_OVERRIDES,
                ),
            ],
        }
    )


if __name__ == "__main__":
    run_experiment(__file__, build)
