from functional import (
    addition_reward,
    addition_rl_dataset,
    build_addition_dataset,
    build_addition_sft_train_dataset,
    compute_addition_metrics,
)
from tenyson import bind_chat_sft_dataset, bind_eval_dataset, run_experiment


def build(exp):
    return exp.run_branches(
        {
            "baseline_branch": [
                exp.eval_stage(
                    "eval_2digit_baseline",
                    dataset=bind_eval_dataset(
                        build_addition_dataset,
                        digits=2,
                        sample_count=100,
                        seed=7,
                    ),
                    metrics=compute_addition_metrics,
                    overrides={
                        "model": {
                            "name": "Qwen/Qwen3-0.6B",
                            "load_in_4bit": False,
                        },
                        "chat_template": {
                            "enable_thinking": False,
                            "stop_strings": ["</answer>"],
                        },
                        "vllm": {
                            "max_tokens": 64,
                        },
                    },
                ),
            ],
            "sft_branch": [
                exp.sft_stage(
                    "sft_2digit_06b",
                    dataset=bind_chat_sft_dataset(
                        build_addition_sft_train_dataset,
                        digits=2,
                        train_sample_count=4096,
                        train_seed=123,
                        benchmark_sample_count=100,
                        benchmark_seed=7,
                    ),
                    overrides={
                        "model": {
                            "name": "Qwen/Qwen3-0.6B",
                            "max_seq_length": 1024,
                            "load_in_4bit": False,
                        },
                        "training": {
                            "hf_repo_id": "goyalayus/arithmetic-2digit-sft_2digit_06b",
                            "loss_on_assistant_only": True,
                            "response_template": "<|im_start|>assistant\n",
                            "val_size": 0,
                            "max_steps": 250,
                            "per_device_train_batch_size": 8,
                            "gradient_accumulation_steps": 4,
                            "learning_rate": 5.0e-5,
                            "logging_steps": 10,
                            "save_steps": 50,
                            "hf_push_every_steps": 50,
                        },
                    },
                ),
                exp.eval_stage(
                    "eval_2digit_after_sft",
                    adapter=exp.adapter("sft_2digit_06b"),
                    dataset=bind_eval_dataset(
                        build_addition_dataset,
                        digits=2,
                        sample_count=100,
                        seed=7,
                    ),
                    metrics=compute_addition_metrics,
                    overrides={
                        "model": {
                            "name": "Qwen/Qwen3-0.6B",
                            "load_in_4bit": False,
                        },
                        "chat_template": {
                            "enable_thinking": False,
                            "stop_strings": ["</answer>"],
                        },
                        "vllm": {
                            "max_tokens": 64,
                        },
                    },
                ),
            ],
            "rl_branch": [
                exp.rl_stage(
                    "rl_2digit_06b",
                    dataset=addition_rl_dataset(
                        digits=2,
                        sample_count=4096,
                        seed=456,
                        benchmark_sample_count=100,
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
                            "hf_repo_id": "goyalayus/arithmetic-2digit-rl_2digit_06b",
                            "max_steps": 150,
                            "per_device_batch_size": 1,
                            "gradient_accumulation_steps": 4,
                            "learning_rate": 5.0e-6,
                            "num_generations": 4,
                            "max_prompt_length": 256,
                            "max_completion_length": 64,
                            "hf_push_every_steps": 50,
                        },
                    },
                ),
                exp.eval_stage(
                    "eval_2digit_after_rl",
                    adapter=exp.adapter("rl_2digit_06b"),
                    dataset=bind_eval_dataset(
                        build_addition_dataset,
                        digits=2,
                        sample_count=100,
                        seed=7,
                    ),
                    metrics=compute_addition_metrics,
                    overrides={
                        "model": {
                            "name": "Qwen/Qwen3-0.6B",
                            "load_in_4bit": False,
                        },
                        "chat_template": {
                            "enable_thinking": False,
                            "stop_strings": ["</answer>"],
                        },
                        "vllm": {
                            "max_tokens": 64,
                        },
                    },
                ),
            ],
        }
    )


if __name__ == "__main__":
    run_experiment(__file__, build)
