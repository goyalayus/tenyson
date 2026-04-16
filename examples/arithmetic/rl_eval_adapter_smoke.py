from functional import (
    addition_reward,
    addition_rl_dataset,
    build_addition_dataset,
    compute_addition_metrics,
)
from tenyson import bind_eval_dataset, run_experiment


def build(exp):
    exp.rl(
        "rl_adapter_smoke",
        run="rl_adapter_smoke",
        dataset=addition_rl_dataset(
            digits=2,
            sample_count=16,
            seed=456,
            benchmark_sample_count=8,
            benchmark_seed=7,
        ),
        reward=addition_reward(),
        overrides={
            "model": {
                "name": "Qwen/Qwen3-0.6B",
                "max_seq_length": 512,
                "load_in_4bit": False,
            },
            "chat_template": {
                "enable_thinking": False,
                "stop_strings": ["</answer>"],
            },
            "vllm": {
                "gpu_memory_utilization": 0.5,
            },
            "training": {
                "hf_repo_id": "goyalayus/arithmetic-rl-adapter-smoke-20260416",
                "max_steps": 1,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 5.0e-6,
                "num_generations": 2,
                "max_prompt_length": 128,
                "max_completion_length": 32,
                "hf_push_every_steps": 1,
                "logging_steps": 1,
            },
        },
    )
    exp.eval(
        "eval_after_rl_adapter_smoke",
        run="eval_after_rl_adapter_smoke",
        adapter=exp.adapter("rl_adapter_smoke"),
        dataset=bind_eval_dataset(
            build_addition_dataset,
            digits=2,
            sample_count=8,
            seed=7,
        ),
        metrics=compute_addition_metrics,
        overrides={
            "evaluation": {
                "batch_size": 8,
            },
            "model": {
                "name": "Qwen/Qwen3-0.6B",
                "max_seq_length": 512,
                "load_in_4bit": False,
            },
            "chat_template": {
                "enable_thinking": False,
                "stop_strings": ["</answer>"],
            },
            "vllm": {
                "gpu_memory_utilization": 0.5,
                "max_tokens": 32,
            },
        },
    )


if __name__ == "__main__":
    run_experiment(__file__, build)
