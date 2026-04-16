from functional import (
    addition_reward,
    addition_rl_dataset,
    build_addition_dataset,
    compute_addition_metrics,
)
from tenyson import AdapterRef, bind_eval_dataset, run_experiment


SEED_SFT_ADAPTER = AdapterRef(
    repo_id="goyalayus/arithmetic-2digit-sft_2digit_06b",
    revision="8960cc9cdf34ee155a4dccc97424a0a1e7b36f08",
)


def build(exp):
    exp.eval(
        "eval_seed_sft_adapter",
        adapter=SEED_SFT_ADAPTER,
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
    )
    exp.rl(
        "rl_seeded_from_sft_smoke",
        adapter=SEED_SFT_ADAPTER,
        dataset=addition_rl_dataset(
            digits=2,
            sample_count=512,
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
                "hf_repo_id": "goyalayus/arithmetic-rl-seeded-from-sft-smoke",
                "max_steps": 1,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 5.0e-6,
                "num_generations": 2,
                "max_prompt_length": 256,
                "max_completion_length": 64,
                "logging_steps": 1,
                "save_steps": 1,
                "hf_push_every_steps": 1,
            },
            "vllm": {
                "standby_mode": False,
                "max_tokens": 64,
            },
        },
    )
    exp.eval(
        "eval_after_seeded_rl_smoke",
        adapter=exp.adapter("rl_seeded_from_sft_smoke"),
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
    )


if __name__ == "__main__":
    run_experiment(__file__, build)
