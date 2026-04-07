from functional import (
    constraint_metrics,
    constraint_reward,
    eval_turn_dataset,
    rl_turn_dataset,
    sft_chat_dataset,
)
from tenyson import run_experiment


def _sft_overrides():
    return {
        "model": {
            "name": "Qwen/Qwen3-0.6B",
            "max_seq_length": 1024,
            "load_in_4bit": False,
            "load_in_8bit": False,
            # Intentionally request the incompatible flag so the runtime has to
            # normalize it away for full finetuning.
            "fast_inference": True,
        },
        "training": {
            "loss_on_assistant_only": True,
            "response_template": "<|im_start|>assistant\n",
            "finetune_mode": "full",
            "max_steps": 2,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5.0e-6,
            "warmup_steps": 0,
            "logging_steps": 1,
            "save_steps": 1,
            "eval_steps": 1,
            "hf_push_every_steps": 1,
            "save_total_limit": 1,
            "val_size": 16,
            "early_stopping_patience": 1,
        },
        "task": {
            "sft_dataset": "goyalayus/wordle-reasoning-sft-prefix-keep-think",
        },
    }


def _eval_overrides():
    return {
        "model": {
            "name": "Qwen/Qwen3-0.6B",
            "max_seq_length": 1024,
            "load_in_4bit": True,
        },
        "vllm": {
            "enabled": True,
            "disable_flashinfer": True,
            "gpu_memory_utilization": 0.6,
            "max_tokens": 64,
        },
        "task": {
            "eval_samples": 16,
            "eval_seed": 42,
        },
    }


def _rl_overrides():
    return {
        "model": {
            "name": "unsloth/Qwen3-0.6B-Base",
            "max_seq_length": 1024,
            "load_in_4bit": False,
            # Intentionally request the default LoRA GRPO path so the runtime
            # has to disable it for full finetuning.
            "fast_inference": True,
        },
        "vllm": {
            "enabled": True,
            "disable_flashinfer": True,
            "gpu_memory_utilization": 0.5,
            "temperature": 1.0,
            "min_p": 0.1,
            "top_p": 1.0,
            "top_k": -1,
        },
        "training": {
            "finetune_mode": "full",
            "max_steps": 2,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5.0e-6,
            "num_generations": 2,
            "max_prompt_length": 256,
            "max_completion_length": 64,
            "hf_push_every_steps": 1,
        },
        "task": {
            "synthetic_samples": 8,
        },
    }


def build(exp):
    exp.sft(
        "sft_full_smoke_06b_autofix",
        dataset=sft_chat_dataset(),
        overrides=_sft_overrides(),
    )
    full_model = exp.artifact("sft_full_smoke_06b_autofix")

    exp.eval(
        "eval_full_after_sft_06b_autofix",
        artifact=full_model,
        dataset=eval_turn_dataset(2),
        metrics=constraint_metrics(),
        overrides=_eval_overrides(),
    )

    exp.rl(
        "rl_full_from_sft_06b_autofix",
        artifact=full_model,
        dataset=rl_turn_dataset(2),
        reward=constraint_reward(),
        overrides=_rl_overrides(),
    )


if __name__ == "__main__":
    run_experiment(__file__, build)
