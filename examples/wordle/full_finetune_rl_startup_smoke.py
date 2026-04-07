from functional import (
    constraint_reward,
    rl_turn_dataset,
)
from tenyson import AdapterRef, run_experiment


_FULL_MODEL_ARTIFACT = AdapterRef(
    repo_id="goyalayus/wordle-lora-20260324-163252-sft_full_smoke",
    revision="f1095a024cb4a9957e6009239638393d8ac1ae85",
    artifact_type="full_model",
)


def _rl_overrides():
    return {
        "model": {
            "name": "unsloth/Qwen3-4B-Base",
            "max_seq_length": 1024,
            "load_in_4bit": False,
            "fast_inference": False,
        },
        "vllm": {
            "enabled": False,
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
    exp.rl(
        "rl_full_from_sft",
        artifact=_FULL_MODEL_ARTIFACT,
        dataset=rl_turn_dataset(2),
        reward=constraint_reward(),
        overrides=_rl_overrides(),
    )


if __name__ == "__main__":
    run_experiment(__file__, build)
