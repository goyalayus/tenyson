from functional import (
    eval_turn_dataset,
    prime_metrics,
    prime_reward,
    rl_turn_dataset,
)
from tenyson import AdapterRef, run_experiment


EXPERIMENT2_SFT = AdapterRef(
    repo_id="goyalayus/wordle-lora-20260324-163252-sft_main",
    revision="30a33278640fcc5bcce216adce59984bfb8f7698",
)


def build(exp):
    exp.eval(
        "baseline",
        adapter=EXPERIMENT2_SFT,
        dataset=eval_turn_dataset(6),
        metrics=prime_metrics(),
        overrides={
            "task": {
                "eval_samples": 100,
                "eval_seed": 42,
            }
        },
    )
    exp.rl(
        "train",
        adapter=EXPERIMENT2_SFT,
        dataset=rl_turn_dataset(6),
        reward=prime_reward(),
        overrides={
            "training": {
                "max_steps": 1000,
                "hf_push_every_steps": 50,
                "num_generations": 2,
                "max_completion_length": 512,
            },
            "vllm": {
                "gpu_memory_utilization": 0.8,
                "max_tokens": 512,
            },
            "task": {
                "synthetic_samples": 4096,
            }
        },
    )
    exp.eval(
        "final",
        adapter=exp.adapter("train"),
        dataset=eval_turn_dataset(6),
        metrics=prime_metrics(),
        overrides={
            "task": {
                "eval_samples": 100,
                "eval_seed": 42,
            }
        },
    )


if __name__ == "__main__":
    run_experiment(__file__, build)
