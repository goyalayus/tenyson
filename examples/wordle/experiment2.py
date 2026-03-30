from functional import (
    eval_turn_dataset,
    prime_metrics,
    prime_reward,
    rl_turn_dataset,
)
from tenyson import run_experiment


def build(exp):
    seed = exp.seed("experiment2_sft")
    exp.eval(
        "baseline",
        adapter=seed,
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
        adapter=seed,
        dataset=rl_turn_dataset(6),
        reward=prime_reward(),
        overrides={
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
