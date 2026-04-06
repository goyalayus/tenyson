from functional import (
    build_three_digit_addition_dataset,
    compute_addition_metrics,
)
from tenyson import run_experiment


def build(exp):
    exp.eval(
        "baseline",
        dataset=build_three_digit_addition_dataset,
        metrics=compute_addition_metrics,
        overrides={
            "vllm": {
                "max_tokens": 32,
            },
        },
    )


if __name__ == "__main__":
    run_experiment(__file__, build)
