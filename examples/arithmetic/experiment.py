from functional import (
    three_digit_addition_eval_dataset,
    three_digit_addition_metrics,
)
from tenyson import AdapterRef, run_experiment


BASE_MODEL = AdapterRef(
    repo_id="unsloth/qwen3-4b-unsloth-bnb-4bit",
    revision="main",
    artifact_type="full_model",
)


def build(exp):
    exp.eval(
        "baseline",
        artifact=BASE_MODEL,
        dataset=three_digit_addition_eval_dataset(),
        metrics=three_digit_addition_metrics(),
        overrides={
            "task": {
                "eval_samples": 100,
                "eval_seed": 7,
            },
            "vllm": {
                "max_tokens": 32,
            },
        },
    )


if __name__ == "__main__":
    run_experiment(__file__, build)
