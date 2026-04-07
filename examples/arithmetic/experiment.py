from functional import (
    build_four_digit_addition_dataset,
    compute_addition_metrics,
)
from tenyson import run_experiment


def build(exp):
    exp.eval(
        "baseline",
        dataset=build_four_digit_addition_dataset,
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
