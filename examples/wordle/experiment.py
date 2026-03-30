from functional import (
    constraint_metrics,
    constraint_reward,
    eval_defaults,
    eval_mixed_dataset,
    eval_turn_dataset,
    rl_defaults,
    rl_mixed_dataset,
    rl_turn_dataset,
    sft_chat_dataset,
    sft_defaults,
)
from tenyson import run_experiment


def build(exp):
    sft_overrides = sft_defaults()
    rl_overrides = rl_defaults()
    eval_overrides = eval_defaults()

    exp.sft(
        "sft_main",
        dataset=sft_chat_dataset(),
        overrides=sft_overrides,
    )
    sft_adapter = exp.adapter("sft_main")

    exp.eval(
        "eval_baseline_mixed",
        adapter=sft_adapter,
        dataset=eval_mixed_dataset(),
        metrics=constraint_metrics(),
        overrides=eval_overrides,
    )

    def build_mixed(branch):
        branch.rl(
            "mixed_rl",
            adapter=sft_adapter,
            dataset=rl_mixed_dataset(),
            reward=constraint_reward(),
            overrides=rl_overrides,
        )
        branch.eval(
            "mixed_final_eval",
            adapter=branch.adapter("mixed_rl"),
            dataset=eval_mixed_dataset(),
            metrics=constraint_metrics(),
            overrides=eval_overrides,
        )

    def build_curriculum(branch):
        branch.rl(
            "curr_rl_t2",
            adapter=sft_adapter,
            dataset=rl_turn_dataset(2),
            reward=constraint_reward(),
            overrides=rl_overrides,
        )
        branch.eval(
            "curr_eval_after_t2_turn2",
            adapter=branch.adapter("curr_rl_t2"),
            dataset=eval_turn_dataset(2),
            metrics=constraint_metrics(),
            overrides=eval_overrides,
        )

        branch.rl(
            "curr_rl_t3",
            adapter=branch.adapter("curr_rl_t2"),
            dataset=rl_turn_dataset(3),
            reward=constraint_reward(),
            overrides=rl_overrides,
        )
        branch.run_parallel(
            "curr_eval_after_t3",
            [
                branch.eval_stage(
                    "curr_eval_after_t3_turn2",
                    adapter=branch.adapter("curr_rl_t3"),
                    dataset=eval_turn_dataset(2),
                    metrics=constraint_metrics(),
                    overrides=eval_overrides,
                ),
                branch.eval_stage(
                    "curr_eval_after_t3_turn3",
                    adapter=branch.adapter("curr_rl_t3"),
                    dataset=eval_turn_dataset(3),
                    metrics=constraint_metrics(),
                    overrides=eval_overrides,
                ),
            ],
        )

        branch.rl(
            "curr_rl_t4",
            adapter=branch.adapter("curr_rl_t3"),
            dataset=rl_turn_dataset(4),
            reward=constraint_reward(),
            overrides=rl_overrides,
        )
        branch.run_parallel(
            "curr_eval_after_t4",
            [
                branch.eval_stage(
                    "curr_eval_after_t4_turn3",
                    adapter=branch.adapter("curr_rl_t4"),
                    dataset=eval_turn_dataset(3),
                    metrics=constraint_metrics(),
                    overrides=eval_overrides,
                ),
                branch.eval_stage(
                    "curr_eval_after_t4_turn4",
                    adapter=branch.adapter("curr_rl_t4"),
                    dataset=eval_turn_dataset(4),
                    metrics=constraint_metrics(),
                    overrides=eval_overrides,
                ),
            ],
        )

        branch.rl(
            "curr_rl_t5",
            adapter=branch.adapter("curr_rl_t4"),
            dataset=rl_turn_dataset(5),
            reward=constraint_reward(),
            overrides=rl_overrides,
        )
        branch.run_parallel(
            "curr_eval_after_t5",
            [
                branch.eval_stage(
                    "curr_eval_after_t5_turn4",
                    adapter=branch.adapter("curr_rl_t5"),
                    dataset=eval_turn_dataset(4),
                    metrics=constraint_metrics(),
                    overrides=eval_overrides,
                ),
                branch.eval_stage(
                    "curr_eval_after_t5_turn5",
                    adapter=branch.adapter("curr_rl_t5"),
                    dataset=eval_turn_dataset(5),
                    metrics=constraint_metrics(),
                    overrides=eval_overrides,
                ),
            ],
        )

        branch.rl(
            "curr_rl_t6",
            adapter=branch.adapter("curr_rl_t5"),
            dataset=rl_turn_dataset(6),
            reward=constraint_reward(),
            overrides=rl_overrides,
        )
        branch.run_parallel(
            "curr_eval_after_t6",
            [
                branch.eval_stage(
                    "curr_eval_after_t6_turn5",
                    adapter=branch.adapter("curr_rl_t6"),
                    dataset=eval_turn_dataset(5),
                    metrics=constraint_metrics(),
                    overrides=eval_overrides,
                ),
                branch.eval_stage(
                    "curr_eval_after_t6_turn6",
                    adapter=branch.adapter("curr_rl_t6"),
                    dataset=eval_turn_dataset(6),
                    metrics=constraint_metrics(),
                    overrides=eval_overrides,
                ),
            ],
        )

        branch.eval(
            "curr_final_eval",
            adapter=branch.adapter("curr_rl_t6"),
            dataset=eval_mixed_dataset(),
            metrics=constraint_metrics(),
            overrides=eval_overrides,
        )

    exp.run_branches(
        {
            "mixed": build_mixed,
            "curriculum": build_curriculum,
        }
    )


if __name__ == "__main__":
    run_experiment(__file__, build)
