from smoke import prepare_wordle_experiment, wordle_smoke_overrides
from tenyson import run_experiment


def build(exp):
    smoke_overrides = wordle_smoke_overrides(
        include_sft=True,
        label="wordle experiment",
    ) or {}
    sft_overrides = smoke_overrides.get("sft")
    rl_overrides = smoke_overrides.get("rl")
    eval_overrides = smoke_overrides.get("eval")

    exp.sft(
        "sft_main",
        run="sft_main",
        overrides=sft_overrides,
    )
    sft_adapter = exp.adapter("sft_main")

    exp.eval(
        "eval_baseline_mixed",
        run="eval_mixed",
        adapter=sft_adapter,
        overrides=eval_overrides,
    )

    def build_mixed(branch):
        branch.rl(
            "mixed_rl",
            run="rl_mixed",
            adapter=sft_adapter,
            overrides=rl_overrides,
        )
        branch.eval(
            "mixed_final_eval",
            run="eval_mixed",
            adapter=branch.adapter("mixed_rl"),
            overrides=eval_overrides,
        )

    def build_curriculum(branch):
        branch.rl(
            "curr_rl_t2",
            run="rl_turn2",
            adapter=sft_adapter,
            overrides=rl_overrides,
        )
        branch.eval(
            "curr_eval_after_t2_turn2",
            run="eval_turn2",
            adapter=branch.adapter("curr_rl_t2"),
            overrides=eval_overrides,
        )

        branch.rl(
            "curr_rl_t3",
            run="rl_turn3",
            adapter=branch.adapter("curr_rl_t2"),
            overrides=rl_overrides,
        )
        branch.run_parallel(
            "curr_eval_after_t3",
            [
                branch.eval_stage(
                    "curr_eval_after_t3_turn2",
                    run="eval_turn2",
                    adapter=branch.adapter("curr_rl_t3"),
                    overrides=eval_overrides,
                ),
                branch.eval_stage(
                    "curr_eval_after_t3_turn3",
                    run="eval_turn3",
                    adapter=branch.adapter("curr_rl_t3"),
                    overrides=eval_overrides,
                ),
            ],
        )

        branch.rl(
            "curr_rl_t4",
            run="rl_turn4",
            adapter=branch.adapter("curr_rl_t3"),
            overrides=rl_overrides,
        )
        branch.run_parallel(
            "curr_eval_after_t4",
            [
                branch.eval_stage(
                    "curr_eval_after_t4_turn3",
                    run="eval_turn3",
                    adapter=branch.adapter("curr_rl_t4"),
                    overrides=eval_overrides,
                ),
                branch.eval_stage(
                    "curr_eval_after_t4_turn4",
                    run="eval_turn4",
                    adapter=branch.adapter("curr_rl_t4"),
                    overrides=eval_overrides,
                ),
            ],
        )

        branch.rl(
            "curr_rl_t5",
            run="rl_turn5",
            adapter=branch.adapter("curr_rl_t4"),
            overrides=rl_overrides,
        )
        branch.run_parallel(
            "curr_eval_after_t5",
            [
                branch.eval_stage(
                    "curr_eval_after_t5_turn4",
                    run="eval_turn4",
                    adapter=branch.adapter("curr_rl_t5"),
                    overrides=eval_overrides,
                ),
                branch.eval_stage(
                    "curr_eval_after_t5_turn5",
                    run="eval_turn5",
                    adapter=branch.adapter("curr_rl_t5"),
                    overrides=eval_overrides,
                ),
            ],
        )

        branch.eval(
            "curr_final_eval",
            run="eval_mixed",
            adapter=branch.adapter("curr_rl_t5"),
            overrides=eval_overrides,
        )

    exp.run_branches(
        {
            "mixed": build_mixed,
            "curriculum": build_curriculum,
        }
    )


if __name__ == "__main__":
    run_experiment(
        __file__,
        build,
        prepare=prepare_wordle_experiment,
        recovery_restart_stage_fallback_env_vars=(
            "TENYSON_WORDLE_RECOVER_RESTART_FROM_STAGE",
        ),
    )
