import os
from pathlib import Path

from tenyson.cloud.aws import AWSManager
from tenyson.experiment import (
    ConfigTemplates,
    ExperimentAborted,
    ExperimentSession,
)
from tenyson.loader import load_task
from tenyson.reporting.builder import ReportBuilder


REPORT_METRICS = (
    "constraint_accuracy",
    "dict_accuracy",
    "format_accuracy",
)


def main() -> None:
    base_dir = Path(__file__).parent
    task = load_task(str(Path(__file__).with_name("wordle_task.py")))

    hf_repo_base = os.getenv("TENYSON_HF_REPO_BASE")
    shared_overrides = (
        {"training": {"hf_repo_base": hf_repo_base}} if hf_repo_base else None
    )
    session = ExperimentSession(
        task=task,
        templates=ConfigTemplates.from_directory(base_dir / "configs"),
        cloud_factory=AWSManager.factory_from_env(
            auto_terminate=True,
            use_spot=True,
        ),
        on_failure="wait",
        shared_overrides=shared_overrides,
    )
    primary_branch = session.branch(cloud=session.create_cloud())

    try:
        primary_branch.run(
            primary_branch.sft(
                "sft_main",
                run_name="wordle_sft_main",
                output_dir="./outputs/wordle_research/sft",
            )
        )
        sft_adapter = primary_branch.require_adapter("sft_main")
        primary_branch.run(
            primary_branch.eval(
                "eval_baseline_mixed",
                run_name="wordle_eval_baseline_mixed",
                output_dir="./outputs/wordle_research/eval_baseline_mixed",
                adapter=sft_adapter,
                overrides={
                    "task": {
                        "min_history_turns": 1,
                        "max_history_turns": 5,
                    }
                },
            )
        )
        branch_results = session.run_branches(
            {
                "mixed": lambda branch: (
                    branch.run(
                        branch.rl(
                            "mixed_rl",
                            run_name="wordle_rl_mixed",
                            output_dir="./outputs/wordle_research/mixed/rl",
                            adapter=sft_adapter,
                            overrides={
                                "task": {
                                    "min_history_turns": 1,
                                    "max_history_turns": 5,
                                }
                            },
                        )
                    ),
                    branch.run(
                        branch.eval(
                            "mixed_final_eval",
                            run_name="wordle_mixed_final_eval",
                            output_dir="./outputs/wordle_research/mixed/eval_final_mixed",
                            adapter=branch.require_adapter("mixed_rl"),
                            overrides={
                                "task": {
                                    "min_history_turns": 1,
                                    "max_history_turns": 5,
                                }
                            },
                        )
                    ),
                ),
                "curriculum": lambda branch: (
                    branch.run(
                        branch.rl(
                            "curr_rl_t2",
                            run_name="wordle_curriculum_rl_t2",
                            output_dir="./outputs/wordle_research/curriculum/rl_t2",
                            adapter=sft_adapter,
                            overrides={
                                "task": {
                                    "min_history_turns": 2,
                                    "max_history_turns": 2,
                                }
                            },
                        )
                    ),
                    branch.run(
                        branch.eval(
                            "curr_eval_after_t2_turn2",
                            run_name="wordle_curr_eval_after_t2_turn2",
                            output_dir="./outputs/wordle_research/curriculum/eval_after_t2_turn2",
                            adapter=branch.require_adapter("curr_rl_t2"),
                            overrides={
                                "task": {
                                    "min_history_turns": 2,
                                    "max_history_turns": 2,
                                    "eval_exact_turns": [2],
                                }
                            },
                        )
                    ),
                    branch.run(
                        branch.rl(
                            "curr_rl_t3",
                            run_name="wordle_curriculum_rl_t3",
                            output_dir="./outputs/wordle_research/curriculum/rl_t3",
                            adapter=branch.require_adapter("curr_rl_t2"),
                            overrides={
                                "task": {
                                    "min_history_turns": 3,
                                    "max_history_turns": 3,
                                }
                            },
                        )
                    ),
                    branch.run_parallel(
                        "curr_eval_after_t3",
                        [
                            branch.eval(
                                "curr_eval_after_t3_turn2",
                                run_name="wordle_curr_eval_after_t3_turn2",
                                output_dir="./outputs/wordle_research/curriculum/eval_after_t3_turn2",
                                adapter=branch.require_adapter("curr_rl_t3"),
                                overrides={
                                    "task": {
                                        "min_history_turns": 2,
                                        "max_history_turns": 2,
                                        "eval_exact_turns": [2],
                                    }
                                },
                            ),
                            branch.eval(
                                "curr_eval_after_t3_turn3",
                                run_name="wordle_curr_eval_after_t3_turn3",
                                output_dir="./outputs/wordle_research/curriculum/eval_after_t3_turn3",
                                adapter=branch.require_adapter("curr_rl_t3"),
                                overrides={
                                    "task": {
                                        "min_history_turns": 3,
                                        "max_history_turns": 3,
                                        "eval_exact_turns": [3],
                                    }
                                },
                            ),
                        ],
                    ),
                    branch.run(
                        branch.rl(
                            "curr_rl_t4",
                            run_name="wordle_curriculum_rl_t4",
                            output_dir="./outputs/wordle_research/curriculum/rl_t4",
                            adapter=branch.require_adapter("curr_rl_t3"),
                            overrides={
                                "task": {
                                    "min_history_turns": 4,
                                    "max_history_turns": 4,
                                }
                            },
                        )
                    ),
                    branch.run_parallel(
                        "curr_eval_after_t4",
                        [
                            branch.eval(
                                "curr_eval_after_t4_turn3",
                                run_name="wordle_curr_eval_after_t4_turn3",
                                output_dir="./outputs/wordle_research/curriculum/eval_after_t4_turn3",
                                adapter=branch.require_adapter("curr_rl_t4"),
                                overrides={
                                    "task": {
                                        "min_history_turns": 3,
                                        "max_history_turns": 3,
                                        "eval_exact_turns": [3],
                                    }
                                },
                            ),
                            branch.eval(
                                "curr_eval_after_t4_turn4",
                                run_name="wordle_curr_eval_after_t4_turn4",
                                output_dir="./outputs/wordle_research/curriculum/eval_after_t4_turn4",
                                adapter=branch.require_adapter("curr_rl_t4"),
                                overrides={
                                    "task": {
                                        "min_history_turns": 4,
                                        "max_history_turns": 4,
                                        "eval_exact_turns": [4],
                                    }
                                },
                            ),
                        ],
                    ),
                    branch.run(
                        branch.rl(
                            "curr_rl_t5",
                            run_name="wordle_curriculum_rl_t5",
                            output_dir="./outputs/wordle_research/curriculum/rl_t5",
                            adapter=branch.require_adapter("curr_rl_t4"),
                            overrides={
                                "task": {
                                    "min_history_turns": 5,
                                    "max_history_turns": 5,
                                }
                            },
                        )
                    ),
                    branch.run_parallel(
                        "curr_eval_after_t5",
                        [
                            branch.eval(
                                "curr_eval_after_t5_turn4",
                                run_name="wordle_curr_eval_after_t5_turn4",
                                output_dir="./outputs/wordle_research/curriculum/eval_after_t5_turn4",
                                adapter=branch.require_adapter("curr_rl_t5"),
                                overrides={
                                    "task": {
                                        "min_history_turns": 4,
                                        "max_history_turns": 4,
                                        "eval_exact_turns": [4],
                                    }
                                },
                            ),
                            branch.eval(
                                "curr_eval_after_t5_turn5",
                                run_name="wordle_curr_eval_after_t5_turn5",
                                output_dir="./outputs/wordle_research/curriculum/eval_after_t5_turn5",
                                adapter=branch.require_adapter("curr_rl_t5"),
                                overrides={
                                    "task": {
                                        "min_history_turns": 5,
                                        "max_history_turns": 5,
                                        "eval_exact_turns": [5],
                                    }
                                },
                            ),
                        ],
                    ),
                    branch.run(
                        branch.eval(
                            "curr_final_eval",
                            run_name="wordle_curriculum_final_eval",
                            output_dir="./outputs/wordle_research/curriculum/eval_final_mixed",
                            adapter=branch.require_adapter("curr_rl_t5"),
                            overrides={
                                "task": {
                                    "min_history_turns": 1,
                                    "max_history_turns": 5,
                                }
                            },
                        )
                    ),
                ),
            }
        )
    except ExperimentAborted as exc:
        print(exc)
        return

    mixed_results = branch_results["mixed"]
    curriculum_results = branch_results["curriculum"]
    all_results = ExperimentSession.combine_results(
        primary_branch.results(),
        mixed_results,
        curriculum_results,
    )

    report = ReportBuilder(
        template_path=str(base_dir / "report_template.md"),
        output_path=str(base_dir / "final_report.md"),
    )
    report.fill_results(all_results, metric_precision=4, wandb_text="run")
    for metric_name in REPORT_METRICS:
        report.fill_metric_delta(
            f"delta_final_{metric_name}",
            mixed_results.get("mixed_final_eval"),
            curriculum_results.get("curr_final_eval"),
            metric_name,
            precision=4,
        )
    report.generate()


if __name__ == "__main__":
    main()
