import os
from pathlib import Path

from tenyson.cloud.aws import AWSManager
from tenyson.experiment import (
    AdapterRef,
    ConfigTemplates,
    ExperimentAborted,
    ExperimentBranch,
    ExperimentSession,
)
from tenyson.loader import load_task
from tenyson.reporting.builder import ReportBuilder


REPORT_METRICS = (
    "constraint_accuracy",
    "dict_accuracy",
    "format_accuracy",
)


def _curriculum_eval_turns(stage_turn: int) -> list[int]:
    if stage_turn == 2:
        return [2]
    if stage_turn == 3:
        return [2, 3]
    if stage_turn == 4:
        return [3, 4]
    if stage_turn == 5:
        return [4, 5]
    raise ValueError(f"Unsupported curriculum stage turn: {stage_turn}")


def _run_mixed_branch(branch: ExperimentBranch, *, adapter: AdapterRef) -> None:
    branch.run(
        branch.rl(
            "mixed_rl",
            run_name="wordle_rl_mixed",
            output_dir="./outputs/wordle_research/mixed/rl",
            adapter=adapter,
            overrides={
                "task": {
                    "min_history_turns": 1,
                    "max_history_turns": 5,
                }
            },
        )
    )
    mixed_adapter = branch.require_adapter("mixed_rl")
    branch.run(
        branch.eval(
            "mixed_final_eval",
            run_name="wordle_mixed_final_eval",
            output_dir="./outputs/wordle_research/mixed/eval_final_mixed",
            adapter=mixed_adapter,
            overrides={
                "task": {
                    "min_history_turns": 1,
                    "max_history_turns": 5,
                }
            },
        )
    )


def _run_curriculum_branch(branch: ExperimentBranch, *, adapter: AdapterRef) -> None:
    current_adapter = adapter

    for stage_turn in [2, 3, 4, 5]:
        rl_key = f"curr_rl_t{stage_turn}"
        branch.run(
            branch.rl(
                rl_key,
                run_name=f"wordle_curriculum_rl_t{stage_turn}",
                output_dir=f"./outputs/wordle_research/curriculum/rl_t{stage_turn}",
                adapter=current_adapter,
                overrides={
                    "task": {
                        "min_history_turns": stage_turn,
                        "max_history_turns": stage_turn,
                    }
                },
            )
        )
        current_adapter = branch.require_adapter(rl_key)

        eval_stages = []
        for turn in _curriculum_eval_turns(stage_turn):
            eval_key = f"curr_eval_after_t{stage_turn}_turn{turn}"
            eval_stages.append(
                branch.eval(
                    eval_key,
                    run_name=f"wordle_curr_eval_after_t{stage_turn}_turn{turn}",
                    output_dir=f"./outputs/wordle_research/curriculum/eval_after_t{stage_turn}_turn{turn}",
                    adapter=current_adapter,
                    overrides={
                        "task": {
                            "min_history_turns": turn,
                            "max_history_turns": turn,
                            "eval_exact_turns": [turn],
                        }
                    },
                )
            )

        if len(eval_stages) == 1:
            branch.run(eval_stages[0])
        else:
            branch.run_parallel(
                label=f"curr_eval_after_t{stage_turn}",
                stages=eval_stages,
            )

    branch.run(
        branch.eval(
            "curr_final_eval",
            run_name="wordle_curriculum_final_eval",
            output_dir="./outputs/wordle_research/curriculum/eval_final_mixed",
            adapter=current_adapter,
            overrides={
                "task": {
                    "min_history_turns": 1,
                    "max_history_turns": 5,
                }
            },
        )
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
                "mixed": lambda branch: _run_mixed_branch(branch, adapter=sft_adapter),
                "curriculum": lambda branch: _run_curriculum_branch(
                    branch,
                    adapter=sft_adapter,
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
