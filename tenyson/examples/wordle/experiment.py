import yaml

from tenyson.cloud.aws import AWSManager
from tenyson.jobs.eval import EvalJob
from tenyson.jobs.rl import RLJob
from tenyson.jobs.sft import SFTJob
from tenyson.pipeline import run_pipeline
from tenyson.examples.wordle.wordle_task import WordleTask


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _before_step(
    label: str,
    step_index: int,
    config: dict,
    previous_results: list,
) -> None:
    """Set init_adapter_repo from the previous step's result for config chaining."""
    if not previous_results:
        return
    prev = previous_results[-1]
    adapter = prev.hf_repo_id or prev.local_output_dir or ""
    config.setdefault("model", {})["init_adapter_repo"] = adapter


def main():
    task = WordleTask()
    cloud = AWSManager(instance_type="g5.2xlarge", auto_terminate=True)

    sft_cfg = load_yaml("tenyson/examples/wordle/configs/sft_config.yaml")
    rl_cfg = load_yaml("tenyson/examples/wordle/configs/rl_config.yaml")
    eval_cfg = load_yaml("tenyson/examples/wordle/configs/eval_config.yaml")

    steps = [
        ("sft", sft_cfg, SFTJob, task),
        ("rl", rl_cfg, RLJob, task),
        ("eval", eval_cfg, EvalJob, task),
    ]

    report_initial_data = {
        "sft_status": "pending",
        "rl_status": "pending",
        "eval_status": "pending",
        "eval_constraint_accuracy": "pending",
        "eval_dict_accuracy": "pending",
        "eval_format_accuracy": "pending",
        "sft_wandb_link": "—",
        "rl_wandb_link": "—",
    }

    run_pipeline(
        steps,
        cloud,
        report_template_path="tenyson/examples/wordle/report_template.md",
        report_output_path="tenyson/examples/wordle/final_report.md",
        report_initial_data=report_initial_data,
        before_step=_before_step,
    )


if __name__ == "__main__":
    main()
