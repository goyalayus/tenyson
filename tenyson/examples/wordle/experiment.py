import copy
from pathlib import Path

import yaml

from tenyson.cloud.aws import AWSManager
from tenyson.jobs.eval import EvalJob
from tenyson.jobs.rl import RLJob
from tenyson.jobs.sft import SFTJob
from tenyson.loader import load_task
from tenyson.pipeline import run_pipeline


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _before_step(
    label: str,
    step_index: int,
    config: dict,
    previous_results: list,
) -> None:
    """Set init_adapter_repo from the most recent previous result that has hf_repo_id."""
    for prev in reversed(previous_results):
        if getattr(prev, "hf_repo_id", None):
            model_cfg = config.setdefault("model", {})
            model_cfg["init_adapter_repo"] = prev.hf_repo_id
            if getattr(prev, "hf_revision", None):
                model_cfg["init_adapter_revision"] = prev.hf_revision
            return


def main():
    base_dir = Path(__file__).parent
    task = load_task(str(Path(__file__).with_name("wordle_task.py")))
    cloud = AWSManager(instance_type="g5.2xlarge", auto_terminate=True)

    sft_cfg = load_yaml(str(base_dir / "configs" / "sft_config.yaml"))
    rl_cfg = load_yaml(str(base_dir / "configs" / "rl_config.yaml"))
    eval_cfg = load_yaml(str(base_dir / "configs" / "eval_config.yaml"))

    rl_mixed_cfg = copy.deepcopy(rl_cfg)
    rl_mixed_cfg.setdefault("training", {})["run_name"] = "wordle_rl_mixed"
    rl_mixed_cfg.setdefault("training", {})["output_dir"] = "./outputs/rl_mixed"

    rl_alt_cfg = copy.deepcopy(rl_cfg)
    rl_alt_cfg.setdefault("training", {})["run_name"] = "wordle_rl_alt"
    rl_alt_cfg.setdefault("training", {})["output_dir"] = "./outputs/rl_alt"

    steps = [
        ("sft", sft_cfg, SFTJob, task),
        ("eval", eval_cfg, EvalJob, task),
        {
            "label": "rl_branches",
            "parallel": [
                ("rl_mixed", rl_mixed_cfg, RLJob, task),
                ("rl_alt", rl_alt_cfg, RLJob, task),
            ],
        },
    ]

    report_initial_data = {
        "sft_status": "pending",
        "rl_mixed_status": "pending",
        "rl_alt_status": "pending",
        "eval_status": "pending",
        "eval_constraint_accuracy": "pending",
        "eval_dict_accuracy": "pending",
        "eval_format_accuracy": "pending",
        "sft_wandb_link": "—",
        "rl_mixed_wandb_link": "—",
        "rl_alt_wandb_link": "—",
    }

    run_pipeline(
        steps,
        cloud,
        report_template_path=str(base_dir / "report_template.md"),
        report_output_path=str(base_dir / "final_report.md"),
        report_initial_data=report_initial_data,
        before_step=_before_step,
    )


if __name__ == "__main__":
    main()
