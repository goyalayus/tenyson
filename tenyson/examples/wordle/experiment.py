import yaml

from tenyson.cloud.aws import AWSManager
from tenyson.jobs.eval import EvalJob
from tenyson.jobs.rl import RLJob
from tenyson.jobs.sft import SFTJob
from tenyson.reporting.builder import ReportBuilder
from tenyson.examples.wordle.wordle_task import WordleTask


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    task = WordleTask()
    cloud = AWSManager(instance_type="g5.2xlarge", auto_terminate=True)

    sft_cfg = load_yaml("tenyson/examples/wordle/configs/sft_config.yaml")
    rl_cfg = load_yaml("tenyson/examples/wordle/configs/rl_config.yaml")
    eval_cfg = load_yaml("tenyson/examples/wordle/configs/eval_config.yaml")

    sft_result = cloud.run(SFTJob(config=sft_cfg, task=task))
    rl_cfg.setdefault("model", {})["init_adapter_repo"] = (
        sft_result.hf_repo_id or sft_result.local_output_dir or ""
    )
    rl_result = cloud.run(RLJob(config=rl_cfg, task=task))
    eval_cfg.setdefault("model", {})["init_adapter_repo"] = (
        rl_result.hf_repo_id or rl_result.local_output_dir or ""
    )
    eval_result = cloud.run(EvalJob(config=eval_cfg, task=task))

    template = "tenyson/examples/wordle/report_template.md"
    output = "tenyson/examples/wordle/final_report.md"
    report = ReportBuilder(template_path=template, output_path=output)
    report.fill(
        {
            "sft_status": sft_result.status,
            "rl_status": rl_result.status,
            "eval_status": eval_result.status,
            "eval_constraint_accuracy": eval_result.metrics.get(
                "constraint_accuracy", "n/a"
            ),
            "eval_dict_accuracy": eval_result.metrics.get("dict_accuracy", "n/a"),
            "eval_format_accuracy": eval_result.metrics.get("format_accuracy", "n/a"),
            # Defaults for optional WandB placeholders so the report still
            # renders cleanly when WandB is not used.
            "sft_wandb_link": "n/a",
            "rl_wandb_link": "n/a",
        }
    )

    # If WandB was enabled for SFT / RL runs, replace the defaults with links.
    if sft_result.wandb_url:
        report.attach_wandb_scalar_link(
            placeholder="sft_wandb_link",
            run_url=sft_result.wandb_url,
            metric_name="SFT run (WandB)",
        )
    if rl_result.wandb_url:
        report.attach_wandb_scalar_link(
            placeholder="rl_wandb_link",
            run_url=rl_result.wandb_url,
            metric_name="RL run (WandB)",
        )
    report.generate()


if __name__ == "__main__":
    main()
