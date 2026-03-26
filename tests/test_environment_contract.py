import tempfile
import unittest
from pathlib import Path

from tenyson.experiment import AdapterRef, ConfigTemplates, ExperimentSession
from tenyson.jobs.result import JobResult
from tenyson.loader import load_task
from tenyson.reporting.fixed import ExperimentReport


class EnvironmentContractTests(unittest.TestCase):
    def test_wordle_named_runs_drive_stage_config(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        task = load_task(str(repo_root / "examples" / "wordle" / "wordle_task.py"))
        self.assertEqual(task.get_environment_name(), "wordle")
        self.assertIn("wordle_rl_turn4", task.list_named_runs("rl"))
        self.assertIn("wordle_eval_turn5", task.list_named_runs("eval"))
        self.assertEqual(task.get_named_run_type("wordle_rl_turn4"), "rl")

        session = ExperimentSession(
            task=task,
            templates=ConfigTemplates(
                {
                    "sft": {"training": {}, "task": {}},
                    "rl": {"training": {}, "task": {}, "model": {}},
                    "eval": {"evaluation": {}, "task": {}, "model": {}},
                }
            ),
            cloud_factory=lambda: object(),
            shared_overrides={"training": {"hf_repo_base": "org/wordle"}},
        )

        rl_stage = session.rl(
            "curr_rl_t4",
            adapter=AdapterRef(repo_id="org/base", revision="main"),
            run="wordle_rl_turn4",
            run_name="wordle_curriculum_rl_t4",
        )
        eval_stage = session.eval(
            "curr_eval_after_t5_turn5",
            adapter=AdapterRef(repo_id="org/base", revision="main"),
            run="wordle_eval_turn5",
            run_name="wordle_curr_eval_after_t5_turn5",
        )

        self.assertEqual(rl_stage.run_type, "rl")
        self.assertEqual(rl_stage.environment_run, "wordle_rl_turn4")
        self.assertEqual(rl_stage.config["task"]["min_history_turns"], 3)
        self.assertEqual(rl_stage.config["task"]["max_history_turns"], 3)
        self.assertEqual(rl_stage.config["model"]["init_adapter_repo"], "org/base")
        self.assertEqual(
            rl_stage.config["_tenyson"]["environment_run"],
            "wordle_rl_turn4",
        )
        self.assertEqual(
            rl_stage.config["training"]["output_dir"],
            "./outputs/wordle_curriculum_rl_t4",
        )
        self.assertIn("words_alpha.txt", rl_stage.config["task"]["wordlists"]["url"])

        self.assertEqual(eval_stage.run_type, "eval")
        self.assertEqual(eval_stage.environment_run, "wordle_eval_turn5")
        self.assertEqual(eval_stage.config["task"]["eval_exact_turns"], [5])
        self.assertEqual(eval_stage.config["task"]["min_history_turns"], 4)
        self.assertEqual(eval_stage.config["task"]["max_history_turns"], 4)
        self.assertEqual(
            eval_stage.config["evaluation"]["output_dir"],
            "./outputs/wordle_curr_eval_after_t5_turn5",
        )


class ExperimentReportTests(unittest.TestCase):
    def test_fixed_report_renders_context_and_stage_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "final_report.md"
            report = ExperimentReport(output_path=report_path)
            report.set_context(
                environment_name="wordle",
                experiment_id="wordle_exp",
                telemetry_backend_ref="wandb://ayush/wordle",
                telemetry_project_url="https://wandb.example/ayush/wordle",
            )
            report.register_stage(
                stage_id="mixed_final_eval",
                run_type="eval",
                run_name="wordle_mixed_final_eval",
                variant="mixed",
            )
            report.mark_stage_running("mixed_final_eval")
            report.update_wandb_link(
                "mixed_final_eval",
                "https://wandb.example/ayush/wordle/runs/run123",
            )
            report.update_result(
                "mixed_final_eval",
                JobResult(
                    run_id="wordle_mixed_final_eval",
                    status="success",
                    total_time_seconds=12.345,
                    metrics={
                        "constraint_accuracy": 0.875,
                        "total_samples": 100,
                    },
                    wandb_url="https://wandb.example/ayush/wordle/runs/run123",
                ),
                metric_precision=4,
            )

            content = report_path.read_text(encoding="utf-8")

        self.assertIn("Environment: `wordle`", content)
        self.assertIn("Experiment ID: `wordle_exp`", content)
        self.assertIn("Telemetry dashboard: [open project](https://wandb.example/ayush/wordle)", content)
        self.assertIn("### 1. mixed_final_eval", content)
        self.assertIn("- Run type: `eval`", content)
        self.assertIn("- Environment run: `mixed`", content)
        self.assertIn("- W&B run: [open run](https://wandb.example/ayush/wordle/runs/run123)", content)
        self.assertIn("- Metric `constraint_accuracy`: `0.8750`", content)
        self.assertIn("- Metric `total_samples`: `100`", content)


if __name__ == "__main__":
    unittest.main()
