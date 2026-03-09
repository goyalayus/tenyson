import tempfile
import unittest
from pathlib import Path

from tenyson.jobs.result import JobResult
from tenyson.reporting.builder import ReportBuilder


class ReportBuilderTests(unittest.TestCase):
    def test_fill_results_and_metric_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = Path(tmpdir) / "template.md"
            output_path = Path(tmpdir) / "report.md"
            template_path.write_text(
                "status={eval_stage_status}\n"
                "wandb={eval_stage_wandb_link}\n"
                "metric={eval_stage_constraint_accuracy}\n"
                "delta={delta_final_constraint_accuracy}\n",
                encoding="utf-8",
            )

            report = ReportBuilder(
                template_path=str(template_path),
                output_path=str(output_path),
            )
            result = JobResult(
                run_id="eval_stage",
                status="success",
                total_time_seconds=1.0,
                metrics={"constraint_accuracy": 0.123456},
                wandb_url="https://wandb.example/run",
            )
            baseline = JobResult(
                run_id="baseline",
                status="success",
                total_time_seconds=1.0,
                metrics={"constraint_accuracy": 0.1},
            )

            report.fill_results({"eval_stage": result}, metric_precision=4)
            report.fill_metric_delta(
                "delta_final_constraint_accuracy",
                result,
                baseline,
                "constraint_accuracy",
                precision=4,
            )
            report.generate()

            content = output_path.read_text(encoding="utf-8")

        self.assertIn("status=success", content)
        self.assertIn("wandb=[run](https://wandb.example/run)", content)
        self.assertIn("metric=0.1235", content)
        self.assertIn("delta=0.0235", content)


if __name__ == "__main__":
    unittest.main()
