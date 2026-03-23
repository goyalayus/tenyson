import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import tenyson.core.notify as notify_module
import tenyson.core.telemetry as telemetry_module
from tenyson.jobs.result import JobResult


class NotifyFailureTests(unittest.TestCase):
    def test_notify_failure_writes_json_log(self) -> None:
        result = JobResult(
            run_id="wordle_sft_main",
            status="failed",
            total_time_seconds=0.0,
            failure_reason="boom",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            notify_module.notify_failure(
                step_label="left/stage",
                result=result,
                failure_log_dir=tmpdir,
                experiment_id="wordle_exp",
            )
            log_files = list(Path(tmpdir).glob("*.json"))

            self.assertEqual(len(log_files), 1)
            payload = json.loads(log_files[0].read_text(encoding="utf-8"))

        self.assertIn("left_stage_wordle_sft_main_", log_files[0].name)
        self.assertEqual(payload["run_id"], "wordle_sft_main")
        self.assertEqual(payload["failure_reason"], "boom")
        self.assertEqual(payload["step_name"], "left/stage")

    def test_notify_failure_posts_webhook_payload(self) -> None:
        result = JobResult(
            run_id="wordle_sft_main",
            status="failed",
            total_time_seconds=0.0,
            failure_reason="boom",
        )

        with patch("urllib.request.urlopen") as urlopen_mock:
            notify_module.notify_failure(
                step_label="sft_main",
                result=result,
                failure_webhook_url="https://example.com/webhook",
                experiment_id="wordle_exp",
            )

        request = urlopen_mock.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["run_id"], "wordle_sft_main")
        self.assertEqual(payload["experiment_id"], "wordle_exp")
        self.assertEqual(payload["failure_reason"], "boom")

    def test_notify_failure_records_wandb_summary(self) -> None:
        result = JobResult(
            run_id="wordle_sft_main",
            status="failed",
            total_time_seconds=1.25,
            failure_reason="boom",
        )
        fake_client = object()
        with patch.object(
            telemetry_module,
            "TelemetryClient",
            return_value=fake_client,
        ) as client_mock, patch.object(
            telemetry_module,
            "record_run_summary",
        ) as summary_mock:
            notify_module.notify_failure(
                step_label="sft_main",
                result=result,
                db_url="wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="sft",
            )

        client_mock.assert_called_once_with(db_url="wandb://ayush/wordle")
        summary_mock.assert_called_once_with(
            client=fake_client,
            experiment_id="wordle_exp",
            phase="sft",
            result=result,
        )


if __name__ == "__main__":
    unittest.main()
