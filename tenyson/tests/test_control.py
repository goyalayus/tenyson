import argparse
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import tenyson.ctl as ctl_module
import tenyson.core.control as control_module
import tenyson.core.telemetry as telemetry_module
from tenyson.core.telemetry import (
    LiveRunInfo,
    TelemetryClient,
    list_live_run_heartbeats,
    start_run_heartbeat,
)
from tenyson.jobs.result import JobResult


class HeartbeatTelemetryTests(unittest.TestCase):
    def test_heartbeat_marks_run_live_and_summary_marks_it_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "telemetry.sqlite")
            db_url = f"sqlite:///{db_path}"
            with patch.object(
                telemetry_module,
                "_validate_shared_db_url",
                lambda _db_url: None,
            ):
                client = TelemetryClient(db_url=db_url)
                start_run_heartbeat(
                    client=client,
                    experiment_id="wordle_exp",
                    run_id="wordle_sft_main",
                    phase="sft",
                    provider="modal",
                )

                live = list_live_run_heartbeats(
                    client=client,
                    experiment_id="wordle_exp",
                    max_age_seconds=90,
                )
                self.assertEqual(len(live), 1)
                self.assertEqual(live[0].run_id, "wordle_sft_main")
                self.assertEqual(live[0].phase, "sft")
                self.assertEqual(live[0].provider, "modal")
                self.assertTrue(live[0].is_active)

                telemetry_module.record_run_summary(
                    client=client,
                    experiment_id="wordle_exp",
                    phase="sft",
                    result=JobResult(
                        run_id="wordle_sft_main",
                        status="success",
                        total_time_seconds=1.0,
                    ),
                )

                live_after = list_live_run_heartbeats(
                    client=client,
                    experiment_id="wordle_exp",
                    max_age_seconds=90,
                )
                self.assertEqual(live_after, [])


class CtlStopTests(unittest.TestCase):
    def test_stop_uses_env_defaults_and_interactive_live_run_selection(self) -> None:
        args = argparse.Namespace(
            db_url=None,
            experiment_id=None,
            run_id=None,
            max_age_seconds=90,
        )
        candidates = [
            LiveRunInfo(
                run_id="wordle_sft_main",
                phase="sft",
                provider="modal",
                status="running",
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ),
            LiveRunInfo(
                run_id="wordle_rl_mixed",
                phase="rl",
                provider="modal",
                status="running",
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ),
        ]

        with patch.dict(
            os.environ,
            {
                "TENYSON_DB_URL": "postgresql+psycopg://db.example/tenyson",
                "TENYSON_EXPERIMENT_ID": "wordle_exp",
            },
            clear=False,
        ), patch.object(
            ctl_module,
            "list_live_runs",
            return_value=candidates,
        ), patch(
            "builtins.input",
            return_value="2",
        ), patch.object(
            ctl_module,
            "request_stop",
            return_value=True,
        ) as request_stop_mock:
            ctl_module._cmd_stop(args)

        request_stop_mock.assert_called_once_with(
            db_url="postgresql+psycopg://db.example/tenyson",
            run_id="wordle_rl_mixed",
            experiment_id="wordle_exp",
            create_if_missing=False,
        )


class WandBStopTests(unittest.TestCase):
    def test_request_stop_uses_wandb_phase_from_live_runs(self) -> None:
        live_runs = [
            LiveRunInfo(
                run_id="wordle_rl_mixed",
                phase="rl",
                provider="modal",
                status="running",
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        ]

        with patch.object(
            control_module,
            "list_live_run_heartbeats",
            return_value=live_runs,
        ), patch.object(
            control_module.wandb_store,
            "set_stop_requested",
            return_value=True,
        ) as set_stop_requested_mock:
            stopped = control_module.request_stop(
                db_url="wandb://ayush/wordle",
                run_id="wordle_rl_mixed",
                experiment_id="wordle_exp",
                create_if_missing=False,
            )

        self.assertTrue(stopped)
        set_stop_requested_mock.assert_called_once_with(
            "wandb://ayush/wordle",
            experiment_id="wordle_exp",
            phase="rl",
            run_name="wordle_rl_mixed",
            requested=True,
            when_iso=set_stop_requested_mock.call_args.kwargs["when_iso"],
        )


if __name__ == "__main__":
    unittest.main()
