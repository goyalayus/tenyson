import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import tenyson.ctl as ctl_module
import tenyson.core.control as control_module
import tenyson.core.telemetry as telemetry_module
from tenyson.core.telemetry import (
    begin_run_attempt,
    LiveRunInfo,
    ManualStopTelemetryCallback,
    TelemetryClient,
    list_live_run_heartbeats,
)


class HeartbeatTelemetryTests(unittest.TestCase):
    def test_rejects_non_wandb_backend_ref(self) -> None:
        with self.assertRaisesRegex(ValueError, "W&B refs"):
            TelemetryClient(db_url="sqlite:///tmp/telemetry.sqlite")

    def test_heartbeat_reads_live_runs_from_wandb_store(self) -> None:
        now = datetime.now(timezone.utc)
        client = TelemetryClient(db_url="wandb://ayush/wordle")
        with patch.object(
            telemetry_module.wandb_store,
            "list_live_runs",
            return_value=[
                {
                    "run_id": "wordle_sft_main",
                    "phase": "sft",
                    "provider": "modal",
                    "status": "running",
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                    "attempt_token": "attempt-1",
                }
            ],
        ):
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
        self.assertEqual(live[0].attempt_token, "attempt-1")


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
                "TENYSON_DB_URL": "wandb://ayush/wordle",
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
            db_url="wandb://ayush/wordle",
            run_id="wordle_rl_mixed",
            experiment_id="wordle_exp",
            create_if_missing=False,
        )

    def test_stop_exits_when_no_active_control_row_exists(self) -> None:
        args = argparse.Namespace(
            db_url="wandb://ayush/wordle",
            experiment_id="wordle_exp",
            run_id="wordle_rl_mixed",
            max_age_seconds=90,
        )
        stderr_buffer = io.StringIO()

        with patch.object(
            ctl_module,
            "request_stop",
            return_value=False,
        ), contextlib.redirect_stderr(stderr_buffer):
            with self.assertRaises(SystemExit) as exc_info:
                ctl_module._cmd_stop(args)

        self.assertEqual(exc_info.exception.code, 1)
        self.assertIn(
            "No active run control row found for run_id=wordle_rl_mixed",
            stderr_buffer.getvalue(),
        )


class CtlControllerLifecycleTests(unittest.TestCase):
    def test_launch_starts_detached_controller_and_writes_state_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                name="Wordle Controller",
                controller_dir=tmpdir,
                cwd="/repo",
                env=["FOO=bar"],
                command=["--", "python3", "examples/wordle/experiment.py"],
            )

            with patch.object(
                ctl_module.subprocess,
                "Popen",
                return_value=SimpleNamespace(pid=43210),
            ) as popen_mock:
                ctl_module._cmd_launch(args)

            pid_path = Path(tmpdir) / "wordle-controller.pid"
            log_path = Path(tmpdir) / "wordle-controller.log"
            metadata_path = Path(tmpdir) / "wordle-controller.json"

            self.assertEqual(pid_path.read_text(encoding="utf-8").strip(), "43210")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["pid"], 43210)
            self.assertEqual(
                metadata["command"],
                ["python3", "examples/wordle/experiment.py"],
            )
            self.assertEqual(metadata["cwd"], "/repo")
            self.assertEqual(metadata["log_path"], str(log_path))
            self.assertTrue(log_path.exists())

            popen_mock.assert_called_once()
            self.assertEqual(
                popen_mock.call_args.args[0],
                ["python3", "examples/wordle/experiment.py"],
            )
            self.assertEqual(popen_mock.call_args.kwargs["cwd"], "/repo")
            self.assertTrue(popen_mock.call_args.kwargs["start_new_session"])
            self.assertEqual(
                popen_mock.call_args.kwargs["stdin"],
                ctl_module.subprocess.DEVNULL,
            )
            self.assertEqual(popen_mock.call_args.kwargs["stderr"], ctl_module.subprocess.STDOUT)
            self.assertEqual(popen_mock.call_args.kwargs["env"]["FOO"], "bar")

    def test_launch_rejects_existing_live_controller(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pid_path = Path(tmpdir) / "wordle.pid"
            pid_path.write_text("123\n", encoding="utf-8")
            args = argparse.Namespace(
                name="wordle",
                controller_dir=tmpdir,
                cwd=".",
                env=[],
                command=["python3", "examples/wordle/experiment.py"],
            )

            with patch.object(ctl_module, "_is_process_alive", return_value=True):
                with self.assertRaises(SystemExit):
                    ctl_module._cmd_launch(args)

    def test_status_reports_running_controller(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "wordle"
            (Path(tmpdir) / "wordle.pid").write_text("456\n", encoding="utf-8")
            (Path(tmpdir) / "wordle.log").write_text("", encoding="utf-8")
            (Path(tmpdir) / "wordle.json").write_text(
                json.dumps(
                    {
                        "name": "wordle",
                        "pid": 456,
                        "command": ["python3", "examples/wordle/experiment.py"],
                        "cwd": "/repo",
                        "log_path": str(base.with_suffix(".log")),
                        "launched_at": "2026-03-24T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(name="wordle", controller_dir=tmpdir)
            stdout_buffer = io.StringIO()

            with patch.object(ctl_module, "_is_process_alive", return_value=True), contextlib.redirect_stdout(
                stdout_buffer
            ):
                ctl_module._cmd_status(args)

            output = stdout_buffer.getvalue()
            self.assertIn("status: running", output)
            self.assertIn("pid: 456", output)
            self.assertIn("python3 examples/wordle/experiment.py", output)

    def test_stop_controller_sends_sigterm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "wordle.pid").write_text("789\n", encoding="utf-8")
            args = argparse.Namespace(name="wordle", controller_dir=tmpdir)

            with patch.object(ctl_module, "_is_process_alive", return_value=True), patch.object(
                ctl_module.os,
                "kill",
            ) as kill_mock:
                ctl_module._cmd_stop_controller(args)

            kill_mock.assert_called_once_with(789, ctl_module.signal.SIGTERM)


class WandBStopTests(unittest.TestCase):
    def test_request_stop_with_explicit_phase_and_attempt_token_short_circuits_live_lookup(self) -> None:
        with patch.object(
            control_module,
            "list_live_run_heartbeats",
            side_effect=AssertionError("live heartbeat lookup should not run"),
        ), patch.object(
            control_module.wandb_store,
            "set_stop_requested",
            return_value=True,
        ) as set_stop_requested_mock:
            stopped = control_module.request_stop(
                db_url="wandb://ayush/wordle",
                run_id="wordle_rl_mixed",
                experiment_id="wordle_exp",
                phase="rl",
                create_if_missing=True,
                attempt_token="attempt-live",
            )

        self.assertTrue(stopped)
        set_stop_requested_mock.assert_called_once_with(
            "wandb://ayush/wordle",
            experiment_id="wordle_exp",
            phase="rl",
            run_name="wordle_rl_mixed",
            requested=True,
            when_iso=set_stop_requested_mock.call_args.kwargs["when_iso"],
            create_if_missing=True,
            attempt_token="attempt-live",
        )

    def test_request_stop_with_explicit_phase_short_circuits_live_lookup(self) -> None:
        with patch.object(
            control_module,
            "list_live_run_heartbeats",
            side_effect=AssertionError("live heartbeat lookup should not run"),
        ), patch.object(
            control_module.wandb_store,
            "set_stop_requested",
            return_value=True,
        ) as set_stop_requested_mock:
            stopped = control_module.request_stop(
                db_url="wandb://ayush/wordle",
                run_id="wordle_rl_mixed",
                experiment_id="wordle_exp",
                phase="rl",
                create_if_missing=True,
            )

        self.assertTrue(stopped)
        set_stop_requested_mock.assert_called_once_with(
            "wandb://ayush/wordle",
            experiment_id="wordle_exp",
            phase="rl",
            run_name="wordle_rl_mixed",
            requested=True,
            when_iso=set_stop_requested_mock.call_args.kwargs["when_iso"],
            create_if_missing=True,
            attempt_token=None,
        )

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
            create_if_missing=True,
            attempt_token=None,
        )

    def test_request_stop_targets_exact_live_attempt_token_when_available(self) -> None:
        live_runs = [
            LiveRunInfo(
                run_id="wordle_rl_mixed",
                phase="rl",
                provider="modal",
                status="running",
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                attempt_token="attempt-live",
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
            create_if_missing=True,
            attempt_token="attempt-live",
        )

    def test_request_stop_dedupes_duplicate_live_attempt_rows(self) -> None:
        now = datetime.now(timezone.utc)
        live_runs = [
            LiveRunInfo(
                run_id="wordle_rl_mixed",
                phase="rl",
                provider="modal",
                status="running",
                is_active=True,
                created_at=now,
                updated_at=now,
                attempt_token="attempt-newest",
            ),
            LiveRunInfo(
                run_id="wordle_rl_mixed",
                phase="rl",
                provider="modal",
                status="running",
                is_active=True,
                created_at=now,
                updated_at=now,
                attempt_token="attempt-newest",
            ),
            LiveRunInfo(
                run_id="wordle_rl_mixed",
                phase="rl",
                provider="modal",
                status="running",
                is_active=True,
                created_at=now,
                updated_at=now,
                attempt_token="attempt-older",
            ),
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
        self.assertEqual(set_stop_requested_mock.call_count, 2)
        self.assertEqual(
            [call.kwargs["attempt_token"] for call in set_stop_requested_mock.call_args_list],
            ["attempt-newest", "attempt-older"],
        )

    def test_request_stop_without_phase_does_not_create_guessed_runs(self) -> None:
        with patch.object(
            control_module,
            "list_live_run_heartbeats",
            return_value=[],
        ), patch.object(
            control_module.wandb_store,
            "set_stop_requested",
            return_value=False,
        ) as set_stop_requested_mock:
            stopped = control_module.request_stop(
                db_url="wandb://ayush/wordle",
                run_id="wordle_unknown",
                experiment_id="wordle_exp",
                create_if_missing=True,
            )

        self.assertFalse(stopped)
        self.assertEqual(set_stop_requested_mock.call_count, 3)
        self.assertEqual(
            [call.kwargs["phase"] for call in set_stop_requested_mock.call_args_list],
            ["sft", "rl", "eval"],
        )
        self.assertTrue(
            all(
                call.kwargs["create_if_missing"] is False
                for call in set_stop_requested_mock.call_args_list
            )
        )


class WandBAttemptTests(unittest.TestCase):
    def test_begin_run_attempt_wandb_sets_heartbeat_without_crashing(self) -> None:
        client = SimpleNamespace(backend="wandb", db_url="wandb://ayush/wordle")

        with patch.object(
            telemetry_module.wandb_store,
            "ensure_run",
            return_value=SimpleNamespace(url="https://wandb.example/run"),
        ) as ensure_run_mock, patch.object(
            telemetry_module,
            "get_run_result",
            side_effect=LookupError("no prior result"),
        ), patch.object(
            telemetry_module.wandb_store,
            "set_stop_requested",
        ) as set_stop_requested_mock, patch.object(
            telemetry_module.wandb_store,
            "update_run_summary",
        ) as update_summary_mock:
            cleared = begin_run_attempt(
                client=client,
                experiment_id="wordle_exp",
                run_id="wordle_sft_main",
                phase="sft",
                attempt_token="abc123",
            )

        self.assertFalse(cleared)
        set_stop_requested_mock.assert_not_called()
        summary_values = update_summary_mock.call_args.args[1]
        self.assertEqual(
            summary_values[telemetry_module.wandb_store.SUMMARY_STATUS], "running"
        )
        self.assertEqual(
            summary_values[telemetry_module.wandb_store.SUMMARY_ATTEMPT_TOKEN],
            "abc123",
        )
        self.assertIsInstance(
            summary_values[telemetry_module.wandb_store.SUMMARY_HEARTBEAT_AT],
            str,
        )
        self.assertEqual(
            ensure_run_mock.call_args.kwargs["attempt_token"],
            "abc123",
        )

    def test_begin_run_attempt_preserves_fresh_stop_request_without_prior_result(self) -> None:
        client = SimpleNamespace(backend="wandb", db_url="wandb://ayush/wordle")

        with patch.object(
            telemetry_module.wandb_store,
            "ensure_run",
            return_value=SimpleNamespace(url="https://wandb.example/run"),
        ), patch.object(
            telemetry_module,
            "get_run_result",
            side_effect=LookupError("no prior result"),
        ), patch.object(
            telemetry_module.wandb_store,
            "fetch_stop_requested",
        ) as fetch_stop_requested_mock, patch.object(
            telemetry_module.wandb_store,
            "set_stop_requested",
        ) as set_stop_requested_mock, patch.object(
            telemetry_module.wandb_store,
            "update_run_summary",
        ):
            cleared = begin_run_attempt(
                client=client,
                experiment_id="wordle_exp",
                run_id="wordle_rl_mixed",
                phase="rl",
                attempt_token="fresh-attempt",
            )

        self.assertFalse(cleared)
        fetch_stop_requested_mock.assert_not_called()
        set_stop_requested_mock.assert_not_called()

    def test_begin_run_attempt_clears_stop_for_prior_terminal_attempt(self) -> None:
        client = SimpleNamespace(backend="wandb", db_url="wandb://ayush/wordle")

        with patch.object(
            telemetry_module.wandb_store,
            "ensure_run",
            return_value=SimpleNamespace(url="https://wandb.example/run"),
        ), patch.object(
            telemetry_module,
            "get_run_result",
            return_value=({}, {"status": "partial", "run_id": "wordle_rl_mixed"}),
        ), patch.object(
            telemetry_module.wandb_store,
            "fetch_stop_requested",
            return_value=True,
        ) as fetch_stop_requested_mock, patch.object(
            telemetry_module.wandb_store,
            "set_stop_requested",
        ) as set_stop_requested_mock, patch.object(
            telemetry_module.wandb_store,
            "update_run_summary",
        ):
            cleared = begin_run_attempt(
                client=client,
                experiment_id="wordle_exp",
                run_id="wordle_rl_mixed",
                phase="rl",
                attempt_token="resume-attempt",
            )

        self.assertTrue(cleared)
        fetch_stop_requested_mock.assert_called_once()
        set_stop_requested_mock.assert_called_once_with(
            "wandb://ayush/wordle",
            experiment_id="wordle_exp",
            phase="rl",
            run_name="wordle_rl_mixed",
            requested=False,
            when_iso=None,
            create_if_missing=False,
            attempt_token="resume-attempt",
        )

    def test_begin_run_attempt_keeps_stop_state_for_non_terminal_prior_attempt(self) -> None:
        client = SimpleNamespace(backend="wandb", db_url="wandb://ayush/wordle")

        with patch.object(
            telemetry_module.wandb_store,
            "ensure_run",
            return_value=SimpleNamespace(url="https://wandb.example/run"),
        ), patch.object(
            telemetry_module,
            "get_run_result",
            return_value=({}, {"status": "running", "run_id": "wordle_rl_mixed"}),
        ), patch.object(
            telemetry_module.wandb_store,
            "fetch_stop_requested",
        ) as fetch_stop_requested_mock, patch.object(
            telemetry_module.wandb_store,
            "set_stop_requested",
        ) as set_stop_requested_mock, patch.object(
            telemetry_module.wandb_store,
            "update_run_summary",
        ):
            cleared = begin_run_attempt(
                client=client,
                experiment_id="wordle_exp",
                run_id="wordle_rl_mixed",
                phase="rl",
                attempt_token="active-attempt",
            )

        self.assertFalse(cleared)
        fetch_stop_requested_mock.assert_not_called()
        set_stop_requested_mock.assert_not_called()


class ManualStopCallbackTests(unittest.TestCase):
    def test_manual_stop_callback_sets_stop_and_save_flags(self) -> None:
        callback = telemetry_module.ManualStopTelemetryCallback(
            run_id="wordle_rl_mixed",
            experiment_id="wordle_exp",
            phase="rl",
            client=SimpleNamespace(db_url="wandb://ayush/wordle"),
            attempt_token="attempt-live",
            check_every_n_steps=2,
        )
        state = SimpleNamespace(global_step=4)
        control = SimpleNamespace(should_training_stop=False, should_save=False)

        with patch.object(
            telemetry_module,
            "run_stop_requested",
            return_value=True,
        ) as run_stop_requested_mock:
            returned_control = callback.on_step_end(
                args=None,
                state=state,
                control=control,
            )

        self.assertIs(returned_control, control)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(control.should_save)
        self.assertTrue(callback.stop_requested)
        self.assertEqual(callback.stop_step, 4)
        run_stop_requested_mock.assert_called_once_with(
            callback.client,
            experiment_id="wordle_exp",
            run_id="wordle_rl_mixed",
            phase="rl",
            attempt_token="attempt-live",
        )


    def test_callback_ignores_poll_errors_and_keeps_training_running(self) -> None:
        callback = ManualStopTelemetryCallback(
            run_id="wordle_sft_main",
            experiment_id="wordle_exp",
            phase="sft",
            client=SimpleNamespace(db_url="wandb://ayush/wordle"),
        )
        state = SimpleNamespace(global_step=3)
        control = SimpleNamespace(should_training_stop=False, should_save=False)

        with patch.object(
            telemetry_module,
            "run_stop_requested",
            side_effect=RuntimeError("wandb unavailable"),
        ):
            returned = callback.on_step_end(None, state, control)

        self.assertIs(returned, control)
        self.assertFalse(callback.stop_requested)
        self.assertIsNone(callback.stop_step)
        self.assertFalse(control.should_training_stop)
        self.assertFalse(control.should_save)


if __name__ == "__main__":
    unittest.main()
