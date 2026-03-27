import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import tenyson.reporting.fixed as fixed_module
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
    def test_fixed_report_rebuild_queries_only_matching_experiment_group(self) -> None:
        calls: dict[str, object] = {}

        class _FakeApi:
            def runs(self, path, filters=None, order=None, per_page=None, lazy=None):
                calls["path"] = path
                calls["filters"] = filters
                calls["order"] = order
                calls["per_page"] = per_page
                calls["lazy"] = lazy
                return []

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            fixed_module.wandb_store,
            "_wandb_api",
            return_value=_FakeApi(),
        ):
            report = ExperimentReport(output_path=Path(tmpdir) / "final_report.md")
            rebuilt = report.rebuild_from_telemetry(
                backend_ref="wandb://ayush/wordle",
                experiment_id="wordle_exp",
            )

        self.assertTrue(rebuilt)
        self.assertEqual(calls["path"], "ayush/wordle")
        self.assertEqual(calls["filters"], {"group": "wordle_exp"})
        self.assertEqual(calls["order"], "-created_at")
        self.assertEqual(calls["per_page"], 200)
        self.assertEqual(calls["lazy"], True)

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

    def test_fixed_report_rebuilds_from_telemetry_and_ignores_stale_active_rows(self) -> None:
        class _FakeRun:
            def __init__(
                self,
                *,
                name: str,
                job_type: str,
                group: str,
                summary: dict,
                config: dict,
                url: str,
                created_at: str,
                updated_at: str,
            ) -> None:
                self.name = name
                self.job_type = job_type
                self.group = group
                self.summary = summary
                self.config = config
                self.url = url
                self.created_at = created_at
                self.updated_at = updated_at

        def _summary(
            *,
            experiment_id: str,
            phase: str,
            run_name: str,
            status: str,
            attempt_token: str,
            heartbeat_at: str,
            wandb_url: str,
            job_result_payload: dict | None = None,
            is_active: bool = False,
        ) -> dict:
            return {
                "tenyson/experiment_id": experiment_id,
                "tenyson/phase": phase,
                "tenyson/run_name": run_name,
                "tenyson/status": status,
                "tenyson/attempt_token": attempt_token,
                "tenyson/heartbeat_at": heartbeat_at,
                "tenyson/is_active": is_active,
                "tenyson/wandb_url": wandb_url,
                "tenyson/job_result_json": (
                    json.dumps(job_result_payload) if job_result_payload else None
                ),
            }

        def _job_result(
            *,
            run_id: str,
            status: str,
            total_time_seconds: float,
            attempt_token: str,
            metrics: dict,
            wandb_url: str,
            hf_repo_id: str | None = None,
            hf_revision: str | None = None,
            failure_reason: str | None = None,
            stopped_early: bool = False,
            processed_samples: int | None = None,
            expected_samples: int | None = None,
        ) -> dict:
            return {
                "run_id": run_id,
                "status": status,
                "total_time_seconds": total_time_seconds,
                "attempt_token": attempt_token,
                "metrics": metrics,
                "wandb_url": wandb_url,
                "hf_repo_id": hf_repo_id,
                "hf_revision": hf_revision,
                "failure_reason": failure_reason,
                "stopped_early": stopped_early,
                "processed_samples": processed_samples,
                "expected_samples": expected_samples,
            }

        experiment_id = "wordle_exp"
        backend_ref = "wandb://ayush/wordle"
        sft_result = _job_result(
            run_id="sft_main",
            status="partial",
            total_time_seconds=1425.27,
            attempt_token="sft-a",
            metrics={"global_step": 256, "train_loss": 0.7413},
            wandb_url="https://wandb.example/runs/sft_main",
            hf_repo_id="org/sft",
            hf_revision="sha-sft",
            failure_reason="Manual stop requested at step 256.",
            stopped_early=True,
        )
        eval_result = _job_result(
            run_id="eval_baseline_mixed",
            status="success",
            total_time_seconds=423.57,
            attempt_token="eval-a",
            metrics={
                "format_accuracy": 0.62,
                "dict_accuracy": 0.52,
                "constraint_accuracy": 0.05,
                "total_samples": 100,
            },
            wandb_url="https://wandb.example/runs/eval_baseline_mixed",
            processed_samples=100,
            expected_samples=100,
        )
        mixed_rl_result = _job_result(
            run_id="mixed_rl",
            status="stopped",
            total_time_seconds=6398.71,
            attempt_token="mixed-rl-a",
            metrics={
                "global_step": 525,
                "train_loss": 0.0008849062340699935,
            },
            wandb_url="https://wandb.example/runs/mixed_rl",
            hf_repo_id="org/mixed-rl",
            hf_revision="sha-mixed",
            failure_reason="Manual stop requested at step 525.",
            stopped_early=True,
        )
        curr_rl_stopped_result = _job_result(
            run_id="curr_rl_t2",
            status="stopped",
            total_time_seconds=5120.5,
            attempt_token="curr-rl-stop",
            metrics={
                "global_step": 410,
                "train_loss": 0.0012,
            },
            wandb_url="https://wandb.example/runs/curr_rl_t2_stopped",
            hf_repo_id="org/curr-rl",
            hf_revision="sha-curr",
            failure_reason="Manual stop requested at step 410.",
            stopped_early=True,
        )
        mixed_final_eval_result = _job_result(
            run_id="mixed_final_eval",
            status="success",
            total_time_seconds=288.58,
            attempt_token="mixed-eval-a",
            metrics={
                "format_accuracy": 0.97,
                "dict_accuracy": 0.95,
                "constraint_accuracy": 0.03,
                "total_samples": 100,
            },
            wandb_url="https://wandb.example/runs/mixed_final_eval",
            processed_samples=100,
            expected_samples=100,
        )
        curr_eval_result = _job_result(
            run_id="curr_eval_after_t2_turn2",
            status="success",
            total_time_seconds=366.14,
            attempt_token="curr-eval-a",
            metrics={
                "format_accuracy": 0.9,
                "dict_accuracy": 0.88,
                "constraint_accuracy": 0.1,
                "total_samples": 100,
            },
            wandb_url="https://wandb.example/runs/curr_eval_after_t2_turn2",
            processed_samples=100,
            expected_samples=100,
        )

        fake_runs = [
            _FakeRun(
                name="sft_main",
                job_type="sft",
                group=experiment_id,
                summary=_summary(
                    experiment_id=experiment_id,
                    phase="sft",
                    run_name="sft_main",
                    status="partial",
                    attempt_token="sft-a",
                    heartbeat_at="2026-03-24T11:03:46+00:00",
                    wandb_url="https://wandb.example/runs/sft_main",
                    job_result_payload=sft_result,
                ),
                config={"_tenyson": {"environment_run": "wordle_sft_main"}},
                url="https://wandb.example/runs/sft_main",
                created_at="2026-03-24T11:03:46+00:00",
                updated_at="2026-03-24T11:30:00+00:00",
            ),
            _FakeRun(
                name="eval_baseline_mixed",
                job_type="eval",
                group=experiment_id,
                summary=_summary(
                    experiment_id=experiment_id,
                    phase="eval",
                    run_name="eval_baseline_mixed",
                    status="success",
                    attempt_token="eval-a",
                    heartbeat_at="2026-03-24T11:53:21+00:00",
                    wandb_url="https://wandb.example/runs/eval_baseline_mixed",
                    job_result_payload=eval_result,
                ),
                config={"_tenyson": {"environment_run": "wordle_eval_mixed"}},
                url="https://wandb.example/runs/eval_baseline_mixed",
                created_at="2026-03-24T11:33:56+00:00",
                updated_at="2026-03-24T11:53:21+00:00",
            ),
            _FakeRun(
                name="mixed_rl",
                job_type="rl",
                group=experiment_id,
                summary=_summary(
                    experiment_id=experiment_id,
                    phase="rl",
                    run_name="mixed_rl",
                    status="stopped",
                    attempt_token="mixed-rl-a",
                    heartbeat_at="2026-03-26T13:23:34+00:00",
                    wandb_url="https://wandb.example/runs/mixed_rl",
                    job_result_payload=mixed_rl_result,
                ),
                config={"_tenyson": {"environment_run": "wordle_rl_mixed"}},
                url="https://wandb.example/runs/mixed_rl",
                created_at="2026-03-26T11:36:44+00:00",
                updated_at="2026-03-26T13:23:34+00:00",
            ),
            _FakeRun(
                name="curr_rl_t2",
                job_type="rl",
                group=experiment_id,
                summary=_summary(
                    experiment_id=experiment_id,
                    phase="rl",
                    run_name="curr_rl_t2",
                    status="running",
                    attempt_token="curr-rl-live",
                    heartbeat_at="2026-03-24T00:00:00+00:00",
                    wandb_url="https://wandb.example/runs/curr_rl_t2_live",
                    is_active=True,
                ),
                config={"_tenyson": {"environment_run": "wordle_rl_turn2"}},
                url="https://wandb.example/runs/curr_rl_t2_live",
                created_at="2026-03-26T11:36:44+00:00",
                updated_at="2026-03-26T13:29:56+00:00",
            ),
            _FakeRun(
                name="curr_rl_t2",
                job_type="rl",
                group=experiment_id,
                summary=_summary(
                    experiment_id=experiment_id,
                    phase="rl",
                    run_name="curr_rl_t2",
                    status="stopped",
                    attempt_token="curr-rl-stop",
                    heartbeat_at="2026-03-25T15:48:08+00:00",
                    wandb_url="https://wandb.example/runs/curr_rl_t2_stopped",
                    job_result_payload=curr_rl_stopped_result,
                ),
                config={"_tenyson": {"environment_run": "wordle_rl_turn2"}},
                url="https://wandb.example/runs/curr_rl_t2_stopped",
                created_at="2026-03-25T15:24:19+00:00",
                updated_at="2026-03-25T15:48:08+00:00",
            ),
            _FakeRun(
                name="mixed_final_eval",
                job_type="eval",
                group=experiment_id,
                summary=_summary(
                    experiment_id=experiment_id,
                    phase="eval",
                    run_name="mixed_final_eval",
                    status="success",
                    attempt_token="mixed-eval-a",
                    heartbeat_at="2026-03-26T13:43:29+00:00",
                    wandb_url="https://wandb.example/runs/mixed_final_eval",
                    job_result_payload=mixed_final_eval_result,
                ),
                config={"_tenyson": {"environment_run": "wordle_eval_mixed"}},
                url="https://wandb.example/runs/mixed_final_eval",
                created_at="2026-03-26T13:38:43+00:00",
                updated_at="2026-03-26T13:43:29+00:00",
            ),
            _FakeRun(
                name="curr_eval_after_t2_turn2",
                job_type="eval",
                group=experiment_id,
                summary=_summary(
                    experiment_id=experiment_id,
                    phase="eval",
                    run_name="curr_eval_after_t2_turn2",
                    status="success",
                    attempt_token="curr-eval-a",
                    heartbeat_at="2026-03-26T13:44:47+00:00",
                    wandb_url="https://wandb.example/runs/curr_eval_after_t2_turn2",
                    job_result_payload=curr_eval_result,
                ),
                config={"_tenyson": {"environment_run": "wordle_eval_turn2"}},
                url="https://wandb.example/runs/curr_eval_after_t2_turn2",
                created_at="2026-03-26T13:38:44+00:00",
                updated_at="2026-03-26T13:44:47+00:00",
            ),
            _FakeRun(
                name="mixed_rl",
                job_type="rl",
                group=experiment_id,
                summary=_summary(
                    experiment_id=experiment_id,
                    phase="rl",
                    run_name="curr_rl_t2",
                    status="failed",
                    attempt_token="corrupted",
                    heartbeat_at="2026-03-24T11:55:14+00:00",
                    wandb_url="https://wandb.example/runs/corrupted",
                    job_result_payload=mixed_rl_result,
                ),
                config={"_tenyson": {"environment_run": "wordle_rl_mixed"}},
                url="https://wandb.example/runs/corrupted",
                created_at="2026-03-24T11:55:14+00:00",
                updated_at="2026-03-24T12:10:00+00:00",
            ),
        ]

        result_lookup = {
            ("sft", "sft_main", "sft-a"): ({}, sft_result),
            ("eval", "eval_baseline_mixed", "eval-a"): ({}, eval_result),
            (
                "rl",
                "mixed_rl",
                "mixed-rl-a",
            ): ({"metrics": {"total_samples": 2100, "rollout_batches": 525}}, mixed_rl_result),
            (
                "rl",
                "curr_rl_t2",
                "curr-rl-stop",
            ): ({"metrics": {"total_samples": 1640, "rollout_batches": 410}}, curr_rl_stopped_result),
            ("eval", "mixed_final_eval", "mixed-eval-a"): ({}, mixed_final_eval_result),
            ("eval", "curr_eval_after_t2_turn2", "curr-eval-a"): ({}, curr_eval_result),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "final_report.md"
            report = ExperimentReport(output_path=report_path)

            with patch(
                "tenyson.reporting.fixed._project_runs_for_backend",
                return_value=fake_runs,
            ), patch(
                "tenyson.reporting.fixed.wandb_store.fetch_run_result",
                side_effect=lambda backend, *, experiment_id, phase, run_name, attempt_token=None: result_lookup.get(
                    (phase, run_name, attempt_token)
                ),
            ):
                rebuilt = report.rebuild_from_telemetry(
                    backend_ref=backend_ref,
                    experiment_id=experiment_id,
                    environment_name="wordle",
                )

            content = report_path.read_text(encoding="utf-8")

        self.assertTrue(rebuilt)
        self.assertEqual(report._stage_order[:2], ["sft_main", "eval_baseline_mixed"])
        self.assertEqual(
            report._stage_order[-2:],
            ["mixed_final_eval", "curr_eval_after_t2_turn2"],
        )
        self.assertCountEqual(
            report._stage_order,
            [
                "sft_main",
                "eval_baseline_mixed",
                "mixed_rl",
                "curr_rl_t2",
                "mixed_final_eval",
                "curr_eval_after_t2_turn2",
            ],
        )
        self.assertEqual(report._stages["curr_rl_t2"].status, "stopped")
        self.assertEqual(report._stages["mixed_rl"].metrics["rollout_batches"], "525")
        self.assertEqual(report._stages["mixed_rl"].metrics["total_samples"], "2100")
        self.assertEqual(
            report._stages["curr_eval_after_t2_turn2"].variant,
            "wordle_eval_turn2",
        )
        self.assertIn("Stage summary: 1 partial, 2 stopped, 3 success", content)
        self.assertIn("### 5. mixed_final_eval", content)
        self.assertIn("### 6. curr_eval_after_t2_turn2", content)
        self.assertIn("- Metric `rollout_batches`: `525`", content)
        self.assertIn("- Metric `total_samples`: `2100`", content)

    def test_fixed_report_rebuild_marks_failed_when_wandb_state_is_terminal(self) -> None:
        class _FakeRun:
            def __init__(
                self,
                *,
                name: str,
                job_type: str,
                group: str,
                summary: dict,
                config: dict,
                url: str,
                created_at: str,
                updated_at: str,
                state: str,
            ) -> None:
                self.name = name
                self.job_type = job_type
                self.group = group
                self.summary = summary
                self.config = config
                self.url = url
                self.created_at = created_at
                self.updated_at = updated_at
                self.state = state

        now = datetime.now(timezone.utc)
        experiment_id = "wordle_exp"
        backend_ref = "wandb://ayush/wordle"
        stale_running_but_failed = _FakeRun(
            name="eval_baseline_mixed",
            job_type="eval",
            group=experiment_id,
            summary={
                "tenyson/experiment_id": experiment_id,
                "tenyson/phase": "eval",
                "tenyson/run_name": "eval_baseline_mixed",
                "tenyson/status": "running",
                "tenyson/attempt_token": "eval-a",
                "tenyson/heartbeat_at": now.isoformat(),
                "tenyson/is_active": True,
                "tenyson/wandb_url": "https://wandb.example/runs/eval_baseline_mixed",
                "tenyson/job_result_json": None,
                "tenyson/failure_reason": None,
            },
            config={"_tenyson": {"environment_run": "wordle_eval_mixed"}},
            url="https://wandb.example/runs/eval_baseline_mixed",
            created_at=(now - timedelta(minutes=1)).isoformat(),
            updated_at=now.isoformat(),
            state="failed",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "final_report.md"
            report = ExperimentReport(output_path=report_path)

            with patch(
                "tenyson.reporting.fixed._project_runs_for_backend",
                return_value=[stale_running_but_failed],
            ), patch(
                "tenyson.reporting.fixed.wandb_store.fetch_run_result",
                return_value=None,
            ):
                rebuilt = report.rebuild_from_telemetry(
                    backend_ref=backend_ref,
                    experiment_id=experiment_id,
                    environment_name="wordle",
                    prefer_terminal_results=True,
                )

            content = report_path.read_text(encoding="utf-8")

        self.assertTrue(rebuilt)
        self.assertEqual(report._stages["eval_baseline_mixed"].status, "failed")
        self.assertIn(
            "state=failed before a canonical tenyson job result was written.",
            report._stages["eval_baseline_mixed"].failure_reason or "",
        )
        self.assertIn("Stage summary: 1 failed", content)


    def test_fixed_report_rebuild_prefers_terminal_results_for_final_snapshot(self) -> None:
        class _FakeRun:
            def __init__(
                self,
                *,
                name: str,
                job_type: str,
                group: str,
                summary: dict,
                config: dict,
                url: str,
                created_at: str,
                updated_at: str,
            ) -> None:
                self.name = name
                self.job_type = job_type
                self.group = group
                self.summary = summary
                self.config = config
                self.url = url
                self.created_at = created_at
                self.updated_at = updated_at

        def _summary(
            *,
            experiment_id: str,
            phase: str,
            run_name: str,
            status: str,
            attempt_token: str,
            heartbeat_at: str,
            wandb_url: str,
            job_result_payload: dict | None = None,
            metrics_payload: dict | None = None,
            is_active: bool = False,
        ) -> dict:
            return {
                "tenyson/experiment_id": experiment_id,
                "tenyson/phase": phase,
                "tenyson/run_name": run_name,
                "tenyson/status": status,
                "tenyson/attempt_token": attempt_token,
                "tenyson/heartbeat_at": heartbeat_at,
                "tenyson/is_active": is_active,
                "tenyson/wandb_url": wandb_url,
                "tenyson/job_result_json": (
                    json.dumps(job_result_payload) if job_result_payload else None
                ),
                "tenyson/metrics_json": (
                    json.dumps(metrics_payload) if metrics_payload else None
                ),
            }

        experiment_id = "wordle_exp"
        backend_ref = "wandb://ayush/wordle"
        now = datetime.now(timezone.utc)
        completed_result = {
            "run_id": "mixed_rl",
            "status": "stopped",
            "total_time_seconds": 512.0,
            "attempt_token": "mixed-done",
            "metrics": {"global_step": 64},
            "wandb_url": "https://wandb.example/runs/mixed_rl_done",
            "failure_reason": "Manual stop requested at step 64.",
            "stopped_early": True,
        }
        fake_runs = [
            _FakeRun(
                name="mixed_rl",
                job_type="rl",
                group=experiment_id,
                summary=_summary(
                    experiment_id=experiment_id,
                    phase="rl",
                    run_name="mixed_rl",
                    status="stopped",
                    attempt_token="mixed-done",
                    heartbeat_at=(now - timedelta(minutes=5)).isoformat(),
                    wandb_url="https://wandb.example/runs/mixed_rl_done",
                    job_result_payload=completed_result,
                    metrics_payload={"global_step": 64},
                ),
                config={"_tenyson": {"environment_run": "wordle_rl_mixed"}},
                url="https://wandb.example/runs/mixed_rl_done",
                created_at=(now - timedelta(minutes=10)).isoformat(),
                updated_at=(now - timedelta(minutes=5)).isoformat(),
            ),
            _FakeRun(
                name="mixed_rl",
                job_type="rl",
                group=experiment_id,
                summary=_summary(
                    experiment_id=experiment_id,
                    phase="rl",
                    run_name="mixed_rl",
                    status="running",
                    attempt_token="mixed-live",
                    heartbeat_at=now.isoformat(),
                    wandb_url="https://wandb.example/runs/mixed_rl_live",
                    is_active=True,
                ),
                config={"_tenyson": {"environment_run": "wordle_rl_mixed"}},
                url="https://wandb.example/runs/mixed_rl_live",
                created_at=(now - timedelta(minutes=1)).isoformat(),
                updated_at=now.isoformat(),
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "final_report.md"
            report = ExperimentReport(output_path=report_path)

            with patch(
                "tenyson.reporting.fixed._project_runs_for_backend",
                return_value=fake_runs,
            ), patch(
                "tenyson.reporting.fixed.wandb_store.fetch_run_result",
                return_value=({"metrics": {"rollout_batches": 42}}, completed_result),
            ):
                rebuilt = report.rebuild_from_telemetry(
                    backend_ref=backend_ref,
                    experiment_id=experiment_id,
                    environment_name="wordle",
                    prefer_terminal_results=True,
                )

        self.assertTrue(rebuilt)
        self.assertEqual(report._stages["mixed_rl"].status, "stopped")
        self.assertEqual(report._stages["mixed_rl"].metrics["rollout_batches"], "42")
        self.assertEqual(report._stages["mixed_rl"].wandb_url, "https://wandb.example/runs/mixed_rl_done")

    def test_fixed_report_rebuild_uses_candidate_payload_when_fetch_misses(self) -> None:
        class _FakeRun:
            def __init__(
                self,
                *,
                name: str,
                job_type: str,
                group: str,
                summary: dict,
                config: dict,
                url: str,
                created_at: str,
                updated_at: str,
            ) -> None:
                self.name = name
                self.job_type = job_type
                self.group = group
                self.summary = summary
                self.config = config
                self.url = url
                self.created_at = created_at
                self.updated_at = updated_at

        def _summary(
            *,
            experiment_id: str,
            phase: str,
            run_name: str,
            status: str,
            attempt_token: str,
            heartbeat_at: str,
            wandb_url: str,
            job_result_payload: dict,
            metrics_payload: dict,
        ) -> dict:
            return {
                "tenyson/experiment_id": experiment_id,
                "tenyson/phase": phase,
                "tenyson/run_name": run_name,
                "tenyson/status": status,
                "tenyson/attempt_token": attempt_token,
                "tenyson/heartbeat_at": heartbeat_at,
                "tenyson/is_active": False,
                "tenyson/wandb_url": wandb_url,
                "tenyson/job_result_json": json.dumps(job_result_payload),
                "tenyson/metrics_json": json.dumps(metrics_payload),
            }

        experiment_id = "wordle_exp"
        backend_ref = "wandb://ayush/wordle"
        now = datetime.now(timezone.utc)
        job_result_payload = {
            "run_id": "eval_baseline_mixed",
            "status": "success",
            "total_time_seconds": 12.34,
            "attempt_token": "eval-a",
            "metrics": {"format_accuracy": 0.62, "dict_accuracy": 0.52},
            "wandb_url": "https://wandb.example/runs/eval_baseline_mixed",
            "processed_samples": 100,
            "expected_samples": 100,
        }
        fake_run = _FakeRun(
            name="eval_baseline_mixed",
            job_type="eval",
            group=experiment_id,
            summary=_summary(
                experiment_id=experiment_id,
                phase="eval",
                run_name="eval_baseline_mixed",
                status="success",
                attempt_token="eval-a",
                heartbeat_at=now.isoformat(),
                wandb_url="https://wandb.example/runs/eval_baseline_mixed",
                job_result_payload=job_result_payload,
                metrics_payload={"format_accuracy": 0.62, "dict_accuracy": 0.52},
            ),
            config={"_tenyson": {"environment_run": "wordle_eval_mixed"}},
            url="https://wandb.example/runs/eval_baseline_mixed",
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "final_report.md"
            report = ExperimentReport(output_path=report_path)

            with patch(
                "tenyson.reporting.fixed._project_runs_for_backend",
                return_value=[fake_run],
            ), patch(
                "tenyson.reporting.fixed.wandb_store.fetch_run_result",
                return_value=None,
            ):
                rebuilt = report.rebuild_from_telemetry(
                    backend_ref=backend_ref,
                    experiment_id=experiment_id,
                    environment_name="wordle",
                    prefer_terminal_results=True,
                )

        self.assertTrue(rebuilt)
        self.assertEqual(report._stages["eval_baseline_mixed"].status, "success")
        self.assertEqual(report._stages["eval_baseline_mixed"].metrics["format_accuracy"], "0.6200")
        self.assertEqual(report._stages["eval_baseline_mixed"].metrics["dict_accuracy"], "0.5200")



if __name__ == "__main__":
    unittest.main()
