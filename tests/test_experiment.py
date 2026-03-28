import copy
from datetime import datetime, timezone
import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import tenyson.experiment as experiment_module
from tenyson.experiment import (
    AdapterRef,
    ConfigTemplates,
    ExperimentAborted,
    ExperimentSession,
)
from tenyson.core.telemetry import LiveRunInfo
from tenyson.jobs.result import JobResult
from tenyson.reporting.builder import ReportBuilder


def _templates() -> ConfigTemplates:
    return ConfigTemplates(
        {
            "sft": {"training": {}, "task": {}},
            "rl": {
                "training": {"epochs": 1},
                "task": {"min_history_turns": 1},
                "model": {"name": "base-model"},
            },
            "eval": {"evaluation": {}, "task": {}, "model": {}},
        }
    )


def _result(run_id: str, *, status: str = "success", **kwargs) -> JobResult:
    return JobResult(
        run_id=run_id,
        status=status,
        total_time_seconds=1.0,
        **kwargs,
    )


def _telemetry_shared_overrides(experiment_id: str = "wordle_exp") -> dict:
    return {
        "telemetry": {
            "backend": "wandb",
            "entity": "demo",
            "project": "tenyson",
            "experiment_id": experiment_id,
        }
    }


class ExperimentSessionTests(unittest.TestCase):
    def test_prompt_recovery_action_empty_stdin_defaults_to_restart(self) -> None:
        last_result = JobResult(
            run_id="wordle_sft_main",
            status="stopped",
            total_time_seconds=0.0,
            hf_repo_id="repo",
            hf_revision="rev",
        )

        with patch.object(sys, "stdin", io.StringIO("")), patch.object(
            sys,
            "stderr",
            io.StringIO(),
        ):
            action = experiment_module._prompt_recovery_action(
                step_label="sft_main",
                run_id="wordle_sft_main",
                job_type="sft",
                last_result=last_result,
            )

        self.assertEqual(action, "restart")

    def test_recovery_controller_lock_refuses_busy_lock(self) -> None:
        if experiment_module.fcntl is None:
            self.skipTest("fcntl is unavailable on this platform")

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {"TENYSON_RECOVERY_LOCK_DIR": tmpdir},
            clear=False,
        ), patch.object(
            experiment_module.fcntl,
            "flock",
            side_effect=BlockingIOError,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                'Recovery controller for experiment_id "wordle_busy_lock" is already running.',
            ):
                ExperimentSession(
                    task=object(),
                    templates=_templates(),
                    cloud_factory=lambda: object(),
                    shared_overrides=_telemetry_shared_overrides(),
                    recovery_experiment_id="wordle_busy_lock",
                )

    def test_stage_building_clones_templates_and_injects_adapter(self) -> None:
        base_rl_template = {
            "training": {"epochs": 1},
            "task": {"min_history_turns": 1},
            "model": {"name": "base-model"},
        }
        templates = ConfigTemplates(
            {
                "sft": {"training": {}, "task": {}},
                "rl": copy.deepcopy(base_rl_template),
                "eval": {"evaluation": {}, "task": {}, "model": {}},
            }
        )
        session = ExperimentSession(
            task=object(),
            templates=templates,
            cloud_factory=lambda: object(),
            shared_overrides={"training": {"hf_repo_base": "org/base"}},
        )

        stage = session.rl(
            "mixed_rl",
            adapter=AdapterRef(repo_id="repo/id", revision="sha123"),
            run_name="wordle_rl_mixed",
            output_dir="./outputs/mixed/rl",
            overrides={"task": {"max_history_turns": 5}},
        )

        self.assertEqual(
            stage.config["training"],
            {
                "epochs": 1,
                "hf_repo_base": "org/base",
                "output_dir": "./outputs/mixed/rl",
                "run_name": "wordle_rl_mixed",
            },
        )
        self.assertEqual(
            stage.config["task"],
            {
                "min_history_turns": 1,
                "max_history_turns": 5,
            },
        )
        self.assertEqual(
            stage.config["model"],
            {
                "name": "base-model",
                "init_adapter_repo": "repo/id",
                "init_adapter_revision": "sha123",
            },
        )
        self.assertEqual(templates.clone("rl"), base_rl_template)

    def test_config_templates_from_directory_uses_default_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            (config_dir / "sft.yaml").write_text(
                "training:\n  epochs: 1\n",
                encoding="utf-8",
            )
            (config_dir / "rl.yaml").write_text(
                "training:\n  epochs: 2\n",
                encoding="utf-8",
            )
            (config_dir / "eval.yaml").write_text(
                "evaluation:\n  batch_size: 4\n",
                encoding="utf-8",
            )

            templates = ConfigTemplates.from_directory(config_dir)

        self.assertEqual(templates.clone("sft")["training"]["epochs"], 1)
        self.assertEqual(templates.clone("rl")["training"]["epochs"], 2)
        self.assertEqual(templates.clone("eval")["evaluation"]["batch_size"], 4)

    def test_run_stage_requires_exact_run_id(self) -> None:
        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=lambda: object(),
        )
        stage = session.sft("sft_main", run_name="wordle_sft_main")

        def fake_run_pipeline(steps, cloud, on_failure):
            return [_result("wrong_run")]

        with patch.object(experiment_module, "run_pipeline", fake_run_pipeline):
            with self.assertRaisesRegex(
                RuntimeError,
                'expected run_id "wordle_sft_main"',
            ):
                session.run_stage(stage, cloud=object())

    def test_run_parallel_matches_results_by_run_id(self) -> None:
        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=lambda: object(),
        )
        adapter = AdapterRef(repo_id="repo/id", revision="sha123")
        stage_turn_2 = session.eval(
            "curr_eval_after_t3_turn2",
            adapter=adapter,
            run_name="wordle_curr_eval_after_t3_turn2",
        )
        stage_turn_3 = session.eval(
            "curr_eval_after_t3_turn3",
            adapter=adapter,
            run_name="wordle_curr_eval_after_t3_turn3",
        )

        def fake_run_pipeline(steps, cloud, on_failure):
            return [
                _result("wordle_curr_eval_after_t3_turn3"),
                _result("wordle_curr_eval_after_t3_turn2"),
            ]

        with patch.object(experiment_module, "run_pipeline", fake_run_pipeline):
            results = session.run_parallel(
                "curr_eval_after_t3",
                [stage_turn_2, stage_turn_3],
                cloud=object(),
            )

        self.assertEqual(
            results["curr_eval_after_t3_turn2"].run_id,
            "wordle_curr_eval_after_t3_turn2",
        )
        self.assertEqual(
            results["curr_eval_after_t3_turn3"].run_id,
            "wordle_curr_eval_after_t3_turn3",
        )

    def test_branch_raises_after_abort_and_keeps_stage_result(self) -> None:
        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=lambda: object(),
        )
        branch = session.branch(cloud=object())
        stage = branch.sft("sft_main", run_name="wordle_sft_main")

        def fake_run_pipeline(steps, cloud, on_failure):
            return [_result("wordle_sft_main", status="failed")]

        with patch.object(experiment_module, "run_pipeline", fake_run_pipeline):
            with self.assertRaises(ExperimentAborted):
                branch.run(stage)

        self.assertEqual(branch.result("sft_main").status, "failed")

    def test_branch_keeps_partial_result_without_aborting(self) -> None:
        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=lambda: object(),
        )
        branch = session.branch(cloud=object())
        stage = branch.sft("sft_main", run_name="wordle_sft_main")

        def fake_run_pipeline(steps, cloud, on_failure):
            return [
                _result(
                    "wordle_sft_main",
                    status="partial",
                    stopped_early=True,
                    failure_reason="Manual stop requested at step 12.",
                    hf_repo_id="repo/id",
                    hf_revision="sha123",
                )
            ]

        with patch.object(experiment_module, "run_pipeline", fake_run_pipeline):
            result = branch.run(stage)

        self.assertEqual(result.status, "partial")
        self.assertTrue(result.stopped_early)
        self.assertEqual(branch.result("sft_main").status, "partial")

    def test_run_branches_returns_each_branch_result_map(self) -> None:
        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=lambda: object(),
        )

        def fake_run_pipeline(steps, cloud, on_failure):
            _label, config, _job_class, _task = steps[0]
            training = config.get("training", {})
            evaluation = config.get("evaluation", {})
            run_id = training.get("run_name") or evaluation.get("run_name")
            return [_result(run_id)]

        with patch.object(experiment_module, "run_pipeline", fake_run_pipeline):
            branch_results = session.run_branches(
                {
                    "left": lambda branch: branch.run(
                        branch.sft("left_stage", run_name="left_run")
                    ),
                    "right": lambda branch: branch.run(
                        branch.sft("right_stage", run_name="right_run")
                    ),
                }
            )

        self.assertEqual(branch_results["left"]["left_stage"].run_id, "left_run")
        self.assertEqual(branch_results["right"]["right_stage"].run_id, "right_run")

    def test_run_stage_updates_report_when_result_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = Path(tmpdir) / "template.md"
            output_path = Path(tmpdir) / "report.md"
            template_path.write_text(
                "status={sft_main_status}\nwandb={sft_main_wandb_link}\n",
                encoding="utf-8",
            )
            report = ReportBuilder(
                template_path=str(template_path),
                output_path=str(output_path),
            )
            report.generate()

            session = ExperimentSession(
                task=object(),
                templates=_templates(),
                cloud_factory=lambda: object(),
                report_builder=report,
            )
            stage = session.sft("sft_main", run_name="wordle_sft_main")

            def fake_run_pipeline(steps, cloud, on_failure):
                return [
                    JobResult(
                        run_id="wordle_sft_main",
                        status="success",
                        total_time_seconds=1.0,
                        wandb_url="https://wandb.example/run",
                    )
                ]

            with patch.object(experiment_module, "run_pipeline", fake_run_pipeline):
                session.run_stage(stage, cloud=object())

            session.close()
            content = output_path.read_text(encoding="utf-8")

        self.assertIn("status=success", content)
        self.assertIn("wandb=[run](https://wandb.example/run)", content)

    def test_session_close_closes_registered_clouds(self) -> None:
        close_calls: list[str] = []

        class FakeCloud:
            def close(self) -> None:
                close_calls.append("closed")

        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=FakeCloud,
        )

        session.create_cloud()
        session.close()

        self.assertEqual(close_calls, ["closed"])

    def test_report_controller_writes_wandb_link_from_telemetry_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = Path(tmpdir) / "template.md"
            output_path = Path(tmpdir) / "report.md"
            template_path.write_text(
                "status={sft_main_status}\nwandb={sft_main_wandb_link}\n",
                encoding="utf-8",
            )
            report = ReportBuilder(
                template_path=str(template_path),
                output_path=str(output_path),
            )
            report.generate()

            controller = experiment_module._ExperimentReportController(
                report,
                poll_interval_seconds=0.01,
            )
            stage = experiment_module.StageSpec(
                id="sft_main",
                config={
                    "training": {"run_name": "wordle_sft_main"},
                    "telemetry": {
                        "backend": "wandb",
                        "entity": "demo",
                        "project": "tenyson",
                        "experiment_id": "wordle_exp",
                    },
                },
                job_class=object,
                task=object(),
                run_type="sft",
                run_name="wordle_sft_main",
            )

            with patch.object(
                experiment_module,
                "TelemetryClient",
                return_value=object(),
            ), patch.object(
                experiment_module,
                "get_run_metadata_wandb_url",
                return_value="https://wandb.example/live",
            ):
                controller.start_stage(stage)
                content = output_path.read_text(encoding="utf-8")
                for _ in range(20):
                    if "wandb=[run](https://wandb.example/live)" in content:
                        break
                    time.sleep(0.02)
                    content = output_path.read_text(encoding="utf-8")
                controller.close()

        self.assertIn("status=running", content)
        self.assertIn("wandb=[run](https://wandb.example/live)", content)

    def test_run_stage_reuses_recovered_success_result_and_updates_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = Path(tmpdir) / "template.md"
            output_path = Path(tmpdir) / "report.md"
            template_path.write_text(
                "status={sft_main_status}\nwandb={sft_main_wandb_link}\n",
                encoding="utf-8",
            )
            report = ReportBuilder(
                template_path=str(template_path),
                output_path=str(output_path),
            )
            report.generate()

            session = ExperimentSession(
                task=object(),
                templates=_templates(),
                cloud_factory=lambda: object(),
                shared_overrides=_telemetry_shared_overrides(),
                recovery_experiment_id="wordle_exp",
                report_builder=report,
            )
            stage = session.sft("sft_main", run_name="wordle_sft_main")
            recovered = _result(
                "wordle_sft_main",
                status="success",
                wandb_url="https://wandb.example/recovered",
                hf_repo_id="repo/id",
                hf_revision="sha123",
            )

            with patch.object(
                experiment_module,
                "TelemetryClient",
                return_value=object(),
            ), patch.object(
                experiment_module,
                "list_live_runs",
                return_value=[],
            ), patch.object(
                experiment_module,
                "get_run_result",
                return_value=({}, dict(recovered.__dict__)),
            ), patch.object(
                experiment_module,
                "run_pipeline",
            ) as run_pipeline_mock:
                result = session.run_stage(stage, cloud=object())

            session.close()
            content = output_path.read_text(encoding="utf-8")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.hf_revision, "sha123")
        run_pipeline_mock.assert_not_called()
        self.assertIn("status=success", content)
        self.assertIn("wandb=[run](https://wandb.example/recovered)", content)

    def test_run_stage_recovery_refuses_when_live_runs_exist(self) -> None:
        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=lambda: object(),
            shared_overrides=_telemetry_shared_overrides(),
            recovery_experiment_id="wordle_exp",
        )
        stage = session.sft("sft_main", run_name="wordle_sft_main")
        live_row = LiveRunInfo(
            run_id="mixed_rl",
            phase="rl",
            provider="modal",
            status="running",
            is_active=True,
            created_at=None,
            updated_at=datetime.now(timezone.utc),
            attempt_token="attempt-1",
        )
        recovered = _result(
            "wordle_sft_main",
            status="stopped",
            hf_repo_id="repo/id",
            hf_revision="sha123",
            failure_reason="Manual stop requested at step 12.",
            stopped_early=True,
            attempt_token="old-attempt-token",
        )

        with patch.object(
            experiment_module,
            "TelemetryClient",
            return_value=object(),
        ), patch.object(
            experiment_module,
            "get_run_result",
            return_value=({}, dict(recovered.__dict__)),
        ), patch.object(
            experiment_module,
            "_prompt_recovery_action",
            return_value="resume",
        ) as prompt_mock, patch.object(
            experiment_module,
            "list_live_runs",
            return_value=[live_row],
        ) as list_live_runs_mock, patch.object(
            experiment_module,
            "run_pipeline",
        ) as run_pipeline_mock:
            with self.assertRaisesRegex(
                RuntimeError,
                r'live runs are still active: mixed_rl \(phase=rl\)',
            ):
                session.run_stage(stage, cloud=object())

        self.assertEqual(
            stage.config["training"]["resume_from_checkpoint"],
            "repo/id:sha123",
        )
        self.assertEqual(
            stage.config["telemetry"]["attempt_token"],
            "old-attempt-token",
        )
        prompt_mock.assert_called_once()
        list_live_runs_mock.assert_called_once_with(
            db_url="wandb://demo/tenyson",
            experiment_id="wordle_exp",
            max_age_seconds=experiment_module._RECOVERY_LIVE_RUN_MAX_AGE_SECONDS,
        )
        run_pipeline_mock.assert_not_called()

    def test_run_stage_recovery_live_run_check_only_happens_once_per_session(self) -> None:
        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=lambda: object(),
            shared_overrides=_telemetry_shared_overrides(),
            recovery_experiment_id="wordle_exp",
        )
        first_stage = session.sft("sft_main", run_name="wordle_sft_main")
        second_stage = session.sft("sft_retry", run_name="wordle_sft_retry")

        def fake_run_pipeline(steps, cloud, on_failure):
            del cloud, on_failure
            _label, config, _job_class, _task = steps[0]
            run_id = config["training"]["run_name"]
            return [_result(run_id, status="success")]

        with patch.object(
            experiment_module,
            "list_live_runs",
            return_value=[],
        ) as list_live_runs_mock, patch.object(
            experiment_module,
            "get_run_result",
            return_value=None,
        ), patch.object(
            experiment_module,
            "run_pipeline",
            side_effect=fake_run_pipeline,
        ) as run_pipeline_mock:
            first_result = session.run_stage(first_stage, cloud=object())
            second_result = session.run_stage(second_stage, cloud=object())

        self.assertEqual(first_result.status, "success")
        self.assertEqual(second_result.status, "success")
        list_live_runs_mock.assert_called_once_with(
            db_url="wandb://demo/tenyson",
            experiment_id="wordle_exp",
            max_age_seconds=experiment_module._RECOVERY_LIVE_RUN_MAX_AGE_SECONDS,
        )
        self.assertEqual(run_pipeline_mock.call_count, 2)

    def test_run_stage_recovery_resume_sets_resume_checkpoint(self) -> None:
        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=lambda: object(),
            shared_overrides=_telemetry_shared_overrides(),
            recovery_experiment_id="wordle_exp",
        )
        stage = session.sft("sft_main", run_name="wordle_sft_main")
        recovered = _result(
            "wordle_sft_main",
            status="stopped",
            hf_repo_id="repo/id",
            hf_revision="sha123",
            failure_reason="Manual stop requested at step 12.",
            stopped_early=True,
            attempt_token="old-attempt-token",
        )

        def fake_run_pipeline(steps, cloud, on_failure):
            _label, config, _job_class, _task = steps[0]
            self.assertEqual(
                config["training"]["resume_from_checkpoint"],
                "repo/id:sha123",
            )
            self.assertEqual(
                config["telemetry"]["attempt_token"],
                "old-attempt-token",
            )
            return [_result("wordle_sft_main", status="success")]

        with patch.object(
            experiment_module,
            "TelemetryClient",
            return_value=object(),
        ), patch.object(
            experiment_module,
            "list_live_runs",
            return_value=[],
        ), patch.object(
            experiment_module,
            "get_run_result",
            return_value=({}, dict(recovered.__dict__)),
        ), patch.object(
            experiment_module,
            "_prompt_recovery_action",
            return_value="resume",
        ) as prompt_mock, patch.object(
            experiment_module,
            "run_pipeline",
            side_effect=fake_run_pipeline,
        ) as run_pipeline_mock:
            result = session.run_stage(stage, cloud=object())

        self.assertEqual(result.status, "success")
        self.assertEqual(
            stage.config["training"]["resume_from_checkpoint"],
            "repo/id:sha123",
        )
        self.assertEqual(
            stage.config["telemetry"]["attempt_token"],
            "old-attempt-token",
        )
        prompt_mock.assert_called_once()
        run_pipeline_mock.assert_called_once()

    def test_run_stage_recovery_continue_skips_rerun(self) -> None:
        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=lambda: object(),
            shared_overrides=_telemetry_shared_overrides(),
            recovery_experiment_id="wordle_exp",
        )
        stage = session.sft("sft_main", run_name="wordle_sft_main")
        recovered = _result(
            "wordle_sft_main",
            status="stopped",
            hf_repo_id="repo/id",
            hf_revision="sha123",
            failure_reason="Manual stop requested at step 12.",
            stopped_early=True,
        )

        def promote_partial(result, *, config, job_type):
            del config, job_type
            result.status = "partial"

        with patch.object(
            experiment_module,
            "TelemetryClient",
            return_value=object(),
        ), patch.object(
            experiment_module,
            "list_live_runs",
            return_value=[],
        ), patch.object(
            experiment_module,
            "get_run_result",
            return_value=({}, dict(recovered.__dict__)),
        ), patch.object(
            experiment_module,
            "_prompt_recovery_action",
            return_value="continue",
        ) as prompt_mock, patch.object(
            experiment_module,
            "_accept_stopped_result",
            side_effect=promote_partial,
        ) as accept_mock, patch.object(
            experiment_module,
            "run_pipeline",
        ) as run_pipeline_mock:
            result = session.run_stage(stage, cloud=object())

        self.assertEqual(result.status, "partial")
        self.assertTrue(result.stopped_early)
        prompt_mock.assert_called_once()
        accept_mock.assert_called_once()
        run_pipeline_mock.assert_not_called()

    def test_run_stage_recovery_restart_clears_resume_checkpoint(self) -> None:
        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=lambda: object(),
            shared_overrides=_telemetry_shared_overrides(),
            recovery_experiment_id="wordle_exp",
        )
        stage = session.sft("sft_main", run_name="wordle_sft_main")
        stage.config.setdefault("training", {})["resume_from_checkpoint"] = "repo/old:rev"
        recovered = _result(
            "wordle_sft_main",
            status="stopped",
            hf_repo_id="repo/id",
            hf_revision="sha123",
            stopped_early=True,
        )

        def fake_run_pipeline(steps, cloud, on_failure):
            _label, config, _job_class, _task = steps[0]
            self.assertNotIn("resume_from_checkpoint", config["training"])
            return [_result("wordle_sft_main", status="success")]

        with patch.object(
            experiment_module,
            "TelemetryClient",
            return_value=object(),
        ), patch.object(
            experiment_module,
            "list_live_runs",
            return_value=[],
        ), patch.object(
            experiment_module,
            "get_run_result",
            return_value=({}, dict(recovered.__dict__)),
        ), patch.object(
            experiment_module,
            "_prompt_recovery_action",
            return_value="restart",
        ), patch.object(
            experiment_module,
            "run_pipeline",
            side_effect=fake_run_pipeline,
        ) as run_pipeline_mock:
            result = session.run_stage(stage, cloud=object())

        self.assertEqual(result.status, "success")
        self.assertNotIn("resume_from_checkpoint", stage.config["training"])
        run_pipeline_mock.assert_called_once()

    def test_run_stage_recovery_restart_set_ignores_prior_success(self) -> None:
        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=lambda: object(),
            shared_overrides=_telemetry_shared_overrides(),
            recovery_experiment_id="wordle_exp",
            recovery_restart_stages=["eval_baseline_mixed"],
        )
        adapter = AdapterRef(repo_id="repo/id", revision="sha123")
        stage = session.eval(
            "eval_baseline_mixed",
            adapter=adapter,
            run_name="eval_baseline_mixed",
        )
        recovered = _result(
            "eval_baseline_mixed",
            status="success",
            processed_samples=25,
            expected_samples=25,
        )

        def fake_run_pipeline(steps, cloud, on_failure):
            del cloud, on_failure
            self.assertEqual(steps[0][0], "eval_baseline_mixed")
            return [
                _result(
                    "eval_baseline_mixed",
                    status="success",
                    processed_samples=100,
                    expected_samples=100,
                )
            ]

        with patch.object(
            experiment_module,
            "TelemetryClient",
            return_value=object(),
        ), patch.object(
            experiment_module,
            "list_live_runs",
            return_value=[],
        ), patch.object(
            experiment_module,
            "get_run_result",
            return_value=({}, dict(recovered.__dict__)),
        ), patch.object(
            experiment_module,
            "run_pipeline",
            side_effect=fake_run_pipeline,
        ) as run_pipeline_mock:
            result = session.run_stage(stage, cloud=object())

        self.assertEqual(result.processed_samples, 100)
        run_pipeline_mock.assert_called_once()

    def test_run_parallel_only_launches_unrecovered_stages(self) -> None:
        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=lambda: object(),
            shared_overrides=_telemetry_shared_overrides(),
            recovery_experiment_id="wordle_exp",
        )
        adapter = AdapterRef(repo_id="repo/id", revision="sha123")
        recovered_stage = session.eval(
            "eval_turn2",
            adapter=adapter,
            run_name="wordle_eval_turn2",
        )
        fresh_stage = session.eval(
            "eval_turn3",
            adapter=adapter,
            run_name="wordle_eval_turn3",
        )
        recovered = _result(
            "wordle_eval_turn2",
            status="success",
            processed_samples=25,
            expected_samples=25,
        )

        def fake_get_run_result(
            client,
            experiment_id,
            run_id,
            phase,
            *,
            include_results_payload=True,
        ):
            del client, experiment_id, phase, include_results_payload
            if run_id == "wordle_eval_turn2":
                return ({}, dict(recovered.__dict__))
            return None

        def fake_run_pipeline(steps, cloud, on_failure):
            self.assertEqual(len(steps), 1)
            self.assertEqual(steps[0]["label"], "eval_pair")
            parallel_steps = steps[0]["parallel"]
            self.assertEqual(len(parallel_steps), 1)
            self.assertEqual(parallel_steps[0][0], "eval_turn3")
            return [
                _result(
                    "wordle_eval_turn3",
                    status="success",
                    processed_samples=25,
                    expected_samples=25,
                )
            ]

        with patch.object(
            experiment_module,
            "TelemetryClient",
            return_value=object(),
        ), patch.object(
            experiment_module,
            "list_live_runs",
            return_value=[],
        ), patch.object(
            experiment_module,
            "get_run_result",
            side_effect=fake_get_run_result,
        ), patch.object(
            experiment_module,
            "run_pipeline",
            side_effect=fake_run_pipeline,
        ) as run_pipeline_mock:
            results = session.run_parallel(
                "eval_pair",
                [recovered_stage, fresh_stage],
                cloud=object(),
            )

        self.assertEqual(results["eval_turn2"].status, "success")
        self.assertEqual(results["eval_turn2"].processed_samples, 25)
        self.assertEqual(results["eval_turn3"].status, "success")
        run_pipeline_mock.assert_called_once()

    def test_run_parallel_recovery_refuses_when_live_runs_exist(self) -> None:
        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=lambda: object(),
            shared_overrides=_telemetry_shared_overrides(),
            recovery_experiment_id="wordle_exp",
        )
        adapter = AdapterRef(repo_id="repo/id", revision="sha123")
        recovered_stage = session.eval(
            "eval_turn2",
            adapter=adapter,
            run_name="wordle_eval_turn2",
        )
        fresh_stage = session.eval(
            "eval_turn3",
            adapter=adapter,
            run_name="wordle_eval_turn3",
        )
        recovered = _result(
            "wordle_eval_turn2",
            status="success",
            processed_samples=25,
            expected_samples=25,
        )
        live_row = LiveRunInfo(
            run_id="curr_rl_t3",
            phase="rl",
            provider="modal",
            status="running",
            is_active=True,
            created_at=None,
            updated_at=datetime.now(timezone.utc),
            attempt_token="attempt-2",
        )

        def fake_get_run_result(
            client,
            experiment_id,
            run_id,
            phase,
            *,
            include_results_payload=True,
        ):
            del client, experiment_id, phase, include_results_payload
            if run_id == "wordle_eval_turn2":
                return ({}, dict(recovered.__dict__))
            return None

        with patch.object(
            experiment_module,
            "TelemetryClient",
            return_value=object(),
        ), patch.object(
            experiment_module,
            "get_run_result",
            side_effect=fake_get_run_result,
        ), patch.object(
            experiment_module,
            "list_live_runs",
            return_value=[live_row],
        ) as list_live_runs_mock, patch.object(
            experiment_module,
            "run_pipeline",
        ) as run_pipeline_mock:
            with self.assertRaisesRegex(
                RuntimeError,
                r'live runs are still active: curr_rl_t3 \(phase=rl\)',
            ):
                session.run_parallel(
                    "eval_pair",
                    [recovered_stage, fresh_stage],
                    cloud=object(),
                )

        list_live_runs_mock.assert_called_once_with(
            db_url="wandb://demo/tenyson",
            experiment_id="wordle_exp",
            max_age_seconds=experiment_module._RECOVERY_LIVE_RUN_MAX_AGE_SECONDS,
        )
        run_pipeline_mock.assert_not_called()

    def test_run_stage_uses_telemetry_result_when_pipeline_result_is_missing(self) -> None:
        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=lambda: object(),
            shared_overrides=_telemetry_shared_overrides(),
        )
        stage = session.sft("sft_main", run_name="wordle_sft_main")
        recovered = _result(
            "wordle_sft_main",
            status="success",
            wandb_url="https://wandb.example/recovered",
            hf_repo_id="repo/id",
            hf_revision="sha123",
        )

        def fake_get_run_result(
            client,
            experiment_id,
            run_id,
            phase,
            *,
            attempt_token=None,
            include_results_payload=True,
        ):
            del client, experiment_id, phase, attempt_token, include_results_payload
            if run_id == "wordle_sft_main":
                return ({}, dict(recovered.__dict__))
            return None

        with patch.object(
            experiment_module,
            "TelemetryClient",
            return_value=object(),
        ), patch.object(
            experiment_module,
            "get_run_result",
            side_effect=fake_get_run_result,
        ), patch.object(
            experiment_module,
            "run_pipeline",
            return_value=[_result("wrong_run", status="failed")],
        ):
            result = session.run_stage(stage, cloud=object())

        self.assertEqual(result.run_id, "wordle_sft_main")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.hf_revision, "sha123")

    def test_run_parallel_uses_telemetry_result_when_pipeline_omits_one_branch_result(self) -> None:
        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=lambda: object(),
            shared_overrides=_telemetry_shared_overrides(),
        )
        adapter = AdapterRef(repo_id="repo/id", revision="sha123")
        left_stage = session.eval(
            "eval_turn2",
            adapter=adapter,
            run_name="wordle_eval_turn2",
        )
        right_stage = session.eval(
            "eval_turn3",
            adapter=adapter,
            run_name="wordle_eval_turn3",
        )
        recovered_left = _result(
            "wordle_eval_turn2",
            status="success",
            processed_samples=25,
            expected_samples=25,
        )

        def fake_get_run_result(
            client,
            experiment_id,
            run_id,
            phase,
            *,
            attempt_token=None,
            include_results_payload=True,
        ):
            del client, experiment_id, phase, attempt_token, include_results_payload
            if run_id == "wordle_eval_turn2":
                return ({}, dict(recovered_left.__dict__))
            return None

        with patch.object(
            experiment_module,
            "TelemetryClient",
            return_value=object(),
        ), patch.object(
            experiment_module,
            "get_run_result",
            side_effect=fake_get_run_result,
        ), patch.object(
            experiment_module,
            "run_pipeline",
            return_value=[
                _result(
                    "wordle_eval_turn3",
                    status="success",
                    processed_samples=25,
                    expected_samples=25,
                )
            ],
        ):
            results = session.run_parallel(
                "eval_pair",
                [left_stage, right_stage],
                cloud=object(),
            )

        self.assertEqual(results["eval_turn2"].status, "success")
        self.assertEqual(results["eval_turn2"].processed_samples, 25)
        self.assertEqual(results["eval_turn3"].status, "success")
        self.assertEqual(results["eval_turn3"].processed_samples, 25)

    def test_recovery_experiment_id_must_match_stage_telemetry_experiment_id(self) -> None:
        session = ExperimentSession(
            task=object(),
            templates=_templates(),
            cloud_factory=lambda: object(),
            shared_overrides=_telemetry_shared_overrides("wordle_exp"),
            recovery_experiment_id="other_exp",
        )
        stage = session.sft("sft_main", run_name="wordle_sft_main")

        with self.assertRaisesRegex(ValueError, "recovery_experiment_id"):
            session.run_stage(stage, cloud=object())


if __name__ == "__main__":
    unittest.main()
