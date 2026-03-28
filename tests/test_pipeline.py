import io
import sys
import unittest
from unittest.mock import patch

import tenyson.pipeline as pipeline_module
from tenyson.jobs.result import JobResult


class SFTJob:
    def __init__(self, config, task):
        self.config = config
        self.task = task


class RLJob:
    def __init__(self, config, task):
        self.config = config
        self.task = task


class SequencedCloud:
    def __init__(self, results):
        self._results = list(results)
        self.jobs = []

    def run(self, job):
        self.jobs.append(job)
        return self._results.pop(0)


class PipelineTests(unittest.TestCase):
    def test_prompt_failure_action_restart_clears_resume_checkpoint(self) -> None:
        config = {"training": {"resume_from_checkpoint": "repo:old"}}
        last_result = JobResult(
            run_id="wordle_sft_main",
            status="failed",
            total_time_seconds=0.0,
            hf_repo_id="repo",
            hf_revision="rev",
        )

        with patch.object(sys, "stdin", io.StringIO("restart\n")), patch.object(
            sys,
            "stderr",
            io.StringIO(),
        ):
            action = pipeline_module._prompt_failure_action(
                step_label="sft_main",
                config=config,
                job_type="sft",
                on_failure="wait",
                last_result=last_result,
            )

        self.assertEqual(action, "restart")
        self.assertNotIn("resume_from_checkpoint", config["training"])

    def test_prompt_failure_action_resume_sets_checkpoint_ref(self) -> None:
        config = {"training": {}}
        last_result = JobResult(
            run_id="wordle_sft_main",
            status="failed",
            total_time_seconds=0.0,
            hf_repo_id="repo",
            hf_revision="rev",
        )

        with patch.object(sys, "stdin", io.StringIO("resume\n")), patch.object(
            sys,
            "stderr",
            io.StringIO(),
        ):
            action = pipeline_module._prompt_failure_action(
                step_label="sft_main",
                config=config,
                job_type="sft",
                on_failure="wait",
                last_result=last_result,
            )

        self.assertEqual(action, "resume")
        self.assertEqual(config["training"]["resume_from_checkpoint"], "repo:rev")

    def test_prompt_failure_action_empty_stdin_waits_for_retry(self) -> None:
        config = {"training": {"resume_from_checkpoint": "repo:old"}}
        last_result = JobResult(
            run_id="wordle_rl_main",
            status="stopped",
            total_time_seconds=0.0,
            hf_repo_id="repo",
            hf_revision="rev",
        )

        class SequencedStdin:
            def __init__(self) -> None:
                self._values = ["", "restart\n"]

            def readline(self) -> str:
                if self._values:
                    return self._values.pop(0)
                return "restart\n"

        stderr = io.StringIO()
        with patch.object(sys, "stdin", SequencedStdin()), patch.object(
            sys,
            "stderr",
            stderr,
        ), patch.object(pipeline_module.time, "sleep") as sleep_mock:
            action = pipeline_module._prompt_failure_action(
                step_label="rl_main",
                config=config,
                job_type="rl",
                on_failure="wait",
                last_result=last_result,
            )

        self.assertEqual(action, "restart")
        self.assertNotIn("resume_from_checkpoint", config["training"])
        sleep_mock.assert_called_once_with(
            pipeline_module._FAILURE_PROMPT_NO_INPUT_WAIT_SECONDS
        )
        self.assertIn("No operator input available", stderr.getvalue())

    def test_prompt_failure_action_continue_accepts_stopped_checkpoint(self) -> None:
        config = {"training": {"resume_from_checkpoint": "repo:old"}}
        last_result = JobResult(
            run_id="wordle_sft_main",
            status="stopped",
            total_time_seconds=0.0,
            hf_repo_id="repo",
            hf_revision="rev",
        )

        stderr = io.StringIO()
        with patch.object(sys, "stdin", io.StringIO("continue\n")), patch.object(
            sys,
            "stderr",
            stderr,
        ):
            action = pipeline_module._prompt_failure_action(
                step_label="sft_main",
                config=config,
                job_type="sft",
                on_failure="wait",
                last_result=last_result,
            )

        self.assertEqual(action, "continue")
        self.assertNotIn("resume_from_checkpoint", config["training"])
        self.assertIn("[continue]", stderr.getvalue())

    def test_prompt_failure_action_continue_accepts_stopped_eval_without_checkpoint(self) -> None:
        config = {"training": {"resume_from_checkpoint": "repo:old"}}
        last_result = JobResult(
            run_id="eval_baseline_mixed",
            status="stopped",
            total_time_seconds=0.0,
        )

        stderr = io.StringIO()
        with patch.object(sys, "stdin", io.StringIO("continue\n")), patch.object(
            sys,
            "stderr",
            stderr,
        ):
            action = pipeline_module._prompt_failure_action(
                step_label="eval_baseline_mixed",
                config=config,
                job_type="eval",
                on_failure="wait",
                last_result=last_result,
            )

        self.assertEqual(action, "continue")
        self.assertNotIn("resume_from_checkpoint", config["training"])
        self.assertIn("[continue]", stderr.getvalue())

    def test_prompt_failure_action_abort_policy_returns_abort(self) -> None:
        config = {"training": {"resume_from_checkpoint": "repo:old"}}
        last_result = JobResult(
            run_id="wordle_sft_main",
            status="failed",
            total_time_seconds=0.0,
            hf_repo_id="repo",
            hf_revision="rev",
        )

        with patch.object(sys, "stderr", io.StringIO()):
            action = pipeline_module._prompt_failure_action(
                step_label="sft_main",
                config=config,
                job_type="sft",
                on_failure="abort",
                last_result=last_result,
            )

        self.assertEqual(action, "abort")
        self.assertEqual(config["training"]["resume_from_checkpoint"], "repo:old")

    def test_prompt_failure_action_rejects_continue_for_failed_run(self) -> None:
        config = {"training": {}}
        last_result = JobResult(
            run_id="wordle_sft_main",
            status="failed",
            total_time_seconds=0.0,
            hf_repo_id="repo",
            hf_revision="rev",
        )

        stderr = io.StringIO()
        with patch.object(sys, "stdin", io.StringIO("continue\nabort\n")), patch.object(
            sys,
            "stderr",
            stderr,
        ):
            action = pipeline_module._prompt_failure_action(
                step_label="sft_main",
                config=config,
                job_type="sft",
                on_failure="wait",
                last_result=last_result,
            )

        self.assertEqual(action, "abort")
        self.assertNotIn("[continue]", stderr.getvalue())
        self.assertIn("Invalid choice", stderr.getvalue())

    def test_prompt_failure_action_rejects_continue_for_stopped_sft_without_checkpoint(self) -> None:
        config = {"training": {}}
        last_result = JobResult(
            run_id="wordle_sft_main",
            status="stopped",
            total_time_seconds=0.0,
        )

        stderr = io.StringIO()
        with patch.object(sys, "stdin", io.StringIO("continue\nabort\n")), patch.object(
            sys,
            "stderr",
            stderr,
        ):
            action = pipeline_module._prompt_failure_action(
                step_label="sft_main",
                config=config,
                job_type="sft",
                on_failure="wait",
                last_result=last_result,
            )

        self.assertEqual(action, "abort")
        self.assertNotIn("[continue]", stderr.getvalue())
        self.assertIn("Invalid choice", stderr.getvalue())

    def test_accept_stopped_result_promotes_status_and_syncs_telemetry(self) -> None:
        result = JobResult(
            run_id="wordle_sft_main",
            status="stopped",
            total_time_seconds=0.0,
            hf_repo_id="repo",
            hf_revision="rev",
            stopped_early=True,
            failure_reason="Manual stop requested at step 12.",
        )
        config = {
            "telemetry": {
                "backend": "wandb",
                "entity": "demo",
                "project": "tenyson",
                "experiment_id": "wordle_exp",
            }
        }

        with patch.object(pipeline_module, "TelemetryClient", return_value=object()) as client_mock, patch.object(
            pipeline_module,
            "record_run_summary",
        ) as record_summary_mock, patch.object(
            pipeline_module,
            "record_run_result",
        ) as record_result_mock:
            pipeline_module._accept_stopped_result(
                result,
                config=config,
                job_type="sft",
            )

        self.assertEqual(result.status, "partial")
        client_mock.assert_called_once_with(db_url="wandb://demo/tenyson")
        record_summary_mock.assert_called_once()
        record_result_mock.assert_called_once()

    def test_abort_parallel_stage_runs_requests_stop_for_sibling_with_attempt_token(self) -> None:
        branches = [
            (
                "left_sft",
                {
                    "training": {"run_name": "left_run"},
                    "telemetry": {
                        "backend": "wandb",
                        "entity": "demo",
                        "project": "tenyson",
                        "experiment_id": "wordle_exp",
                    },
                },
                SFTJob,
                object(),
            ),
            (
                "right_rl",
                {
                    "training": {"run_name": "right_run"},
                    "telemetry": {
                        "backend": "wandb",
                        "entity": "demo",
                        "project": "tenyson",
                        "experiment_id": "wordle_exp",
                        "attempt_token": "attempt-right",
                    },
                },
                RLJob,
                object(),
            ),
        ]

        with patch.object(pipeline_module, "request_stop") as request_stop_mock:
            pipeline_module._abort_parallel_stage_runs(
                branches,
                source_run_id="left_run",
            )

        request_stop_mock.assert_called_once_with(
            db_url="wandb://demo/tenyson",
            run_id="right_run",
            experiment_id="wordle_exp",
            phase="rl",
            create_if_missing=True,
            attempt_token="attempt-right",
        )

    def test_validate_pipeline_run_names_rejects_duplicates(self) -> None:
        duplicate_steps = [
            ("first", {"training": {"run_name": "dup_run"}}, SFTJob, object()),
            ("second", {"training": {"run_name": "dup_run"}}, SFTJob, object()),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate run_name"):
            pipeline_module._validate_pipeline_run_names(duplicate_steps)

    def test_run_pipeline_retries_failed_step_after_restart_choice(self) -> None:
        cloud = SequencedCloud(
            [
                JobResult(
                    run_id="wordle_sft_main",
                    status="failed",
                    total_time_seconds=0.0,
                    failure_reason="boom",
                ),
                JobResult(
                    run_id="wordle_sft_main",
                    status="success",
                    total_time_seconds=1.5,
                ),
            ]
        )
        config = {
            "training": {"run_name": "wordle_sft_main"},
            "telemetry": {"experiment_id": "wordle_exp"},
        }

        with patch.object(pipeline_module, "notify_failure") as notify_failure_mock, patch.object(
            pipeline_module,
            "_prompt_failure_action",
            return_value="restart",
        ) as prompt_mock:
            results = pipeline_module.run_pipeline(
                [("sft_main", config, SFTJob, object())],
                cloud,
                on_failure="wait",
            )

        self.assertEqual([result.status for result in results], ["failed", "success"])
        notify_failure_mock.assert_called_once()
        prompt_mock.assert_called_once()
        self.assertEqual(len(cloud.jobs), 2)

    def test_run_pipeline_moves_to_next_step_after_continue_choice(self) -> None:
        cloud = SequencedCloud(
            [
                JobResult(
                    run_id="wordle_sft_main",
                    status="stopped",
                    total_time_seconds=0.3,
                    hf_repo_id="repo",
                    hf_revision="rev",
                    failure_reason="Manual stop requested at step 12.",
                    stopped_early=True,
                ),
                JobResult(
                    run_id="wordle_rl_main",
                    status="success",
                    total_time_seconds=1.2,
                ),
            ]
        )
        sft_config = {
            "training": {"run_name": "wordle_sft_main"},
            "telemetry": {"experiment_id": "wordle_exp"},
        }
        rl_config = {
            "training": {"run_name": "wordle_rl_main"},
            "telemetry": {"experiment_id": "wordle_exp"},
        }

        with patch.object(pipeline_module, "notify_failure") as notify_failure_mock, patch.object(
            pipeline_module,
            "_prompt_failure_action",
            return_value="continue",
        ) as prompt_mock:
            results = pipeline_module.run_pipeline(
                [
                    ("sft_main", sft_config, SFTJob, object()),
                    ("rl_main", rl_config, RLJob, object()),
                ],
                cloud,
                on_failure="wait",
            )

        self.assertEqual([result.status for result in results], ["partial", "success"])
        self.assertTrue(results[0].stopped_early)
        self.assertEqual(results[0].failure_reason, "Manual stop requested at step 12.")
        notify_failure_mock.assert_called_once()
        prompt_mock.assert_called_once()
        self.assertEqual(len(cloud.jobs), 2)

    def test_run_pipeline_abort_policy_returns_after_failed_step_without_prompt(self) -> None:
        cloud = SequencedCloud(
            [
                JobResult(
                    run_id="wordle_sft_main",
                    status="failed",
                    total_time_seconds=0.0,
                    failure_reason="boom",
                ),
            ]
        )
        config = {
            "training": {"run_name": "wordle_sft_main"},
            "telemetry": {"experiment_id": "wordle_exp"},
        }

        with patch.object(pipeline_module, "notify_failure") as notify_failure_mock, patch.object(
            pipeline_module,
            "_prompt_failure_action",
        ) as prompt_mock:
            results = pipeline_module.run_pipeline(
                [("sft_main", config, SFTJob, object())],
                cloud,
                on_failure="abort",
            )

        self.assertEqual([result.status for result in results], ["failed"])
        notify_failure_mock.assert_called_once()
        prompt_mock.assert_not_called()
        self.assertEqual(len(cloud.jobs), 1)

    def test_normalize_on_failure_policy_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "on_failure must be either"):
            pipeline_module._normalize_on_failure_policy("maybe")

    def test_before_step_can_mutate_config_before_execution(self) -> None:
        cloud = SequencedCloud(
            [
                JobResult(
                    run_id="wordle_sft_main",
                    status="success",
                    total_time_seconds=0.5,
                )
            ]
        )
        config = {"training": {"run_name": "wordle_sft_main"}}
        seen_previous_results = []

        def before_step(label, step_index, step_config, previous_results):
            seen_previous_results.append((label, step_index, list(previous_results)))
            step_config.setdefault("training", {})["extra_flag"] = "set"

        pipeline_module.run_pipeline(
            [("sft_main", config, SFTJob, object())],
            cloud,
            on_failure="abort",
            before_step=before_step,
        )

        self.assertEqual(seen_previous_results, [("sft_main", 0, [])])
        self.assertEqual(cloud.jobs[0].config["training"]["extra_flag"], "set")


if __name__ == "__main__":
    unittest.main()
