import os
import sys
import unittest
from unittest.mock import patch

from tenyson.core.execution_policy import require_gpu_provider_runtime
import tenyson.runner as runner_module


class _BaseFakeJob:
    instances = []

    def __init__(self, config, task):
        self.config = config
        self.task = task
        self.run_called = False
        type(self).instances.append(self)

    def run(self):
        self.run_called = True


class FakeSFTJob(_BaseFakeJob):
    instances = []


class FakeRLJob(_BaseFakeJob):
    instances = []


class FakeEvalJob(_BaseFakeJob):
    instances = []


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeSFTJob.instances.clear()
        FakeRLJob.instances.clear()
        FakeEvalJob.instances.clear()

    def test_resolve_task_uses_file_loader_for_paths(self) -> None:
        with patch.object(runner_module, "load_task", return_value="file-task") as load_task_mock, patch.object(
            runner_module,
            "load_task_from_spec",
            return_value="spec-task",
        ) as load_task_from_spec_mock:
            task = runner_module._resolve_task("tasks/wordle.py")

        self.assertEqual(task, "file-task")
        load_task_mock.assert_called_once_with("tasks/wordle.py")
        load_task_from_spec_mock.assert_not_called()

    def test_resolve_task_uses_symbol_loader_for_module_specs(self) -> None:
        with patch.object(runner_module, "load_task", return_value="file-task") as load_task_mock, patch.object(
            runner_module,
            "load_task_from_spec",
            return_value="spec-task",
        ) as load_task_from_spec_mock:
            task = runner_module._resolve_task("module.path:Task")

        self.assertEqual(task, "spec-task")
        load_task_from_spec_mock.assert_called_once_with("module.path:Task")
        load_task_mock.assert_not_called()

    def test_main_builds_rl_job_and_applies_resume_checkpoint(self) -> None:
        with patch.object(runner_module, "require_gpu_provider_runtime"), patch.object(
            runner_module,
            "load_config",
            return_value={},
        ) as load_config_mock, patch.object(
            runner_module,
            "_resolve_task",
            return_value="task",
        ), patch.object(
            runner_module,
            "SFTJob",
            FakeSFTJob,
        ), patch.object(
            runner_module,
            "RLJob",
            FakeRLJob,
        ), patch.object(
            runner_module,
            "EvalJob",
            FakeEvalJob,
        ), patch.object(
            sys,
            "argv",
            [
                "runner",
                "--job-type",
                "rl",
                "--config",
                "config.yaml",
                "--task-module",
                "module.path:Task",
                "--resume-from-checkpoint",
                "repo:rev",
            ],
        ):
            runner_module.main()

        self.assertEqual(load_config_mock.call_args.args[0], os.path.abspath("config.yaml"))
        self.assertEqual(len(FakeRLJob.instances), 1)
        job = FakeRLJob.instances[0]
        self.assertEqual(job.config["training"]["resume_from_checkpoint"], "repo:rev")
        self.assertEqual(job.task, "task")
        self.assertTrue(job.run_called)

    def test_main_builds_eval_job_without_training_resume_injection(self) -> None:
        with patch.object(runner_module, "require_gpu_provider_runtime"), patch.object(
            runner_module,
            "load_config",
            return_value={"evaluation": {"run_name": "wordle_eval_main"}},
        ), patch.object(
            runner_module,
            "_resolve_task",
            return_value="task",
        ), patch.object(
            runner_module,
            "SFTJob",
            FakeSFTJob,
        ), patch.object(
            runner_module,
            "RLJob",
            FakeRLJob,
        ), patch.object(
            runner_module,
            "EvalJob",
            FakeEvalJob,
        ), patch.object(
            sys,
            "argv",
            [
                "runner",
                "--job-type",
                "eval",
                "--config",
                "config.yaml",
                "--task-module",
                "module.path:Task",
                "--resume-from-checkpoint",
                "repo:rev",
            ],
        ):
            runner_module.main()

        self.assertEqual(len(FakeEvalJob.instances), 1)
        job = FakeEvalJob.instances[0]
        self.assertNotIn("training", job.config)
        self.assertTrue(job.run_called)


class ExecutionPolicyTests(unittest.TestCase):
    def test_require_gpu_provider_runtime_accepts_modal(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TENYSON_EXECUTION_MODE": "cloud",
                "TENYSON_GPU_PROVIDER": "modal",
            },
            clear=True,
        ):
            provider = require_gpu_provider_runtime()

        self.assertEqual(provider, "modal")

    def test_require_gpu_provider_runtime_rejects_local_execution(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Local execution is disabled"):
                require_gpu_provider_runtime()


if __name__ == "__main__":
    unittest.main()
