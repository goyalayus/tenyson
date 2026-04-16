import contextlib
from datetime import datetime, timedelta, timezone
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tenyson.cloud.manager import CloudManager
from tenyson.cloud.modal import (
    ActiveModalLaunch,
    GitRepoSource,
    ModalManager,
    _bind_modal_run_remote,
    _build_modal_launcher_command,
    _build_clone_repo_command,
    _finish_local_failed_run_record,
    _modal_run_remote,
    _normalize_modal_python_version,
    _normalize_git_clone_url,
    _resolve_modal_python_version,
    _run_subprocess_with_streaming_logs,
    _wait_for_modal_function_call,
    _wait_for_run_start_or_terminal_result,
    _write_temp_config_payload,
)
from tenyson.cloud.runtime_deps import (
    REMOTE_RUNTIME_PACKAGES,
    resolve_runtime_dependency_profile,
    runtime_pip_install_command,
)
from tenyson.jobs.eval import EvalJob
from tenyson.jobs.sft import SFTJob
from tenyson.loader import load_task


class ModalManagerEnvTests(unittest.TestCase):
    def test_from_env_uses_expected_env_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            manager = ModalManager.from_env()

        self.assertEqual(manager.gpu, "A100")
        self.assertEqual(manager.timeout, 86400)
        self.assertTrue(manager.auto_terminate)
        self.assertFalse(manager.serialized)

    def test_from_env_reads_modal_specific_settings(self) -> None:
        env = {
            "TENYSON_MODAL_GPU": "A100-80GB",
            "TENYSON_MODAL_TIMEOUT": "7200",
        }
        with patch.dict(os.environ, env, clear=True):
            manager = ModalManager.from_env()

        self.assertEqual(manager.gpu, "A100-80GB")
        self.assertEqual(manager.timeout, 7200)

    def test_from_env_reads_serialized_override(self) -> None:
        with patch.dict(
            os.environ,
            {"TENYSON_MODAL_SERIALIZED": "true"},
            clear=True,
        ):
            manager = ModalManager.from_env()

        self.assertTrue(manager.serialized)

    def test_from_env_reads_python_version_override(self) -> None:
        with patch.dict(
            os.environ,
            {"TENYSON_MODAL_PYTHON_VERSION": "3.12"},
            clear=True,
        ):
            manager = ModalManager.from_env()

        self.assertEqual(manager.python_version, "3.12")

    def test_from_env_rejects_invalid_python_version(self) -> None:
        with patch.dict(
            os.environ,
            {"TENYSON_MODAL_PYTHON_VERSION": "three.twelve"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "Modal Python version"):
                ModalManager.from_env()

    def test_resolve_local_project_root_prefers_explicit_override(self) -> None:
        manager = ModalManager()
        with patch.dict(
            os.environ,
            {"TENYSON_LOCAL_PROJECT_ROOT": "/tmp/tenyson-root"},
            clear=True,
        ):
            self.assertEqual(manager._resolve_local_project_root(), "/tmp/tenyson-root")

    def test_from_env_rejects_invalid_timeout(self) -> None:
        with patch.dict(
            os.environ, {"TENYSON_MODAL_TIMEOUT": "not-an-int"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "TENYSON_MODAL_TIMEOUT"):
                ModalManager.from_env()

    def test_resolve_modal_python_version_defaults_to_3_12_on_python_3_10(self) -> None:
        fake_version = SimpleNamespace(major=3, minor=10)
        with patch.dict(os.environ, {}, clear=True), patch(
            "tenyson.cloud.modal.sys.version_info",
            fake_version,
        ):
            self.assertEqual(_resolve_modal_python_version(), "3.12")

    def test_resolve_modal_python_version_uses_current_minor_when_supported(self) -> None:
        fake_version = SimpleNamespace(major=3, minor=12)
        with patch.dict(os.environ, {}, clear=True), patch(
            "tenyson.cloud.modal.sys.version_info",
            fake_version,
        ):
            self.assertEqual(_resolve_modal_python_version(), "3.12")

    def test_normalize_modal_python_version_rejects_invalid_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "Modal Python version"):
            _normalize_modal_python_version("3")

    def test_factory_from_env_freezes_git_source_across_managers(self) -> None:
        status_calls = 0

        def fake_git(repo_root: str, *args: str) -> str:
            nonlocal status_calls
            mapping = {
                ("remote", "get-url", "origin"): "https://github.com/goyalayus/tenyson.git",
                ("rev-parse", "HEAD"): "abc123",
                ("ls-remote", "origin"): "abc123\trefs/heads/main",
            }
            if args == ("status", "--porcelain", "--untracked-files=all"):
                status_calls += 1
                if status_calls > 1:
                    return " M README.md"
                return ""
            return mapping[args]

        with patch.dict(os.environ, {}, clear=True), patch.object(
            ModalManager,
            "_resolve_local_project_root",
            return_value="/repo",
        ), patch(
            "tenyson.cloud.modal._run_git_command",
            side_effect=fake_git,
        ):
            factory = ModalManager.factory_from_env()
            manager_one = factory()
            manager_two = factory()
            source_one = manager_one._resolve_git_source("/repo")
            source_two = manager_two._resolve_git_source("/repo")

        self.assertEqual(status_calls, 1)
        self.assertEqual(source_one, GitRepoSource("https://github.com/goyalayus/tenyson.git", "abc123"))
        self.assertEqual(source_two, source_one)

    def test_factory_from_env_uses_explicit_git_source_without_repo_probe(self) -> None:
        explicit_source = GitRepoSource(
            clone_url="https://github.com/example/repo.git",
            commit="abc123",
        )

        with patch.object(
            ModalManager,
            "_resolve_local_project_root",
            side_effect=AssertionError("should not probe repo root"),
        ):
            factory = ModalManager.factory_from_env(git_source=explicit_source)
            manager = factory()

        self.assertEqual(manager._resolve_git_source("/repo"), explicit_source)


class CloudManagerDefaultTests(unittest.TestCase):
    def test_cloud_manager_defaults_to_modal(self) -> None:
        manager = CloudManager()
        self.assertIsInstance(manager, ModalManager)
        self.assertEqual(manager.gpu, "A100")


class ModalTaskSpecTests(unittest.TestCase):
    def test_resolve_task_spec_prefers_repo_relative_file_for_loaded_task(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        task_path = repo_root / "examples" / "wordle" / "functional.py"
        task = load_task(str(task_path))

        manager = ModalManager()
        task_spec = manager._resolve_task_spec(task, str(repo_root))

        self.assertEqual(task_spec, "examples/wordle/functional.py")


class ModalFunctionOptionsTests(unittest.TestCase):
    def test_resolve_modal_gpu_request_supports_t4(self) -> None:
        manager = ModalManager(gpu="T4")

        self.assertEqual(manager._resolve_modal_gpu_request(), "T4")

    def test_run_remote_options_do_not_serialize_by_default(self) -> None:
        manager = ModalManager()

        options = manager._build_run_remote_function_options(
            image="fake-image",
            gpu_request="A100-40GB",
            secrets=[],
        )

        self.assertFalse(options["serialized"])
        self.assertEqual(options["gpu"], "A100-40GB")
        self.assertEqual(options["timeout"], 86400)

    def test_run_remote_options_preserve_serialized_override(self) -> None:
        manager = ModalManager(serialized=True)

        options = manager._build_run_remote_function_options(
            image="fake-image",
            gpu_request="A100-40GB",
            secrets=[],
        )

        self.assertTrue(options["serialized"])

    def test_bind_modal_run_remote_uses_module_scope_function(self) -> None:
        captured: dict[str, object] = {}

        class FakeApp:
            def function(self, **options: object):
                captured["options"] = options

                def decorator(fn: object) -> object:
                    captured["function"] = fn
                    return fn

                return decorator

        bound = _bind_modal_run_remote(FakeApp(), {"serialized": False})

        self.assertIs(bound, _modal_run_remote)
        self.assertIs(captured["function"], _modal_run_remote)
        self.assertEqual(captured["options"], {"serialized": False})

    def test_build_modal_launcher_command_passes_expected_flags(self) -> None:
        command = _build_modal_launcher_command(
            python_executable="/usr/bin/python3",
            job_type="rl",
            config_path="/tmp/job.yaml",
            task_spec="examples/wordle/functional.py",
            gpu="A100",
            timeout=7200,
            serialized=False,
        )

        self.assertEqual(
            command,
            [
                "/usr/bin/python3",
                "-m",
                "tenyson.cloud.modal_launcher",
                "--job-type",
                "rl",
                "--config",
                "/tmp/job.yaml",
                "--task-spec",
                "examples/wordle/functional.py",
                "--gpu",
                "A100",
                "--timeout",
                "7200",
                "--serialized",
                "false",
            ],
        )

    def test_write_temp_config_payload_round_trips_yaml(self) -> None:
        path = _write_temp_config_payload("eval", {"training": {"steps": 4}})
        try:
            self.assertTrue(path.endswith(".yaml"))
            self.assertIn(tempfile.gettempdir(), path)
            self.assertIn("steps: 4", Path(path).read_text(encoding="utf-8"))
        finally:
            Path(path).unlink(missing_ok=True)


class ModalGitSourceTests(unittest.TestCase):
    def test_clone_repo_command_is_single_line_shell_safe(self) -> None:
        command = _build_clone_repo_command()
        self.assertIn("python3 -c", command)
        self.assertNotIn("\n", command)

    def test_normalize_git_clone_url_converts_github_ssh(self) -> None:
        clone_url = _normalize_git_clone_url("git@github.com:goyalayus/tenyson.git")
        self.assertEqual(clone_url, "https://github.com/goyalayus/tenyson.git")

    def test_resolve_git_source_uses_clean_head_commit(self) -> None:
        manager = ModalManager()

        def fake_git(repo_root: str, *args: str) -> str:
            mapping = {
                ("remote", "get-url", "origin"): "git@github.com:goyalayus/tenyson.git",
                ("status", "--porcelain", "--untracked-files=all"): "",
                ("rev-parse", "HEAD"): "abc123",
                ("ls-remote", "origin"): "abc123\trefs/heads/main",
            }
            return mapping[args]

        with patch("tenyson.cloud.modal._run_git_command", side_effect=fake_git), patch.dict(
            os.environ, {}, clear=True
        ):
            source = manager._resolve_git_source("/repo")

        self.assertEqual(source.clone_url, "https://github.com/goyalayus/tenyson.git")
        self.assertEqual(source.commit, "abc123")

    def test_resolve_git_source_pushes_clean_unpushed_head_to_snapshot_ref(self) -> None:
        manager = ModalManager()

        def fake_git(repo_root: str, *args: str) -> str:
            mapping = {
                ("remote", "get-url", "origin"): "https://github.com/goyalayus/tenyson.git",
                ("status", "--porcelain", "--untracked-files=all"): "",
                ("rev-parse", "HEAD"): "abc123",
                ("ls-remote", "origin"): "def456\trefs/heads/main",
            }
            return mapping[args]

        with patch(
            "tenyson.cloud.modal._run_git_command",
            side_effect=fake_git,
        ), patch(
            "tenyson.cloud.modal._push_commit_to_remote_ref",
        ) as push_commit, patch.dict(os.environ, {}, clear=True):
            source = manager._resolve_git_source("/repo")

        push_commit.assert_called_once_with(
            "/repo",
            remote_name="origin",
            commit="abc123",
            remote_ref="refs/heads/tenyson/runs/abc123",
        )
        self.assertEqual(source.clone_url, "https://github.com/goyalayus/tenyson.git")
        self.assertEqual(source.commit, "abc123")

    def test_resolve_git_source_allows_ignorable_tracked_docs_changes(self) -> None:
        manager = ModalManager()

        def fake_git(repo_root: str, *args: str) -> str:
            mapping = {
                ("remote", "get-url", "origin"): "https://github.com/goyalayus/tenyson.git",
                (
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ): " M README.md\n M docs/probe.md",
                ("rev-parse", "HEAD"): "abc123",
                ("ls-remote", "origin"): "abc123\trefs/heads/main",
            }
            return mapping[args]

        with patch("tenyson.cloud.modal._run_git_command", side_effect=fake_git), patch.dict(
            os.environ, {}, clear=True
        ):
            source = manager._resolve_git_source("/repo")

        self.assertEqual(source.clone_url, "https://github.com/goyalayus/tenyson.git")
        self.assertEqual(source.commit, "abc123")

    def test_resolve_git_source_snapshots_dirty_tracked_changes(self) -> None:
        manager = ModalManager()

        def fake_git(repo_root: str, *args: str) -> str:
            mapping = {
                ("remote", "get-url", "origin"): "https://github.com/goyalayus/tenyson.git",
                (
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ): " M README.md\n M src/tenyson/cloud/modal.py",
                ("rev-parse", "HEAD"): "abc123",
            }
            return mapping[args]

        with patch(
            "tenyson.cloud.modal._run_git_command",
            side_effect=fake_git,
        ), patch(
            "tenyson.cloud.modal._create_snapshot_commit",
            return_value="snap123",
        ) as create_snapshot, patch(
            "tenyson.cloud.modal._push_commit_to_remote_ref",
        ) as push_commit, patch.dict(os.environ, {}, clear=True):
            source = manager._resolve_git_source("/repo")

        create_snapshot.assert_called_once_with("/repo", parent_commit="abc123")
        push_commit.assert_called_once_with(
            "/repo",
            remote_name="origin",
            commit="snap123",
            remote_ref="refs/heads/tenyson/runs/snap123",
        )
        self.assertEqual(source.clone_url, "https://github.com/goyalayus/tenyson.git")
        self.assertEqual(source.commit, "snap123")

    def test_resolve_git_source_snapshots_untracked_code_files(self) -> None:
        manager = ModalManager()

        def fake_git(repo_root: str, *args: str) -> str:
            mapping = {
                ("remote", "get-url", "origin"): "https://github.com/goyalayus/tenyson.git",
                ("status", "--porcelain", "--untracked-files=all"): "?? src/new_helper.py",
                ("rev-parse", "HEAD"): "abc123",
            }
            return mapping[args]

        with patch(
            "tenyson.cloud.modal._run_git_command",
            side_effect=fake_git,
        ), patch(
            "tenyson.cloud.modal._create_snapshot_commit",
            return_value="snap456",
        ), patch(
            "tenyson.cloud.modal._push_commit_to_remote_ref",
        ) as push_commit, patch.dict(os.environ, {}, clear=True):
            source = manager._resolve_git_source("/repo")

        push_commit.assert_called_once_with(
            "/repo",
            remote_name="origin",
            commit="snap456",
            remote_ref="refs/heads/tenyson/runs/snap456",
        )
        self.assertEqual(source.commit, "snap456")

    def test_resolve_git_source_uses_frozen_env_snapshot_without_git_commands(self) -> None:
        manager = ModalManager()
        env = {
            "TENYSON_GIT_SOURCE_FROZEN": "1",
            "TENYSON_GIT_REPO_URL": "git@github.com:goyalayus/tenyson.git",
            "TENYSON_GIT_COMMIT": "abc123",
        }

        with patch.dict(os.environ, env, clear=True), patch(
            "tenyson.cloud.modal._run_git_command",
            side_effect=AssertionError("git should not be queried"),
        ):
            source = manager._resolve_git_source("/repo")

        self.assertEqual(source.clone_url, "https://github.com/goyalayus/tenyson.git")
        self.assertEqual(source.commit, "abc123")


class ModalGitSnapshotIntegrationTests(unittest.TestCase):
    def _init_repo_pair(self):
        remote_dir = tempfile.TemporaryDirectory()
        repo_dir = tempfile.TemporaryDirectory()
        remote_path = Path(remote_dir.name)
        repo_path = Path(repo_dir.name)

        subprocess.run(["git", "init", "--bare", str(remote_path)], check=True)
        subprocess.run(["git", "init", str(repo_path)], check=True)
        subprocess.run(
            ["git", "-C", str(repo_path), "config", "user.name", "Tenyson Tests"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_path), "config", "user.email", "tests@example.com"],
            check=True,
        )
        (repo_path / "app.py").write_text("print('initial')\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo_path), "add", "app.py"], check=True)
        subprocess.run(
            ["git", "-C", str(repo_path), "commit", "-m", "initial"],
            check=True,
        )
        subprocess.run(["git", "-C", str(repo_path), "branch", "-M", "main"], check=True)
        subprocess.run(
            ["git", "-C", str(repo_path), "remote", "add", "origin", str(remote_path)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_path), "push", "-u", "origin", "main"],
            check=True,
        )
        return remote_dir, repo_dir, remote_path, repo_path

    def test_snapshot_commit_captures_untracked_code_without_touching_head(self) -> None:
        remote_dir, repo_dir, remote_path, repo_path = self._init_repo_pair()
        self.addCleanup(remote_dir.cleanup)
        self.addCleanup(repo_dir.cleanup)

        (repo_path / "src").mkdir()
        (repo_path / "src" / "new_helper.py").write_text(
            "VALUE = 7\n",
            encoding="utf-8",
        )
        original_head = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        manager = ModalManager()
        with patch.dict(
            os.environ,
            {"TENYSON_GIT_REPO_URL": "https://example.invalid/repo.git"},
            clear=False,
        ):
            source = manager._resolve_git_source(str(repo_path))

        self.assertEqual(source.clone_url, "https://example.invalid/repo.git")
        self.assertNotEqual(source.commit, original_head)

        head_after = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(head_after, original_head)

        checkout_dir = tempfile.TemporaryDirectory()
        self.addCleanup(checkout_dir.cleanup)
        checkout_path = Path(checkout_dir.name)
        subprocess.run(
            ["git", "clone", str(remote_path), str(checkout_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout_path), "checkout", "--detach", source.commit],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            (checkout_path / "src" / "new_helper.py").read_text(encoding="utf-8"),
            "VALUE = 7\n",
        )


class ModalSubprocessStreamingTests(unittest.TestCase):
    def test_streaming_helper_forwards_output_and_keeps_error_tail(self) -> None:
        cmd = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('stdout-line', flush=True); "
                "print('stderr-line', file=sys.stderr, flush=True); "
                "raise SystemExit(3)"
            ),
        ]
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
            stderr_buffer
        ):
            returncode, details = _run_subprocess_with_streaming_logs(
                cmd,
                env=os.environ.copy(),
            )

        self.assertEqual(returncode, 3)
        self.assertIn("stdout-line", stdout_buffer.getvalue())
        self.assertIn("stderr-line", stderr_buffer.getvalue())
        self.assertIn("stderr-line", details)

    def test_streaming_helper_emits_on_line_and_terminates_on_interrupt(self) -> None:
        fake_process = SimpleNamespace(
            stdout=io.StringIO("first-line\n"),
            stderr=io.StringIO(""),
            wait=unittest.mock.Mock(side_effect=KeyboardInterrupt),
            terminate=unittest.mock.Mock(),
            kill=unittest.mock.Mock(),
        )
        seen_lines: list[str] = []

        with patch("tenyson.cloud.modal.subprocess.Popen", return_value=fake_process):
            with self.assertRaises(KeyboardInterrupt):
                _run_subprocess_with_streaming_logs(
                    [sys.executable, "-c", "print('hi')"],
                    env=os.environ.copy(),
                    on_line=seen_lines.append,
                )

        fake_process.terminate.assert_called_once_with()
        self.assertEqual(seen_lines, ["first-line"])

    def test_modal_launcher_runs_in_repo_subprocess(self) -> None:
        manager = ModalManager(gpu="A100", timeout=7200, serialized=True)

        with patch(
            "tenyson.cloud.modal._write_temp_config_payload",
            return_value="/tmp/fake-job.yaml",
        ), patch(
            "tenyson.cloud.modal._run_subprocess_with_streaming_logs",
            return_value=(0, ""),
        ) as run_subprocess, patch(
            "tenyson.cloud.modal.os.unlink"
        ) as unlink, patch.object(
            manager,
            "_resolve_git_source",
            return_value=GitRepoSource(
                clone_url="https://github.com/example/repo.git",
                commit="abc123",
            ),
        ):
            manager._run_modal_job_via_launcher(
                job_type="rl",
                config_payload={"training": {"steps": 4}},
                task_spec="examples/wordle/functional.py",
                local_project_root="/repo",
                backend_ref="wandb://demo/tenyson",
                experiment_id="wordle_exp",
                run_name="mixed_rl",
                attempt_token="attempt-123",
            )

        run_subprocess.assert_called_once()
        cmd = run_subprocess.call_args.args[0]
        kwargs = run_subprocess.call_args.kwargs
        self.assertIn("tenyson.cloud.modal_launcher", cmd)
        self.assertIsNone(kwargs["cwd"])
        self.assertTrue(kwargs["close_fds"])
        self.assertEqual(kwargs["env"]["PYTHONUNBUFFERED"], "1")
        self.assertEqual(kwargs["env"]["TENYSON_LOCAL_PROJECT_ROOT"], "/repo")
        self.assertEqual(
            kwargs["env"]["TENYSON_GIT_REPO_URL"],
            "https://github.com/example/repo.git",
        )
        self.assertEqual(kwargs["env"]["TENYSON_GIT_COMMIT"], "abc123")
        self.assertEqual(kwargs["env"]["TENYSON_GIT_SOURCE_FROZEN"], "1")
        self.assertTrue(kwargs["env"]["PYTHONPATH"].startswith("/repo/src"))
        unlink.assert_called_once_with("/tmp/fake-job.yaml")

    def test_modal_launcher_raises_with_subprocess_tail(self) -> None:
        manager = ModalManager()
        git_source = GitRepoSource(
            clone_url="https://github.com/example/repo.git",
            commit="abc123",
        )

        with patch(
            "tenyson.cloud.modal._write_temp_config_payload",
            return_value="/tmp/fake-job.yaml",
        ), patch(
            "tenyson.cloud.modal._run_subprocess_with_streaming_logs",
            return_value=(1, "boom"),
        ), patch("tenyson.cloud.modal.os.unlink"), patch.object(
            manager,
            "_resolve_git_source",
            return_value=git_source,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Modal launcher subprocess failed with code 1. boom",
            ):
                manager._run_modal_job_via_launcher(
                    job_type="eval",
                    config_payload={"eval": {"samples": 10}},
                    task_spec="examples/wordle/functional.py",
                    local_project_root="/repo",
                    backend_ref="wandb://demo/tenyson",
                    experiment_id="wordle_exp",
                    run_name="eval_baseline_mixed",
                    attempt_token="attempt-123",
                )

    def test_modal_launcher_keeps_active_launch_on_interrupt_and_close_requests_stop(self) -> None:
        manager = ModalManager()
        git_source = GitRepoSource(
            clone_url="https://github.com/example/repo.git",
            commit="abc123",
        )

        def fake_run_subprocess(*args, **kwargs):
            kwargs["on_line"](
                "[ModalManager] Spawned Modal app ap-live "
                "with function call fc-live."
            )
            raise KeyboardInterrupt

        with patch(
            "tenyson.cloud.modal._write_temp_config_payload",
            return_value="/tmp/fake-job.yaml",
        ), patch(
            "tenyson.cloud.modal._run_subprocess_with_streaming_logs",
            side_effect=fake_run_subprocess,
        ), patch("tenyson.cloud.modal.os.unlink"), patch.object(
            manager,
            "_resolve_git_source",
            return_value=git_source,
        ):
            with self.assertRaises(KeyboardInterrupt):
                manager._run_modal_job_via_launcher(
                    job_type="sft",
                    config_payload={"training": {"steps": 4}},
                    task_spec="examples/wordle/functional.py",
                    local_project_root="/repo",
                    backend_ref="wandb://demo/tenyson",
                    experiment_id="wordle_exp",
                    run_name="sft_main",
                    attempt_token="attempt-123",
                )

        launches = manager._snapshot_active_launches()
        self.assertEqual(len(launches), 1)
        launch = launches[0]
        self.assertEqual(launch.app_id, "ap-live")
        self.assertEqual(launch.function_call_id, "fc-live")
        self.assertEqual(launch.backend_ref, "wandb://demo/tenyson")
        self.assertEqual(launch.experiment_id, "wordle_exp")
        self.assertEqual(launch.phase, "sft")
        self.assertEqual(launch.run_name, "sft_main")
        self.assertEqual(launch.attempt_token, "attempt-123")
        self.assertIsNotNone(launch.launched_at)

        with patch("tenyson.cloud.modal.request_stop", return_value=True) as request_stop_mock, patch(
            "tenyson.core.telemetry.TelemetryClient",
            return_value=object(),
        ) as client_mock, patch(
            "tenyson.core.telemetry.wait_for_run_result",
            return_value=({}, {"status": "stopped"}),
        ) as wait_mock, patch("tenyson.cloud.modal.subprocess.run") as stop_app_mock:
            manager.close()

        request_stop_mock.assert_called_once_with(
            db_url="wandb://demo/tenyson",
            run_id="sft_main",
            experiment_id="wordle_exp",
            phase="sft",
            attempt_token="attempt-123",
            create_if_missing=True,
        )
        client_mock.assert_called_once_with(db_url="wandb://demo/tenyson")
        wait_mock.assert_called_once_with(
            client=client_mock.return_value,
            experiment_id="wordle_exp",
            run_id="sft_main",
            phase="sft",
            timeout_seconds=90.0,
            poll_interval_seconds=2.0,
            attempt_token="attempt-123",
            min_attempt_updated_at=launch.launched_at,
            include_results_payload=False,
        )
        stop_app_mock.assert_not_called()
        self.assertEqual(manager._snapshot_active_launches(), [])

    def test_modal_launcher_keeps_active_launch_after_successful_spawn(self) -> None:
        manager = ModalManager()
        git_source = GitRepoSource(
            clone_url="https://github.com/example/repo.git",
            commit="abc123",
        )

        def fake_run_subprocess(*args, **kwargs):
            kwargs["on_line"](
                "[ModalManager] Spawned detached Modal app ap-live "
                "with function call fc-live."
            )
            return (0, "")

        with patch(
            "tenyson.cloud.modal._write_temp_config_payload",
            return_value="/tmp/fake-job.yaml",
        ), patch(
            "tenyson.cloud.modal._run_subprocess_with_streaming_logs",
            side_effect=fake_run_subprocess,
        ), patch("tenyson.cloud.modal.os.unlink"), patch.object(
            manager,
            "_resolve_git_source",
            return_value=git_source,
        ):
            manager._run_modal_job_via_launcher(
                job_type="eval",
                config_payload={"evaluation": {"samples": 4}},
                task_spec="examples/wordle/functional.py",
                local_project_root="/repo",
                backend_ref="wandb://demo/tenyson",
                experiment_id="wordle_exp",
                run_name="eval_baseline_mixed",
                attempt_token="attempt-123",
            )

        launches = manager._snapshot_active_launches()
        self.assertEqual(len(launches), 1)
        launch = launches[0]
        self.assertEqual(launch.app_id, "ap-live")
        self.assertEqual(launch.function_call_id, "fc-live")
        self.assertEqual(launch.backend_ref, "wandb://demo/tenyson")
        self.assertEqual(launch.experiment_id, "wordle_exp")
        self.assertEqual(launch.phase, "eval")
        self.assertEqual(launch.run_name, "eval_baseline_mixed")
        self.assertEqual(launch.attempt_token, "attempt-123")
        self.assertIsNotNone(launch.launched_at)

    def test_modal_close_hard_stops_app_after_grace_timeout(self) -> None:
        manager = ModalManager()
        manager._remember_active_launch(
            ActiveModalLaunch(
                app_id="ap-live",
                function_call_id="fc-live",
                backend_ref="wandb://demo/tenyson",
                experiment_id="wordle_exp",
                phase="rl",
                run_name="mixed_rl",
                attempt_token="attempt-123",
            )
        )

        with patch("tenyson.cloud.modal.request_stop", return_value=True), patch(
            "tenyson.core.telemetry.TelemetryClient",
            return_value=object(),
        ), patch(
            "tenyson.core.telemetry.wait_for_run_result",
            side_effect=TimeoutError("still running"),
        ), patch("tenyson.cloud.modal.subprocess.run") as stop_app_mock:
            manager.close()

        stop_app_mock.assert_called_once()
        stop_cmd = stop_app_mock.call_args.args[0]
        self.assertEqual(
            stop_cmd,
            [sys.executable, "-m", "modal", "app", "stop", "ap-live"],
        )
        self.assertEqual(manager._snapshot_active_launches(), [])

    def test_modal_close_waits_for_multiple_launches_in_parallel(self) -> None:
        manager = ModalManager()
        manager._remember_active_launch(
            ActiveModalLaunch(
                app_id="ap-live-1",
                function_call_id="fc-live-1",
                backend_ref="wandb://demo/tenyson",
                experiment_id="wordle_exp",
                phase="rl",
                run_name="mixed_rl",
                attempt_token="attempt-123",
            )
        )
        manager._remember_active_launch(
            ActiveModalLaunch(
                app_id="ap-live-2",
                function_call_id="fc-live-2",
                backend_ref="wandb://demo/tenyson",
                experiment_id="wordle_exp",
                phase="rl",
                run_name="curr_rl_t2",
                attempt_token="attempt-456",
            )
        )

        barrier = threading.Barrier(2, timeout=1.0)

        def fake_wait_for_run_result(**kwargs):
            barrier.wait()
            return {}, {"status": "stopped", "run_id": kwargs["run_id"]}

        with patch("tenyson.cloud.modal.request_stop", return_value=True), patch(
            "tenyson.core.telemetry.TelemetryClient",
            return_value=object(),
        ), patch(
            "tenyson.core.telemetry.wait_for_run_result",
            side_effect=fake_wait_for_run_result,
        ) as wait_mock, patch("tenyson.cloud.modal.subprocess.run") as stop_app_mock:
            manager.close()

        self.assertEqual(wait_mock.call_count, 2)
        stop_app_mock.assert_not_called()
        self.assertEqual(manager._snapshot_active_launches(), [])


class ModalDetachedLaunchTests(unittest.TestCase):
    def test_wait_for_modal_function_call_retries_timeout_until_success(self) -> None:
        attempts: list[float] = []

        class FakeFunctionCall:
            def __init__(self) -> None:
                self._calls = 0

            def get(self, timeout=None):
                attempts.append(timeout)
                self._calls += 1
                if self._calls == 1:
                    raise TimeoutError("still running")
                return {"ok": True}

        fake_call = FakeFunctionCall()
        fake_modal = SimpleNamespace(
            FunctionCall=SimpleNamespace(from_id=lambda function_call_id: fake_call)
        )

        with patch.dict(sys.modules, {"modal": fake_modal}):
            _wait_for_modal_function_call(
                "fc-123",
                poll_timeout_seconds=0.25,
                overall_timeout_seconds=1.0,
            )

        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], 0.25)

    def test_wait_for_run_start_or_terminal_result_returns_terminal_result(self) -> None:
        with patch(
            "tenyson.core.telemetry.get_run_result",
            return_value=({}, {"status": "success", "run_id": "mixed_rl"}),
        ), patch("tenyson.core.telemetry.list_live_run_heartbeats") as live_mock:
            result = _wait_for_run_start_or_terminal_result(
                client=object(),
                experiment_id="wordle_exp",
                run_id="mixed_rl",
                phase="rl",
                timeout_seconds=5.0,
                attempt_token="attempt-123",
            )

        self.assertEqual(result, {"status": "success", "run_id": "mixed_rl"})
        live_mock.assert_not_called()

    def test_wait_for_run_start_or_terminal_result_returns_none_after_live_heartbeat(self) -> None:
        with patch(
            "tenyson.core.telemetry.get_run_result",
            return_value=None,
        ), patch(
            "tenyson.core.telemetry.list_live_run_heartbeats",
            return_value=[
                SimpleNamespace(
                    run_id="mixed_rl",
                    phase="rl",
                    attempt_token="attempt-123",
                )
            ],
        ):
            result = _wait_for_run_start_or_terminal_result(
                client=object(),
                experiment_id="wordle_exp",
                run_id="mixed_rl",
                phase="rl",
                timeout_seconds=5.0,
                attempt_token="attempt-123",
            )

        self.assertIsNone(result)

    def test_wait_for_run_start_or_terminal_result_ignores_stale_live_heartbeat(self) -> None:
        launch_started_at = datetime.now(timezone.utc)
        stale_heartbeat = launch_started_at - timedelta(seconds=60)
        fresh_heartbeat = launch_started_at + timedelta(seconds=1)

        with patch(
            "tenyson.core.telemetry.get_run_result",
            return_value=None,
        ), patch(
            "tenyson.core.telemetry.list_live_run_heartbeats",
            side_effect=[
                [
                    SimpleNamespace(
                        run_id="mixed_rl",
                        phase="rl",
                        attempt_token="attempt-123",
                        updated_at=stale_heartbeat,
                        created_at=stale_heartbeat,
                    )
                ],
                [
                    SimpleNamespace(
                        run_id="mixed_rl",
                        phase="rl",
                        attempt_token="attempt-123",
                        updated_at=fresh_heartbeat,
                        created_at=fresh_heartbeat,
                    )
                ],
            ],
        ):
            result = _wait_for_run_start_or_terminal_result(
                client=object(),
                experiment_id="wordle_exp",
                run_id="mixed_rl",
                phase="rl",
                timeout_seconds=5.0,
                poll_interval_seconds=0.01,
                attempt_token="attempt-123",
                min_attempt_updated_at=launch_started_at,
            )

        self.assertIsNone(result)

    def test_run_modal_job_returns_after_detached_spawn_without_polling_function_call(self) -> None:
        captured: dict[str, object] = {}

        class FakeImageBuilder:
            def run_commands(self, *args, **kwargs):
                return self

            def env(self, *args, **kwargs):
                return self

        class FakeSecret:
            @staticmethod
            def from_dict(value):
                return {"secret": dict(value)}

        class FakeEnableOutput:
            def __enter__(self):
                captured["enable_output_entered"] = True
                return self

            def __exit__(self, exc_type, exc, tb):
                captured["enable_output_exited"] = True
                return False

        class FakeBoundFunction:
            def spawn(self, *args):
                captured["spawn_args"] = args
                return SimpleNamespace(object_id="fc-123")

        class FakeRunContext:
            def __init__(self, app):
                self.app = app

            def __enter__(self):
                captured["run_entered"] = True
                return self.app

            def __exit__(self, exc_type, exc, tb):
                captured["run_exited"] = True
                return False

        class FakeApp:
            def __init__(self, description):
                captured["app_description"] = description
                self.app_id = "ap-test"

            def function(self, **options):
                captured["function_options"] = options

                def decorator(fn):
                    captured["decorated_function"] = fn
                    return FakeBoundFunction()

                return decorator

            def run(self, **kwargs):
                captured["run_kwargs"] = kwargs
                return FakeRunContext(self)

        def fake_debian_slim(*, python_version):
            captured["python_version"] = python_version
            return FakeImageBuilder()

        fake_modal = SimpleNamespace(
            App=FakeApp,
            Image=SimpleNamespace(debian_slim=fake_debian_slim),
            Secret=FakeSecret,
            enable_output=lambda: FakeEnableOutput(),
        )
        manager = ModalManager(timeout=7200)

        with patch.dict(sys.modules, {"modal": fake_modal}), patch(
            "tenyson.cloud.modal.runtime_pip_install_command",
            return_value="python3 -m pip install unsloth vllm",
        ), patch.object(
            manager,
            "_resolve_local_project_root",
            return_value="/repo",
        ), patch.object(
            manager,
            "_resolve_git_source",
            return_value=SimpleNamespace(
                clone_url="https://github.com/example/repo.git",
                commit="abc123",
            ),
        ), patch("tenyson.cloud.modal._wait_for_modal_function_call") as wait_mock:
            manager._run_modal_job(
                job_type="rl",
                config_payload={"training": {"steps": 4}},
                task_spec="examples/wordle/functional.py",
            )

        self.assertTrue(captured["enable_output_entered"])
        self.assertEqual(captured["run_kwargs"], {"detach": True})
        self.assertTrue(captured["run_exited"])
        self.assertEqual(
            captured["spawn_args"],
            ("rl", {"training": {"steps": 4}}, "examples/wordle/functional.py"),
        )
        self.assertEqual(captured["python_version"], manager.python_version)
        wait_mock.assert_not_called()


class RuntimeDependencyTests(unittest.TestCase):
    def test_remote_runtime_packages_include_unsloth_and_vllm(self) -> None:
        self.assertIn("unsloth", REMOTE_RUNTIME_PACKAGES)
        self.assertIn("vllm", REMOTE_RUNTIME_PACKAGES)

    def test_runtime_pip_install_command_includes_core_gpu_runtime_deps(self) -> None:
        command = runtime_pip_install_command()
        self.assertIn("unsloth", command)
        self.assertIn("vllm", command)

    def test_runtime_pip_install_command_t4_profile_uses_colab_compat_stack(self) -> None:
        command = runtime_pip_install_command(profile="modal_t4_colab_compat")
        self.assertIn("uv pip install --system", command)
        self.assertIn("vllm==0.9.2", command)
        self.assertIn("triton==3.2.0", command)
        self.assertIn("python3 -m pip uninstall -y", command)

    def test_runtime_pip_install_command_rejects_unknown_profile(self) -> None:
        with self.assertRaises(ValueError):
            runtime_pip_install_command(profile="unknown-profile")

    def test_resolve_runtime_dependency_profile_uses_t4_env(self) -> None:
        with patch.dict(os.environ, {"TENYSON_MODAL_GPU": "T4"}, clear=True):
            self.assertEqual(
                resolve_runtime_dependency_profile(),
                "modal_t4_colab_compat",
            )

    def test_resolve_runtime_dependency_profile_prefers_explicit_gpu(self) -> None:
        with patch.dict(os.environ, {"TENYSON_MODAL_GPU": "T4"}, clear=True):
            self.assertEqual(
                resolve_runtime_dependency_profile(gpu="A100"),
                "default",
            )

    def test_modal_manager_resolves_t4_runtime_profile(self) -> None:
        self.assertEqual(
            ModalManager(gpu="T4")._resolve_runtime_dependency_profile(),
            "modal_t4_colab_compat",
        )
        self.assertEqual(
            ModalManager(gpu="A100")._resolve_runtime_dependency_profile(),
            "default",
        )


class ModalManagerRunTests(unittest.TestCase):
    def test_finish_local_failed_run_record_finishes_matching_active_run(self) -> None:
        fake_run = SimpleNamespace(id="expected-run-id")
        fake_wandb = SimpleNamespace(finish=lambda: None)

        with patch(
            "tenyson.cloud.modal.wandb_store.active_run",
            return_value=fake_run,
        ), patch(
            "tenyson.cloud.modal.wandb_store.build_run_id",
            return_value="expected-run-id",
        ), patch.dict(sys.modules, {"wandb": fake_wandb}), patch.object(
            fake_wandb,
            "finish",
        ) as finish_mock:
            _finish_local_failed_run_record(
                experiment_id="wordle_exp",
                phase="eval",
                run_name="eval_baseline_mixed",
                attempt_token="attempt-123",
            )

        finish_mock.assert_called_once_with()

    def test_run_fails_locally_when_hf_preflight_fails_before_modal_launch(self) -> None:
        manager = ModalManager()
        job = SFTJob(
            {
                "telemetry": {
                    "entity": "ayush",
                    "project": "wordle",
                    "experiment_id": "wordle_exp",
                },
                "training": {
                    "run_name": "sft_main",
                    "hf_repo_base": "org/base",
                },
            },
            task=SimpleNamespace(),
        )

        with patch.object(
            manager,
            "_resolve_local_project_root",
            return_value="/repo",
        ), patch.object(
            manager,
            "_resolve_task_spec",
            return_value="examples/wordle/functional.py",
        ), patch(
            "tenyson.cloud.modal.preflight_cloud_hf_push",
            side_effect=ValueError("HF_TOKEN must be exported before launch."),
        ), patch.object(
            manager,
            "_run_modal_job_via_launcher",
        ) as launch_mock, patch(
            "tenyson.core.telemetry.TelemetryClient",
            return_value=object(),
        ), patch(
            "tenyson.core.telemetry.record_run_summary"
        ) as record_summary_mock, patch(
            "tenyson.core.telemetry.record_run_result"
        ) as record_result_mock, patch(
            "tenyson.cloud.modal._finish_local_failed_run_record"
        ) as finish_run_mock:
            result = manager.run(job)

        self.assertEqual(result.status, "failed")
        self.assertIn("HF_TOKEN", str(result.failure_reason))
        launch_mock.assert_not_called()
        record_summary_mock.assert_called_once()
        record_result_mock.assert_called_once()
        finish_run_mock.assert_called_once_with(
            experiment_id="wordle_exp",
            phase="sft",
            run_name="sft_main",
            attempt_token=None,
        )

    def test_run_waits_for_job_result_without_fetching_results_payload(self) -> None:
        manager = ModalManager()
        manager._remember_active_launch(
            ActiveModalLaunch(
                app_id="ap-live",
                function_call_id="fc-live",
                backend_ref="wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="eval",
                run_name="eval_baseline_mixed",
                attempt_token="attempt-123",
            )
        )
        job = EvalJob(
            {
                "telemetry": {
                    "entity": "ayush",
                    "project": "wordle",
                    "experiment_id": "wordle_exp",
                    "attempt_token": "attempt-123",
                },
                "evaluation": {
                    "run_name": "eval_baseline_mixed",
                },
            },
            task=SimpleNamespace(),
        )

        with patch.object(
            manager,
            "_resolve_local_project_root",
            return_value="/repo",
        ), patch.object(
            manager,
            "_resolve_task_spec",
            return_value="examples/wordle/functional.py",
        ), patch.object(
            manager,
            "_run_modal_job_via_launcher",
            return_value=None,
        ), patch(
            "tenyson.core.telemetry.TelemetryClient",
            return_value=object(),
        ), patch(
            "tenyson.core.telemetry.get_run_result",
            return_value=None,
        ), patch(
            "tenyson.core.telemetry.list_live_run_heartbeats",
            return_value=[
                SimpleNamespace(
                    run_id="eval_baseline_mixed",
                    phase="eval",
                    attempt_token="attempt-123",
                    updated_at=datetime.now(timezone.utc) + timedelta(seconds=60),
                    created_at=datetime.now(timezone.utc) + timedelta(seconds=60),
                )
            ],
        ), patch(
            "tenyson.core.telemetry.wait_for_run_result",
            return_value=(
                {},
                {
                    "run_id": "eval_baseline_mixed",
                    "status": "success",
                    "total_time_seconds": 1.0,
                },
            ),
        ) as wait_mock:
            result = manager.run(job)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.run_id, "eval_baseline_mixed")
        self.assertEqual(
            wait_mock.call_args.kwargs["timeout_seconds"],
            15,
        )
        self.assertIsNotNone(wait_mock.call_args.kwargs["min_attempt_updated_at"])
        self.assertEqual(
            wait_mock.call_args.kwargs["include_results_payload"],
            False,
        )
        self.assertEqual(manager._snapshot_active_launches(), [])

    def test_run_ignores_launch_status_poll_error_and_waits_for_terminal_result(self) -> None:
        manager = ModalManager()
        manager._remember_active_launch(
            ActiveModalLaunch(
                app_id="ap-live",
                function_call_id="fc-live",
                backend_ref="wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="eval",
                run_name="eval_baseline_mixed",
                attempt_token="attempt-123",
            )
        )
        job = EvalJob(
            {
                "telemetry": {
                    "entity": "ayush",
                    "project": "wordle",
                    "experiment_id": "wordle_exp",
                    "attempt_token": "attempt-123",
                },
                "evaluation": {
                    "run_name": "eval_baseline_mixed",
                },
            },
            task=SimpleNamespace(),
        )

        with patch.object(
            manager,
            "_resolve_local_project_root",
            return_value="/repo",
        ), patch.object(
            manager,
            "_resolve_task_spec",
            return_value="examples/wordle/functional.py",
        ), patch.object(
            manager,
            "_run_modal_job_via_launcher",
            return_value=None,
        ), patch(
            "tenyson.core.telemetry.TelemetryClient",
            return_value=object(),
        ), patch(
            "tenyson.cloud.modal._wait_for_run_start_or_terminal_result",
            return_value=None,
        ), patch(
            "tenyson.core.telemetry.list_live_run_heartbeats",
            side_effect=RuntimeError("graphql timeout"),
        ), patch(
            "tenyson.core.telemetry.get_run_result",
            return_value=None,
        ), patch(
            "tenyson.core.telemetry.wait_for_run_result",
            return_value=(
                {},
                {
                    "run_id": "eval_baseline_mixed",
                    "status": "success",
                    "total_time_seconds": 1.0,
                },
            ),
        ) as wait_mock, patch(
            "tenyson.core.telemetry.record_run_summary"
        ) as record_summary_mock, patch(
            "tenyson.core.telemetry.record_run_result"
        ) as record_result_mock:
            result = manager.run(job)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.run_id, "eval_baseline_mixed")
        self.assertEqual(wait_mock.call_count, 1)
        self.assertEqual(wait_mock.call_args.kwargs["include_results_payload"], False)
        record_summary_mock.assert_not_called()
        record_result_mock.assert_not_called()
        self.assertEqual(manager._snapshot_active_launches(), [])

    def test_run_recovers_terminal_result_after_launcher_teardown_crash(self) -> None:
        manager = ModalManager()
        job = EvalJob(
            {
                "telemetry": {
                    "entity": "ayush",
                    "project": "wordle",
                    "experiment_id": "wordle_exp",
                    "attempt_token": "attempt-123",
                },
                "evaluation": {
                    "run_name": "eval_baseline_mixed",
                },
            },
            task=SimpleNamespace(),
        )

        with patch.object(
            manager,
            "_resolve_local_project_root",
            return_value="/repo",
        ), patch.object(
            manager,
            "_resolve_task_spec",
            return_value="examples/wordle/functional.py",
        ), patch.object(
            manager,
            "_run_modal_job_via_launcher",
            side_effect=RuntimeError(
                "Tenyson job failed inside Modal with code -11. "
                "frame #6: c10::cuda::MemPool::~MemPool()"
            ),
        ), patch(
            "tenyson.core.telemetry.TelemetryClient",
            return_value=object(),
        ), patch(
            "tenyson.core.telemetry.wait_for_run_result",
            return_value=(
                {},
                {
                    "run_id": "eval_baseline_mixed",
                    "status": "success",
                    "total_time_seconds": 3.5,
                },
            ),
        ) as wait_mock, patch(
            "tenyson.core.telemetry.record_run_summary"
        ) as record_summary_mock, patch(
            "tenyson.core.telemetry.record_run_result"
        ) as record_result_mock:
            result = manager.run(job)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.run_id, "eval_baseline_mixed")
        self.assertEqual(wait_mock.call_count, 1)
        self.assertEqual(wait_mock.call_args.kwargs["timeout_seconds"], 20)
        self.assertIsNotNone(wait_mock.call_args.kwargs["min_attempt_updated_at"])
        record_summary_mock.assert_not_called()
        record_result_mock.assert_not_called()

    def test_run_recovers_terminal_result_when_attempt_token_matches(self) -> None:
        manager = ModalManager()
        job = EvalJob(
            {
                "telemetry": {
                    "entity": "ayush",
                    "project": "wordle",
                    "experiment_id": "wordle_exp",
                    "attempt_token": "attempt-123",
                },
                "evaluation": {
                    "run_name": "eval_baseline_mixed",
                },
            },
            task=SimpleNamespace(),
        )

        with patch.object(
            manager,
            "_resolve_local_project_root",
            return_value="/repo",
        ), patch.object(
            manager,
            "_resolve_task_spec",
            return_value="examples/wordle/functional.py",
        ), patch.object(
            manager,
            "_run_modal_job_via_launcher",
            side_effect=RuntimeError("Modal launcher subprocess failed with code 1."),
        ), patch(
            "tenyson.core.telemetry.TelemetryClient",
            return_value=object(),
        ), patch(
            "tenyson.core.telemetry.wait_for_run_result",
            return_value=(
                {},
                {
                    "run_id": "eval_baseline_mixed",
                    "status": "success",
                    "total_time_seconds": 4.0,
                    "attempt_token": "attempt-123",
                },
            ),
        ) as wait_mock, patch(
            "tenyson.core.telemetry.record_run_summary"
        ) as record_summary_mock, patch(
            "tenyson.core.telemetry.record_run_result"
        ) as record_result_mock:
            result = manager.run(job)

        self.assertEqual(result.status, "success")
        self.assertEqual(wait_mock.call_count, 1)
        self.assertEqual(wait_mock.call_args.kwargs["attempt_token"], "attempt-123")
        self.assertIsNotNone(wait_mock.call_args.kwargs["min_attempt_updated_at"])
        record_summary_mock.assert_not_called()
        record_result_mock.assert_not_called()

    def test_run_hands_off_to_live_telemetry_when_launcher_fails_after_detached_start(self) -> None:
        manager = ModalManager(timeout=60)
        manager._remember_active_launch(
            ActiveModalLaunch(
                app_id="ap-live",
                function_call_id="fc-live",
                backend_ref="wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="eval",
                run_name="eval_baseline_mixed",
                attempt_token="attempt-123",
            )
        )
        job = EvalJob(
            {
                "telemetry": {
                    "entity": "ayush",
                    "project": "wordle",
                    "experiment_id": "wordle_exp",
                    "attempt_token": "attempt-123",
                },
                "evaluation": {
                    "run_name": "eval_baseline_mixed",
                },
            },
            task=SimpleNamespace(),
        )

        with patch.object(
            manager,
            "_resolve_local_project_root",
            return_value="/repo",
        ), patch.object(
            manager,
            "_resolve_task_spec",
            return_value="examples/wordle/functional.py",
        ), patch.object(
            manager,
            "_run_modal_job_via_launcher",
            side_effect=RuntimeError("Modal launcher subprocess failed with code 1."),
        ), patch(
            "tenyson.core.telemetry.TelemetryClient",
            return_value=object(),
        ), patch(
            "tenyson.cloud.modal._wait_for_run_start_or_terminal_result",
            return_value=None,
        ) as handoff_mock, patch(
            "tenyson.core.telemetry.wait_for_run_result",
            side_effect=[
                TimeoutError("terminal result not written yet"),
                (
                    {},
                    {
                        "run_id": "eval_baseline_mixed",
                        "status": "success",
                        "total_time_seconds": 5.0,
                        "attempt_token": "attempt-123",
                    },
                ),
            ],
        ) as wait_mock, patch(
            "tenyson.core.telemetry.record_run_summary"
        ) as record_summary_mock, patch(
            "tenyson.core.telemetry.record_run_result"
        ) as record_result_mock:
            result = manager.run(job)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.run_id, "eval_baseline_mixed")
        self.assertEqual(handoff_mock.call_count, 1)
        self.assertEqual(wait_mock.call_count, 2)
        self.assertEqual(wait_mock.call_args_list[0].kwargs["timeout_seconds"], 20)
        self.assertEqual(wait_mock.call_args_list[0].kwargs["attempt_token"], "attempt-123")
        self.assertIsNotNone(
            wait_mock.call_args_list[0].kwargs["min_attempt_updated_at"]
        )
        self.assertEqual(wait_mock.call_args_list[1].kwargs["timeout_seconds"], 15)
        self.assertEqual(wait_mock.call_args_list[1].kwargs["attempt_token"], "attempt-123")
        self.assertIsNotNone(
            wait_mock.call_args_list[1].kwargs["min_attempt_updated_at"]
        )
        record_summary_mock.assert_not_called()
        record_result_mock.assert_not_called()
        self.assertEqual(manager._snapshot_active_launches(), [])

    def test_run_records_failure_when_teardown_crash_has_no_recoverable_result(self) -> None:
        manager = ModalManager()
        job = EvalJob(
            {
                "telemetry": {
                    "entity": "ayush",
                    "project": "wordle",
                    "experiment_id": "wordle_exp",
                    "attempt_token": "attempt-123",
                },
                "evaluation": {
                    "run_name": "eval_baseline_mixed",
                },
            },
            task=SimpleNamespace(),
        )

        with patch.object(
            manager,
            "_resolve_local_project_root",
            return_value="/repo",
        ), patch.object(
            manager,
            "_resolve_task_spec",
            return_value="examples/wordle/functional.py",
        ), patch.object(
            manager,
            "_run_modal_job_via_launcher",
            side_effect=RuntimeError(
                "Tenyson job failed inside Modal with code -11. "
                "frame #6: c10::cuda::MemPool::~MemPool()"
            ),
        ), patch(
            "tenyson.core.telemetry.TelemetryClient",
            return_value=object(),
        ), patch(
            "tenyson.core.telemetry.wait_for_run_result",
            side_effect=TimeoutError("run result missing"),
        ), patch(
            "tenyson.core.telemetry.record_run_summary"
        ) as record_summary_mock, patch(
            "tenyson.core.telemetry.record_run_result"
        ) as record_result_mock, patch(
            "tenyson.cloud.modal._finish_local_failed_run_record"
        ) as finish_run_mock:
            result = manager.run(job)

        self.assertEqual(result.status, "failed")
        self.assertIn("code -11", str(result.failure_reason))
        self.assertEqual(record_summary_mock.call_count, 1)
        self.assertEqual(record_result_mock.call_count, 1)
        finish_run_mock.assert_called_once_with(
            experiment_id="wordle_exp",
            phase="eval",
            run_name="eval_baseline_mixed",
            attempt_token="attempt-123",
        )

    def test_run_records_failure_when_detached_modal_call_crashes_after_handoff(self) -> None:
        manager = ModalManager(timeout=60)
        manager._remember_active_launch(
            ActiveModalLaunch(
                app_id="ap-live",
                function_call_id="fc-live",
                backend_ref="wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="eval",
                run_name="eval_baseline_mixed",
                attempt_token="attempt-123",
                launched_at=datetime.now(timezone.utc),
            )
        )
        job = EvalJob(
            {
                "telemetry": {
                    "entity": "ayush",
                    "project": "wordle",
                    "experiment_id": "wordle_exp",
                    "attempt_token": "attempt-123",
                },
                "evaluation": {
                    "run_name": "eval_baseline_mixed",
                },
            },
            task=SimpleNamespace(),
        )

        with patch.object(
            manager,
            "_resolve_local_project_root",
            return_value="/repo",
        ), patch.object(
            manager,
            "_resolve_task_spec",
            return_value="examples/wordle/functional.py",
        ), patch.object(
            manager,
            "_run_modal_job_via_launcher",
            return_value=None,
        ), patch(
            "tenyson.core.telemetry.TelemetryClient",
            return_value=object(),
        ), patch(
            "tenyson.cloud.modal._wait_for_run_start_or_terminal_result",
            return_value=None,
        ), patch(
            "tenyson.core.telemetry.list_live_run_heartbeats",
            return_value=[
                SimpleNamespace(
                    run_id="eval_baseline_mixed",
                    phase="eval",
                    attempt_token="attempt-123",
                    updated_at=datetime.now(timezone.utc) + timedelta(seconds=60),
                    created_at=datetime.now(timezone.utc) + timedelta(seconds=60),
                )
            ],
        ), patch(
            "tenyson.core.telemetry.get_run_result",
            return_value=None,
        ), patch(
            "tenyson.core.telemetry.wait_for_run_result",
            side_effect=[
                TimeoutError("terminal result not written yet"),
                TimeoutError("terminal result still missing"),
            ],
        ) as wait_mock, patch(
            "tenyson.cloud.modal._probe_modal_function_call",
            return_value=("failed", "worker crashed"),
        ) as probe_mock, patch(
            "tenyson.core.telemetry.record_run_summary"
        ) as record_summary_mock, patch(
            "tenyson.core.telemetry.record_run_result"
        ) as record_result_mock, patch(
            "tenyson.cloud.modal._finish_local_failed_run_record"
        ) as finish_run_mock:
            result = manager.run(job)

        self.assertEqual(result.status, "failed")
        self.assertIn("worker crashed", str(result.failure_reason))
        self.assertEqual(wait_mock.call_count, 2)
        self.assertEqual(
            wait_mock.call_args_list[0].kwargs["timeout_seconds"],
            15,
        )
        self.assertEqual(
            wait_mock.call_args_list[1].kwargs["timeout_seconds"],
            20,
        )
        probe_mock.assert_called_once_with("fc-live")
        self.assertEqual(record_summary_mock.call_count, 1)
        self.assertEqual(record_result_mock.call_count, 1)
        finish_run_mock.assert_called_once_with(
            experiment_id="wordle_exp",
            phase="eval",
            run_name="eval_baseline_mixed",
            attempt_token="attempt-123",
        )
        self.assertEqual(manager._snapshot_active_launches(), [])


if __name__ == "__main__":
    unittest.main()
