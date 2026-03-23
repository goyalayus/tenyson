import contextlib
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from tenyson.cloud.manager import CloudManager
from tenyson.cloud.modal import (
    ModalManager,
    _normalize_git_clone_url,
    _run_subprocess_with_streaming_logs,
)
from tenyson.cloud.runtime_deps import REMOTE_RUNTIME_PACKAGES, runtime_pip_install_command
from tenyson.loader import load_task


class ModalManagerEnvTests(unittest.TestCase):
    def test_from_env_uses_expected_env_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            manager = ModalManager.from_env()

        self.assertEqual(manager.gpu, "A100")
        self.assertEqual(manager.timeout, 86400)
        self.assertTrue(manager.auto_terminate)

    def test_from_env_reads_modal_specific_settings(self) -> None:
        env = {
            "TENYSON_MODAL_GPU": "A100-80GB",
            "TENYSON_MODAL_TIMEOUT": "7200",
        }
        with patch.dict(os.environ, env, clear=True):
            manager = ModalManager.from_env()

        self.assertEqual(manager.gpu, "A100-80GB")
        self.assertEqual(manager.timeout, 7200)

    def test_from_env_rejects_invalid_timeout(self) -> None:
        with patch.dict(
            os.environ, {"TENYSON_MODAL_TIMEOUT": "not-an-int"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "TENYSON_MODAL_TIMEOUT"):
                ModalManager.from_env()


class CloudManagerDefaultTests(unittest.TestCase):
    def test_cloud_manager_defaults_to_modal(self) -> None:
        manager = CloudManager()
        self.assertIsInstance(manager, ModalManager)
        self.assertEqual(manager.gpu, "A100")


class ModalTaskSpecTests(unittest.TestCase):
    def test_resolve_task_spec_prefers_repo_relative_file_for_loaded_task(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        task_path = repo_root / "examples" / "wordle" / "wordle_task.py"
        task = load_task(str(task_path))

        manager = ModalManager()
        task_spec = manager._resolve_task_spec(task, str(repo_root))

        self.assertEqual(task_spec, "examples/wordle/wordle_task.py")


class ModalGitSourceTests(unittest.TestCase):
    def test_normalize_git_clone_url_converts_github_ssh(self) -> None:
        clone_url = _normalize_git_clone_url("git@github.com:goyalayus/tenyson.git")
        self.assertEqual(clone_url, "https://github.com/goyalayus/tenyson.git")

    def test_resolve_git_source_uses_clean_head_commit(self) -> None:
        manager = ModalManager()

        def fake_git(repo_root: str, *args: str) -> str:
            mapping = {
                ("remote", "get-url", "origin"): "git@github.com:goyalayus/tenyson.git",
                ("status", "--porcelain", "--untracked-files=no"): "",
                ("rev-parse", "HEAD"): "abc123",
            }
            return mapping[args]

        with patch("tenyson.cloud.modal._run_git_command", side_effect=fake_git), patch.dict(
            os.environ, {}, clear=True
        ):
            source = manager._resolve_git_source("/repo")

        self.assertEqual(source.clone_url, "https://github.com/goyalayus/tenyson.git")
        self.assertEqual(source.commit, "abc123")

    def test_resolve_git_source_rejects_dirty_tracked_changes(self) -> None:
        manager = ModalManager()

        def fake_git(repo_root: str, *args: str) -> str:
            mapping = {
                ("remote", "get-url", "origin"): "https://github.com/goyalayus/tenyson.git",
                ("status", "--porcelain", "--untracked-files=no"): " M src/tenyson/cloud/modal.py",
            }
            return mapping[args]

        with patch("tenyson.cloud.modal._run_git_command", side_effect=fake_git), patch.dict(
            os.environ, {}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "git-backed only"):
                manager._resolve_git_source("/repo")


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


class RuntimeDependencyTests(unittest.TestCase):
    def test_remote_runtime_packages_include_unsloth_and_vllm(self) -> None:
        self.assertIn("unsloth", REMOTE_RUNTIME_PACKAGES)
        self.assertIn("vllm", REMOTE_RUNTIME_PACKAGES)

    def test_runtime_pip_install_command_includes_core_gpu_runtime_deps(self) -> None:
        command = runtime_pip_install_command()
        self.assertIn("unsloth", command)
        self.assertIn("vllm", command)


if __name__ == "__main__":
    unittest.main()
