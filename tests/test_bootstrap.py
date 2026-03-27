import os
import runpy
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tenyson.bootstrap import (
    ensure_local_controller_environment,
    is_truthy,
    load_env_file,
    missing_controller_packages,
    resolve_project_root,
)
from tenyson.core.run_config import shared_overrides_from_env


class BootstrapTests(unittest.TestCase):
    def test_is_truthy(self) -> None:
        self.assertTrue(is_truthy("1"))
        self.assertTrue(is_truthy("TRUE"))
        self.assertTrue(is_truthy(" yes "))
        self.assertFalse(is_truthy("0"))
        self.assertFalse(is_truthy("false"))
        self.assertFalse(is_truthy(None))

    def test_resolve_project_root_from_repo_file(self) -> None:
        project_root = resolve_project_root(__file__)
        self.assertTrue((project_root / "pyproject.toml").is_file())
        self.assertTrue((project_root / "src" / "tenyson").is_dir())

    def test_missing_controller_packages_uses_mapping(self) -> None:
        missing = missing_controller_packages(
            {
                "json": "json",
                "definitely_missing_module_xyz": "fake-pkg",
            }
        )
        self.assertIn("fake-pkg", missing)
        self.assertNotIn("json", missing)

    def test_ensure_local_controller_environment_respects_skip_env(self) -> None:
        with patch.dict(os.environ, {"TENYSON_SKIP_LOCAL_BOOTSTRAP": "true"}, clear=False):
            installed = ensure_local_controller_environment(anchor_file=__file__)
        self.assertEqual(list(installed), [])

    def test_load_env_file_parses_dotenv_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "# comment",
                        "TENYSON_EXPERIMENT_ID=exp_123",
                        "TENYSON_WANDB_ENTITY=ayush_g",
                        'TENYSON_WANDB_PROJECT="wordle-research"',
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                loaded = load_env_file(env_file)

        self.assertEqual(
            loaded,
            {
                "TENYSON_EXPERIMENT_ID": "exp_123",
                "TENYSON_WANDB_ENTITY": "ayush_g",
                "TENYSON_WANDB_PROJECT": "wordle-research",
            },
        )

    def test_load_env_file_ignores_shell_export_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "export TENYSON_EXPERIMENT_ID=exp_123",
                        "TENYSON_WANDB_ENTITY=ayush_g",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                loaded = load_env_file(env_file)

        self.assertEqual(loaded, {"TENYSON_WANDB_ENTITY": "ayush_g"})

    def test_load_env_file_preserves_existing_env_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "TENYSON_EXPERIMENT_ID=from_file\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"TENYSON_EXPERIMENT_ID": "from_shell"},
                clear=False,
            ):
                loaded = load_env_file(env_file)
                current = os.environ["TENYSON_EXPERIMENT_ID"]

        self.assertEqual(current, "from_shell")
        self.assertEqual(loaded, {})

    def test_load_env_file_can_override_existing_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "TENYSON_EXPERIMENT_ID=from_file\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"TENYSON_EXPERIMENT_ID": "from_shell"},
                clear=False,
            ):
                loaded = load_env_file(env_file, override=True)
                current = os.environ["TENYSON_EXPERIMENT_ID"]

        self.assertEqual(current, "from_file")
        self.assertEqual(loaded, {"TENYSON_EXPERIMENT_ID": "from_file"})

    def test_wordle_experiment_import_preserves_shell_env(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        experiment_path = repo_root / "examples" / "wordle" / "experiment.py"
        env_path = experiment_path.with_name(".env")
        original_is_file = Path.is_file
        original_read_text = Path.read_text

        def fake_is_file(path: Path) -> bool:
            if path == env_path:
                return True
            return original_is_file(path)

        def fake_read_text(path: Path, *args, **kwargs) -> str:
            if path == env_path:
                return (
                    "TENYSON_EXPERIMENT_ID=from_file\n"
                    "TENYSON_WANDB_ENTITY=from_file_entity\n"
                )
            return original_read_text(path, *args, **kwargs)

        patched_env = dict(os.environ)
        patched_env["TENYSON_EXPERIMENT_ID"] = "from_shell"
        patched_env.pop("TENYSON_WANDB_ENTITY", None)

        with patch(
            "tenyson.bootstrap.ensure_local_controller_environment",
            return_value=[],
        ):
            with patch.object(Path, "is_file", fake_is_file):
                with patch.object(Path, "read_text", fake_read_text):
                    with patch.dict(os.environ, patched_env, clear=True):
                        module_globals = runpy.run_path(
                            str(experiment_path),
                            run_name="__tenyson_wordle_env_test__",
                        )
                        experiment_id_after_import = os.environ["TENYSON_EXPERIMENT_ID"]
                        wandb_entity_after_import = os.environ.get("TENYSON_WANDB_ENTITY")
                        module_globals["_load_example_env"](env_path)
                        experiment_id_after_load = os.environ["TENYSON_EXPERIMENT_ID"]
                        wandb_entity_after_load = os.environ["TENYSON_WANDB_ENTITY"]

        self.assertEqual(experiment_id_after_import, "from_shell")
        self.assertIsNone(wandb_entity_after_import)
        self.assertEqual(experiment_id_after_load, "from_shell")
        self.assertEqual(wandb_entity_after_load, "from_file_entity")

    def test_wordle_smoke_identity_uses_isolated_defaults(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        experiment_path = repo_root / "examples" / "wordle" / "experiment.py"

        with patch(
            "tenyson.bootstrap.ensure_local_controller_environment",
            return_value=[],
        ):
            with patch.dict(
                os.environ,
                {
                    "TENYSON_WORDLE_SMOKE": "true",
                    "TENYSON_EXPERIMENT_ID": "from_file",
                    "TENYSON_HF_REPO_BASE": "org/wordle-lora",
                },
                clear=True,
            ):
                module_globals = runpy.run_path(
                    str(experiment_path),
                    run_name="__tenyson_wordle_smoke_default_test__",
                )
                module_globals["_configure_smoke_identity"](
                    base_dir=experiment_path.parent,
                    loaded_env={
                        "TENYSON_EXPERIMENT_ID": "from_file",
                        "TENYSON_HF_REPO_BASE": "org/wordle-lora",
                    },
                )
                experiment_id = os.environ["TENYSON_EXPERIMENT_ID"]
                hf_repo_base = os.environ["TENYSON_HF_REPO_BASE"]
                report_path = os.environ["TENYSON_WORDLE_REPORT_PATH"]

        self.assertTrue(experiment_id.startswith("wordle_smoke_"))
        self.assertNotEqual(experiment_id, "from_file")
        self.assertEqual(hf_repo_base, "org/wordle-lora-smoke")
        self.assertTrue(report_path.endswith(f"smoke_reports/{experiment_id}.md"))

    def test_wordle_smoke_identity_preserves_explicit_shell_targets(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        experiment_path = repo_root / "examples" / "wordle" / "experiment.py"

        with patch(
            "tenyson.bootstrap.ensure_local_controller_environment",
            return_value=[],
        ):
            with patch.dict(
                os.environ,
                {
                    "TENYSON_WORDLE_SMOKE": "true",
                    "TENYSON_EXPERIMENT_ID": "from_shell",
                    "TENYSON_HF_REPO_BASE": "org/custom-smoke",
                    "TENYSON_WORDLE_REPORT_PATH": "/tmp/custom-smoke-report.md",
                },
                clear=True,
            ):
                module_globals = runpy.run_path(
                    str(experiment_path),
                    run_name="__tenyson_wordle_smoke_shell_test__",
                )
                module_globals["_configure_smoke_identity"](
                    base_dir=experiment_path.parent,
                    loaded_env={},
                )
                experiment_id = os.environ["TENYSON_EXPERIMENT_ID"]
                hf_repo_base = os.environ["TENYSON_HF_REPO_BASE"]
                report_path = os.environ["TENYSON_WORDLE_REPORT_PATH"]

        self.assertEqual(experiment_id, "from_shell")
        self.assertEqual(hf_repo_base, "org/custom-smoke")
        self.assertEqual(report_path, "/tmp/custom-smoke-report.md")

    def test_wordle_experiment_sigterm_handler_raises_keyboardinterrupt(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        experiment_path = repo_root / "examples" / "wordle" / "experiment.py"

        with patch(
            "tenyson.bootstrap.ensure_local_controller_environment",
            return_value=[],
        ):
            module_globals = runpy.run_path(
                str(experiment_path),
                run_name="__tenyson_wordle_sigterm_test__",
            )

        with self.assertRaises(KeyboardInterrupt):
            module_globals["_graceful_shutdown_signal_handler"](signal.SIGTERM, None)

    def test_wordle_experiment_main_rebuilds_report_on_keyboardinterrupt(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        experiment_path = repo_root / "examples" / "wordle" / "experiment.py"

        with patch(
            "tenyson.bootstrap.ensure_local_controller_environment",
            return_value=[],
        ):
            module_globals = runpy.run_path(
                str(experiment_path),
                run_name="__tenyson_wordle_interrupt_test__",
            )

        close_mock = Mock()
        fake_task = object()
        fake_report = Mock()
        fake_branch = Mock()
        fake_branch.sft.return_value = "fake-sft-stage"
        fake_branch.run.side_effect = KeyboardInterrupt

        class FakeSession:
            def __init__(self, *args, **kwargs) -> None:
                del args, kwargs

            def create_cloud(self):
                return object()

            def branch(self, cloud=None):
                del cloud
                return fake_branch

            def close(self):
                close_mock()

        class FakeConfigTemplates:
            @staticmethod
            def from_directory(*args, **kwargs):
                del args, kwargs
                return object()

        class FakeModalManager:
            @staticmethod
            def factory_from_env(**kwargs):
                del kwargs
                return lambda: object()

        rebuild_mock = Mock()
        main_fn = module_globals["main"]
        main_globals = main_fn.__globals__
        main_globals["_install_graceful_shutdown_handlers"] = lambda: None
        main_globals["_load_example_env"] = lambda path=None: {}
        main_globals["load_task"] = lambda path: fake_task
        main_globals["ExperimentReport"] = lambda output_path: fake_report
        main_globals["_wordle_smoke_overrides"] = lambda: {}
        main_globals["shared_overrides_from_env"] = lambda: None
        main_globals["ConfigTemplates"] = FakeConfigTemplates
        main_globals["ModalManager"] = FakeModalManager
        main_globals["ExperimentSession"] = FakeSession
        main_globals["_rebuild_report_from_telemetry"] = rebuild_mock

        main_fn()

        close_mock.assert_called_once_with()
        rebuild_mock.assert_called_once_with(fake_report, fake_task)

    def test_wordle_experiment_main_ignores_keyboardinterrupt_during_report_rebuild(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        experiment_path = repo_root / "examples" / "wordle" / "experiment.py"

        with patch(
            "tenyson.bootstrap.ensure_local_controller_environment",
            return_value=[],
        ):
            module_globals = runpy.run_path(
                str(experiment_path),
                run_name="__tenyson_wordle_rebuild_interrupt_test__",
            )

        close_mock = Mock()
        fake_task = object()
        fake_report = Mock()
        fake_branch = Mock()
        fake_branch.sft.return_value = "fake-sft-stage"
        fake_branch.run.side_effect = KeyboardInterrupt

        class FakeSession:
            def __init__(self, *args, **kwargs) -> None:
                del args, kwargs

            def create_cloud(self):
                return object()

            def branch(self, cloud=None):
                del cloud
                return fake_branch

            def close(self):
                close_mock()

        class FakeConfigTemplates:
            @staticmethod
            def from_directory(*args, **kwargs):
                del args, kwargs
                return object()

        class FakeModalManager:
            @staticmethod
            def factory_from_env(**kwargs):
                del kwargs
                return lambda: object()

        main_fn = module_globals["main"]
        main_globals = main_fn.__globals__
        main_globals["_install_graceful_shutdown_handlers"] = lambda: None
        main_globals["_load_example_env"] = lambda path=None: {}
        main_globals["load_task"] = lambda path: fake_task
        main_globals["ExperimentReport"] = lambda output_path: fake_report
        main_globals["_wordle_smoke_overrides"] = lambda: {}
        main_globals["shared_overrides_from_env"] = lambda: None
        main_globals["ConfigTemplates"] = FakeConfigTemplates
        main_globals["ModalManager"] = FakeModalManager
        main_globals["ExperimentSession"] = FakeSession
        main_globals["_rebuild_report_from_telemetry"] = Mock(
            side_effect=KeyboardInterrupt
        )

        main_fn()

        close_mock.assert_called_once_with()

    def test_wordle_experiment_main_forces_exit_after_sigterm_shutdown(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        experiment_path = repo_root / "examples" / "wordle" / "experiment.py"

        with patch(
            "tenyson.bootstrap.ensure_local_controller_environment",
            return_value=[],
        ):
            module_globals = runpy.run_path(
                str(experiment_path),
                run_name="__tenyson_wordle_forced_exit_test__",
            )

        close_mock = Mock()
        fake_task = object()
        fake_report = Mock()
        fake_branch = Mock()
        fake_branch.sft.return_value = "fake-sft-stage"
        fake_branch.run.side_effect = KeyboardInterrupt

        class FakeSession:
            def __init__(self, *args, **kwargs) -> None:
                del args, kwargs

            def create_cloud(self):
                return object()

            def branch(self, cloud=None):
                del cloud
                return fake_branch

            def close(self):
                close_mock()

        class FakeConfigTemplates:
            @staticmethod
            def from_directory(*args, **kwargs):
                del args, kwargs
                return object()

        class FakeModalManager:
            @staticmethod
            def factory_from_env(**kwargs):
                del kwargs
                return lambda: object()

        rebuild_mock = Mock()
        main_fn = module_globals["main"]
        main_globals = main_fn.__globals__
        main_globals["_FORCED_STOP_REQUESTED"] = True
        main_globals["_install_graceful_shutdown_handlers"] = lambda: None
        main_globals["_load_example_env"] = lambda path=None: {}
        main_globals["load_task"] = lambda path: fake_task
        main_globals["ExperimentReport"] = lambda output_path: fake_report
        main_globals["_wordle_smoke_overrides"] = lambda: {}
        main_globals["shared_overrides_from_env"] = lambda: None
        main_globals["ConfigTemplates"] = FakeConfigTemplates
        main_globals["ModalManager"] = FakeModalManager
        main_globals["ExperimentSession"] = FakeSession
        main_globals["_rebuild_report_from_telemetry"] = rebuild_mock
        main_globals["_wait_for_wordle_live_runs_to_finish"] = lambda report, timeout_seconds: None

        with patch.object(main_globals["os"], "_exit", side_effect=SystemExit(0)) as exit_mock:
            with self.assertRaises(SystemExit):
                main_fn()

        close_mock.assert_called_once_with()
        rebuild_mock.assert_called_once_with(fake_report, fake_task)
        exit_mock.assert_called_once_with(0)

    def test_wordle_forced_stop_wait_exits_when_live_runs_clear(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        experiment_path = repo_root / "examples" / "wordle" / "experiment.py"

        with patch(
            "tenyson.bootstrap.ensure_local_controller_environment",
            return_value=[],
        ):
            module_globals = runpy.run_path(
                str(experiment_path),
                run_name="__tenyson_wordle_wait_live_runs_test__",
            )

        report = Mock()
        report.experiment_id = "wordle_exp"
        report.telemetry_backend_ref = "wandb://demo/tenyson"
        wait_fn = module_globals["_wait_for_wordle_live_runs_to_finish"]

        live_row = Mock(run_id="sft_main")
        with patch.dict(
            module_globals["os"].environ,
            {"TENYSON_EXPERIMENT_ID": "wordle_exp"},
            clear=False,
        ), patch.object(
            module_globals["time"],
            "sleep",
        ) as sleep_mock, patch.dict(
            wait_fn.__globals__,
            {"list_live_runs": Mock(side_effect=[[live_row], []])},
        ):
            wait_fn(report, timeout_seconds=5.0, poll_interval_seconds=0.1)

        sleep_mock.assert_called_once_with(0.1)


class SharedOverridesTests(unittest.TestCase):
    def test_shared_overrides_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TENYSON_HF_REPO_BASE": "org/repo",
                "TENYSON_EXPERIMENT_ID": "exp_123",
                "TENYSON_WANDB_ENTITY": "wandb-entity",
                "TENYSON_WANDB_PROJECT": "wandb-project",
            },
            clear=False,
        ):
            overrides = shared_overrides_from_env()

        self.assertEqual(
            overrides,
            {
                "training": {"hf_repo_base": "org/repo"},
                "telemetry": {
                    "experiment_id": "exp_123",
                    "entity": "wandb-entity",
                    "project": "wandb-project",
                },
            },
        )

    def test_shared_overrides_from_env_returns_none_when_empty(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TENYSON_HF_REPO_BASE": "",
                "TENYSON_EXPERIMENT_ID": "",
                "TENYSON_WANDB_ENTITY": "",
                "TENYSON_WANDB_PROJECT": "",
            },
            clear=False,
        ):
            overrides = shared_overrides_from_env()

        self.assertIsNone(overrides)


if __name__ == "__main__":
    unittest.main()
