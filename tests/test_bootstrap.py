import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
