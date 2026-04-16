import os
import runpy
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tenyson.bootstrap import (
    ensure_local_controller_environment,
    is_truthy,
    load_env_file,
    missing_controller_packages,
    resolve_project_root,
)
from tenyson.core.experiment_runtime import (
    install_sigterm_handler,
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

    def test_load_env_file_overrides_existing_env_by_default(self) -> None:
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

        self.assertEqual(current, "from_file")
        self.assertEqual(loaded, {"TENYSON_EXPERIMENT_ID": "from_file"})

    def test_load_env_file_can_preserve_existing_env_when_override_is_false(self) -> None:
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
                loaded = load_env_file(env_file, override=False)
                current = os.environ["TENYSON_EXPERIMENT_ID"]

        self.assertEqual(current, "from_shell")
        self.assertEqual(loaded, {})

    def test_install_sigterm_handler_raises_keyboardinterrupt(self) -> None:
        previous_handler = signal.getsignal(signal.SIGTERM)
        try:
            install_sigterm_handler(label="demo experiment")
            handler = signal.getsignal(signal.SIGTERM)
            with self.assertRaises(KeyboardInterrupt):
                handler(signal.SIGTERM, None)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)

    def test_wordle_experiment_main_delegates_flow_to_run_experiment(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        experiment_path = repo_root / "examples" / "wordle" / "experiment.py"

        captured: dict[str, object] = {}

        def fake_run_experiment(anchor_file, build, **kwargs):
            captured["anchor_file"] = anchor_file
            captured["build"] = build
            captured["kwargs"] = kwargs

        original_sys_path = list(sys.path)
        try:
            sys.path.insert(0, str(experiment_path.parent))
            with patch("tenyson.run_experiment", new=fake_run_experiment):
                runpy.run_path(
                    str(experiment_path),
                    run_name="__main__",
                )
        finally:
            sys.path[:] = original_sys_path

        self.assertEqual(
            Path(str(captured["anchor_file"])).resolve(),
            experiment_path.resolve(),
        )
        self.assertNotIn("prepare", captured["kwargs"])
        self.assertNotIn("recovery_restart_stage_fallback_env_vars", captured["kwargs"])

        top_level_calls: list[tuple[str, str, tuple[str, ...]]] = []
        branch_calls: list[tuple[str, str, tuple[str, ...]]] = []
        parallel_calls: list[tuple[str, list[str]]] = []

        class FakeBranch:
            def rl(self, stage_id: str, **kwargs):
                branch_calls.append(
                    ("rl", stage_id, tuple(sorted(kwargs.keys())))
                )
                return SimpleNamespace(id=stage_id)

            def eval(self, stage_id: str, **kwargs):
                branch_calls.append(
                    ("eval", stage_id, tuple(sorted(kwargs.keys())))
                )
                return SimpleNamespace(id=stage_id)

            def eval_stage(self, stage_id: str, **kwargs):
                branch_calls.append(
                    ("eval_stage", stage_id, tuple(sorted(kwargs.keys())))
                )
                return SimpleNamespace(id=stage_id)

            def run(self, stage):
                branch_calls.append(("run", stage.id, stage.id))
                return SimpleNamespace(id=stage.id)

            def run_parallel(self, label: str, stages):
                parallel_calls.append((label, [stage.id for stage in stages]))
                return {stage.id: SimpleNamespace(id=stage.id) for stage in stages}

            def adapter(self, stage_id: str):
                return f"adapter:{stage_id}"

        class FakeExperiment:
            def sft(self, stage_id: str, **kwargs):
                top_level_calls.append(
                    ("sft", stage_id, tuple(sorted(kwargs.keys())))
                )
                return SimpleNamespace(id=stage_id)

            def adapter(self, stage_id: str):
                return f"adapter:{stage_id}"

            def eval(self, stage_id: str, **kwargs):
                top_level_calls.append(
                    ("eval", stage_id, tuple(sorted(kwargs.keys())))
                )
                return SimpleNamespace(id=stage_id)

            def run_branches(self, branches):
                self_branch = FakeBranch()
                for builder in branches.values():
                    builder(self_branch)

        captured["build"](FakeExperiment())

        self.assertEqual(
            top_level_calls,
            [
                ("sft", "sft_main", ("dataset", "overrides")),
                (
                    "eval",
                    "eval_baseline_mixed",
                    ("adapter", "dataset", "metrics", "overrides"),
                ),
            ],
        )
        self.assertIn(
            (
                "rl",
                "mixed_rl",
                ("adapter", "dataset", "overrides", "reward"),
            ),
            branch_calls,
        )
        self.assertIn(
            (
                "eval",
                "mixed_final_eval",
                ("adapter", "dataset", "metrics", "overrides"),
            ),
            branch_calls,
        )
        self.assertIn(("curr_eval_after_t3", ["curr_eval_after_t3_turn2", "curr_eval_after_t3_turn3"]), parallel_calls)


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
