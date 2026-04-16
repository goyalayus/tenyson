import sys
import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import tenyson.experiment as experiment_module
from tenyson.core.experiment_runtime import LocalExperimentContext, bootstrap_local_experiment
from tenyson.core.experiment_runner import run_experiment
from tenyson.core.functional import load_functional_manifest
from tenyson.experiment import AdapterRef


class FunctionalManifestTests(unittest.TestCase):
    def test_load_functional_manifest_derives_short_run_aliases_and_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            functional_path = base_dir / "functional.py"
            functional_path.write_text(
                "\n".join(
                    [
                        "from tenyson.core.environment import EnvironmentDefinition, EnvironmentRunSpec",
                        "from tenyson.experiment import AdapterRef",
                        "",
                        "ENVIRONMENT = EnvironmentDefinition(",
                        '    name=\"demo\",',
                        "    runs={",
                        '        \"demo_eval_turn6\": EnvironmentRunSpec(run_type=\"eval\"),',
                        '        \"demo_rl_turn6\": EnvironmentRunSpec(run_type=\"rl\"),',
                        "    },",
                        ")",
                        "SEEDS = {",
                        '    \"base\": AdapterRef(repo_id=\"org/demo\", revision=\"rev1\"),',
                        "}",
                    ]
                ),
                encoding="utf-8",
            )

            manifest = load_functional_manifest(functional_path)

        self.assertEqual(manifest.resolve_run("eval_turn6"), "demo_eval_turn6")
        self.assertEqual(manifest.resolve_run("rl_turn6"), "demo_rl_turn6")
        self.assertEqual(
            manifest.resolve_seed("base"),
            AdapterRef(repo_id="org/demo", revision="rev1"),
        )

    def test_load_functional_manifest_accepts_full_model_seed_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            functional_path = base_dir / "functional.py"
            functional_path.write_text(
                "\n".join(
                    [
                        "from tenyson.core.plugin import TemplateTaskPlugin",
                        "",
                        'TASK = TemplateTaskPlugin(environment_name="demo")',
                        "SEEDS = {",
                        '    "base": {',
                        '        "repo_id": "org/demo-full",',
                        '        "revision": "rev9",',
                        '        "artifact_type": "full_model",',
                        "    },",
                        "}",
                    ]
                ),
                encoding="utf-8",
            )

            manifest = load_functional_manifest(functional_path)

        self.assertEqual(
            manifest.resolve_seed("base"),
            AdapterRef(
                repo_id="org/demo-full",
                revision="rev9",
                artifact_type="full_model",
            ),
        )

    def test_load_functional_manifest_accepts_template_task_without_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            functional_path = base_dir / "functional.py"
            functional_path.write_text(
                "\n".join(
                    [
                        "from tenyson.core.plugin import TemplateTaskPlugin",
                        "from tenyson.experiment import AdapterRef",
                        "",
                        'TASK = TemplateTaskPlugin(environment_name=\"demo\")',
                        "SEEDS = {",
                        '    \"base\": AdapterRef(repo_id=\"org/demo\", revision=\"rev1\"),',
                        "}",
                    ]
                ),
                encoding="utf-8",
            )

            manifest = load_functional_manifest(functional_path)

        self.assertEqual(manifest.task.get_environment_name(), "demo")
        self.assertEqual(manifest.resolve_run("eval_turn6"), "eval_turn6")
        self.assertEqual(
            manifest.resolve_seed("base"),
            AdapterRef(repo_id="org/demo", revision="rev1"),
        )


class BootstrapRuntimeTests(unittest.TestCase):
    def test_bootstrap_local_experiment_loads_root_env_and_overrides_shell_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            anchor = project_root / "examples" / "demo" / "experiment.py"
            anchor.parent.mkdir(parents=True, exist_ok=True)
            anchor.write_text("print('demo')\n", encoding="utf-8")
            (project_root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (project_root / "src" / "tenyson").mkdir(parents=True, exist_ok=True)
            (project_root / ".env").write_text(
                "TENYSON_EXPERIMENT_ID=from_root_env\n",
                encoding="utf-8",
            )

            with patch(
                "tenyson.core.experiment_runtime.ensure_local_controller_environment",
                return_value=[],
            ), patch.dict(
                os.environ,
                {"TENYSON_EXPERIMENT_ID": "from_shell_env"},
                clear=True,
            ):
                original_sys_path = list(sys.path)
                try:
                    context = bootstrap_local_experiment(anchor)
                    resolved_experiment_id = os.environ["TENYSON_EXPERIMENT_ID"]
                finally:
                    sys.path[:] = original_sys_path

        self.assertEqual(context.project_root, project_root)
        self.assertEqual(context.env_path, project_root / ".env")
        self.assertEqual(context.loaded_env, {"TENYSON_EXPERIMENT_ID": "from_root_env"})
        self.assertEqual(context.base_dir, anchor.parent)
        self.assertEqual(resolved_experiment_id, "from_root_env")


class RunExperimentTests(unittest.TestCase):
    def test_run_experiment_executes_build_with_seed_and_run_alias_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            anchor_file = base_dir / "experiment2.py"
            anchor_file.write_text("# placeholder\n", encoding="utf-8")
            functional_path = base_dir / "functional.py"
            functional_path.write_text("# placeholder\n", encoding="utf-8")

            context = LocalExperimentContext(
                anchor_file=anchor_file,
                project_root=base_dir,
                base_dir=base_dir,
                env_path=base_dir / ".env",
                loaded_env={},
            )

            fake_task = SimpleNamespace(get_environment_name=lambda: "demo")
            manifest = SimpleNamespace(
                task=fake_task,
                resolve_seed=lambda alias: AdapterRef(repo_id=f"seed/{alias}", revision="r1"),
                resolve_run=lambda run_name: {
                    "eval_turn6": "demo_eval_turn6",
                    "rl_turn6": "demo_rl_turn6",
                }[run_name],
            )

            fake_report = Mock()
            fake_report.output_path = base_dir / "final_report_experiment2.md"
            fake_report.experiment_id = "demo_experiment2_123"
            fake_report.telemetry_backend_ref = "wandb://demo/project"

            created_stages: dict[str, dict] = {}
            run_order: list[str] = []

            class FakeBranch:
                def sft(self, stage_id: str, **kwargs):
                    created_stages[stage_id] = {"run_type": "sft", **kwargs}
                    return SimpleNamespace(id=stage_id)

                def rl(self, stage_id: str, **kwargs):
                    created_stages[stage_id] = {"run_type": "rl", **kwargs}
                    return SimpleNamespace(id=stage_id)

                def eval(self, stage_id: str, **kwargs):
                    created_stages[stage_id] = {"run_type": "eval", **kwargs}
                    return SimpleNamespace(id=stage_id)

                def run(self, stage):
                    run_order.append(stage.id)
                    return SimpleNamespace(stage_id=stage.id)

                def run_parallel(self, label: str, stages):
                    del label
                    for stage in stages:
                        run_order.append(stage.id)
                    return {stage.id: SimpleNamespace(stage_id=stage.id) for stage in stages}

                def require_adapter(self, stage_id: str):
                    return AdapterRef(repo_id=f"adapter/{stage_id}", revision="rev")

                def result(self, stage_id: str):
                    return SimpleNamespace(id=stage_id)

            fake_branch = FakeBranch()

            class FakeSession:
                def __init__(self) -> None:
                    self.closed = False

                def branch(self):
                    return fake_branch

                def run_branches(self, branches):
                    del branches
                    return {}

                def close(self):
                    self.closed = True

            fake_session = FakeSession()
            rebuild_mock = Mock()

            def build(exp):
                seed = exp.seed("experiment2_sft")
                exp.eval("baseline", run="eval_turn6", artifact=seed)
                exp.rl("train", run="rl_turn6", artifact=seed)
                exp.eval("final", run="eval_turn6", artifact=exp.artifact("train"))

            with patch(
                "tenyson.core.experiment_runner.bootstrap_local_experiment",
                return_value=context,
            ), patch(
                "tenyson.core.experiment_runner.load_functional_manifest",
                return_value=manifest,
            ), patch(
                "tenyson.core.experiment_runner.install_sigterm_handler",
                return_value=None,
            ), patch(
                "tenyson.core.experiment_runner.configure_experiment_identity",
                return_value="demo_experiment2_123",
            ), patch(
                "tenyson.core.experiment_runner.build_experiment_report",
                return_value=fake_report,
            ), patch(
                "tenyson.core.experiment_runner.resolve_recovery_experiment_id",
                return_value=None,
            ), patch(
                "tenyson.core.experiment_runner.create_modal_experiment_session",
                return_value=fake_session,
            ), patch(
                "tenyson.core.experiment_runner._rebuild_report_from_telemetry",
                rebuild_mock,
            ):
                run_experiment(anchor_file, build)

        self.assertEqual(created_stages["baseline"]["run"], "demo_eval_turn6")
        self.assertEqual(created_stages["train"]["run"], "demo_rl_turn6")
        self.assertEqual(
            created_stages["baseline"]["artifact"],
            AdapterRef(repo_id="seed/experiment2_sft", revision="r1"),
        )
        self.assertEqual(
            created_stages["final"]["artifact"],
            AdapterRef(repo_id="adapter/train", revision="rev"),
        )
        self.assertEqual(run_order, ["baseline", "train", "final"])
        self.assertTrue(fake_session.closed)
        rebuild_mock.assert_called_once()

    def test_run_experiment_calls_prepare_before_identity_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            anchor_file = base_dir / "experiment.py"
            anchor_file.write_text("# placeholder\n", encoding="utf-8")
            functional_path = base_dir / "functional.py"
            functional_path.write_text("# placeholder\n", encoding="utf-8")

            context = LocalExperimentContext(
                anchor_file=anchor_file,
                project_root=base_dir,
                base_dir=base_dir,
                env_path=base_dir / ".env",
                loaded_env={},
            )

            fake_task = SimpleNamespace(get_environment_name=lambda: "demo")
            manifest = SimpleNamespace(
                task=fake_task,
                resolve_seed=lambda alias: AdapterRef(repo_id=f"seed/{alias}", revision="r1"),
                resolve_run=lambda run_name: run_name,
            )

            fake_report = Mock()
            fake_report.output_path = base_dir / "final_report.md"
            fake_report.experiment_id = "demo_experiment_123"
            fake_report.telemetry_backend_ref = "wandb://demo/project"

            class FakeBranch:
                def eval(self, stage_id: str, **kwargs):
                    del kwargs
                    return SimpleNamespace(id=stage_id)

                def run(self, stage):
                    return SimpleNamespace(stage_id=stage.id)

                def require_adapter(self, stage_id: str):
                    return AdapterRef(repo_id=f"adapter/{stage_id}", revision="rev")

            fake_branch = FakeBranch()

            class FakeSession:
                def branch(self):
                    return fake_branch

                def run_branches(self, branches):
                    del branches
                    return {}

                def close(self):
                    return None

            fake_session = FakeSession()
            observed_prepare: list[tuple[LocalExperimentContext, object]] = []

            def prepare(context_arg, manifest_arg):
                observed_prepare.append((context_arg, manifest_arg))
                os.environ["TENYSON_EXPERIMENT_ID"] = "prepared_experiment"

            def configure_identity(**kwargs):
                self.assertEqual(os.environ["TENYSON_EXPERIMENT_ID"], "prepared_experiment")
                return "prepared_experiment"

            with patch(
                "tenyson.core.experiment_runner.bootstrap_local_experiment",
                return_value=context,
            ), patch(
                "tenyson.core.experiment_runner.load_functional_manifest",
                return_value=manifest,
            ), patch(
                "tenyson.core.experiment_runner.install_sigterm_handler",
                return_value=None,
            ), patch(
                "tenyson.core.experiment_runner.configure_experiment_identity",
                side_effect=configure_identity,
            ), patch(
                "tenyson.core.experiment_runner.build_experiment_report",
                return_value=fake_report,
            ), patch(
                "tenyson.core.experiment_runner.resolve_recovery_experiment_id",
                return_value=None,
            ), patch(
                "tenyson.core.experiment_runner.create_modal_experiment_session",
                return_value=fake_session,
            ), patch(
                "tenyson.core.experiment_runner._rebuild_report_from_telemetry",
                Mock(),
            ):
                run_experiment(
                    anchor_file,
                    lambda exp: exp.eval("baseline", run="eval_turn6"),
                    prepare=prepare,
                )

        self.assertEqual(observed_prepare, [(context, manifest)])

    def test_run_experiment_accepts_direct_stage_sequences_in_branches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            anchor_file = base_dir / "experiment.py"
            anchor_file.write_text("# placeholder\n", encoding="utf-8")
            functional_path = base_dir / "functional.py"
            functional_path.write_text("# placeholder\n", encoding="utf-8")

            context = LocalExperimentContext(
                anchor_file=anchor_file,
                project_root=base_dir,
                base_dir=base_dir,
                env_path=base_dir / ".env",
                loaded_env={},
            )

            fake_task = SimpleNamespace(get_environment_name=lambda: "demo")
            manifest = SimpleNamespace(
                task=fake_task,
                resolve_seed=lambda alias: AdapterRef(repo_id=f"seed/{alias}", revision="r1"),
                resolve_run=lambda run_name: {
                    "sft_main": "demo_sft_main",
                    "eval_turn2": "demo_eval_turn2",
                }[run_name],
            )

            fake_report = Mock()
            fake_report.output_path = base_dir / "final_report.md"
            fake_report.experiment_id = "demo_experiment_123"
            fake_report.telemetry_backend_ref = "wandb://demo/project"

            created_stages: dict[str, dict[str, object]] = {}
            observed_branch_stage_ids: list[str] = []

            class FakeBranch:
                def sft(self, stage_id: str, **kwargs):
                    created_stages[stage_id] = {"run_type": "sft", **kwargs}
                    return SimpleNamespace(id=stage_id)

                def eval(self, stage_id: str, **kwargs):
                    created_stages[stage_id] = {"run_type": "eval", **kwargs}
                    return SimpleNamespace(id=stage_id)

                def require_adapter(self, stage_id: str):
                    raise KeyError(stage_id)

            fake_branch = FakeBranch()

            class FakeSession:
                def branch(self):
                    return fake_branch

                def run_branches(self, branches):
                    plan = branches["curriculum"]
                    if callable(plan):
                        raise AssertionError("Expected a direct stage sequence, not a branch builder.")
                    observed_branch_stage_ids.extend(stage.id for stage in plan)
                    return {
                        "curriculum": {
                            stage.id: SimpleNamespace(stage_id=stage.id)
                            for stage in plan
                        }
                    }

                def close(self):
                    return None

            def build(exp):
                return exp.run_branches(
                    {
                        "curriculum": [
                            exp.sft_stage("sft_main", run="sft_main"),
                            exp.eval_stage(
                                "eval_after_sft",
                                run="eval_turn2",
                                artifact=exp.adapter("sft_main"),
                            ),
                        ]
                    }
                )

            with patch(
                "tenyson.core.experiment_runner.bootstrap_local_experiment",
                return_value=context,
            ), patch(
                "tenyson.core.experiment_runner.load_functional_manifest",
                return_value=manifest,
            ), patch(
                "tenyson.core.experiment_runner.install_sigterm_handler",
                return_value=None,
            ), patch(
                "tenyson.core.experiment_runner.configure_experiment_identity",
                return_value="demo_experiment_123",
            ), patch(
                "tenyson.core.experiment_runner.build_experiment_report",
                return_value=fake_report,
            ), patch(
                "tenyson.core.experiment_runner.resolve_recovery_experiment_id",
                return_value=None,
            ), patch(
                "tenyson.core.experiment_runner.create_modal_experiment_session",
                return_value=FakeSession(),
            ), patch(
                "tenyson.core.experiment_runner._rebuild_report_from_telemetry",
                Mock(),
            ):
                branch_results = run_experiment(anchor_file, build)

        self.assertEqual(observed_branch_stage_ids, ["sft_main", "eval_after_sft"])
        self.assertEqual(created_stages["sft_main"]["run"], "demo_sft_main")
        self.assertEqual(created_stages["eval_after_sft"]["run"], "demo_eval_turn2")
        self.assertIsInstance(
            created_stages["eval_after_sft"]["artifact"],
            experiment_module._DeferredStageArtifactRef,
        )
        self.assertEqual(
            created_stages["eval_after_sft"]["artifact"].stage_id,
            "sft_main",
        )
        self.assertIn("curriculum", branch_results)

    def test_run_experiment_plans_restart_stages_from_build_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            anchor_file = base_dir / "experiment.py"
            anchor_file.write_text("# placeholder\n", encoding="utf-8")
            functional_path = base_dir / "functional.py"
            functional_path.write_text("# placeholder\n", encoding="utf-8")

            context = LocalExperimentContext(
                anchor_file=anchor_file,
                project_root=base_dir,
                base_dir=base_dir,
                env_path=base_dir / ".env",
                loaded_env={},
            )

            fake_task = SimpleNamespace(get_environment_name=lambda: "demo")
            manifest = SimpleNamespace(
                task=fake_task,
                resolve_seed=lambda alias: AdapterRef(repo_id=f"seed/{alias}", revision="r1"),
                resolve_run=lambda run_name: {
                    "sft_main": "demo_sft_main",
                    "rl_turn2": "demo_rl_turn2",
                    "eval_turn2": "demo_eval_turn2",
                }[run_name],
            )

            fake_report = Mock()
            fake_report.output_path = base_dir / "final_report.md"
            fake_report.experiment_id = "demo_experiment_123"
            fake_report.telemetry_backend_ref = "wandb://demo/project"

            created_stages: list[str] = []
            captured_session_kwargs: dict[str, object] = {}

            class FakeBranch:
                def sft(self, stage_id: str, **kwargs):
                    del kwargs
                    created_stages.append(stage_id)
                    return SimpleNamespace(id=stage_id)

                def rl(self, stage_id: str, **kwargs):
                    del kwargs
                    created_stages.append(stage_id)
                    return SimpleNamespace(id=stage_id)

                def eval(self, stage_id: str, **kwargs):
                    del kwargs
                    created_stages.append(stage_id)
                    return SimpleNamespace(id=stage_id)

                def run(self, stage):
                    return SimpleNamespace(stage_id=stage.id)

                def require_adapter(self, stage_id: str):
                    return AdapterRef(repo_id=f"adapter/{stage_id}", revision="rev")

            fake_branch = FakeBranch()

            class FakeSession:
                def branch(self):
                    return fake_branch

                def run_branches(self, branches):
                    del branches
                    return {}

                def close(self):
                    return None

            def create_session(**kwargs):
                captured_session_kwargs.update(kwargs)
                return FakeSession()

            def build(exp):
                exp.sft("sft_main", run="sft_main")
                sft_artifact = exp.artifact("sft_main")

                def build_branch(branch):
                    branch.run(
                        branch.rl(
                            "curr_rl_t2",
                            run="rl_turn2",
                            artifact=sft_artifact,
                        )
                    )
                    branch.run(
                        branch.eval(
                            "curr_eval_after_t2_turn2",
                            run="eval_turn2",
                            artifact=branch.require_artifact("curr_rl_t2"),
                        )
                    )

                exp.run_branches({"curriculum": build_branch})

            with patch(
                "tenyson.core.experiment_runner.bootstrap_local_experiment",
                return_value=context,
            ), patch(
                "tenyson.core.experiment_runner.load_functional_manifest",
                return_value=manifest,
            ), patch(
                "tenyson.core.experiment_runner.install_sigterm_handler",
                return_value=None,
            ), patch(
                "tenyson.core.experiment_runner.configure_experiment_identity",
                return_value="demo_experiment_123",
            ), patch(
                "tenyson.core.experiment_runner.build_experiment_report",
                return_value=fake_report,
            ), patch(
                "tenyson.core.experiment_runner.resolve_recovery_experiment_id",
                return_value="demo_experiment_123",
            ), patch(
                "tenyson.core.experiment_runner.create_modal_experiment_session",
                side_effect=create_session,
            ), patch(
                "tenyson.core.experiment_runner._rebuild_report_from_telemetry",
                Mock(),
            ), patch.dict(
                os.environ,
                {"TENYSON_RECOVER_RESTART_FROM_STAGE": "curr_rl_t2"},
                clear=False,
            ):
                run_experiment(anchor_file, build)

        self.assertEqual(
            captured_session_kwargs["recovery_restart_stages"],
            ["curr_rl_t2", "curr_eval_after_t2_turn2"],
        )
        self.assertEqual(created_stages, ["sft_main"])


if __name__ == "__main__":
    unittest.main()
