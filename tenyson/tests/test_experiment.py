import copy
import unittest
from unittest.mock import patch

import tenyson.experiment as experiment_module
from tenyson.experiment import AdapterRef, ConfigTemplates, ExperimentSession
from tenyson.jobs.result import JobResult


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


def _result(run_id: str, *, status: str = "success") -> JobResult:
    return JobResult(run_id=run_id, status=status, total_time_seconds=1.0)


class ExperimentSessionTests(unittest.TestCase):
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
        )

        stage = session.rl(
            "mixed_rl",
            adapter=AdapterRef(repo_id="repo/id", revision="sha123"),
            run_name="wordle_rl_mixed",
            output_dir="./outputs/mixed/rl",
            overrides={
                "training": {"hf_repo_base": "org/base"},
                "task": {"max_history_turns": 5},
            },
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


if __name__ == "__main__":
    unittest.main()
