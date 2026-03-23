import unittest

from datasets import Dataset

from tenyson.core.environment import (
    DatasetHooks,
    EnvironmentDefinition,
    EnvironmentRunSpec,
    EnvironmentTaskAdapter,
    RubricHooks,
    bind_environment_run,
    resolve_bound_environment_run,
)


def _dataset_for(label: str):
    return Dataset.from_dict({"value": [label]})


class EnvironmentCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = EnvironmentDefinition(
            name="toy",
            runs={
                "sft_main": EnvironmentRunSpec(
                    run_type="sft",
                    datasets=DatasetHooks(
                        primary=lambda config, tokenizer: _dataset_for("sft-main"),
                    ),
                ),
                "sft_alt": EnvironmentRunSpec(
                    run_type="sft",
                    datasets=DatasetHooks(
                        primary=lambda config, tokenizer: _dataset_for("sft-alt"),
                    ),
                ),
                "eval_main": EnvironmentRunSpec(
                    run_type="eval",
                    datasets=DatasetHooks(
                        primary=lambda config, tokenizer: _dataset_for("eval-main"),
                    ),
                    rubric=RubricHooks(
                        compute_metrics=lambda prompts, completions, dataset_rows, config, tokenizer: {
                            "count": len(prompts)
                        }
                    ),
                ),
            },
        )
        self.task = EnvironmentTaskAdapter(self.environment)

    def test_bind_environment_run_selects_runtime_spec(self) -> None:
        config = {}
        bind_environment_run(config, "sft_alt")

        dataset = self.task.get_sft_dataset(config, tokenizer=None)

        self.assertEqual(resolve_bound_environment_run(config), "sft_alt")
        self.assertEqual(dataset[0]["value"], "sft-alt")

    def test_resolve_run_spec_requires_name_when_multiple_runs_exist(self) -> None:
        with self.assertRaisesRegex(ValueError, "has multiple \"sft\" runs"):
            self.environment.resolve_run_spec("sft")

    def test_compute_metrics_uses_bound_eval_run(self) -> None:
        config = {}
        bind_environment_run(config, "eval_main")

        metrics = self.task.compute_metrics(
            prompts=["a", "b"],
            completions=["x", "y"],
            dataset_rows=_dataset_for("eval-main"),
            config=config,
            tokenizer=None,
        )

        self.assertEqual(metrics, {"count": 2})


if __name__ == "__main__":
    unittest.main()
