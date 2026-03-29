import unittest
from unittest.mock import Mock, patch

from datasets import Dataset

from tenyson.core.chat_sft import (
    build_chat_messages_formatting_func,
    load_hub_chat_sft_train_eval_split,
)
from tenyson.core.environment import (
    DatasetHooks,
    EnvironmentDefinition,
    EnvironmentRunSpec,
    EnvironmentTaskAdapter,
    RubricHooks,
    bind_environment_run,
    build_run_family,
    merge_config_overrides,
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

    def test_merge_config_overrides_deep_merges_without_mutating_inputs(self) -> None:
        left = {"task": {"window": {"min": 1}, "reward": "a"}}
        right = {"task": {"window": {"max": 3}}}

        merged = merge_config_overrides(left, right)

        self.assertEqual(
            merged,
            {"task": {"window": {"min": 1, "max": 3}, "reward": "a"}},
        )
        self.assertEqual(left, {"task": {"window": {"min": 1}, "reward": "a"}})
        self.assertEqual(right, {"task": {"window": {"max": 3}}})

    def test_build_run_family_generates_named_specs(self) -> None:
        runs = build_run_family(
            prefix="toy_eval_turn",
            run_type="eval",
            values=[2, 3],
            datasets=DatasetHooks(
                primary=lambda config, tokenizer: _dataset_for("eval"),
            ),
            rubric=RubricHooks(
                compute_metrics=lambda prompts, completions, dataset_rows, config, tokenizer: {
                    "count": len(prompts)
                }
            ),
            base_config_overrides={"task": {"window": "base"}},
            config_for_value=lambda turn: {"task": {"turn": int(turn)}},
        )

        self.assertEqual(sorted(runs), ["toy_eval_turn2", "toy_eval_turn3"])
        self.assertEqual(
            runs["toy_eval_turn2"].config_overrides,
            {"task": {"window": "base", "turn": 2}},
        )
        self.assertEqual(runs["toy_eval_turn3"].run_type, "eval")

    def test_load_hub_chat_sft_train_eval_split_validates_and_splits(self) -> None:
        dataset = Dataset.from_dict(
            {
                "messages": [
                    [{"role": "user", "content": f"prompt-{idx}"}]
                    for idx in range(20)
                ]
            }
        )

        with patch("tenyson.core.chat_sft.load_dataset", return_value=dataset):
            train_ds, eval_ds = load_hub_chat_sft_train_eval_split(
                {
                    "task": {
                        "sft_train_samples": 5,
                    },
                    "training": {
                        "val_size": 2,
                        "seed": 123,
                    },
                },
                default_dataset="org/demo-chat",
            )

        self.assertEqual(len(train_ds), 5)
        self.assertIsNotNone(eval_ds)
        self.assertEqual(len(eval_ds), 2)

    def test_load_hub_chat_sft_train_eval_split_rejects_invalid_messages_shape(self) -> None:
        dataset = Dataset.from_dict(
            {
                "messages": [
                    [{"role": "user", "content": "ok"}],
                    [{"role": "", "content": "still text"}],
                ]
            }
        )

        with patch("tenyson.core.chat_sft.load_dataset", return_value=dataset):
            with self.assertRaisesRegex(ValueError, 'must have a non-empty string "role"'):
                load_hub_chat_sft_train_eval_split(
                    {"training": {}},
                    default_dataset="org/demo-chat",
                )

    def test_build_chat_messages_formatting_func_uses_chat_template(self) -> None:
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: f"formatted:{messages[0]['role']}"
        formatter = build_chat_messages_formatting_func(tokenizer=tokenizer)

        output = formatter(
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ]
            }
        )

        self.assertEqual(output, ["formatted:user"])
        tokenizer.apply_chat_template.assert_called_once()


if __name__ == "__main__":
    unittest.main()
