import unittest
from unittest.mock import Mock, patch

from datasets import Dataset

from tenyson.core.chat_sft import (
    build_chat_messages_formatting_func,
    load_hub_chat_sft_train_eval_split,
)
from tenyson.core.plugin import TaskPlugin
from tenyson.core.stage_templates import (
    STAGE_TEMPLATE_CONFIG_KEY,
    EvalDatasetTemplate,
    EvalMetricsTemplate,
    RLRewardTemplate,
    RLDatasetTemplate,
    bind_stage_templates_from_config,
    template_factory_ref,
)
from tenyson.experiment import AdapterRef, ConfigTemplates, ExperimentSession
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


def _serialized_eval_dataset_template() -> EvalDatasetTemplate:
    return EvalDatasetTemplate(
        build=lambda config: _dataset_for("serialized-eval"),
        factory_ref=template_factory_ref(__name__, "_serialized_eval_dataset_template"),
    )


def _serialized_rl_dataset_template() -> RLDatasetTemplate:
    return RLDatasetTemplate(
        build=lambda config: _dataset_for("serialized-rl"),
        factory_ref=template_factory_ref(__name__, "_serialized_rl_dataset_template"),
    )


def _serialized_rl_reward_template() -> RLRewardTemplate:
    return RLRewardTemplate(
        build=lambda config, tokenizer: [
            lambda prompts, completions, **kwargs: [1.0 for _ in completions]
        ],
        factory_ref=template_factory_ref(__name__, "_serialized_rl_reward_template"),
    )


def _serialized_eval_metrics_template() -> EvalMetricsTemplate:
    return EvalMetricsTemplate(
        compute=lambda prompts, completions, dataset_rows, config, tokenizer: {
            "source": "serialized",
            "count": len(prompts),
        },
        factory_ref=template_factory_ref(__name__, "_serialized_eval_metrics_template"),
    )


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

    def test_explicit_stage_templates_override_task_behavior_without_default_run_overrides(self) -> None:
        class DummyTask(TaskPlugin):
            def get_run_config_overrides(self, run_type, *, variant=None):
                del run_type, variant
                return {"task": {"source": "default-run-overrides"}}

            def get_sft_dataset(self, config, tokenizer):
                del config, tokenizer
                return _dataset_for("base-sft")

            def get_rl_dataset(self, config):
                del config
                return _dataset_for("base-rl")

            def get_reward_funcs(self, config, tokenizer):
                del config, tokenizer
                return [lambda prompts, completions, **kwargs: [0.0 for _ in completions]]

            def get_eval_dataset(self, config):
                del config
                return _dataset_for("base-eval")

            def compute_metrics(
                self,
                prompts,
                completions,
                dataset_rows,
                config,
                tokenizer,
            ):
                del prompts, completions, dataset_rows, config, tokenizer
                return {"source": "base"}

        session = ExperimentSession(
            task=DummyTask(),
            templates=ConfigTemplates(
                {
                    "sft": {"training": {}, "task": {}},
                    "rl": {"training": {}, "task": {}, "model": {}},
                    "eval": {"evaluation": {}, "task": {}, "model": {}},
                }
            ),
            cloud_factory=lambda: object(),
        )

        rl_stage = session.rl(
            "explicit_rl",
            adapter=AdapterRef(repo_id="org/base", revision="main"),
            dataset=_serialized_rl_dataset_template(),
            reward=_serialized_rl_reward_template(),
        )
        eval_stage = session.eval(
            "explicit_eval",
            adapter=AdapterRef(repo_id="org/base", revision="main"),
            dataset=_serialized_eval_dataset_template(),
            metrics=_serialized_eval_metrics_template(),
        )

        self.assertNotIn("source", rl_stage.config.get("task", {}))
        self.assertEqual(rl_stage.task.get_rl_dataset(rl_stage.config)[0]["value"], "serialized-rl")
        self.assertEqual(eval_stage.task.get_eval_dataset(eval_stage.config)[0]["value"], "serialized-eval")
        self.assertEqual(
            eval_stage.task.compute_metrics(
                prompts=["a", "b"],
                completions=["x", "y"],
                dataset_rows=_dataset_for("explicit-eval"),
                config=eval_stage.config,
                tokenizer=None,
            ),
            {"source": "serialized", "count": 2},
        )

    def test_explicit_stage_templates_round_trip_through_config_payload(self) -> None:
        class DummyTask(TaskPlugin):
            def get_sft_dataset(self, config, tokenizer):
                del config, tokenizer
                return _dataset_for("base-sft")

            def get_rl_dataset(self, config):
                del config
                return _dataset_for("base-rl")

            def get_reward_funcs(self, config, tokenizer):
                del config, tokenizer
                return [lambda prompts, completions, **kwargs: [0.0 for _ in completions]]

            def get_eval_dataset(self, config):
                del config
                return _dataset_for("base-eval")

            def compute_metrics(
                self,
                prompts,
                completions,
                dataset_rows,
                config,
                tokenizer,
            ):
                del prompts, completions, dataset_rows, config, tokenizer
                return {"source": "base"}

        session = ExperimentSession(
            task=DummyTask(),
            templates=ConfigTemplates(
                {
                    "sft": {"training": {}, "task": {}},
                    "rl": {"training": {}, "task": {}, "model": {}},
                    "eval": {"evaluation": {}, "task": {}, "model": {}},
                }
            ),
            cloud_factory=lambda: object(),
        )

        stage = session.eval(
            "serialized_eval",
            adapter=AdapterRef(repo_id="org/base", revision="main"),
            dataset=_serialized_eval_dataset_template(),
            metrics=_serialized_eval_metrics_template(),
        )

        payload = stage.config.get(STAGE_TEMPLATE_CONFIG_KEY)
        self.assertIsInstance(payload, dict)
        self.assertEqual(
            payload["eval_dataset"]["factory"],
            "_serialized_eval_dataset_template",
        )
        rebound_task = bind_stage_templates_from_config(DummyTask(), stage.config)
        self.assertEqual(
            rebound_task.get_eval_dataset(stage.config)[0]["value"],
            "serialized-eval",
        )
        self.assertEqual(
            rebound_task.compute_metrics(
                prompts=["a", "b", "c"],
                completions=["x", "y", "z"],
                dataset_rows=_dataset_for("serialized-eval"),
                config=stage.config,
                tokenizer=None,
            ),
            {"source": "serialized", "count": 3},
        )


if __name__ == "__main__":
    unittest.main()
