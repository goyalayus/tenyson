import unittest
from unittest.mock import Mock, patch

from datasets import Dataset

from tenyson.core.chat_sft import (
    bind_chat_sft_dataset,
    build_chat_messages_formatting_func,
    chat_sft_dataset_fn,
    load_hub_chat_sft_train_eval_split,
)
from tenyson.core.plugin import TaskPlugin
from tenyson.core.stage_templates import (
    EvalMetricsContext,
    STAGE_TEMPLATE_CONFIG_KEY,
    EvalDatasetTemplate,
    EvalMetricsTemplate,
    RLRewardTemplate,
    RLDatasetTemplate,
    bind_eval_dataset,
    bind_stage_templates_from_config,
    eval_dataset_fn,
    eval_dataset_template,
    eval_metrics_fn,
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


def _chat_messages_dataset(*, sample_count: int) -> Dataset:
    return Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": f"prompt-{index}"},
                    {"role": "assistant", "content": f"answer-{index}"},
                ]
                for index in range(sample_count)
            ]
        }
    )


@chat_sft_dataset_fn
def _decorated_chat_messages_dataset(*, sample_count: int) -> Dataset:
    return _chat_messages_dataset(sample_count=sample_count)


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


@eval_dataset_fn
def _dataset_from_sample_count(*, sample_count: int) -> Dataset:
    return Dataset.from_dict({"value": [f"sample-count:{sample_count}"]})


@eval_dataset_fn
def _dataset_from_digits_and_sample_count(
    *,
    digits: int,
    sample_count: int,
) -> Dataset:
    return Dataset.from_dict({"value": [f"digits:{digits}-sample-count:{sample_count}"]})


@eval_dataset_fn
def _dataset_from_digits_sample_count_and_seed(
    *,
    digits: int,
    sample_count: int,
    seed: int,
) -> Dataset:
    return Dataset.from_dict(
        {
            "value": [
                f"digits:{digits}-sample-count:{sample_count}-seed:{seed}"
            ]
        }
    )


@eval_metrics_fn
def _plain_eval_metrics(
    prompts,
    completions,
    dataset_rows,
    config,
    tokenizer,
):
    del dataset_rows, config, tokenizer
    return {"source": "plain", "count": len(prompts), "completions": len(completions)}


@eval_metrics_fn
def _context_eval_metrics(ctx: EvalMetricsContext):
    return {
        "source": "context",
        "count": len(ctx.completions),
        "prompt_count": len(ctx.prompts),
    }


@eval_dataset_template
def _decorated_eval_dataset_template() -> EvalDatasetTemplate:
    return EvalDatasetTemplate(
        build=_dataset_from_sample_count
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

    def test_compute_metrics_accepts_context_style_eval_rubric(self) -> None:
        environment = EnvironmentDefinition(
            name="toy",
            runs={
                "eval_main": EnvironmentRunSpec(
                    run_type="eval",
                    datasets=DatasetHooks(
                        primary=lambda config, tokenizer: _dataset_for("eval-main"),
                    ),
                    rubric=RubricHooks(
                        compute_metrics=lambda ctx: {
                            "count": len(ctx.prompts),
                            "completions": len(ctx.completions),
                        }
                    ),
                ),
            },
        )
        task = EnvironmentTaskAdapter(environment)
        config = {}

        metrics = task.compute_metrics(
            prompts=["a", "b", "c"],
            completions=["x", "y", "z"],
            dataset_rows=_dataset_for("eval-main"),
            config=config,
            tokenizer=None,
        )

        self.assertEqual(metrics, {"count": 3, "completions": 3})

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

    def test_bind_chat_sft_dataset_binds_train_builder_and_formatting(self) -> None:
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: f"formatted:{messages[0]['role']}"

        template = bind_chat_sft_dataset(
            _chat_messages_dataset,
            sample_count=3,
        )
        train_dataset = template.train({}, None)
        formatter = template.formatting({}, tokenizer)

        self.assertEqual(len(train_dataset), 3)
        self.assertEqual(
            formatter(train_dataset[0]),
            ["formatted:user"],
        )
        tokenizer.apply_chat_template.assert_called_once()

    def test_chat_sft_dataset_fn_marks_plain_builder_for_binding(self) -> None:
        self.assertTrue(
            getattr(_decorated_chat_messages_dataset, "__tenyson_chat_sft_dataset_fn__", False)
        )

        template = bind_chat_sft_dataset(
            _decorated_chat_messages_dataset,
            sample_count=2,
        )
        train_dataset = template.train({}, None)

        self.assertEqual(len(train_dataset), 2)

    def test_sft_stage_accepts_bound_chat_sft_dataset_and_round_trips(self) -> None:
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
                    "sft": {"training": {}, "task": {}, "model": {"name": "Qwen/Qwen3-0.6B"}},
                    "rl": {"training": {}, "task": {}, "model": {}},
                    "eval": {"evaluation": {}, "task": {}, "model": {}},
                }
            ),
            cloud_factory=lambda: object(),
        )

        stage = session.sft(
            "bound_chat_sft",
            dataset=bind_chat_sft_dataset(
                _chat_messages_dataset,
                sample_count=4,
            ),
        )

        self.assertEqual(len(stage.task.get_sft_dataset(stage.config, None)), 4)

        payload = stage.config.get(STAGE_TEMPLATE_CONFIG_KEY)
        self.assertIsInstance(payload, dict)
        self.assertEqual(
            payload["sft_dataset"]["factory"],
            "_bound_chat_sft_dataset_from_callable_ref",
        )

        rebound_task = bind_stage_templates_from_config(DummyTask(), stage.config)
        self.assertEqual(len(rebound_task.get_sft_dataset(stage.config, None)), 4)

    def test_eval_dataset_template_reads_named_kwargs_from_evaluation_section(self) -> None:
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
            "plain_eval_builder",
            adapter=AdapterRef(repo_id="org/base", revision="main"),
            dataset=_decorated_eval_dataset_template(),
            overrides={"evaluation": {"sample_count": 12}},
        )

        self.assertEqual(
            stage.task.get_eval_dataset(stage.config)[0]["value"],
            "sample-count:12",
        )

    def test_eval_metrics_fn_rejects_bad_signature_early(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "must accept either one EvalMetricsContext parameter",
        ):
            @eval_metrics_fn
            def _bad_metrics(completions, dataset_rows):
                del completions, dataset_rows
                return {"metrics": {}}

    def test_eval_dataset_fn_rejects_var_kwargs_early(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "cannot use \\*\\*kwargs",
        ):
            @eval_dataset_fn
            def _bad_dataset_builder(**kwargs):
                del kwargs
                return _dataset_for("bad")

    def test_eval_stage_accepts_plain_module_level_functions(self) -> None:
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
            "plain_functions",
            adapter=AdapterRef(repo_id="org/base", revision="main"),
            dataset=_dataset_from_sample_count,
            metrics=_plain_eval_metrics,
            overrides={"evaluation": {"sample_count": 11}},
        )

        self.assertEqual(
            stage.task.get_eval_dataset(stage.config)[0]["value"],
            "sample-count:11",
        )
        self.assertEqual(
            stage.task.compute_metrics(
                prompts=["a", "b"],
                completions=["x", "y"],
                dataset_rows=_dataset_for("plain"),
                config=stage.config,
                tokenizer=None,
            ),
            {"source": "plain", "count": 2, "completions": 2},
        )

        payload = stage.config.get(STAGE_TEMPLATE_CONFIG_KEY)
        self.assertIsInstance(payload, dict)
        self.assertEqual(
            payload["eval_dataset"]["factory"],
            "_eval_template_from_callable_ref",
        )
        self.assertEqual(
            payload["eval_metrics"]["factory"],
            "_eval_template_from_callable_ref",
        )

        rebound_task = bind_stage_templates_from_config(DummyTask(), stage.config)
        self.assertEqual(
            rebound_task.get_eval_dataset(stage.config)[0]["value"],
            "sample-count:11",
        )
        self.assertEqual(
            rebound_task.compute_metrics(
                prompts=["a"],
                completions=["x"],
                dataset_rows=_dataset_for("plain"),
                config=stage.config,
                tokenizer=None,
            ),
            {"source": "plain", "count": 1, "completions": 1},
        )

    def test_eval_stage_accepts_context_metrics_function(self) -> None:
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
            "context_metrics",
            adapter=AdapterRef(repo_id="org/base", revision="main"),
            dataset=_dataset_from_sample_count,
            metrics=_context_eval_metrics,
            overrides={"evaluation": {"sample_count": 3}},
        )

        self.assertEqual(
            stage.task.compute_metrics(
                prompts=["a", "b"],
                completions=["x", "y"],
                dataset_rows=_dataset_for("plain"),
                config=stage.config,
                tokenizer=None,
            ),
            {"source": "context", "count": 2, "prompt_count": 2},
        )

        rebound_task = bind_stage_templates_from_config(DummyTask(), stage.config)
        self.assertEqual(
            rebound_task.compute_metrics(
                prompts=["a"],
                completions=["x"],
                dataset_rows=_dataset_for("plain"),
                config=stage.config,
                tokenizer=None,
            ),
            {"source": "context", "count": 1, "prompt_count": 1},
        )

    def test_bind_eval_dataset_binds_fixed_kwargs_and_round_trips(self) -> None:
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
            "bound_eval_builder",
            adapter=AdapterRef(repo_id="org/base", revision="main"),
            dataset=bind_eval_dataset(_dataset_from_digits_and_sample_count, digits=4),
            overrides={"evaluation": {"sample_count": 11}},
        )

        self.assertEqual(
            stage.task.get_eval_dataset(stage.config)[0]["value"],
            "digits:4-sample-count:11",
        )

        payload = stage.config.get(STAGE_TEMPLATE_CONFIG_KEY)
        self.assertIsInstance(payload, dict)
        self.assertEqual(
            payload["eval_dataset"]["factory"],
            "_eval_template_from_callable_ref",
        )
        self.assertEqual(
            payload["eval_dataset"]["kwargs"]["bound_kwargs"],
            {"digits": 4},
        )

        rebound_task = bind_stage_templates_from_config(DummyTask(), stage.config)
        self.assertEqual(
            rebound_task.get_eval_dataset(stage.config)[0]["value"],
            "digits:4-sample-count:11",
        )

    def test_bind_eval_dataset_can_bind_all_builder_kwargs(self) -> None:
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
            "fully_bound_eval_builder",
            adapter=AdapterRef(repo_id="org/base", revision="main"),
            dataset=bind_eval_dataset(
                _dataset_from_digits_sample_count_and_seed,
                digits=4,
                sample_count=11,
                seed=7,
            ),
        )

        self.assertEqual(
            stage.task.get_eval_dataset(stage.config)[0]["value"],
            "digits:4-sample-count:11-seed:7",
        )

        payload = stage.config.get(STAGE_TEMPLATE_CONFIG_KEY)
        self.assertIsInstance(payload, dict)
        self.assertEqual(
            payload["eval_dataset"]["kwargs"]["bound_kwargs"],
            {"digits": 4, "sample_count": 11, "seed": 7},
        )

        rebound_task = bind_stage_templates_from_config(DummyTask(), stage.config)
        self.assertEqual(
            rebound_task.get_eval_dataset(stage.config)[0]["value"],
            "digits:4-sample-count:11-seed:7",
        )

    def test_eval_stage_rejects_lambda_dataset_builder(self) -> None:
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

        with self.assertRaisesRegex(TypeError, "module-level named function"):
            session.eval(
                "lambda_dataset",
                adapter=AdapterRef(repo_id="org/base", revision="main"),
                dataset=lambda *, sample_count: _dataset_for(f"lambda:{sample_count}"),
            )

    def test_eval_dataset_template_requires_missing_evaluation_parameter(self) -> None:
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
            "missing_eval_parameter",
            adapter=AdapterRef(repo_id="org/base", revision="main"),
            dataset=_decorated_eval_dataset_template(),
            overrides={"evaluation": {}},
        )

        with self.assertRaisesRegex(ValueError, 'evaluation\\["sample_count"\\]'):
            stage.task.get_eval_dataset(stage.config)

    def test_legacy_eval_dataset_builder_with_config_still_works(self) -> None:
        @eval_dataset_template
        def _legacy_eval_dataset_template() -> EvalDatasetTemplate:
            return EvalDatasetTemplate(
                build=lambda config: _dataset_for(
                    f'legacy:{config["evaluation"]["sample_count"]}'
                )
            )

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
            "legacy_eval_builder",
            adapter=AdapterRef(repo_id="org/base", revision="main"),
            dataset=_legacy_eval_dataset_template(),
            overrides={"evaluation": {"sample_count": 5}},
        )

        self.assertEqual(
            stage.task.get_eval_dataset(stage.config)[0]["value"],
            "legacy:5",
        )

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

    def test_decorated_eval_template_auto_infers_factory_ref_for_round_trip(self) -> None:
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
            "decorated_eval",
            adapter=AdapterRef(repo_id="org/base", revision="main"),
            dataset=_decorated_eval_dataset_template(),
            overrides={"evaluation": {"sample_count": 9}},
        )

        payload = stage.config.get(STAGE_TEMPLATE_CONFIG_KEY)
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["eval_dataset"]["module"], __name__)
        self.assertEqual(
            payload["eval_dataset"]["factory"],
            "_decorated_eval_dataset_template",
        )

        rebound_task = bind_stage_templates_from_config(DummyTask(), stage.config)
        self.assertEqual(
            rebound_task.get_eval_dataset(stage.config)[0]["value"],
            "sample-count:9",
        )


if __name__ == "__main__":
    unittest.main()
