import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

from datasets import Dataset
import torch

from tenyson.core.telemetry import RLRolloutTracker
import tenyson.jobs.eval as eval_module
import tenyson.jobs.rl as rl_module
from tenyson.jobs.eval import (
    EvalJob,
    _require_eval_vllm_config,
    _resolve_eval_fast_inference_requested,
    _resolve_eval_model_load_kwargs,
)
from tenyson.jobs.rl import (
    _build_grpo_vllm_overrides,
    _build_vllm_sampling_params_kwargs,
    _build_vllm_generation_kwargs,
    _ensure_trl_vllm_guided_decoding_compat,
    _ensure_trl_vllm_sampling_params_compat,
    _RewardTelemetryCollector,
    _require_rl_vllm_config,
    _resolve_unsloth_model_load_kwargs,
    _resolve_grpo_max_completion_length,
    RLJob,
)
from tenyson.jobs.reporting_utils import normalize_report_to


def _load_wordle_task_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "examples" / "wordle" / "wordle_task.py"
    )
    spec = importlib.util.spec_from_file_location("wordle_task_for_tests", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class WordleParserTests(unittest.TestCase):
    def test_parse_history_accepts_unicode_arrow(self) -> None:
        wordle_task = _load_wordle_task_module()
        prompt = (
            "Prior turns and feedback:\n"
            "Turn 1: [crane] → X X G G G\n\n"
            "Enter your next guess."
        )

        history_rows = wordle_task._parse_history_from_prompt(prompt)

        self.assertEqual(history_rows, [("crane", "XXGGG")])

    def test_build_constraints_accepts_spaced_feedback(self) -> None:
        wordle_task = _load_wordle_task_module()

        compact = wordle_task.build_constraints("crane", "XXGGG")
        spaced = wordle_task.build_constraints("crane", "X X G G G")

        self.assertEqual(compact, spaced)

    def test_wordlists_relative_paths_resolve_from_task_file(self) -> None:
        wordle_task = _load_wordle_task_module()

        solutions, allowed = wordle_task.get_wordlists(
            {
                "task": {
                    "wordlists": {
                        "solutions": "wordlists/wordle_solutions.txt",
                        "allowed": "wordlists/wordle_allowed_guesses.txt",
                    }
                }
            }
        )

        self.assertGreater(len(solutions), 0)
        self.assertGreater(len(allowed), 0)

    def test_single_word_source_url_filters_and_reuses_five_letter_words(self) -> None:
        wordle_task = _load_wordle_task_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "words.txt"
            source_path.write_text(
                "a\n"
                "apple\n"
                "zebra\n"
                "co-op\n"
                "planet\n"
                "APPLE\n",
                encoding="utf-8",
            )

            solutions, allowed = wordle_task.get_wordlists(
                {
                    "task": {
                        "wordlists": {
                            "url": source_path.as_uri(),
                        }
                    }
                }
            )

        self.assertEqual(solutions, ["apple", "zebra"])
        self.assertEqual(allowed, ["apple", "zebra"])


    def test_fixed_turn_eval_named_run_matches_prompt_turn_number(self) -> None:
        wordle_task = _load_wordle_task_module()
        config = {
            "task": {
                "eval_samples": 1,
                "eval_seed": 42,
                "wordlists": {
                    "solutions": "wordlists/wordle_solutions.txt",
                    "allowed": "wordlists/wordle_allowed_guesses.txt",
                },
                "min_history_turns": 1,
                "max_history_turns": 1,
                "eval_exact_turns": [2],
            }
        }

        dataset = wordle_task.get_eval_dataset(config)
        row = dataset[0]
        prompt = row["prompt"]

        self.assertEqual(row["history_len"], 1)
        self.assertIn("This is turn 2 of the game.", prompt)

    def test_generated_secret_satisfies_constraints_from_raw_history_rows(self) -> None:
        wordle_task = _load_wordle_task_module()
        config = {
            "task": {
                "wordlists": {
                    "solutions": "wordlists/wordle_solutions.txt",
                    "allowed": "wordlists/wordle_allowed_guesses.txt",
                },
                "min_history_turns": 1,
                "max_history_turns": 5,
            },
            "training": {"seed": 123},
        }

        dataset = wordle_task.generate_synthetic_wordle_dataset(
            config, seed=123, n_samples=25
        )

        for row in dataset:
            ac = wordle_task.aggregate_constraints(row["history_rows"])
            sat, totals, _ = wordle_task.compute_sat_count(row["secret"], ac)
            self.assertEqual(sat, sum(totals.values()))

    def test_invalid_word_gets_no_constraint_reward(self) -> None:
        wordle_task = _load_wordle_task_module()
        config = {
            "task": {
                "wordlists": {
                    "solutions": "wordlists/wordle_solutions.txt",
                    "allowed": "wordlists/wordle_allowed_guesses.txt",
                },
                "rewards": {
                    "format": 0.2,
                    "dict": 0.2,
                    "repeat_penalty": -0.5,
                    "constraint": 0.1,
                    "overlength_penalty": -0.5,
                },
            }
        }
        solutions, allowed = wordle_task.get_wordlists(config)
        valid_set = set(solutions) | set(allowed)
        self.assertNotIn("irkls", valid_set)

        prompt = wordle_task.build_prompt_text(
            [
                ("bravo", "X G X X X"),
                ("wooer", "X X X X Y"),
                ("mambo", "X X X X X"),
            ]
        )

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(text)

        scored = wordle_task.score_completion(
            prompt,
            "<think>test</think><guess>[irkls]</guess>",
            valid_set,
            config["task"],
            FakeTokenizer(),
        )

        self.assertEqual(scored["is_wordle_valid"], 0)
        self.assertEqual(scored["reward_constraints"], 0.0)
        self.assertEqual(
            scored["reward_total"],
            scored["reward_format"] + scored["reward_overlength"],
        )

    def test_constraint_reward_normalizes_by_total_constraints(self) -> None:
        wordle_task = _load_wordle_task_module()
        prompt = wordle_task.build_prompt_text([("abcde", "X X X X X")])

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(text)

        scored = wordle_task.score_completion(
            prompt_text=prompt,
            completion_text="<think>test</think><guess>[fghij]</guess>",
            valid_set={"fghij"},
            task_cfg={
                "rewards": {
                    "constraint": 0.5,
                    "constraint_perfect_bonus": 0.2,
                }
            },
            tokenizer=FakeTokenizer(),
        )

        self.assertEqual(scored["sat_count"], 5)
        self.assertEqual(sum(scored["totals"].values()), 5)
        self.assertAlmostEqual(scored["constraint_ratio"], 1.0)
        self.assertEqual(scored["is_perfect_constraint_match"], 1)
        self.assertAlmostEqual(scored["reward_constraints"], 0.7)

    def test_compute_metrics_reports_avg_constraint_reward(self) -> None:
        wordle_task = _load_wordle_task_module()
        prompts = [
            wordle_task.build_prompt_text([("abcde", "X X X X X")]),
            wordle_task.build_prompt_text([("abcde", "X X X X X")]),
        ]
        completions = [
            "<think>test</think><guess>[fghij]</guess>",
            "<think>test</think><guess>[abcde]</guess>",
        ]

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(text)

        with tempfile.TemporaryDirectory() as tmpdir:
            solutions_path = Path(tmpdir) / "solutions.txt"
            allowed_path = Path(tmpdir) / "allowed.txt"
            solutions_path.write_text("fghij\nabcde\n", encoding="utf-8")
            allowed_path.write_text("fghij\nabcde\n", encoding="utf-8")

            metrics_payload = wordle_task.compute_metrics(
                prompts=prompts,
                completions=completions,
                dataset_rows=Dataset.from_list(
                    [
                        {"id": 0, "secret": "mango"},
                        {"id": 1, "secret": "pearl"},
                    ]
                ),
                config={
                    "task": {
                        "wordlists": {
                            "solutions": str(solutions_path),
                            "allowed": str(allowed_path),
                        },
                        "rewards": {
                            "constraint": 0.5,
                            "constraint_perfect_bonus": 0.2,
                        },
                    }
                },
                tokenizer=FakeTokenizer(),
            )

        metrics = metrics_payload["metrics"]
        self.assertAlmostEqual(metrics["avg_constraint_reward"], 0.35)
        self.assertAlmostEqual(metrics["constraint_accuracy"], 0.5)
        self.assertEqual(metrics["total_samples"], 2)

    def test_resolve_reward_max_output_tokens_prefers_training_limit(self) -> None:
        wordle_task = _load_wordle_task_module()

        resolved = wordle_task.resolve_reward_max_output_tokens(
            {
                "training": {"max_completion_length": 512},
                "vllm": {"max_tokens": 1024},
                "task": {"rewards": {}},
            },
            task_cfg={"rewards": {}},
        )

        self.assertEqual(resolved, 512)


class RewardTelemetryCollectorTests(unittest.TestCase):
    def test_build_results_payload_records_weighted_rollouts_for_wandb(self) -> None:
        collector = _RewardTelemetryCollector(
            experiment_id="wordle_exp",
            run_id="mixed_rl",
            reward_component_names=["format_exact", "wordle_strict"],
            rollout_tracker=RLRolloutTracker(),
        )
        collector.set_reward_weights([1.0, 0.5])

        prompts = [
            {"content": "Prompt one"},
            {"content": "Prompt two"},
        ]
        completions = [
            {"content": "<guess>crate</guess>"},
            {"content": "<guess>zzzzz</guess>"},
        ]
        trainer_state = SimpleNamespace(global_step=12)

        collector.record_component(
            component_name="format_exact",
            prompts=prompts,
            completions=completions,
            rewards=[0.2, 0.0],
            kwargs={"trainer_state": trainer_state},
        )

        pending_payload = collector.build_results_payload()
        self.assertEqual(pending_payload["metrics"]["total_samples"], 0)
        self.assertEqual(pending_payload["metrics"]["rollout_batches"], 0)
        self.assertEqual(pending_payload["detailed_results"], [])

        collector.record_component(
            component_name="wordle_strict",
            prompts=prompts,
            completions=completions,
            rewards=[0.8, -1.0],
            kwargs={"trainer_state": trainer_state},
        )

        payload = collector.build_results_payload()
        self.assertEqual(payload["metrics"]["total_samples"], 2)
        self.assertEqual(payload["metrics"]["rollout_batches"], 1)

        first_row = payload["detailed_results"][0]
        second_row = payload["detailed_results"][1]

        self.assertEqual(first_row["global_step"], 12)
        self.assertEqual(first_row["rollout_step"], 1)
        self.assertEqual(first_row["prompt"], "Prompt one")
        self.assertEqual(first_row["completion"], "<guess>crate</guess>")
        self.assertEqual(
            first_row["reward_components"],
            {
                "format_exact": 0.2,
                "wordle_strict": 0.4,
            },
        )
        self.assertAlmostEqual(first_row["reward_total"], 0.6)
        self.assertAlmostEqual(first_row["reward"], 0.6)

        self.assertEqual(second_row["global_step"], 12)
        self.assertEqual(second_row["rollout_step"], 1)
        self.assertEqual(second_row["prompt"], "Prompt two")
        self.assertEqual(second_row["completion"], "<guess>zzzzz</guess>")
        self.assertEqual(
            second_row["reward_components"],
            {
                "format_exact": 0.0,
                "wordle_strict": -0.5,
            },
        )
        self.assertAlmostEqual(second_row["reward_total"], -0.5)
        self.assertAlmostEqual(second_row["reward"], -0.5)


class RLConfigHelpersTests(unittest.TestCase):
    def test_resolve_grpo_max_completion_length_uses_vllm_fallback(self) -> None:
        length = _resolve_grpo_max_completion_length(
            {},
            {"enabled": True, "max_tokens": 1536},
        )

        self.assertEqual(length, 1536)

    def test_resolve_grpo_max_completion_length_prefers_training_limit(self) -> None:
        length = _resolve_grpo_max_completion_length(
            {"max_completion_length": 1024},
            {"enabled": True, "max_tokens": 2048},
        )

        self.assertEqual(length, 1024)

    def test_build_vllm_generation_kwargs_sets_stop_behavior(self) -> None:
        kwargs = _build_vllm_generation_kwargs(
            SimpleNamespace(eos_token="<eos>"),
            {"enabled": True},
        )

        self.assertEqual(
            kwargs,
            {"include_stop_str_in_output": True, "stop": ["<eos>"]},
        )

    def test_build_vllm_sampling_params_kwargs_matches_reference_shape(self) -> None:
        kwargs = _build_vllm_sampling_params_kwargs(
            SimpleNamespace(eos_token="<eos>"),
            {
                "enabled": True,
                "top_p": 0.95,
                "top_k": -1,
                "min_p": 0.1,
            },
            seed=1234,
        )

        self.assertEqual(
            kwargs,
            {
                "top_p": 0.95,
                "top_k": -1,
                "min_p": 0.1,
                "seed": 1234,
                "include_stop_str_in_output": True,
                "stop": ["<eos>"],
            },
        )

    def test_build_grpo_vllm_overrides_maps_live_trl_fields(self) -> None:
        overrides = _build_grpo_vllm_overrides(
            SimpleNamespace(eos_token="<eos>"),
            {
                "enabled": True,
                "top_p": 0.95,
                "top_k": -1,
                "min_p": 0.1,
                "gpu_memory_utilization": 0.8,
            },
        )

        self.assertEqual(overrides["top_p"], 0.95)
        self.assertEqual(overrides["top_k"], -1)
        self.assertEqual(overrides["min_p"], 0.1)
        self.assertEqual(
            overrides["generation_kwargs"],
            {"include_stop_str_in_output": True, "stop": ["<eos>"]},
        )

    def test_build_grpo_vllm_overrides_prefers_explicit_sampling_params(self) -> None:
        module_name = "vllm"
        original_module = sys.modules.get(module_name)

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_module = ModuleType("vllm")
        fake_module.SamplingParams = FakeSamplingParams
        sys.modules[module_name] = fake_module
        try:
            overrides = _build_grpo_vllm_overrides(
                SimpleNamespace(eos_token="<eos>"),
                {
                    "enabled": True,
                    "top_p": 0.95,
                    "top_k": -1,
                    "min_p": 0.1,
                    "gpu_memory_utilization": 0.8,
                },
                seed=1234,
                prefer_explicit_sampling_params=True,
            )

            self.assertIn("vllm_sampling_params", overrides)
            self.assertNotIn("generation_kwargs", overrides)
            self.assertNotIn("top_p", overrides)
            self.assertEqual(
                overrides["vllm_sampling_params"].kwargs,
                {
                    "top_p": 0.95,
                    "top_k": -1,
                    "min_p": 0.1,
                    "seed": 1234,
                    "include_stop_str_in_output": True,
                    "stop": ["<eos>"],
                },
            )
        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

    def test_build_grpo_vllm_overrides_ignores_trl_engine_fields(self) -> None:
        overrides = _build_grpo_vllm_overrides(
            SimpleNamespace(eos_token="<eos>"),
            {
                "enabled": True,
                "mode": "server",
                "server_timeout": 12.5,
                "gpu_memory_utilization": 0.8,
            },
        )

        self.assertNotIn("use_vllm", overrides)
        self.assertNotIn("vllm_mode", overrides)
        self.assertNotIn("vllm_gpu_memory_utilization", overrides)
        self.assertEqual(
            overrides["generation_kwargs"],
            {"include_stop_str_in_output": True, "stop": ["<eos>"]},
        )

    def test_resolve_unsloth_model_load_kwargs_preserves_fast_inference_for_grpo_vllm(self) -> None:
        kwargs = _resolve_unsloth_model_load_kwargs(
            {"fast_inference": True},
            {
                "enabled": True,
                "gpu_memory_utilization": 0.8,
            },
        )

        self.assertEqual(
            kwargs,
            {
                "fast_inference": True,
                "gpu_memory_utilization": 0.8,
            },
        )

    def test_resolve_unsloth_model_load_kwargs_preserves_fast_inference_without_grpo_vllm(self) -> None:
        kwargs = _resolve_unsloth_model_load_kwargs(
            {"fast_inference": True},
            {
                "enabled": False,
                "gpu_memory_utilization": 0.8,
            },
        )

        self.assertEqual(
            kwargs,
            {
                "fast_inference": True,
                "gpu_memory_utilization": 0.8,
            },
        )

    def test_require_rl_vllm_config_rejects_disabled_vllm(self) -> None:
        with self.assertRaisesRegex(ValueError, "RL requires vLLM"):
            _require_rl_vllm_config(
                {"fast_inference": True},
                {"enabled": False},
            )

    def test_require_rl_vllm_config_rejects_server_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires vllm.mode=colocate"):
            _require_rl_vllm_config(
                {"fast_inference": True},
                {"enabled": True, "mode": "server"},
            )

    def test_require_rl_vllm_config_rejects_colocate_without_fast_inference(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires model.fast_inference=true"):
            _require_rl_vllm_config(
                {"fast_inference": False},
                {"enabled": True, "mode": "colocate"},
            )

    def test_trl_vllm_compat_aliases_guided_decoding_params(self) -> None:
        module_name = "vllm.sampling_params"
        original_module = sys.modules.get(module_name)
        fake_sampling_params = SimpleNamespace(
            StructuredOutputsParams=object(),
        )
        sys.modules[module_name] = fake_sampling_params
        try:
            _ensure_trl_vllm_guided_decoding_compat()
            self.assertIs(
                fake_sampling_params.GuidedDecodingParams,
                fake_sampling_params.StructuredOutputsParams,
            )
        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

    def test_trl_vllm_compat_preserves_existing_guided_decoding_params(self) -> None:
        module_name = "vllm.sampling_params"
        original_module = sys.modules.get(module_name)
        guided = object()
        structured = object()
        fake_sampling_params = SimpleNamespace(
            GuidedDecodingParams=guided,
            StructuredOutputsParams=structured,
        )
        sys.modules[module_name] = fake_sampling_params
        try:
            _ensure_trl_vllm_guided_decoding_compat()
            self.assertIs(fake_sampling_params.GuidedDecodingParams, guided)
        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

    def test_trl_vllm_sampling_params_compat_drops_truncate_prompt_tokens(self) -> None:
        module_name = "vllm.sampling_params"
        original_module = sys.modules.get(module_name)

        class FakeSamplingParams:
            def __init__(self, *, temperature=None):
                self.temperature = temperature

        fake_sampling_params = SimpleNamespace(SamplingParams=FakeSamplingParams)
        sys.modules[module_name] = fake_sampling_params
        try:
            _ensure_trl_vllm_sampling_params_compat()
            params = fake_sampling_params.SamplingParams(
                temperature=0.7,
                truncate_prompt_tokens=128,
            )
            self.assertEqual(params.temperature, 0.7)
        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

    def test_trl_vllm_sampling_params_compat_preserves_supported_kwarg(self) -> None:
        module_name = "vllm.sampling_params"
        original_module = sys.modules.get(module_name)

        class FakeSamplingParams:
            def __init__(self, *, temperature=None, truncate_prompt_tokens=None):
                self.temperature = temperature
                self.truncate_prompt_tokens = truncate_prompt_tokens

        fake_sampling_params = SimpleNamespace(SamplingParams=FakeSamplingParams)
        sys.modules[module_name] = fake_sampling_params
        try:
            _ensure_trl_vllm_sampling_params_compat()
            params = fake_sampling_params.SamplingParams(
                temperature=0.7,
                truncate_prompt_tokens=128,
            )
            self.assertEqual(params.temperature, 0.7)
            self.assertEqual(params.truncate_prompt_tokens, 128)
        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

    def test_normalize_report_to_strips_none_from_list_for_wandb(self) -> None:
        normalized = normalize_report_to(
            ["none", "tensorboard"],
            telemetry_backend="wandb",
        )

        self.assertEqual(normalized, ["tensorboard", "wandb"])

    def test_normalize_report_to_defaults_none_string_to_wandb(self) -> None:
        normalized = normalize_report_to(
            "none",
            telemetry_backend="wandb",
        )

        self.assertEqual(normalized, ["wandb"])


class _DummyBatch(dict):
    def to(self, _device):
        return self


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, texts, return_tensors="pt", padding=True):
        del return_tensors, padding
        rows = []
        for text in texts:
            if text == "ab":
                rows.append([1, 2])
            elif text == "c":
                rows.append([0, 3])
            else:
                raise AssertionError(f"Unexpected prompt: {text!r}")
        return _DummyBatch(
            {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.tensor(
                    [[1 if token != 0 else 0 for token in row] for row in rows],
                    dtype=torch.long,
                ),
            }
        )

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        mapping = {4: "X", 5: "Y", 6: "Z"}
        values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return "".join(mapping.get(token, "") for token in values if token in mapping)


class _DummyModel:
    device = "cpu"

    def fast_generate(self, *args, **kwargs):
        raise AssertionError("fast_generate should not be used when vllm.enabled is false")

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        completions = torch.tensor([[4, 5], [6, 0]], dtype=torch.long)
        return torch.cat([input_ids, completions], dim=1)


class EvalFallbackTests(unittest.TestCase):
    def test_resolve_eval_fast_inference_defaults_to_vllm_setting(self) -> None:
        requested = _resolve_eval_fast_inference_requested(
            {},
            {"enabled": True},
        )

        self.assertTrue(requested)

    def test_resolve_eval_fast_inference_respects_explicit_model_override(self) -> None:
        requested = _resolve_eval_fast_inference_requested(
            {"fast_inference": False},
            {"enabled": True},
        )

        self.assertFalse(requested)

    def test_resolve_eval_model_load_kwargs_omits_gpu_memory_without_fast_inference(self) -> None:
        kwargs = _resolve_eval_model_load_kwargs(
            {"fast_inference": False},
            {"enabled": True, "gpu_memory_utilization": 0.8},
        )

        self.assertEqual(kwargs, {"fast_inference": False})

    def test_require_eval_vllm_config_rejects_disabled_vllm(self) -> None:
        with self.assertRaisesRegex(ValueError, "Eval requires vLLM"):
            _require_eval_vllm_config(
                {},
                {"enabled": False},
            )

    def test_require_eval_vllm_config_rejects_disabled_fast_inference(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires vLLM fast inference"):
            _require_eval_vllm_config(
                {"fast_inference": False},
                {"enabled": True},
            )

    def test_build_sampling_params_requires_vllm_runtime(self) -> None:
        job = EvalJob(
            config={"evaluation": {"run_name": "eval_test"}, "vllm": {"enabled": True}},
            task=object(),
        )
        job._vllm_runtime_enabled = False

        with self.assertRaisesRegex(RuntimeError, "Eval requires vLLM"):
            job._build_sampling_params(SimpleNamespace(eos_token="<eos>"))

    def test_generate_batch_refuses_transformers_fallback(self) -> None:
        job = EvalJob(
            config={
                "evaluation": {"run_name": "eval_test"},
                "vllm": {"enabled": True, "max_tokens": 2, "temperature": 0.0},
            },
            task=object(),
        )

        with self.assertRaisesRegex(RuntimeError, "Refusing to fall back"):
            job._generate_batch(
                _DummyModel(),
                _DummyTokenizer(),
                ["ab", "c"],
                sampling_params=None,
            )

    def test_build_model_and_tokenizer_fails_when_vllm_startup_breaks(self) -> None:
        module_name = "unsloth"
        original_module = sys.modules.get(module_name)
        calls = []

        class FakeModel:
            def train(self, mode):
                self.mode = mode
                return self

        class FakeFastLanguageModel:
            @staticmethod
            def from_pretrained(**kwargs):
                calls.append(kwargs)
                if kwargs.get("fast_inference", False):
                    raise RuntimeError(
                        "<function standalone_compile> does not have the attribute 'FakeTensorMode'"
                    )
                return FakeModel(), SimpleNamespace()

            @staticmethod
            def for_inference(model):
                setattr(model, "for_inference_called", True)
                return model

        sys.modules[module_name] = SimpleNamespace(FastLanguageModel=FakeFastLanguageModel)
        try:
            job = EvalJob(
                config={
                    "evaluation": {"run_name": "eval_test"},
                    "model": {
                        "name": "Qwen/Qwen3-4B",
                        "load_in_4bit": True,
                        "fast_inference": True,
                    },
                    "vllm": {"enabled": True, "gpu_memory_utilization": 0.8},
                },
                task=object(),
            )

            with unittest.mock.patch.object(
                eval_module,
                "normalize_tokenizer_special_tokens",
            ) as normalize_mock, self.assertRaisesRegex(
                RuntimeError,
                "Eval requires vLLM and will now abort",
            ):
                job._build_model_and_tokenizer()

        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["fast_inference"])
        self.assertEqual(calls[0]["gpu_memory_utilization"], 0.8)
        self.assertTrue(job._vllm_runtime_enabled)
        normalize_mock.assert_not_called()


class RLFallbackTests(unittest.TestCase):
    def test_build_model_and_tokenizer_fails_when_vllm_startup_breaks(self) -> None:
        module_name = "unsloth"
        original_module = sys.modules.get(module_name)
        original_standby = os.environ.get("UNSLOTH_VLLM_STANDBY")
        calls = []
        standby_values = []

        class FakeModel:
            def train(self):
                self.mode = "train"
                return self

        class FakeFastLanguageModel:
            @staticmethod
            def from_pretrained(**kwargs):
                standby_values.append(os.environ.get("UNSLOTH_VLLM_STANDBY"))
                calls.append(kwargs)
                if kwargs.get("fast_inference", False):
                    raise RuntimeError(
                        "<function standalone_compile> does not have the attribute 'FakeTensorMode'"
                    )
                return FakeModel(), SimpleNamespace()

            @staticmethod
            def get_peft_model(model, **kwargs):
                model.peft_kwargs = kwargs
                return model

        sys.modules[module_name] = SimpleNamespace(FastLanguageModel=FakeFastLanguageModel)
        try:
            os.environ.pop("UNSLOTH_VLLM_STANDBY", None)
            job = RLJob(
                config={
                    "training": {"run_name": "rl_test", "seed": 3407},
                    "model": {
                        "name": "Qwen/Qwen3-4B",
                        "load_in_4bit": True,
                        "fast_inference": True,
                    },
                    "lora": {
                        "r": 16,
                        "target_modules": ["up_proj", "gate_proj", "down_proj"],
                        "alpha": 32,
                        "dropout": 0.0,
                        "bias": "none",
                    },
                    "vllm": {"enabled": True, "gpu_memory_utilization": 0.8},
                },
                task=object(),
            )

            with unittest.mock.patch(
                "tenyson.jobs.rl.normalize_tokenizer_special_tokens"
            ) as normalize_mock, self.assertRaisesRegex(
                RuntimeError,
                "RL requires vLLM and will now abort",
            ):
                job._build_model_and_tokenizer()
        finally:
            if original_standby is None:
                os.environ.pop("UNSLOTH_VLLM_STANDBY", None)
            else:
                os.environ["UNSLOTH_VLLM_STANDBY"] = original_standby
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

        self.assertEqual(os.environ.get("UNSLOTH_VLLM_STANDBY"), original_standby)
        self.assertEqual(len(calls), 1)
        self.assertEqual(standby_values, ["1"])
        self.assertTrue(calls[0]["fast_inference"])
        self.assertEqual(calls[0]["gpu_memory_utilization"], 0.8)
        self.assertTrue(job._vllm_runtime_enabled)
        normalize_mock.assert_not_called()

    def test_build_model_and_tokenizer_keeps_unsloth_lora_when_loading_init_adapter(
        self,
    ) -> None:
        unsloth_module_name = "unsloth"
        original_unsloth_module = sys.modules.get(unsloth_module_name)
        unsloth_calls = []
        strict_load_calls = []

        class FakeModel:
            def load_lora(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

        class FakeFastLanguageModel:
            @staticmethod
            def from_pretrained(**kwargs):
                unsloth_calls.append(kwargs)
                return FakeModel(), SimpleNamespace()

            @staticmethod
            def get_peft_model(model, **kwargs):
                model.peft_kwargs = kwargs
                return model

        sys.modules[unsloth_module_name] = SimpleNamespace(
            FastLanguageModel=FakeFastLanguageModel
        )

        adapter = SimpleNamespace(
            repo_id="org/adapter",
            resolved_revision="rev-123",
            weights_in_repo="adapter_model.safetensors",
            config={
                "base_model_name_or_path": "Qwen/Qwen3-4B",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": ["up_proj", "gate_proj", "down_proj"],
            },
        )

        try:
            job = RLJob(
                config={
                    "training": {"run_name": "rl_test", "seed": 3407},
                    "model": {
                        "name": "unsloth/Qwen3-4B-Base",
                        "fast_inference": True,
                        "init_adapter_repo": "org/adapter",
                        "init_adapter_revision": "main",
                    },
                    "lora": {
                        "r": 16,
                        "target_modules": ["up_proj", "gate_proj", "down_proj"],
                        "alpha": 32,
                        "dropout": 0.0,
                        "bias": "none",
                    },
                    "vllm": {"enabled": True, "gpu_memory_utilization": 0.8},
                },
                task=object(),
            )

            with unittest.mock.patch.object(
                rl_module,
                "download_hf_lora_adapter",
                return_value=adapter,
            ), unittest.mock.patch.object(
                rl_module,
                "resolve_hf_lora_runtime_kwargs",
                return_value={
                    "r": 16,
                    "target_modules": ["up_proj", "gate_proj", "down_proj"],
                    "lora_alpha": 32,
                    "lora_dropout": 0.0,
                    "bias": "none",
                },
            ), unittest.mock.patch.object(
                rl_module,
                "strict_load_hf_lora_adapter_weights",
                side_effect=lambda model, adapter_arg: strict_load_calls.append(
                    adapter_arg
                )
                or 7,
            ), unittest.mock.patch.object(
                rl_module,
                "normalize_tokenizer_special_tokens",
            ) as normalize_mock:
                model, _tokenizer = job._build_model_and_tokenizer()
        finally:
            if original_unsloth_module is None:
                sys.modules.pop(unsloth_module_name, None)
            else:
                sys.modules[unsloth_module_name] = original_unsloth_module

        self.assertEqual(len(unsloth_calls), 1)
        self.assertEqual(unsloth_calls[0]["model_name"], "Qwen/Qwen3-4B")
        self.assertEqual(unsloth_calls[0]["max_seq_length"], 2048)
        self.assertFalse(unsloth_calls[0]["load_in_4bit"])
        self.assertTrue(unsloth_calls[0]["fast_inference"])
        self.assertEqual(unsloth_calls[0]["gpu_memory_utilization"], 0.8)
        self.assertEqual(
            getattr(model, "peft_kwargs", None),
            {
                "r": 16,
                "target_modules": ["up_proj", "gate_proj", "down_proj"],
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "bias": "none",
                "use_gradient_checkpointing": "unsloth",
                "random_state": 3407,
            },
        )
        self.assertTrue(callable(getattr(model, "load_lora", None)))
        self.assertEqual(strict_load_calls, [adapter])
        normalize_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
