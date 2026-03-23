import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import torch

from tenyson.jobs.eval import EvalJob
from tenyson.jobs.rl import (
    _build_grpo_vllm_overrides,
    _build_vllm_sampling_params_kwargs,
    _build_vllm_generation_kwargs,
    _ensure_trl_vllm_guided_decoding_compat,
    _ensure_trl_vllm_sampling_params_compat,
    _resolve_unsloth_model_load_kwargs,
    _resolve_grpo_max_completion_length,
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


class RLConfigHelpersTests(unittest.TestCase):
    def test_resolve_grpo_max_completion_length_uses_vllm_fallback(self) -> None:
        length = _resolve_grpo_max_completion_length(
            {},
            {"enabled": True, "max_tokens": 1536},
        )

        self.assertEqual(length, 1536)

    def test_resolve_grpo_max_completion_length_rejects_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            _resolve_grpo_max_completion_length(
                {"max_completion_length": 1024},
                {"enabled": True, "max_tokens": 2048},
            )

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

        self.assertTrue(overrides["use_vllm"])
        self.assertEqual(overrides["vllm_mode"], "colocate")
        self.assertEqual(overrides["top_p"], 0.95)
        self.assertEqual(overrides["top_k"], -1)
        self.assertEqual(overrides["min_p"], 0.1)
        self.assertEqual(overrides["vllm_gpu_memory_utilization"], 0.8)
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

            self.assertTrue(overrides["use_vllm"])
            self.assertEqual(overrides["vllm_mode"], "colocate")
            self.assertEqual(overrides["vllm_gpu_memory_utilization"], 0.8)
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

    def test_build_grpo_vllm_overrides_supports_server_mode(self) -> None:
        overrides = _build_grpo_vllm_overrides(
            SimpleNamespace(eos_token="<eos>"),
            {
                "enabled": True,
                "mode": "server",
                "server_timeout": 12.5,
            },
        )

        self.assertTrue(overrides["use_vllm"])
        self.assertEqual(overrides["vllm_mode"], "server")
        self.assertEqual(overrides["vllm_server_host"], "127.0.0.1")
        self.assertEqual(overrides["vllm_server_port"], 8000)
        self.assertEqual(overrides["vllm_server_timeout"], 12.5)

    def test_build_grpo_vllm_overrides_rejects_unknown_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "vllm.mode must be either"):
            _build_grpo_vllm_overrides(
                SimpleNamespace(eos_token="<eos>"),
                {
                    "enabled": True,
                    "mode": "mystery",
                },
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

    def test_resolve_unsloth_model_load_kwargs_disables_fast_inference_for_vllm_server_mode(self) -> None:
        kwargs = _resolve_unsloth_model_load_kwargs(
            {"fast_inference": True},
            {
                "enabled": True,
                "mode": "server",
                "gpu_memory_utilization": 0.8,
            },
        )

        self.assertEqual(kwargs, {"fast_inference": False})

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
    def test_build_sampling_params_skips_vllm_when_disabled(self) -> None:
        job = EvalJob(
            config={"evaluation": {"run_name": "eval_test"}, "vllm": {"enabled": False}},
            task=object(),
        )

        sampling_params = job._build_sampling_params(SimpleNamespace(eos_token="<eos>"))

        self.assertIsNone(sampling_params)

    def test_generate_batch_uses_transformers_path_when_vllm_disabled(self) -> None:
        job = EvalJob(
            config={
                "evaluation": {"run_name": "eval_test"},
                "vllm": {"enabled": False, "max_tokens": 2, "temperature": 0.0},
            },
            task=object(),
        )

        completions = job._generate_batch(
            _DummyModel(),
            _DummyTokenizer(),
            ["ab", "c"],
            sampling_params=None,
        )

        self.assertEqual(completions, ["XY", "Z"])


if __name__ == "__main__":
    unittest.main()
