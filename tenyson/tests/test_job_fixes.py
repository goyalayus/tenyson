import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tenyson.jobs.eval import EvalJob
from tenyson.jobs.rl import (
    _build_grpo_vllm_overrides,
    _build_vllm_generation_kwargs,
    _resolve_grpo_max_completion_length,
)


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
        self.assertEqual(overrides["top_p"], 0.95)
        self.assertEqual(overrides["top_k"], -1)
        self.assertEqual(overrides["min_p"], 0.1)
        self.assertEqual(overrides["vllm_gpu_memory_utilization"], 0.8)
        self.assertEqual(
            overrides["generation_kwargs"],
            {"include_stop_str_in_output": True, "stop": ["<eos>"]},
        )


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
