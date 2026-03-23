from types import SimpleNamespace
import unittest

import torch

from tenyson.core.hf_adapter import (
    HfLoraAdapter,
    _resolve_adapter_artifact_paths,
    resolve_hf_lora_runtime_kwargs,
    strict_load_hf_lora_adapter_weights,
)


class FakeModel:
    def __init__(self):
        self._state = {
            "layer.lora_A.default.weight": torch.zeros(2, 3),
            "layer.lora_B.default.weight": torch.zeros(3, 2),
            "base.weight": torch.zeros(1),
        }
        self.loaded = None

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state_dict, strict=False):
        self.loaded = dict(state_dict)
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


class HFAdapterTests(unittest.TestCase):
    def test_resolve_adapter_artifact_paths_prefers_latest_checkpoint_pair(self) -> None:
        config_path, weights_path = _resolve_adapter_artifact_paths(
            [
                "checkpoint-2/adapter_config.json",
                "checkpoint-2/adapter_model.safetensors",
                "checkpoint-5/adapter_config.json",
                "checkpoint-5/adapter_model.safetensors",
            ]
        )

        self.assertEqual(config_path, "checkpoint-5/adapter_config.json")
        self.assertEqual(weights_path, "checkpoint-5/adapter_model.safetensors")

    def test_resolve_hf_lora_runtime_kwargs_validates_expected_layout(self) -> None:
        adapter = HfLoraAdapter(
            repo_id="org/repo",
            requested_revision="main",
            resolved_revision="abc123",
            config_in_repo="adapter_config.json",
            weights_in_repo="adapter_model.safetensors",
            config={
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": ["q_proj", "v_proj"],
            },
            state_dict={"layer.lora_A.weight": torch.zeros(1)},
        )

        runtime_kwargs = resolve_hf_lora_runtime_kwargs(
            adapter,
            expected_r=16,
            expected_alpha=32,
            expected_dropout=0.0,
            expected_bias="none",
            expected_target_modules=["v_proj", "q_proj"],
        )

        self.assertEqual(runtime_kwargs["r"], 16)
        self.assertEqual(runtime_kwargs["lora_alpha"], 32.0)
        self.assertEqual(runtime_kwargs["target_modules"], ["q_proj", "v_proj"])

        with self.assertRaisesRegex(ValueError, "does not match the configured LoRA layout"):
            resolve_hf_lora_runtime_kwargs(adapter, expected_r=8)

    def test_strict_load_hf_lora_adapter_weights_normalizes_default_suffixes(self) -> None:
        adapter = HfLoraAdapter(
            repo_id="org/repo",
            requested_revision="main",
            resolved_revision="abc123",
            config_in_repo="adapter_config.json",
            weights_in_repo="adapter_model.safetensors",
            config={},
            state_dict={
                "layer.lora_A.weight": torch.ones(2, 3),
                "layer.lora_B.weight": torch.ones(3, 2),
            },
        )
        model = FakeModel()

        loaded_count = strict_load_hf_lora_adapter_weights(model, adapter)

        self.assertEqual(loaded_count, 2)
        self.assertIn("layer.lora_A.default.weight", model.loaded)
        self.assertIn("layer.lora_B.default.weight", model.loaded)


if __name__ == "__main__":
    unittest.main()
