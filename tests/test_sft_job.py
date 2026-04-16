import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tenyson.jobs.sft import (
    SFTJob,
    _enable_best_model_tracking,
    _enable_unsloth_full_finetune_training_mode,
    _push_final_adapter_snapshot,
    _push_final_model_snapshot,
    _require_full_finetune_model_config,
    _resolve_finetune_mode,
    _resolve_early_stopping_settings,
    _reject_removed_sft_packing_setting,
)


class SFTJobHelperTests(unittest.TestCase):
    def test_resolve_finetune_mode_defaults_to_lora(self) -> None:
        self.assertEqual(_resolve_finetune_mode({}), "lora")

    def test_resolve_finetune_mode_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "finetune_mode"):
            _resolve_finetune_mode({"finetune_mode": "mystery"})

    def test_require_full_finetune_model_config_rejects_4bit(self) -> None:
        with self.assertRaisesRegex(ValueError, "load_in_4bit=false"):
            _require_full_finetune_model_config({"load_in_4bit": True})

    def test_enable_unsloth_full_finetune_training_mode_falls_back_without_kwargs(self) -> None:
        calls = []

        class FakeModel:
            def for_training(self, *args, **kwargs):
                calls.append((args, kwargs))
                if kwargs:
                    raise TypeError("kwargs not accepted")

        _enable_unsloth_full_finetune_training_mode(
            FakeModel(),
            gradient_checkpointing="unsloth",
        )

        self.assertEqual(
            calls,
            [
                ((), {"use_gradient_checkpointing": "unsloth"}),
                ((), {}),
            ],
        )

    def test_resolve_early_stopping_settings_requires_eval_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires an SFT eval dataset"):
            _resolve_early_stopping_settings(
                {"early_stopping_patience": 2},
                has_eval_dataset=False,
                save_steps=20,
                eval_steps=None,
            )

    def test_resolve_early_stopping_settings_requires_matching_save_and_eval_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            _resolve_early_stopping_settings(
                {"early_stopping_patience": 2},
                has_eval_dataset=True,
                save_steps=20,
                eval_steps=10,
            )

    def test_resolve_early_stopping_settings_returns_patience_and_threshold(self) -> None:
        settings = _resolve_early_stopping_settings(
            {
                "early_stopping_patience": 3,
                "early_stopping_min_delta": 0.125,
            },
            has_eval_dataset=True,
            save_steps=20,
            eval_steps=20,
        )

        self.assertEqual(
            settings,
            {
                "patience": 3,
                "min_delta": 0.125,
            },
        )

    def test_enable_best_model_tracking_sets_eval_loss_tracking(self) -> None:
        training_args = SimpleNamespace(
            load_best_model_at_end=False,
            metric_for_best_model=None,
            greater_is_better=True,
        )

        _enable_best_model_tracking(training_args)

        self.assertTrue(training_args.load_best_model_at_end)
        self.assertEqual(training_args.metric_for_best_model, "eval_loss")
        self.assertFalse(training_args.greater_is_better)

    def test_push_final_adapter_snapshot_pushes_model_and_tokenizer(self) -> None:
        model = object()
        tokenizer = object()

        with patch("tenyson.jobs.sft.push_pretrained_snapshot_to_hub") as push_mock:
            _push_final_adapter_snapshot(
                repo_id="org/repo",
                run_name="wordle_sft_main",
                model=model,
                tokenizer=tokenizer,
                step=42,
                best_checkpoint="/tmp/checkpoint-40",
            )

        push_mock.assert_called_once()
        args, kwargs = push_mock.call_args
        self.assertEqual(args[0], "org/repo")
        self.assertIs(kwargs["model"], model)
        self.assertIs(kwargs["tokenizer"], tokenizer)
        self.assertIn("final best-model sync", kwargs["commit_message"])
        self.assertIn("checkpoint-40", kwargs["commit_message"])
        self.assertIn("step=42", kwargs["commit_message"])

    def test_push_final_model_snapshot_mentions_model(self) -> None:
        model = object()

        with patch("tenyson.jobs.sft.push_pretrained_snapshot_to_hub") as push_mock:
            _push_final_model_snapshot(
                repo_id="org/repo",
                run_name="wordle_sft_main",
                model=model,
                tokenizer=None,
                step=7,
                artifact_label="model",
            )

        push_mock.assert_called_once()
        args, kwargs = push_mock.call_args
        self.assertEqual(args[0], "org/repo")
        self.assertIs(kwargs["model"], model)
        self.assertIsNone(kwargs["tokenizer"])
        self.assertIn("final model sync", kwargs["commit_message"])

    def test_reject_removed_sft_packing_setting_allows_missing_key(self) -> None:
        _reject_removed_sft_packing_setting(
            {
                "loss_on_assistant_only": True,
                "response_template": "<|im_start|>assistant\n",
            }
        )

    def test_reject_removed_sft_packing_setting_rejects_true(self) -> None:
        with self.assertRaisesRegex(ValueError, "no longer supported for SFT"):
            _reject_removed_sft_packing_setting({"packing": True})

    def test_reject_removed_sft_packing_setting_rejects_false_too(self) -> None:
        with self.assertRaisesRegex(ValueError, "Remove this field from the config"):
            _reject_removed_sft_packing_setting({"packing": False})

    def test_build_model_and_tokenizer_full_mode_skips_lora(self) -> None:
        module_name = "unsloth"
        original_module = sys.modules.get(module_name)
        calls = []

        class FakeModel:
            def for_training(self, *args, **kwargs):
                calls.append(("for_training", args, kwargs))

        class FakeFastLanguageModel:
            @staticmethod
            def from_pretrained(**kwargs):
                calls.append(("from_pretrained", kwargs))
                return FakeModel(), SimpleNamespace()

            @staticmethod
            def get_peft_model(model, **kwargs):
                calls.append(("get_peft_model", kwargs))
                return model

        sys.modules[module_name] = SimpleNamespace(FastLanguageModel=FakeFastLanguageModel)
        try:
            job = SFTJob(
                config={
                    "training": {
                        "run_name": "sft_full_test",
                        "finetune_mode": "full",
                    },
                    "model": {
                        "name": "Qwen/Qwen3-4B",
                        "fast_inference": True,
                        "load_in_4bit": False,
                        "load_in_8bit": False,
                    },
                    "lora": {"gradient_checkpointing": "unsloth"},
                },
                task=object(),
            )

            with patch(
                "tenyson.jobs.sft.normalize_tokenizer_special_tokens"
            ) as normalize_mock:
                model, _tokenizer, seq_len = job._build_model_and_tokenizer()
        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

        self.assertIsInstance(model, FakeModel)
        self.assertEqual(seq_len, 2048)
        self.assertEqual(calls[0][0], "from_pretrained")
        self.assertTrue(calls[0][1]["full_finetuning"])
        self.assertFalse(calls[0][1]["fast_inference"])
        self.assertNotIn("trust_remote_code", calls[0][1])
        self.assertFalse(job.config["model"]["fast_inference"])
        self.assertEqual(calls[0][1]["model_name"], "Qwen/Qwen3-4B")
        self.assertNotIn("get_peft_model", [call[0] for call in calls])
        normalize_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
