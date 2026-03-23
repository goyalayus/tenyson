from types import SimpleNamespace
import unittest

from tenyson.jobs.sft import (
    _enable_best_model_tracking,
    _resolve_requested_attn_implementation,
    _resolve_assistant_only_strategy,
    _resolve_sft_special_tokens_kwargs,
    _validate_packing_attention_implementation,
    _push_final_adapter_snapshot,
    _resolve_early_stopping_settings,
)


class FakePusher:
    def __init__(self):
        self.calls = []

    def push_to_hub(self, repo_id, commit_message):
        self.calls.append((repo_id, commit_message))


class SFTJobHelperTests(unittest.TestCase):
    def test_resolve_requested_attn_implementation_defaults_to_flash_attention_for_packing(self) -> None:
        resolved = _resolve_requested_attn_implementation(
            {"packing": True},
            {},
        )

        self.assertEqual(resolved, "flash_attention_2")

    def test_resolve_requested_attn_implementation_respects_explicit_override(self) -> None:
        resolved = _resolve_requested_attn_implementation(
            {"packing": True},
            {"attn_implementation": "flash_attention_3"},
        )

        self.assertEqual(resolved, "flash_attention_3")

    def test_validate_packing_attention_implementation_rejects_non_flash_attention(self) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(_attn_implementation="flex_attention")
        )

        with self.assertRaisesRegex(RuntimeError, "cross-contaminate packed samples"):
            _validate_packing_attention_implementation(
                {"packing": True},
                model,
            )

    def test_validate_packing_attention_implementation_allows_supported_flash_attention(self) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(_attn_implementation="flash_attention_2")
        )

        _validate_packing_attention_implementation(
            {"packing": True},
            model,
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
        model = FakePusher()
        tokenizer = FakePusher()

        _push_final_adapter_snapshot(
            repo_id="org/repo",
            run_name="wordle_sft_main",
            model=model,
            tokenizer=tokenizer,
            step=42,
            best_checkpoint="/tmp/checkpoint-40",
        )

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(tokenizer.calls), 1)
        self.assertEqual(model.calls[0][0], "org/repo")
        self.assertIn("final best-model sync", model.calls[0][1])
        self.assertIn("checkpoint-40", model.calls[0][1])
        self.assertIn("step=42", model.calls[0][1])

    def test_assistant_only_strategy_uses_response_template_collator_for_text_runs(self) -> None:
        strategy = _resolve_assistant_only_strategy(
            {
                "loss_on_assistant_only": True,
                "response_template": "<|im_start|>assistant\n",
                "packing": False,
            },
            train_sample={"text": "plain formatted sample"},
        )

        self.assertFalse(strategy["use_native_assistant_only_loss"])
        self.assertFalse(strategy["use_manual_assistant_masks"])
        self.assertTrue(strategy["use_response_template_collator"])

    def test_assistant_only_strategy_uses_manual_masks_for_packed_builtin_rows(self) -> None:
        strategy = _resolve_assistant_only_strategy(
            {
                "loss_on_assistant_only": True,
                "response_template": "<|im_start|>assistant\n",
                "packing": True,
            },
            train_sample={
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ]
            },
        )

        self.assertFalse(strategy["use_native_assistant_only_loss"])
        self.assertTrue(strategy["use_manual_assistant_masks"])
        self.assertFalse(strategy["use_response_template_collator"])

    def test_assistant_only_strategy_rejects_packed_non_builtin_runs(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires one of Tenyson's built-in SFT dataset schemas",
        ):
            _resolve_assistant_only_strategy(
                {
                    "loss_on_assistant_only": True,
                    "response_template": "<|im_start|>assistant\n",
                    "packing": True,
                },
                train_sample={"text": "plain formatted sample"},
            )

    def test_assistant_only_strategy_uses_manual_masks_without_response_template_for_builtin_rows(self) -> None:
        strategy = _resolve_assistant_only_strategy(
            {
                "loss_on_assistant_only": True,
                "packing": False,
            },
            train_sample={
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ]
            },
        )

        self.assertFalse(strategy["use_native_assistant_only_loss"])
        self.assertTrue(strategy["use_manual_assistant_masks"])
        self.assertFalse(strategy["use_response_template_collator"])

    def test_resolve_sft_special_tokens_kwargs_uses_tokenizer_tokens(self) -> None:
        tokenizer = SimpleNamespace(eos_token="<|im_end|>", pad_token="<|PAD_TOKEN|>")

        resolved = _resolve_sft_special_tokens_kwargs(
            tokenizer,
            accepted_fields={"eos_token", "pad_token", "run_name"},
        )

        self.assertEqual(
            resolved,
            {
                "eos_token": "<|im_end|>",
                "pad_token": "<|PAD_TOKEN|>",
            },
        )


if __name__ == "__main__":
    unittest.main()
