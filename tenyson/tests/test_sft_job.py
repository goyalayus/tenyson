from types import SimpleNamespace
import unittest

from tenyson.jobs.sft import (
    _enable_best_model_tracking,
    _push_final_adapter_snapshot,
    _resolve_early_stopping_settings,
)


class FakePusher:
    def __init__(self):
        self.calls = []

    def push_to_hub(self, repo_id, commit_message):
        self.calls.append((repo_id, commit_message))


class SFTJobHelperTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
