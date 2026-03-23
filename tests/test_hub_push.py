from types import SimpleNamespace
import unittest

from tenyson.core.hub_push import PeriodicHubPushCallback


class FakePusher:
    def __init__(self):
        self.calls = []

    def push_to_hub(self, repo_id, commit_message):
        self.calls.append((repo_id, commit_message))


class HubPushTests(unittest.TestCase):
    def test_push_every_steps_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "push_every_steps"):
            PeriodicHubPushCallback("org/repo", "run", 0)

    def test_on_step_end_pushes_once_per_interval(self) -> None:
        callback = PeriodicHubPushCallback("org/repo", "wordle_run", 2)
        model = FakePusher()
        state = SimpleNamespace(global_step=2)
        control = SimpleNamespace()

        callback.on_step_end(None, state, control, model=model)
        callback.on_step_end(None, state, control, model=model)

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][0], "org/repo")
        self.assertIn("periodic push", model.calls[0][1])

    def test_on_train_end_pushes_model_and_tokenizer(self) -> None:
        model = FakePusher()
        tokenizer = FakePusher()
        callback = PeriodicHubPushCallback(
            "org/repo",
            "wordle_run",
            5,
            tokenizer=tokenizer,
        )
        state = SimpleNamespace(global_step=7)
        control = SimpleNamespace()

        callback.on_train_end(None, state, control, model=model)

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(tokenizer.calls), 1)
        self.assertIn("final push", model.calls[0][1])
        self.assertIn("final push", tokenizer.calls[0][1])


if __name__ == "__main__":
    unittest.main()
