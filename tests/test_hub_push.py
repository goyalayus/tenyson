from types import SimpleNamespace
import unittest
from unittest.mock import patch

from huggingface_hub.errors import HfHubHTTPError
from requests import Response
from requests.exceptions import ReadTimeout

from tenyson.core.hub_push import PeriodicHubPushCallback, ensure_hf_repo


class FakePusher:
    def __init__(self):
        self.calls = []

    def push_to_hub(self, repo_id, commit_message):
        self.calls.append((repo_id, commit_message))


class HubPushTests(unittest.TestCase):
    def test_ensure_hf_repo_retries_transient_request_failures(self) -> None:
        calls = {"count": 0}

        def flaky_create_repo(*args, **kwargs):
            del args, kwargs
            calls["count"] += 1
            if calls["count"] < 3:
                raise ReadTimeout("temporary timeout")
            return None

        with patch("tenyson.core.hub_push.create_repo", side_effect=flaky_create_repo), patch(
            "tenyson.core.hub_push.time.sleep"
        ) as sleep_mock:
            ensure_hf_repo("org/repo", max_attempts=5, initial_backoff_seconds=0.1)

        self.assertEqual(calls["count"], 3)
        self.assertEqual(sleep_mock.call_count, 2)

    def test_ensure_hf_repo_retries_retryable_hf_http_errors(self) -> None:
        response = Response()
        response.status_code = 504
        response.url = "https://huggingface.co/api/repos/create"
        retryable_error = HfHubHTTPError("gateway timeout", response=response)

        with patch(
            "tenyson.core.hub_push.create_repo",
            side_effect=[retryable_error, None],
        ) as create_repo_mock, patch("tenyson.core.hub_push.time.sleep") as sleep_mock:
            ensure_hf_repo("org/repo", max_attempts=3, initial_backoff_seconds=0.1)

        self.assertEqual(create_repo_mock.call_count, 2)
        sleep_mock.assert_called_once()

    def test_ensure_hf_repo_does_not_retry_non_retryable_hf_http_errors(self) -> None:
        response = Response()
        response.status_code = 400
        response.url = "https://huggingface.co/api/repos/create"
        non_retryable_error = HfHubHTTPError("bad request", response=response)

        with patch(
            "tenyson.core.hub_push.create_repo",
            side_effect=non_retryable_error,
        ) as create_repo_mock, patch("tenyson.core.hub_push.time.sleep") as sleep_mock:
            with self.assertRaises(HfHubHTTPError):
                ensure_hf_repo("org/repo", max_attempts=3, initial_backoff_seconds=0.1)

        create_repo_mock.assert_called_once()
        sleep_mock.assert_not_called()

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
