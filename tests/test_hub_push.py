from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from huggingface_hub.errors import HfHubHTTPError
from requests import Response
from requests.exceptions import ReadTimeout

from tenyson.core.hub_push import (
    PeriodicHubPushCallback,
    _resolve_stale_root_artifact_files,
    ensure_hf_repo,
    push_pretrained_snapshot_to_hub,
)


class FakeSaveable:
    def __init__(self, files):
        self.files = dict(files)
        self.calls = []

    def save_pretrained(self, output_dir):
        root = Path(output_dir)
        self.calls.append(str(root))
        for relative_path, content in self.files.items():
            target = root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")


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

    def test_resolve_stale_root_artifact_files_deletes_old_model_shards_and_index(self) -> None:
        stale_files = _resolve_stale_root_artifact_files(
            remote_files=[
                "README.md",
                "config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json",
                "tokenizer.json",
                "checkpoint-10/model.safetensors",
            ],
            local_files=[
                "config.json",
                "model.safetensors",
                "tokenizer.json",
            ],
            include_tokenizer=True,
        )

        self.assertEqual(
            stale_files,
            [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json",
            ],
        )

    def test_resolve_stale_root_artifact_files_preserves_tokenizer_when_not_uploading_one(self) -> None:
        stale_files = _resolve_stale_root_artifact_files(
            remote_files=[
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
            local_files=[
                "config.json",
                "model.safetensors",
            ],
            include_tokenizer=False,
        )

        self.assertEqual(stale_files, [])

    def test_push_every_steps_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "push_every_steps"):
            PeriodicHubPushCallback("org/repo", "run", 0)

    def test_push_pretrained_snapshot_uploads_current_files_and_deletes_stale_root_artifacts(self) -> None:
        model = FakeSaveable(
            {
                "config.json": "{}",
                "model.safetensors": "weights",
            }
        )
        tokenizer = FakeSaveable(
            {
                "tokenizer.json": "{}",
            }
        )

        class FakeApi:
            def __init__(self) -> None:
                self.upload_kwargs = None
                self.uploaded_files = None
                self.commit_kwargs = None

            def list_repo_files(self, repo_id):
                return [
                    "README.md",
                    "config.json",
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                    "model.safetensors.index.json",
                    "tokenizer.json",
                ]

            def create_commit(self, **kwargs):
                self.commit_kwargs = kwargs

            def upload_folder(self, **kwargs):
                self.upload_kwargs = kwargs
                folder_path = Path(kwargs["folder_path"])
                self.uploaded_files = sorted(
                    path.relative_to(folder_path).as_posix()
                    for path in folder_path.rglob("*")
                    if path.is_file()
                )

        fake_api = FakeApi()

        with patch("tenyson.core.hub_push.ensure_hf_repo") as ensure_repo_mock:
            with patch("tenyson.core.hub_push.HfApi", return_value=fake_api):
                push_pretrained_snapshot_to_hub(
                    "org/repo",
                    model=model,
                    tokenizer=tokenizer,
                    commit_message="sync snapshot",
                )

        ensure_repo_mock.assert_called_once_with("org/repo")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(tokenizer.calls), 1)
        self.assertEqual(fake_api.upload_kwargs["repo_id"], "org/repo")
        self.assertEqual(fake_api.upload_kwargs["commit_message"], "sync snapshot")
        self.assertIsNone(fake_api.upload_kwargs.get("delete_patterns"))
        self.assertEqual(
            [op.path_in_repo for op in fake_api.commit_kwargs["operations"]],
            [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json",
            ],
        )
        self.assertEqual(
            fake_api.uploaded_files,
            [
                "config.json",
                "model.safetensors",
                "tokenizer.json",
            ],
        )

    def test_on_step_end_pushes_once_per_interval(self) -> None:
        callback = PeriodicHubPushCallback("org/repo", "wordle_run", 2)
        model = object()
        state = SimpleNamespace(global_step=2)
        control = SimpleNamespace()

        with patch(
            "tenyson.core.hub_push.push_pretrained_snapshot_to_hub"
        ) as push_mock:
            callback.on_step_end(None, state, control, model=model)
            callback.on_step_end(None, state, control, model=model)

        push_mock.assert_called_once()
        _, kwargs = push_mock.call_args
        self.assertEqual(kwargs["model"], model)
        self.assertIsNone(kwargs["tokenizer"])
        self.assertIn("periodic push", kwargs["commit_message"])

    def test_on_train_end_pushes_model_and_tokenizer(self) -> None:
        tokenizer = object()
        callback = PeriodicHubPushCallback(
            "org/repo",
            "wordle_run",
            5,
            tokenizer=tokenizer,
        )
        model = object()
        state = SimpleNamespace(global_step=7)
        control = SimpleNamespace()

        with patch(
            "tenyson.core.hub_push.push_pretrained_snapshot_to_hub"
        ) as push_mock:
            callback.on_train_end(None, state, control, model=model)

        push_mock.assert_called_once()
        _, kwargs = push_mock.call_args
        self.assertEqual(kwargs["model"], model)
        self.assertEqual(kwargs["tokenizer"], tokenizer)
        self.assertIn("final push", kwargs["commit_message"])


if __name__ == "__main__":
    unittest.main()
