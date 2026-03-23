import os
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import tenyson.core.wandb_store as wandb_store


class FakeSummary(dict):
    def update(self, values=None):
        if values:
            dict.update(self, values)


class FakeRun:
    def __init__(self, run_id: str, url: str | None = None):
        self.id = run_id
        self.url = url or f"https://wandb.example/runs/{run_id}"
        self.summary = FakeSummary()
        self.logged_artifacts = []

    def log_artifact(self, artifact):
        self.logged_artifacts.append(artifact)


class FakeArtifact:
    def __init__(self, name, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path, name=None):
        self.files.append((path, name))


class FakeApi:
    def __init__(self, module):
        self._module = module

    def run(self, path):
        if self._module.api_run_error is not None:
            raise self._module.api_run_error
        return self._module.api_run

    def runs(self, path):
        return self._module.api_runs


def build_fake_wandb_module(current_run=None, api_run=None):
    module = ModuleType("wandb")
    module.run = current_run
    module.init_calls = []
    module.finish_calls = 0
    module.api_run = api_run
    module.api_runs = []
    module.api_run_error = None

    def init(**kwargs):
        run = FakeRun(kwargs["id"])
        run.init_kwargs = kwargs
        module.run = run
        module.init_calls.append(kwargs)
        return run

    def finish():
        module.finish_calls += 1
        module.run = None

    module.init = init
    module.finish = finish
    module.Api = lambda: FakeApi(module)
    module.Artifact = FakeArtifact
    return module


class WandBStoreTests(unittest.TestCase):
    def test_ensure_run_initializes_run_and_updates_summary(self) -> None:
        fake_wandb = build_fake_wandb_module()
        with patch.dict(sys.modules, {"wandb": fake_wandb}), patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            run = wandb_store.ensure_run(
                "wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="sft",
                run_name="wordle_sft_main",
                config={"training": {"max_steps": 2}},
            )

        self.assertEqual(len(fake_wandb.init_calls), 1)
        init_kwargs = fake_wandb.init_calls[0]
        self.assertEqual(
            init_kwargs["id"],
            wandb_store.build_run_id("wordle_exp", "sft", "wordle_sft_main"),
        )
        self.assertEqual(init_kwargs["name"], "wordle_sft_main")
        self.assertEqual(init_kwargs["group"], "wordle_exp")
        self.assertEqual(init_kwargs["job_type"], "sft")
        self.assertEqual(run.summary[wandb_store.SUMMARY_EXPERIMENT_ID], "wordle_exp")
        self.assertEqual(run.summary[wandb_store.SUMMARY_PHASE], "sft")
        self.assertEqual(run.summary[wandb_store.SUMMARY_RUN_NAME], "wordle_sft_main")

    def test_ensure_run_finishes_mismatched_active_run_before_init(self) -> None:
        fake_wandb = build_fake_wandb_module(current_run=FakeRun("some-other-run"))

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            wandb_store.ensure_run(
                "wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="rl",
                run_name="wordle_rl_main",
            )

        self.assertEqual(fake_wandb.finish_calls, 1)
        self.assertEqual(len(fake_wandb.init_calls), 1)

    def test_resolve_run_prefers_matching_active_run(self) -> None:
        expected_id = wandb_store.build_run_id("wordle_exp", "eval", "wordle_eval")
        matching_run = FakeRun(expected_id)

        with patch.object(wandb_store, "active_run", return_value=matching_run), patch.object(
            wandb_store,
            "fetch_run",
        ) as fetch_run_mock:
            resolved = wandb_store.resolve_run(
                "wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="eval",
                run_name="wordle_eval",
            )

        self.assertIs(resolved, matching_run)
        fetch_run_mock.assert_not_called()

    def test_fetch_run_url_filters_by_attempt_token(self) -> None:
        run = FakeRun("run123")
        run.summary[wandb_store.SUMMARY_ATTEMPT_TOKEN] = "abc123"

        with patch.object(wandb_store, "fetch_run", return_value=run):
            self.assertEqual(
                wandb_store.fetch_run_url(
                    "wandb://ayush/wordle",
                    experiment_id="wordle_exp",
                    phase="sft",
                    run_name="wordle_sft_main",
                    attempt_token="abc123",
                ),
                run.url,
            )
            self.assertIsNone(
                wandb_store.fetch_run_url(
                    "wandb://ayush/wordle",
                    experiment_id="wordle_exp",
                    phase="sft",
                    run_name="wordle_sft_main",
                    attempt_token="wrong-token",
                )
            )

    def test_set_stop_requested_passes_create_if_missing_to_resolve_run(self) -> None:
        run = FakeRun("run123")
        with patch.object(
            wandb_store,
            "resolve_run",
            return_value=run,
        ) as resolve_run_mock:
            stopped = wandb_store.set_stop_requested(
                "wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="sft",
                run_name="wordle_sft_main",
                requested=True,
                when_iso="2026-03-22T10:00:00+00:00",
                create_if_missing=True,
            )

        self.assertTrue(stopped)
        self.assertEqual(run.summary[wandb_store.SUMMARY_STOP_REQUESTED], True)
        self.assertEqual(
            run.summary[wandb_store.SUMMARY_STOP_REQUESTED_AT],
            "2026-03-22T10:00:00+00:00",
        )
        self.assertEqual(resolve_run_mock.call_args.kwargs["create_if_missing"], True)


if __name__ == "__main__":
    unittest.main()
