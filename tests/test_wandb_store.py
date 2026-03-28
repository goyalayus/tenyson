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
    def __init__(
        self,
        run_id: str,
        url: str | None = None,
        *,
        name: str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        created_at=None,
        updated_at=None,
        state: str | None = None,
    ):
        self.id = run_id
        self.url = url or f"https://wandb.example/runs/{run_id}"
        self.summary = FakeSummary()
        self.logged_artifacts = []
        self.name = name or run_id
        self.group = group or ""
        self.job_type = job_type or ""
        self.created_at = created_at
        self.updated_at = updated_at
        self.state = state

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

    def runs(self, path, filters=None, order=None, per_page=None, lazy=None):
        self._module.last_runs_call = {
            "path": path,
            "filters": filters,
            "order": order,
            "per_page": per_page,
            "lazy": lazy,
        }
        return self._module.api_runs


def build_fake_wandb_module(current_run=None, api_run=None):
    module = ModuleType("wandb")
    module.run = current_run
    module.init_calls = []
    module.finish_calls = 0
    module.api_run = api_run
    module.api_runs = []
    module.api_run_error = None
    module.last_runs_call = None
    module.api_calls = []

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
    def _api_factory(*args, **kwargs):
        module.api_calls.append({"args": args, "kwargs": kwargs})
        return FakeApi(module)

    module.Api = _api_factory
    module.Artifact = FakeArtifact
    return module


class WandBStoreTests(unittest.TestCase):
    def test_fetch_run_uses_configured_api_timeout(self) -> None:
        fake_wandb = build_fake_wandb_module(api_run=FakeRun("run123"))
        with patch.dict(sys.modules, {"wandb": fake_wandb}), patch.dict(
            os.environ,
            {"TENYSON_WANDB_API_TIMEOUT": "17"},
            clear=False,
        ):
            resolved = wandb_store.fetch_run(
                "wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="sft",
                run_name="wordle_sft_main",
            )

        self.assertEqual(resolved.id, "run123")
        self.assertEqual(fake_wandb.api_calls[0]["kwargs"].get("timeout"), 17)

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

    def test_ensure_run_uses_attempt_specific_run_id_when_present(self) -> None:
        fake_wandb = build_fake_wandb_module()
        with patch.dict(sys.modules, {"wandb": fake_wandb}), patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            run = wandb_store.ensure_run(
                "wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="rl",
                run_name="wordle_rl_main",
                attempt_token="abc123",
            )

        self.assertEqual(len(fake_wandb.init_calls), 1)
        self.assertEqual(
            fake_wandb.init_calls[0]["id"],
            wandb_store.build_run_id(
                "wordle_exp",
                "rl",
                "wordle_rl_main",
                attempt_token="abc123",
            ),
        )
        self.assertEqual(run.summary[wandb_store.SUMMARY_ATTEMPT_TOKEN], "abc123")

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

    def test_fetch_run_without_attempt_token_returns_latest_matching_attempt(self) -> None:
        older = FakeRun(
            "older-run",
            name="wordle_rl_main",
            group="wordle_exp",
            job_type="rl",
            updated_at="2026-03-25T10:00:00+00:00",
        )
        older.summary.update(
            {
                wandb_store.SUMMARY_EXPERIMENT_ID: "wordle_exp",
                wandb_store.SUMMARY_PHASE: "rl",
                wandb_store.SUMMARY_RUN_NAME: "wordle_rl_main",
                wandb_store.SUMMARY_ATTEMPT_TOKEN: "older",
                wandb_store.SUMMARY_HEARTBEAT_AT: "2026-03-25T10:00:00+00:00",
            }
        )
        newer = FakeRun(
            "newer-run",
            name="wordle_rl_main",
            group="wordle_exp",
            job_type="rl",
            updated_at="2026-03-25T11:00:00+00:00",
        )
        newer.summary.update(
            {
                wandb_store.SUMMARY_EXPERIMENT_ID: "wordle_exp",
                wandb_store.SUMMARY_PHASE: "rl",
                wandb_store.SUMMARY_RUN_NAME: "wordle_rl_main",
                wandb_store.SUMMARY_ATTEMPT_TOKEN: "newer",
                wandb_store.SUMMARY_HEARTBEAT_AT: "2026-03-25T11:00:00+00:00",
            }
        )
        fake_wandb = build_fake_wandb_module(api_run=older)
        fake_wandb.api_runs = [older, newer]

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            resolved = wandb_store.fetch_run(
                "wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="rl",
                run_name="wordle_rl_main",
            )

        self.assertIs(resolved, newer)

    def test_fetch_run_ignores_inconsistent_summary_run_name(self) -> None:
        corrupted = FakeRun(
            "corrupted-run",
            name="mixed_rl",
            group="wordle_exp",
            job_type="rl",
            updated_at="2026-03-25T12:30:25+00:00",
        )
        corrupted.summary.update(
            {
                wandb_store.SUMMARY_EXPERIMENT_ID: "wordle_exp",
                wandb_store.SUMMARY_PHASE: "rl",
                wandb_store.SUMMARY_RUN_NAME: "curr_rl_t2",
                wandb_store.SUMMARY_ATTEMPT_TOKEN: "bad",
                wandb_store.SUMMARY_HEARTBEAT_AT: "2026-03-25T12:30:25+00:00",
            }
        )
        valid = FakeRun(
            "valid-run",
            name="curr_rl_t2",
            group="wordle_exp",
            job_type="rl",
            updated_at="2026-03-26T13:29:56+00:00",
        )
        valid.summary.update(
            {
                wandb_store.SUMMARY_EXPERIMENT_ID: "wordle_exp",
                wandb_store.SUMMARY_PHASE: "rl",
                wandb_store.SUMMARY_RUN_NAME: "curr_rl_t2",
                wandb_store.SUMMARY_ATTEMPT_TOKEN: "good",
                wandb_store.SUMMARY_HEARTBEAT_AT: "2026-03-26T13:29:56+00:00",
            }
        )
        fake_wandb = build_fake_wandb_module(api_run=valid)
        fake_wandb.api_runs = [corrupted, valid]

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            resolved = wandb_store.fetch_run(
                "wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="rl",
                run_name="curr_rl_t2",
            )

        self.assertIs(resolved, valid)

    def test_list_live_runs_excludes_terminal_wandb_states(self) -> None:
        now = wandb_store._parse_datetime("2026-03-27T12:00:00+00:00")
        stale = FakeRun(
            "failed-run",
            name="wordle_eval_main",
            group="wordle_exp",
            job_type="eval",
            updated_at="2026-03-27T12:00:00+00:00",
            state="failed",
        )
        stale.summary.update(
            {
                wandb_store.SUMMARY_EXPERIMENT_ID: "wordle_exp",
                wandb_store.SUMMARY_PHASE: "eval",
                wandb_store.SUMMARY_RUN_NAME: "wordle_eval_main",
                wandb_store.SUMMARY_STATUS: "running",
                wandb_store.SUMMARY_IS_ACTIVE: True,
                wandb_store.SUMMARY_HEARTBEAT_AT: "2026-03-27T12:00:00+00:00",
            }
        )
        live = FakeRun(
            "running-run",
            name="wordle_rl_main",
            group="wordle_exp",
            job_type="rl",
            updated_at="2026-03-27T11:59:45+00:00",
            state="running",
        )
        live.summary.update(
            {
                wandb_store.SUMMARY_EXPERIMENT_ID: "wordle_exp",
                wandb_store.SUMMARY_PHASE: "rl",
                wandb_store.SUMMARY_RUN_NAME: "wordle_rl_main",
                wandb_store.SUMMARY_STATUS: "running",
                wandb_store.SUMMARY_IS_ACTIVE: True,
                wandb_store.SUMMARY_HEARTBEAT_AT: "2026-03-27T11:59:45+00:00",
                wandb_store.SUMMARY_PROVIDER: "modal",
                wandb_store.SUMMARY_ATTEMPT_TOKEN: "attempt-live",
            }
        )
        fake_wandb = build_fake_wandb_module()
        fake_wandb.api_runs = [stale, live]

        with patch.dict(sys.modules, {"wandb": fake_wandb}), patch.object(
            wandb_store,
            "_utc_now",
            return_value=now,
        ):
            rows = wandb_store.list_live_runs(
                "wandb://ayush/wordle",
                experiment_id="wordle_exp",
                max_age_seconds=90,
            )

        self.assertEqual(
            rows,
            [
                {
                    "run_id": "wordle_rl_main",
                    "phase": "rl",
                    "provider": "modal",
                    "status": "running",
                    "is_active": True,
                    "attempt_token": "attempt-live",
                    "created_at": None,
                    "updated_at": wandb_store._parse_datetime("2026-03-27T11:59:45+00:00"),
                }
            ],
        )
        self.assertEqual(fake_wandb.last_runs_call["filters"], {"group": "wordle_exp"})
        self.assertEqual(fake_wandb.last_runs_call["order"], "-created_at")
        self.assertEqual(fake_wandb.last_runs_call["per_page"], 200)
        self.assertEqual(fake_wandb.last_runs_call["lazy"], True)

    def test_fetch_run_result_prefers_latest_completed_attempt_over_newer_active_one(self) -> None:
        completed = FakeRun(
            "completed-run",
            name="sft_main",
            group="wordle_exp",
            job_type="sft",
            updated_at="2026-03-25T12:30:10+00:00",
        )
        completed.summary.update(
            {
                wandb_store.SUMMARY_EXPERIMENT_ID: "wordle_exp",
                wandb_store.SUMMARY_PHASE: "sft",
                wandb_store.SUMMARY_RUN_NAME: "sft_main",
                wandb_store.SUMMARY_ATTEMPT_TOKEN: "done",
                wandb_store.SUMMARY_HEARTBEAT_AT: "2026-03-25T12:30:10+00:00",
                wandb_store.SUMMARY_JOB_RESULT_JSON: '{"status":"partial","run_id":"sft_main"}',
                wandb_store.SUMMARY_RESULTS_JSON: '{"metrics":{"global_step":256}}',
            }
        )
        active = FakeRun(
            "active-run",
            name="sft_main",
            group="wordle_exp",
            job_type="sft",
            updated_at="2026-03-26T18:53:17+00:00",
        )
        active.summary.update(
            {
                wandb_store.SUMMARY_EXPERIMENT_ID: "wordle_exp",
                wandb_store.SUMMARY_PHASE: "sft",
                wandb_store.SUMMARY_RUN_NAME: "sft_main",
                wandb_store.SUMMARY_ATTEMPT_TOKEN: "live",
                wandb_store.SUMMARY_HEARTBEAT_AT: "2026-03-26T18:53:17+00:00",
                wandb_store.SUMMARY_STATUS: "running",
                wandb_store.SUMMARY_IS_ACTIVE: True,
            }
        )
        fake_wandb = build_fake_wandb_module(api_run=active)
        fake_wandb.api_runs = [completed, active]

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            result = wandb_store.fetch_run_result(
                "wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="sft",
                run_name="sft_main",
            )

        self.assertEqual(
            result,
            ({"metrics": {"global_step": 256}}, {"status": "partial", "run_id": "sft_main"}),
        )

    def test_fetch_run_result_prefers_newer_completed_attempt_over_older_canonical_result(self) -> None:
        canonical = FakeRun(
            "canonical-run",
            name="mixed_rl",
            group="wordle_exp",
            job_type="rl",
            updated_at="2026-03-25T12:00:00+00:00",
        )
        canonical.summary.update(
            {
                wandb_store.SUMMARY_EXPERIMENT_ID: "wordle_exp",
                wandb_store.SUMMARY_PHASE: "rl",
                wandb_store.SUMMARY_RUN_NAME: "mixed_rl",
                wandb_store.SUMMARY_HEARTBEAT_AT: "2026-03-25T12:00:00+00:00",
                wandb_store.SUMMARY_JOB_RESULT_JSON: '{"status":"failed","run_id":"mixed_rl"}',
                wandb_store.SUMMARY_RESULTS_JSON: '{"metrics":{"reward":0.1}}',
            }
        )
        retried = FakeRun(
            "retried-run",
            name="mixed_rl",
            group="wordle_exp",
            job_type="rl",
            updated_at="2026-03-26T12:00:00+00:00",
        )
        retried.summary.update(
            {
                wandb_store.SUMMARY_EXPERIMENT_ID: "wordle_exp",
                wandb_store.SUMMARY_PHASE: "rl",
                wandb_store.SUMMARY_RUN_NAME: "mixed_rl",
                wandb_store.SUMMARY_ATTEMPT_TOKEN: "retry",
                wandb_store.SUMMARY_HEARTBEAT_AT: "2026-03-26T12:00:00+00:00",
                wandb_store.SUMMARY_JOB_RESULT_JSON: '{"status":"stopped","run_id":"mixed_rl"}',
                wandb_store.SUMMARY_RESULTS_JSON: '{"metrics":{"reward":0.4}}',
            }
        )
        fake_wandb = build_fake_wandb_module(api_run=canonical)
        fake_wandb.api_runs = [canonical, retried]

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            result = wandb_store.fetch_run_result(
                "wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="rl",
                run_name="mixed_rl",
            )

        self.assertEqual(
            result,
            ({"metrics": {"reward": 0.4}}, {"status": "stopped", "run_id": "mixed_rl"}),
        )

    def test_fetch_run_result_can_skip_artifact_download(self) -> None:
        completed = FakeRun(
            "completed-run",
            name="eval_baseline_mixed",
            group="wordle_exp",
            job_type="eval",
            updated_at="2026-03-25T12:30:10+00:00",
        )
        completed.summary.update(
            {
                wandb_store.SUMMARY_EXPERIMENT_ID: "wordle_exp",
                wandb_store.SUMMARY_PHASE: "eval",
                wandb_store.SUMMARY_RUN_NAME: "eval_baseline_mixed",
                wandb_store.SUMMARY_HEARTBEAT_AT: "2026-03-25T12:30:10+00:00",
                wandb_store.SUMMARY_JOB_RESULT_JSON: (
                    '{"status":"success","run_id":"eval_baseline_mixed",'
                    '"metrics":{"format_accuracy":0.62}}'
                ),
                wandb_store.SUMMARY_RESULT_ARTIFACT: "artifact-name",
            }
        )
        fake_wandb = build_fake_wandb_module(api_run=completed)
        fake_wandb.api_runs = [completed]

        with patch.dict(sys.modules, {"wandb": fake_wandb}), patch.object(
            wandb_store,
            "fetch_artifact_results",
            side_effect=AssertionError("artifact download should be skipped"),
        ):
            result = wandb_store.fetch_run_result(
                "wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="eval",
                run_name="eval_baseline_mixed",
                include_results_payload=False,
            )

        self.assertEqual(
            result,
            ({}, {"status": "success", "run_id": "eval_baseline_mixed", "metrics": {"format_accuracy": 0.62}}),
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
        self.assertEqual(
            run.summary[wandb_store.SUMMARY_CONTROL_TARGET_EXPERIMENT_ID],
            "wordle_exp",
        )
        self.assertEqual(
            run.summary[wandb_store.SUMMARY_CONTROL_TARGET_PHASE],
            "sft",
        )
        self.assertEqual(
            run.summary[wandb_store.SUMMARY_CONTROL_TARGET_RUN_NAME],
            "wordle_sft_main",
        )
        self.assertEqual(resolve_run_mock.call_args.kwargs["create_if_missing"], True)
        self.assertEqual(
            resolve_run_mock.call_args.kwargs["experiment_id"],
            wandb_store._stop_control_experiment_id("wordle_exp"),
        )
        self.assertEqual(
            resolve_run_mock.call_args.kwargs["phase"],
            wandb_store.CONTROL_PHASE,
        )
        self.assertEqual(
            resolve_run_mock.call_args.kwargs["run_name"],
            wandb_store._stop_control_run_name("sft", "wordle_sft_main"),
        )

    def test_set_stop_requested_passes_attempt_token_to_resolve_run(self) -> None:
        run = FakeRun("run123")
        with patch.object(
            wandb_store,
            "resolve_run",
            return_value=run,
        ) as resolve_run_mock:
            stopped = wandb_store.set_stop_requested(
                "wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="rl",
                run_name="wordle_rl_main",
                requested=True,
                when_iso="2026-03-22T10:00:00+00:00",
                create_if_missing=False,
                attempt_token="attempt-live",
            )

        self.assertTrue(stopped)
        self.assertEqual(
            resolve_run_mock.call_args.kwargs["attempt_token"],
            "attempt-live",
        )
        self.assertEqual(
            resolve_run_mock.call_args.kwargs["experiment_id"],
            wandb_store._stop_control_experiment_id("wordle_exp"),
        )
        self.assertEqual(
            resolve_run_mock.call_args.kwargs["phase"],
            wandb_store.CONTROL_PHASE,
        )
        self.assertEqual(
            resolve_run_mock.call_args.kwargs["run_name"],
            wandb_store._stop_control_run_name("rl", "wordle_rl_main"),
        )

    def test_set_stop_requested_falls_back_to_legacy_live_run_when_control_run_missing(self) -> None:
        live_run = FakeRun("live-run")

        def fake_resolve_run(
            backend_ref,
            *,
            experiment_id,
            phase,
            run_name,
            create_if_missing=False,
            attempt_token=None,
        ):
            _unused = (backend_ref, phase, run_name, create_if_missing, attempt_token)
            if experiment_id == wandb_store._stop_control_experiment_id("wordle_exp"):
                raise LookupError("no control record yet")
            return live_run

        with patch.object(
            wandb_store,
            "resolve_run",
            side_effect=fake_resolve_run,
        ) as resolve_run_mock:
            stopped = wandb_store.set_stop_requested(
                "wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="sft",
                run_name="sft_main",
                requested=True,
                when_iso="2026-03-22T10:00:00+00:00",
                create_if_missing=False,
            )

        self.assertTrue(stopped)
        self.assertEqual(live_run.summary[wandb_store.SUMMARY_STOP_REQUESTED], True)
        self.assertEqual(
            live_run.summary[wandb_store.SUMMARY_STOP_REQUESTED_AT],
            "2026-03-22T10:00:00+00:00",
        )
        self.assertNotIn(wandb_store.SUMMARY_CONTROL_KIND, live_run.summary)
        self.assertEqual(resolve_run_mock.call_count, 2)
        self.assertEqual(
            resolve_run_mock.call_args_list[0].kwargs["experiment_id"],
            wandb_store._stop_control_experiment_id("wordle_exp"),
        )
        self.assertEqual(
            resolve_run_mock.call_args_list[1].kwargs["experiment_id"],
            "wordle_exp",
        )
        self.assertEqual(
            resolve_run_mock.call_args_list[1].kwargs["run_name"],
            "sft_main",
        )

    def test_fetch_stop_request_state_prefers_control_run(self) -> None:
        control = FakeRun("control-run")
        control.summary.update(
            {
                wandb_store.SUMMARY_STOP_REQUESTED: True,
                wandb_store.SUMMARY_STOP_REQUESTED_AT: "2026-03-27T00:00:00+00:00",
            }
        )
        main = FakeRun("main-run")
        main.summary.update(
            {
                wandb_store.SUMMARY_STOP_REQUESTED: False,
                wandb_store.SUMMARY_STOP_REQUESTED_AT: None,
            }
        )

        def fake_fetch_run(
            backend_ref,
            *,
            experiment_id,
            phase,
            run_name,
            attempt_token=None,
        ):
            _unused = (backend_ref, phase, run_name, attempt_token)
            if experiment_id == wandb_store._stop_control_experiment_id("wordle_exp"):
                return control
            return main

        with patch.object(wandb_store, "fetch_run", side_effect=fake_fetch_run):
            requested, requested_at = wandb_store.fetch_stop_request_state(
                "wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="rl",
                run_name="mixed_rl",
                attempt_token="attempt-live",
            )

        self.assertTrue(requested)
        self.assertEqual(requested_at, "2026-03-27T00:00:00+00:00")

    def test_fetch_stop_request_state_falls_back_to_legacy_live_run(self) -> None:
        legacy = FakeRun("legacy-run")
        legacy.summary.update(
            {
                wandb_store.SUMMARY_STOP_REQUESTED: True,
                wandb_store.SUMMARY_STOP_REQUESTED_AT: "2026-03-27T00:00:00+00:00",
            }
        )

        def fake_fetch_run(
            backend_ref,
            *,
            experiment_id,
            phase,
            run_name,
            attempt_token=None,
        ):
            _unused = (backend_ref, phase, run_name, attempt_token)
            if experiment_id == wandb_store._stop_control_experiment_id("wordle_exp"):
                raise LookupError("no control record yet")
            return legacy

        with patch.object(wandb_store, "fetch_run", side_effect=fake_fetch_run):
            requested, requested_at = wandb_store.fetch_stop_request_state(
                "wandb://ayush/wordle",
                experiment_id="wordle_exp",
                phase="sft",
                run_name="sft_main",
            )

        self.assertTrue(requested)
        self.assertEqual(requested_at, "2026-03-27T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
