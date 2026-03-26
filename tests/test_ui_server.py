import json
import sys
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from types import ModuleType
from urllib.request import urlopen
from unittest.mock import patch

import tenyson.core.wandb_store as wandb_store
import tenyson.ui.server as ui_server


class FakeSummary(dict):
    def update(self, values=None):
        if values:
            dict.update(self, values)


class FakeRun:
    def __init__(
        self,
        *,
        experiment_id: str,
        phase: str,
        run_name: str,
        created_at: datetime,
        updated_at: datetime,
        summary: dict,
        history_rows: list[dict] | None = None,
        config: dict | None = None,
    ) -> None:
        self.id = wandb_store.build_run_id(experiment_id, phase, run_name)
        self.name = run_name
        self.job_type = phase
        self.group = experiment_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.summary = FakeSummary(summary)
        self.config = config or {}
        self.url = summary.get(
            wandb_store.SUMMARY_WANDB_URL,
            f"https://wandb.example/runs/{self.id}",
        )
        self._history_rows = list(history_rows or [])

    def scan_history(self, page_size=None):  # noqa: ARG002
        for row in self._history_rows:
            yield row


class FakeApi:
    def __init__(self, module):
        self._module = module

    def runs(self, path):  # noqa: ARG002
        return list(self._module.api_runs)

    def run(self, path):
        run_id = str(path).split("/")[-1]
        return self._module.api_runs_by_id[run_id]


def build_fake_wandb_module(runs: list[FakeRun]) -> ModuleType:
    module = ModuleType("wandb")
    module.api_runs = list(runs)
    module.api_runs_by_id = {run.id: run for run in runs}
    module.Api = lambda: FakeApi(module)
    return module


def make_run(
    *,
    experiment_id: str,
    phase: str,
    run_name: str,
    created_at: datetime,
    heartbeat_at: datetime | None = None,
    status: str = "success",
    is_active: bool = False,
    metrics: dict | None = None,
    results_payload: dict | None = None,
    job_result_payload: dict | None = None,
    history_rows: list[dict] | None = None,
    config: dict | None = None,
    provider: str = "modal",
) -> FakeRun:
    metrics_payload = dict(metrics or {})
    job_payload = {
        "run_id": run_name,
        "status": status,
        "total_time_seconds": 42.0,
    }
    if job_result_payload:
        job_payload.update(job_result_payload)

    summary = {
        wandb_store.SUMMARY_EXPERIMENT_ID: experiment_id,
        wandb_store.SUMMARY_PHASE: phase,
        wandb_store.SUMMARY_RUN_NAME: run_name,
        wandb_store.SUMMARY_STATUS: status,
        wandb_store.SUMMARY_IS_ACTIVE: is_active,
        wandb_store.SUMMARY_PROVIDER: provider,
        wandb_store.SUMMARY_HEARTBEAT_AT: (heartbeat_at or created_at).isoformat(),
        wandb_store.SUMMARY_METRICS_JSON: json.dumps(metrics_payload),
        wandb_store.SUMMARY_JOB_RESULT_JSON: json.dumps(job_payload),
        wandb_store.SUMMARY_PROJECT_URL: "https://wandb.example/ayush/wordle",
    }
    if results_payload is not None:
        summary[wandb_store.SUMMARY_RESULTS_JSON] = json.dumps(results_payload)

    return FakeRun(
        experiment_id=experiment_id,
        phase=phase,
        run_name=run_name,
        created_at=created_at,
        updated_at=heartbeat_at or created_at,
        summary=summary,
        history_rows=history_rows,
        config=config,
    )


class DashboardDataServiceTests(unittest.TestCase):
    def test_list_experiments_groups_runs_by_experiment_and_activity(self) -> None:
        now = datetime.now(timezone.utc)
        older = make_run(
            experiment_id="wordle_old",
            phase="sft",
            run_name="wordle_sft_main",
            created_at=now - timedelta(hours=4),
            heartbeat_at=now - timedelta(hours=3),
            status="success",
            metrics={"train/loss": 1.2},
        )
        newer = make_run(
            experiment_id="wordle_new",
            phase="eval",
            run_name="wordle_eval_mixed",
            created_at=now - timedelta(minutes=45),
            heartbeat_at=now - timedelta(minutes=1),
            status="running",
            is_active=True,
            metrics={"constraint_accuracy": 0.71},
        )

        fake_wandb = build_fake_wandb_module([older, newer])

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            service = ui_server.DashboardDataService(
                backend_ref="wandb://ayush/wordle",
                cache_ttl_seconds=0.0,
            )
            experiments = service.list_experiments()

        self.assertEqual([item["experiment_id"] for item in experiments], ["wordle_new", "wordle_old"])
        self.assertEqual(experiments[0]["active_run_count"], 1)
        self.assertEqual(experiments[0]["status_counts"]["running"], 1)
        self.assertEqual(experiments[0]["phase_counts"]["eval"], 1)

    def test_get_experiment_snapshot_and_run_detail_include_eval_payloads(self) -> None:
        now = datetime.now(timezone.utc)
        sft_run = make_run(
            experiment_id="wordle_exp",
            phase="sft",
            run_name="wordle_sft_main",
            created_at=now - timedelta(hours=2),
            heartbeat_at=now - timedelta(hours=1, minutes=50),
            status="success",
            metrics={"train/loss": 0.82},
            history_rows=[
                {"_step": 1, "train/loss": 1.2, "train/global_step": 8},
                {"_step": 2, "train/loss": 0.82, "train/global_step": 16},
            ],
        )
        eval_run = make_run(
            experiment_id="wordle_exp",
            phase="eval",
            run_name="wordle_eval_mixed",
            created_at=now - timedelta(minutes=20),
            heartbeat_at=now - timedelta(minutes=3),
            status="success",
            metrics={
                "constraint_accuracy": 0.875,
                "dict_accuracy": 0.95,
                "format_accuracy": 1.0,
                "total_samples": 40,
            },
            results_payload={
                "metrics": {
                    "constraint_accuracy": 0.875,
                    "dict_accuracy": 0.95,
                    "format_accuracy": 1.0,
                    "total_samples": 40,
                },
                "detailed_results": [
                    {
                        "prompt": "Prompt one",
                        "completion": "<answer>crate</answer>",
                        "parsed_guess": "crate",
                        "format_ok": True,
                        "dict_ok": True,
                        "consistent": True,
                    },
                    {
                        "prompt": "Prompt two",
                        "completion": "just guessing",
                        "parsed_guess": None,
                        "format_ok": False,
                        "dict_ok": False,
                        "consistent": False,
                    },
                ],
            },
            history_rows=[
                {
                    "_step": 1,
                    "constraint_accuracy": 0.75,
                    "dict_accuracy": 0.88,
                    "format_accuracy": 1.0,
                    "total_samples": 20,
                },
                {
                    "_step": 2,
                    "constraint_accuracy": 0.875,
                    "dict_accuracy": 0.95,
                    "format_accuracy": 1.0,
                    "total_samples": 40,
                },
            ],
            config={"eval": {"max_new_tokens": 96}},
        )

        fake_wandb = build_fake_wandb_module([sft_run, eval_run])

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            service = ui_server.DashboardDataService(
                backend_ref="wandb://ayush/wordle",
                default_experiment_id="wordle_exp",
                cache_ttl_seconds=0.0,
            )
            snapshot = service.get_experiment_snapshot("wordle_exp")
            detail = service.get_run_detail(
                experiment_id="wordle_exp",
                phase="eval",
                run_name="wordle_eval_mixed",
            )

        self.assertEqual(
            [run["run_name"] for run in snapshot["runs"]],
            ["wordle_eval_mixed", "wordle_sft_main"],
        )
        self.assertEqual(snapshot["experiment"]["active_run_count"], 0)
        self.assertEqual(detail["summary"]["phase"], "eval")
        self.assertEqual(detail["result_payload"]["detailed_results"][0]["parsed_guess"], "crate")
        self.assertEqual(detail["history"]["default_keys"][0], "constraint_accuracy")
        self.assertEqual(detail["history"]["rows"][-1]["total_samples"], 40.0)
        self.assertEqual(detail["config"]["eval"]["max_new_tokens"], 96)

    def test_get_run_detail_preserves_rl_rollout_reward_components(self) -> None:
        now = datetime.now(timezone.utc)
        rl_run = make_run(
            experiment_id="wordle_exp",
            phase="rl",
            run_name="wordle_rl_mixed",
            created_at=now - timedelta(minutes=30),
            heartbeat_at=now - timedelta(minutes=2),
            status="success",
            metrics={"train/reward": 0.18, "train/global_step": 24},
            results_payload={
                "metrics": {"total_samples": 2, "rollout_batches": 1},
                "detailed_results": [
                    {
                        "global_step": 24,
                        "rollout_step": 1,
                        "prompt": "Prompt one",
                        "completion": "<guess>crate</guess>",
                        "reward": 0.6,
                        "reward_total": 0.6,
                        "reward_components": {
                            "format_exact": 0.2,
                            "wordle_strict": 0.4,
                        },
                    }
                ],
            },
            history_rows=[
                {"_step": 1, "train/reward": 0.12, "train/global_step": 8},
                {"_step": 2, "train/reward": 0.18, "train/global_step": 24},
            ],
            config={"task": {"wordlists": {"solutions": "a.txt", "allowed": "b.txt"}}},
        )

        fake_wandb = build_fake_wandb_module([rl_run])

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            service = ui_server.DashboardDataService(
                backend_ref="wandb://ayush/wordle",
                default_experiment_id="wordle_exp",
                cache_ttl_seconds=0.0,
            )
            detail = service.get_run_detail(
                experiment_id="wordle_exp",
                phase="rl",
                run_name="wordle_rl_mixed",
            )

        row = detail["result_payload"]["detailed_results"][0]
        self.assertEqual(row["rollout_step"], 1)
        self.assertEqual(row["reward_total"], 0.6)
        self.assertEqual(
            row["reward_components"],
            {
                "format_exact": 0.2,
                "wordle_strict": 0.4,
            },
        )


class DashboardRequestHandlerTests(unittest.TestCase):
    def test_http_endpoints_serve_json_and_static_assets(self) -> None:
        now = datetime.now(timezone.utc)
        eval_run = make_run(
            experiment_id="wordle_exp",
            phase="eval",
            run_name="wordle_eval_mixed",
            created_at=now - timedelta(minutes=10),
            heartbeat_at=now - timedelta(minutes=1),
            status="running",
            is_active=True,
            metrics={"constraint_accuracy": 0.9, "total_samples": 12},
            results_payload={
                "metrics": {"constraint_accuracy": 0.9, "total_samples": 12},
                "detailed_results": [],
            },
            history_rows=[
                {"_step": 1, "constraint_accuracy": 0.8, "total_samples": 6},
                {"_step": 2, "constraint_accuracy": 0.9, "total_samples": 12},
            ],
        )
        fake_wandb = build_fake_wandb_module([eval_run])

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            service = ui_server.DashboardDataService(
                backend_ref="wandb://ayush/wordle",
                default_experiment_id="wordle_exp",
                cache_ttl_seconds=0.0,
            )
            config = ui_server.DashboardServerConfig(
                backend_ref="wandb://ayush/wordle",
                experiment_id="wordle_exp",
                host="127.0.0.1",
                port=0,
                refresh_seconds=7.0,
            )
            handler = ui_server._build_request_handler(service=service, config=config)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urlopen(f"{base_url}/api/config", timeout=5) as response:
                    config_payload = json.loads(response.read().decode("utf-8"))
                with urlopen(
                    f"{base_url}/api/run?experiment_id=wordle_exp&phase=eval&run_name=wordle_eval_mixed",
                    timeout=5,
                ) as response:
                    run_payload = json.loads(response.read().decode("utf-8"))
                with urlopen(f"{base_url}/assets/app.js", timeout=5) as response:
                    app_js = response.read().decode("utf-8")
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertEqual(config_payload["default_experiment_id"], "wordle_exp")
        self.assertEqual(config_payload["refresh_seconds"], 7.0)
        self.assertEqual(run_payload["summary"]["run_name"], "wordle_eval_mixed")
        self.assertEqual(run_payload["result_payload"]["metrics"]["constraint_accuracy"], 0.9)
        self.assertIn("Tenyson Dashboard", app_js)


if __name__ == "__main__":
    unittest.main()
