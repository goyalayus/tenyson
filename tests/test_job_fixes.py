import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import patch

from datasets import Dataset
import torch

import tenyson.core.telemetry as telemetry_module
from tenyson.core.chat_generation import prepare_generation_prompts
from tenyson.core.telemetry import RLRolloutTracker
import tenyson.jobs.eval as eval_module
import tenyson.jobs.rl as rl_module
from tenyson.jobs.eval import (
    EvalJob,
    _require_eval_vllm_config,
    _resolve_eval_fast_inference_requested,
    _resolve_eval_model_load_kwargs,
    _should_force_transformers_eval_for_adapter,
)
from tenyson.jobs.rl import (
    _RAW_PROMPT_COLUMN,
    _build_grpo_vllm_overrides,
    _build_vllm_sampling_params_kwargs,
    _build_vllm_generation_kwargs,
    _ensure_trl_vllm_guided_decoding_compat,
    _ensure_trl_vllm_sampling_params_compat,
    _prepare_rl_generation_dataset,
    _RewardTelemetryCollector,
    _require_rl_vllm_config,
    _resolve_unsloth_model_load_kwargs,
    _resolve_grpo_max_completion_length,
    _wrap_reward_func_for_telemetry,
    RLJob,
)
from tenyson.jobs.reporting_utils import normalize_report_to


class _FakeChatTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, **kwargs):
        recorded_messages = [dict(message) for message in messages]
        self.calls.append(
            {
                "messages": recorded_messages,
                **dict(kwargs),
            }
        )
        last_user_message = next(
            (
                str(message["content"])
                for message in reversed(recorded_messages)
                if message.get("role") == "user"
            ),
            "",
        )
        last_assistant_message = next(
            (
                str(message["content"])
                for message in reversed(recorded_messages)
                if message.get("role") == "assistant"
            ),
            "",
        )
        if kwargs.get("continue_final_message"):
            return (
                f"CHAT_CONTINUE::{last_user_message}::assistant="
                f"{last_assistant_message}::thinking="
                f"{kwargs.get('enable_thinking', 'default')}"
            )
        return (
            f"CHAT::{last_user_message}::thinking="
            f"{kwargs.get('enable_thinking', 'default')}"
        )


class ChatGenerationPromptTests(unittest.TestCase):
    def test_prepare_generation_prompts_wraps_plain_prompt_with_chat_template(self) -> None:
        tokenizer = _FakeChatTokenizer()

        prepared_prompts = prepare_generation_prompts(
            [{"prompt": "What is 2 + 2?"}],
            tokenizer,
            {
                "chat_template": {
                    "enabled": True,
                    "enable_thinking": False,
                }
            },
        )

        self.assertEqual(len(prepared_prompts), 1)
        self.assertEqual(prepared_prompts[0].raw_prompt_text, "What is 2 + 2?")
        self.assertEqual(
            prepared_prompts[0].generation_prompt,
            "CHAT::What is 2 + 2?::thinking=False",
        )
        self.assertEqual(
            tokenizer.calls,
            [
                {
                    "messages": [
                        {"role": "user", "content": "What is 2 + 2?"},
                    ],
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                }
            ],
        )

    def test_prepare_generation_prompts_uses_messages_for_generation_but_prompt_for_metrics(
        self,
    ) -> None:
        tokenizer = _FakeChatTokenizer()

        prepared_prompts = prepare_generation_prompts(
            [
                {
                    "prompt": "raw metric prompt",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "What is 2 + 2?"},
                    ],
                }
            ],
            tokenizer,
            {"chat_template": {"enabled": True}},
        )

        self.assertEqual(prepared_prompts[0].raw_prompt_text, "raw metric prompt")
        self.assertEqual(
            prepared_prompts[0].generation_prompt,
            "CHAT::What is 2 + 2?::thinking=default",
        )

    def test_prepare_generation_prompts_continues_prefilled_assistant_message(
        self,
    ) -> None:
        tokenizer = _FakeChatTokenizer()

        prepared_prompts = prepare_generation_prompts(
            [
                {
                    "prompt": "raw metric prompt",
                    "messages": [
                        {"role": "user", "content": "What is 314 + 592?"},
                        {"role": "assistant", "content": "<answer>"},
                    ],
                }
            ],
            tokenizer,
            {"chat_template": {"enabled": True, "enable_thinking": False}},
        )

        self.assertEqual(prepared_prompts[0].raw_prompt_text, "raw metric prompt")
        self.assertEqual(
            prepared_prompts[0].generation_prompt,
            "CHAT_CONTINUE::What is 314 + 592?::assistant=<answer>::thinking=False",
        )
        self.assertEqual(
            tokenizer.calls,
            [
                {
                    "messages": [
                        {"role": "user", "content": "What is 314 + 592?"},
                        {"role": "assistant", "content": "<answer>"},
                    ],
                    "tokenize": False,
                    "add_generation_prompt": False,
                    "continue_final_message": True,
                    "enable_thinking": False,
                }
            ],
        )


class EvalChatTemplatePromptTests(unittest.TestCase):
    def test_eval_job_keeps_raw_prompts_for_metrics_but_templates_generation(self) -> None:
        tokenizer = _FakeChatTokenizer()
        dataset = Dataset.from_list([{"prompt": "Prompt one"}])
        job = EvalJob(
            config={
                "evaluation": {"run_name": "eval_test"},
                "chat_template": {
                    "enabled": True,
                    "enable_thinking": False,
                },
            },
            task=object(),
        )

        self.assertEqual(job._extract_prompts(dataset), ["Prompt one"])
        self.assertEqual(
            job._build_generation_prompts(dataset, tokenizer),
            ["CHAT::Prompt one::thinking=False"],
        )


class EvalStopTests(unittest.TestCase):
    def test_eval_run_stops_early_when_stop_requested(self) -> None:
        dataset = Dataset.from_list(
            [
                {"prompt": "prompt-1"},
                {"prompt": "prompt-2"},
            ]
        )

        class FakeTask:
            def get_eval_dataset(self, config):
                return dataset

            def compute_metrics(
                self,
                prompts,
                completions,
                dataset_rows,
                config,
                tokenizer,
            ):
                self.last_prompts = list(prompts)
                self.last_completions = list(completions)
                self.last_rows = dataset_rows
                return {"metrics": {"format_accuracy": 1.0}}

        task = FakeTask()
        job = EvalJob(
            config={
                "evaluation": {"run_name": "eval_test", "batch_size": 1},
                "telemetry": {
                    "backend": "wandb",
                    "entity": "demo",
                    "project": "tenyson",
                    "experiment_id": "wordle_exp",
                    "attempt_token": "attempt-eval",
                },
            },
            task=task,
        )

        fake_client = SimpleNamespace(db_url="wandb://demo/tenyson")

        with patch.object(eval_module, "require_gpu_provider_runtime"), patch.object(
            telemetry_module,
            "resolve_required_telemetry_context",
            return_value=("wandb://demo/tenyson", "wordle_exp"),
        ), patch.object(
            telemetry_module,
            "TelemetryClient",
            return_value=fake_client,
        ), patch.object(
            telemetry_module,
            "ensure_wandb_telemetry_run",
            return_value=None,
        ), patch.object(
            telemetry_module,
            "begin_run_attempt",
            return_value=False,
        ), patch.object(
            telemetry_module,
            "start_run_heartbeat",
        ) as start_run_heartbeat_mock, patch.object(
            telemetry_module,
            "beat_run_heartbeat",
        ) as beat_run_heartbeat_mock, patch.object(
            telemetry_module,
            "run_stop_requested",
            side_effect=[False, True],
        ) as run_stop_requested_mock, patch.object(
            telemetry_module,
            "record_run_summary",
        ) as record_run_summary_mock, patch.object(
            telemetry_module,
            "record_run_result",
        ) as record_run_result_mock, patch.object(
            job,
            "_build_model_and_tokenizer",
            return_value=(object(), object()),
        ), patch.object(
            job,
            "_build_sampling_params",
            return_value=SimpleNamespace(temperature=0.0),
        ), patch.object(
            job,
            "_extract_prompts",
            return_value=["prompt-1", "prompt-2"],
        ), patch.object(
            job,
            "_build_generation_prompts",
            return_value=["prompt-1", "prompt-2"],
        ), patch.object(
            job,
            "_generate_batch",
            side_effect=[["completion-1"]],
        ) as generate_batch_mock:
            result = job.run()

        self.assertEqual(result.status, "stopped")
        self.assertTrue(result.stopped_early)
        self.assertEqual(result.processed_samples, 1)
        self.assertEqual(result.expected_samples, 2)
        self.assertIn("Manual stop requested", result.failure_reason or "")
        self.assertEqual(task.last_prompts, ["prompt-1"])
        self.assertEqual(task.last_completions, ["completion-1"])
        self.assertEqual(len(task.last_rows), 1)
        self.assertEqual(generate_batch_mock.call_count, 1)
        self.assertEqual(run_stop_requested_mock.call_count, 2)
        start_run_heartbeat_mock.assert_called_once_with(
            fake_client,
            "wordle_exp",
            "eval_test",
            "eval",
            attempt_token="attempt-eval",
        )
        beat_run_heartbeat_mock.assert_called_once_with(
            client=fake_client,
            experiment_id="wordle_exp",
            run_id="eval_test",
            phase="eval",
            attempt_token="attempt-eval",
        )
        run_stop_requested_mock.assert_called_with(
            fake_client,
            experiment_id="wordle_exp",
            run_id="eval_test",
            phase="eval",
            attempt_token="attempt-eval",
        )
        recorded_summary_result = record_run_summary_mock.call_args.kwargs["result"]
        recorded_payload = record_run_result_mock.call_args.kwargs["job_result_payload"]
        self.assertEqual(recorded_summary_result.status, "stopped")
        self.assertTrue(recorded_summary_result.stopped_early)
        self.assertEqual(recorded_payload.status, "stopped")
        self.assertTrue(recorded_payload.stopped_early)



class RLJobTelemetryStartupTests(unittest.TestCase):
    def test_rl_run_starts_telemetry_before_model_build(self) -> None:
        job = RLJob(
            config={
                "training": {
                    "run_name": "rl_test",
                    "hf_repo_base": "goyalayus/rl-test",
                },
                "telemetry": {
                    "backend": "wandb",
                    "entity": "demo",
                    "project": "tenyson",
                    "experiment_id": "wordle_exp",
                    "attempt_token": "attempt-rl",
                },
                "vllm": {"enabled": True},
            },
            task=SimpleNamespace(),
        )

        order: list[str] = []
        fake_client = SimpleNamespace(backend="wandb", db_url="wandb://demo/tenyson")

        def remember(label: str):
            def wrapped(*args, **kwargs):
                del args, kwargs
                order.append(label)
                return False if label == "begin_run_attempt" else None

            return wrapped

        def fake_build_model_and_tokenizer():
            self.assertEqual(
                order,
                [
                    "ensure_wandb_telemetry_run",
                    "begin_run_attempt",
                    "start_run_heartbeat",
                ],
            )
            raise RuntimeError("stop after startup ordering check")

        with patch.dict(os.environ, {"HF_TOKEN": "hf-token"}, clear=False), patch.object(
            rl_module, "require_gpu_provider_runtime"
        ), patch.object(
            telemetry_module,
            "resolve_required_telemetry_context",
            return_value=("wandb://demo/tenyson", "wordle_exp"),
        ), patch.object(
            telemetry_module,
            "TelemetryClient",
            return_value=fake_client,
        ), patch.object(
            telemetry_module,
            "ensure_wandb_telemetry_run",
            side_effect=remember("ensure_wandb_telemetry_run"),
        ), patch.object(
            telemetry_module,
            "begin_run_attempt",
            side_effect=remember("begin_run_attempt"),
        ), patch.object(
            telemetry_module,
            "start_run_heartbeat",
            side_effect=remember("start_run_heartbeat"),
        ), patch.object(
            job,
            "_build_model_and_tokenizer",
            side_effect=fake_build_model_and_tokenizer,
        ), self.assertRaisesRegex(RuntimeError, "startup ordering check"):
            job.run()


class EvalJobFailureTelemetryTests(unittest.TestCase):
    def test_eval_run_records_failed_result_when_startup_raises(self) -> None:
        job = EvalJob(
            config={
                "evaluation": {"run_name": "eval_test", "batch_size": 1},
                "telemetry": {
                    "backend": "wandb",
                    "entity": "demo",
                    "project": "tenyson",
                    "experiment_id": "wordle_exp",
                    "attempt_token": "attempt-eval",
                },
            },
            task=SimpleNamespace(),
        )

        fake_client = SimpleNamespace(db_url="wandb://demo/tenyson")
        fake_wandb_run = SimpleNamespace(url="https://wandb.example/runs/eval_test")
        fake_wandb = SimpleNamespace(finish=lambda: None)

        with patch.object(eval_module, "require_gpu_provider_runtime"), patch.object(
            telemetry_module,
            "resolve_required_telemetry_context",
            return_value=("wandb://demo/tenyson", "wordle_exp"),
        ), patch.object(
            telemetry_module,
            "TelemetryClient",
            return_value=fake_client,
        ), patch.object(
            telemetry_module,
            "ensure_wandb_telemetry_run",
            return_value=fake_wandb_run,
        ), patch.object(
            telemetry_module,
            "begin_run_attempt",
            return_value=False,
        ), patch.object(
            telemetry_module,
            "start_run_heartbeat",
        ), patch.object(
            telemetry_module,
            "record_run_summary",
        ) as record_run_summary_mock, patch.object(
            telemetry_module,
            "record_run_result",
        ) as record_run_result_mock, patch.object(
            job,
            "_build_model_and_tokenizer",
            side_effect=RuntimeError("adapter download 504"),
        ), patch.dict(sys.modules, {"wandb": fake_wandb}):
            result = job.run()

        self.assertEqual(result.status, "failed")
        self.assertIn("adapter download 504", result.failure_reason or "")
        self.assertEqual(result.wandb_url, "https://wandb.example/runs/eval_test")
        recorded_summary_result = record_run_summary_mock.call_args.kwargs["result"]
        recorded_payload = record_run_result_mock.call_args.kwargs["job_result_payload"]
        self.assertEqual(recorded_summary_result.status, "failed")
        self.assertEqual(recorded_payload.status, "failed")
        self.assertIn("adapter download 504", recorded_payload.failure_reason or "")


class RewardTelemetryCollectorTests(unittest.TestCase):
    def test_build_results_payload_records_weighted_rollouts_for_wandb(self) -> None:
        collector = _RewardTelemetryCollector(
            experiment_id="wordle_exp",
            run_id="mixed_rl",
            reward_component_names=["format_exact", "wordle_strict"],
            rollout_tracker=RLRolloutTracker(),
        )
        collector.set_reward_weights([1.0, 0.5])

        prompts = [
            {"content": "Prompt one"},
            {"content": "Prompt two"},
        ]
        completions = [
            {"content": "<guess>crate</guess>"},
            {"content": "<guess>zzzzz</guess>"},
        ]
        trainer_state = SimpleNamespace(global_step=12)

        collector.record_component(
            component_name="format_exact",
            prompts=prompts,
            completions=completions,
            rewards=[0.2, 0.0],
            kwargs={"trainer_state": trainer_state},
        )

        pending_payload = collector.build_results_payload()
        self.assertEqual(pending_payload["metrics"]["total_samples"], 0)
        self.assertEqual(pending_payload["metrics"]["rollout_batches"], 0)
        self.assertEqual(pending_payload["detailed_results"], [])

        collector.record_component(
            component_name="wordle_strict",
            prompts=prompts,
            completions=completions,
            rewards=[0.8, -1.0],
            kwargs={"trainer_state": trainer_state},
        )

        payload = collector.build_results_payload()
        self.assertEqual(payload["metrics"]["total_samples"], 2)
        self.assertEqual(payload["metrics"]["rollout_batches"], 1)

        first_row = payload["detailed_results"][0]
        second_row = payload["detailed_results"][1]

        self.assertEqual(first_row["global_step"], 12)
        self.assertEqual(first_row["rollout_step"], 1)
        self.assertEqual(first_row["prompt"], "Prompt one")
        self.assertEqual(first_row["completion"], "<guess>crate</guess>")
        self.assertEqual(
            first_row["reward_components"],
            {
                "format_exact": 0.2,
                "wordle_strict": 0.4,
            },
        )
        self.assertAlmostEqual(first_row["reward_total"], 0.6)
        self.assertAlmostEqual(first_row["reward"], 0.6)

        self.assertEqual(second_row["global_step"], 12)
        self.assertEqual(second_row["rollout_step"], 1)
        self.assertEqual(second_row["prompt"], "Prompt two")
        self.assertEqual(second_row["completion"], "<guess>zzzzz</guess>")
        self.assertEqual(
            second_row["reward_components"],
            {
                "format_exact": 0.0,
                "wordle_strict": -0.5,
            },
        )
        self.assertAlmostEqual(second_row["reward_total"], -0.5)
        self.assertAlmostEqual(second_row["reward"], -0.5)


class RLChatTemplatePromptTests(unittest.TestCase):
    def test_prepare_rl_generation_dataset_rewrites_prompt_and_keeps_raw_prompt_column(
        self,
    ) -> None:
        tokenizer = _FakeChatTokenizer()
        dataset = Dataset.from_list(
            [
                {
                    "prompt": "Prompt one",
                    "difficulty": "easy",
                }
            ]
        )

        prepared_dataset, prompt_lookup = _prepare_rl_generation_dataset(
            dataset,
            tokenizer,
            {
                "chat_template": {
                    "enabled": True,
                    "enable_thinking": False,
                }
            },
        )

        self.assertEqual(len(prepared_dataset), 1)
        self.assertEqual(
            prepared_dataset[0]["prompt"],
            "CHAT::Prompt one::thinking=False",
        )
        self.assertEqual(
            prepared_dataset[0][_RAW_PROMPT_COLUMN],
            "Prompt one",
        )
        self.assertEqual(
            prompt_lookup,
            {"CHAT::Prompt one::thinking=False": "Prompt one"},
        )

    def test_reward_wrapper_restores_raw_prompt_text_before_scoring_and_logging(
        self,
    ) -> None:
        captured: dict[str, object] = {}

        def reward_func(prompts, completions, **kwargs):
            captured["prompts"] = list(prompts)
            captured["completions"] = list(completions)
            captured["prompt_kw"] = list(kwargs["prompt"])
            captured["raw_prompt_kw"] = list(kwargs[_RAW_PROMPT_COLUMN])
            return [0.75]

        collector = _RewardTelemetryCollector(
            experiment_id="wordle_exp",
            run_id="rl_chat_test",
            reward_component_names=["format_exact"],
            rollout_tracker=RLRolloutTracker(),
        )
        wrapped_reward = _wrap_reward_func_for_telemetry(
            reward_func,
            component_name="format_exact",
            collector=collector,
            prompt_lookup={"CHAT::Prompt one::thinking=False": "Prompt one"},
        )

        rewards = wrapped_reward(
            ["CHAT::Prompt one::thinking=False"],
            ["<answer>4</answer>"],
            trainer_state=SimpleNamespace(global_step=3),
        )

        self.assertEqual(rewards, [0.75])
        self.assertEqual(captured["prompts"], ["Prompt one"])
        self.assertEqual(captured["completions"], ["<answer>4</answer>"])
        self.assertEqual(captured["prompt_kw"], ["Prompt one"])
        self.assertEqual(captured["raw_prompt_kw"], ["Prompt one"])

        payload = collector.build_results_payload()
        self.assertEqual(payload["metrics"]["total_samples"], 1)
        self.assertEqual(
            payload["detailed_results"][0]["prompt"],
            "Prompt one",
        )

    def test_reward_wrapper_filters_framework_only_kwargs_for_strict_reward_funcs(
        self,
    ) -> None:
        captured: dict[str, object] = {}

        def reward_func(prompts, completions):
            captured["prompts"] = list(prompts)
            captured["completions"] = list(completions)
            return [0.25]

        collector = _RewardTelemetryCollector(
            experiment_id="wordle_exp",
            run_id="rl_chat_test",
            reward_component_names=["format_exact"],
            rollout_tracker=RLRolloutTracker(),
        )
        wrapped_reward = _wrap_reward_func_for_telemetry(
            reward_func,
            component_name="format_exact",
            collector=collector,
            prompt_lookup={"CHAT::Prompt one::thinking=False": "Prompt one"},
        )

        rewards = wrapped_reward(
            ["CHAT::Prompt one::thinking=False"],
            ["<answer>4</answer>"],
            trainer_state=SimpleNamespace(global_step=3),
        )

        self.assertEqual(rewards, [0.25])
        self.assertEqual(captured["prompts"], ["Prompt one"])
        self.assertEqual(captured["completions"], ["<answer>4</answer>"])


class RLConfigHelpersTests(unittest.TestCase):
    def test_resolve_grpo_max_completion_length_uses_vllm_fallback(self) -> None:
        length = _resolve_grpo_max_completion_length(
            {},
            {"enabled": True, "max_tokens": 1536},
        )

        self.assertEqual(length, 1536)

    def test_resolve_grpo_max_completion_length_prefers_training_limit(self) -> None:
        length = _resolve_grpo_max_completion_length(
            {"max_completion_length": 1024},
            {"enabled": True, "max_tokens": 2048},
        )

        self.assertEqual(length, 1024)

    def test_build_vllm_generation_kwargs_sets_stop_behavior(self) -> None:
        kwargs = _build_vllm_generation_kwargs(
            SimpleNamespace(eos_token="<eos>"),
            {"enabled": True},
        )

        self.assertEqual(
            kwargs,
            {"include_stop_str_in_output": True, "stop": ["<eos>"]},
        )

    def test_build_vllm_generation_kwargs_appends_custom_stop_strings(self) -> None:
        kwargs = _build_vllm_generation_kwargs(
            SimpleNamespace(eos_token="<eos>"),
            {"enabled": True},
            stop_strings=["</answer>"],
        )

        self.assertEqual(
            kwargs,
            {
                "include_stop_str_in_output": True,
                "stop": ["<eos>", "</answer>"],
            },
        )

    def test_build_vllm_sampling_params_kwargs_matches_reference_shape(self) -> None:
        kwargs = _build_vllm_sampling_params_kwargs(
            SimpleNamespace(eos_token="<eos>"),
            {
                "enabled": True,
                "top_p": 0.95,
                "top_k": -1,
                "min_p": 0.1,
            },
            seed=1234,
        )

        self.assertEqual(
            kwargs,
            {
                "top_p": 0.95,
                "top_k": -1,
                "min_p": 0.1,
                "seed": 1234,
                "include_stop_str_in_output": True,
                "stop": ["<eos>"],
            },
        )

    def test_build_vllm_sampling_params_kwargs_appends_custom_stop_strings(
        self,
    ) -> None:
        kwargs = _build_vllm_sampling_params_kwargs(
            SimpleNamespace(eos_token="<eos>"),
            {
                "enabled": True,
                "top_p": 0.95,
                "top_k": -1,
            },
            seed=1234,
            stop_strings=["</answer>"],
        )

        self.assertEqual(
            kwargs,
            {
                "top_p": 0.95,
                "top_k": -1,
                "seed": 1234,
                "include_stop_str_in_output": True,
                "stop": ["<eos>", "</answer>"],
            },
        )

    def test_build_grpo_vllm_overrides_maps_live_trl_fields(self) -> None:
        overrides = _build_grpo_vllm_overrides(
            SimpleNamespace(eos_token="<eos>"),
            {
                "enabled": True,
                "top_p": 0.95,
                "top_k": -1,
                "min_p": 0.1,
                "gpu_memory_utilization": 0.8,
            },
        )

        self.assertEqual(overrides["top_p"], 0.95)
        self.assertEqual(overrides["top_k"], -1)
        self.assertEqual(overrides["min_p"], 0.1)
        self.assertEqual(
            overrides["generation_kwargs"],
            {"include_stop_str_in_output": True, "stop": ["<eos>"]},
        )

    def test_build_grpo_vllm_overrides_prefers_explicit_sampling_params(self) -> None:
        module_name = "vllm"
        original_module = sys.modules.get(module_name)

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_module = ModuleType("vllm")
        fake_module.SamplingParams = FakeSamplingParams
        sys.modules[module_name] = fake_module
        try:
            overrides = _build_grpo_vllm_overrides(
                SimpleNamespace(eos_token="<eos>"),
                {
                    "enabled": True,
                    "top_p": 0.95,
                    "top_k": -1,
                    "min_p": 0.1,
                    "gpu_memory_utilization": 0.8,
                },
                seed=1234,
                prefer_explicit_sampling_params=True,
            )

            self.assertIn("vllm_sampling_params", overrides)
            self.assertNotIn("generation_kwargs", overrides)
            self.assertNotIn("top_p", overrides)
            self.assertEqual(
                overrides["vllm_sampling_params"].kwargs,
                {
                    "top_p": 0.95,
                    "top_k": -1,
                    "min_p": 0.1,
                    "seed": 1234,
                    "include_stop_str_in_output": True,
                    "stop": ["<eos>"],
                },
            )
        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

    def test_build_grpo_vllm_overrides_ignores_trl_engine_fields(self) -> None:
        overrides = _build_grpo_vllm_overrides(
            SimpleNamespace(eos_token="<eos>"),
            {
                "enabled": True,
                "mode": "server",
                "server_timeout": 12.5,
                "gpu_memory_utilization": 0.8,
            },
        )

        self.assertNotIn("use_vllm", overrides)
        self.assertNotIn("vllm_mode", overrides)
        self.assertNotIn("vllm_gpu_memory_utilization", overrides)
        self.assertEqual(
            overrides["generation_kwargs"],
            {"include_stop_str_in_output": True, "stop": ["<eos>"]},
        )

    def test_resolve_unsloth_model_load_kwargs_preserves_fast_inference_for_grpo_vllm(self) -> None:
        kwargs = _resolve_unsloth_model_load_kwargs(
            {"fast_inference": True},
            {
                "enabled": True,
                "gpu_memory_utilization": 0.8,
            },
        )

        self.assertEqual(
            kwargs,
            {
                "fast_inference": True,
                "gpu_memory_utilization": 0.8,
            },
        )

    def test_resolve_unsloth_model_load_kwargs_preserves_fast_inference_without_grpo_vllm(self) -> None:
        kwargs = _resolve_unsloth_model_load_kwargs(
            {"fast_inference": True},
            {
                "enabled": False,
                "gpu_memory_utilization": 0.8,
            },
        )

        self.assertEqual(
            kwargs,
            {
                "fast_inference": True,
                "gpu_memory_utilization": 0.8,
            },
        )

    def test_require_rl_vllm_config_rejects_disabled_vllm(self) -> None:
        with self.assertRaisesRegex(ValueError, "RL requires vLLM"):
            _require_rl_vllm_config(
                {"fast_inference": True},
                {"enabled": False},
            )

    def test_require_rl_vllm_config_allows_full_mode_without_vllm(self) -> None:
        _require_rl_vllm_config(
            {"fast_inference": False},
            {"enabled": False},
            allow_full_finetune_fallback=True,
        )

    def test_require_rl_vllm_config_rejects_fast_inference_in_full_mode_fallback(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Full RL finetuning fallback requires model.fast_inference=false",
        ):
            _require_rl_vllm_config(
                {"fast_inference": True},
                {"enabled": False},
                allow_full_finetune_fallback=True,
            )

    def test_require_rl_vllm_config_rejects_server_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires vllm.mode=colocate"):
            _require_rl_vllm_config(
                {"fast_inference": True},
                {"enabled": True, "mode": "server"},
            )

    def test_require_rl_vllm_config_rejects_colocate_without_fast_inference(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires model.fast_inference=true"):
            _require_rl_vllm_config(
                {"fast_inference": False},
                {"enabled": True, "mode": "colocate"},
            )

    def test_trl_vllm_compat_aliases_guided_decoding_params(self) -> None:
        module_name = "vllm.sampling_params"
        original_module = sys.modules.get(module_name)
        fake_sampling_params = SimpleNamespace(
            StructuredOutputsParams=object(),
        )
        sys.modules[module_name] = fake_sampling_params
        try:
            _ensure_trl_vllm_guided_decoding_compat()
            self.assertIs(
                fake_sampling_params.GuidedDecodingParams,
                fake_sampling_params.StructuredOutputsParams,
            )
        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

    def test_trl_vllm_compat_preserves_existing_guided_decoding_params(self) -> None:
        module_name = "vllm.sampling_params"
        original_module = sys.modules.get(module_name)
        guided = object()
        structured = object()
        fake_sampling_params = SimpleNamespace(
            GuidedDecodingParams=guided,
            StructuredOutputsParams=structured,
        )
        sys.modules[module_name] = fake_sampling_params
        try:
            _ensure_trl_vllm_guided_decoding_compat()
            self.assertIs(fake_sampling_params.GuidedDecodingParams, guided)
        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

    def test_trl_vllm_sampling_params_compat_drops_truncate_prompt_tokens(self) -> None:
        module_name = "vllm.sampling_params"
        original_module = sys.modules.get(module_name)

        class FakeSamplingParams:
            def __init__(self, *, temperature=None):
                self.temperature = temperature

        fake_sampling_params = SimpleNamespace(SamplingParams=FakeSamplingParams)
        sys.modules[module_name] = fake_sampling_params
        try:
            _ensure_trl_vllm_sampling_params_compat()
            params = fake_sampling_params.SamplingParams(
                temperature=0.7,
                truncate_prompt_tokens=128,
            )
            self.assertEqual(params.temperature, 0.7)
        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

    def test_trl_vllm_sampling_params_compat_preserves_supported_kwarg(self) -> None:
        module_name = "vllm.sampling_params"
        original_module = sys.modules.get(module_name)

        class FakeSamplingParams:
            def __init__(self, *, temperature=None, truncate_prompt_tokens=None):
                self.temperature = temperature
                self.truncate_prompt_tokens = truncate_prompt_tokens

        fake_sampling_params = SimpleNamespace(SamplingParams=FakeSamplingParams)
        sys.modules[module_name] = fake_sampling_params
        try:
            _ensure_trl_vllm_sampling_params_compat()
            params = fake_sampling_params.SamplingParams(
                temperature=0.7,
                truncate_prompt_tokens=128,
            )
            self.assertEqual(params.temperature, 0.7)
            self.assertEqual(params.truncate_prompt_tokens, 128)
        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

    def test_normalize_report_to_strips_none_from_list_for_wandb(self) -> None:
        normalized = normalize_report_to(
            ["none", "tensorboard"],
            telemetry_backend="wandb",
        )

        self.assertEqual(normalized, ["tensorboard", "wandb"])

    def test_normalize_report_to_defaults_none_string_to_wandb(self) -> None:
        normalized = normalize_report_to(
            "none",
            telemetry_backend="wandb",
        )

        self.assertEqual(normalized, ["wandb"])


class _DummyBatch(dict):
    def to(self, _device):
        return self


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, texts, return_tensors="pt", padding=True):
        del return_tensors, padding
        rows = []
        for text in texts:
            if text == "ab":
                rows.append([1, 2])
            elif text == "c":
                rows.append([0, 3])
            else:
                raise AssertionError(f"Unexpected prompt: {text!r}")
        return _DummyBatch(
            {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.tensor(
                    [[1 if token != 0 else 0 for token in row] for row in rows],
                    dtype=torch.long,
                ),
            }
        )

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        mapping = {4: "X", 5: "Y", 6: "Z"}
        values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return "".join(mapping.get(token, "") for token in values if token in mapping)


class _DummyModel:
    device = "cpu"

    def fast_generate(self, *args, **kwargs):
        raise AssertionError("fast_generate should not be used when vllm.enabled is false")

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        completions = torch.tensor([[4, 5], [6, 0]], dtype=torch.long)
        return torch.cat([input_ids, completions], dim=1)


class EvalFallbackTests(unittest.TestCase):
    def test_resolve_eval_fast_inference_defaults_to_vllm_setting(self) -> None:
        requested = _resolve_eval_fast_inference_requested(
            {},
            {"enabled": True},
        )

        self.assertTrue(requested)

    def test_resolve_eval_fast_inference_respects_explicit_model_override(self) -> None:
        requested = _resolve_eval_fast_inference_requested(
            {"fast_inference": False},
            {"enabled": True},
        )

        self.assertFalse(requested)

    def test_resolve_eval_fast_inference_forces_transformers_for_init_adapter(self) -> None:
        requested = _resolve_eval_fast_inference_requested(
            {
                "fast_inference": True,
                "init_adapter_repo": "org/adapter",
            },
            {"enabled": True},
        )

        self.assertFalse(requested)

    def test_should_force_transformers_eval_for_adapter_detects_seed_adapter(self) -> None:
        self.assertTrue(
            _should_force_transformers_eval_for_adapter(
                {"init_adapter_repo": "org/adapter"}
            )
        )
        self.assertFalse(
            _should_force_transformers_eval_for_adapter(
                {"init_artifact_type": "full_model", "init_model_repo": "org/full"}
            )
        )

    def test_resolve_eval_model_load_kwargs_omits_gpu_memory_without_fast_inference(self) -> None:
        kwargs = _resolve_eval_model_load_kwargs(
            {"fast_inference": False},
            {"enabled": True, "gpu_memory_utilization": 0.8},
        )

        self.assertEqual(kwargs, {"fast_inference": False})

    def test_require_eval_vllm_config_rejects_disabled_vllm(self) -> None:
        with self.assertRaisesRegex(ValueError, "Eval requires vLLM"):
            _require_eval_vllm_config(
                {},
                {"enabled": False},
            )

    def test_require_eval_vllm_config_allows_seed_adapter_without_vllm(self) -> None:
        _require_eval_vllm_config(
            {"init_adapter_repo": "org/adapter"},
            {"enabled": False},
        )

    def test_require_eval_vllm_config_rejects_disabled_fast_inference(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires vLLM fast inference"):
            _require_eval_vllm_config(
                {"fast_inference": False},
                {"enabled": True},
            )

    def test_build_sampling_params_returns_none_without_vllm_runtime(self) -> None:
        job = EvalJob(
            config={"evaluation": {"run_name": "eval_test"}, "vllm": {"enabled": True}},
            task=object(),
        )
        job._vllm_runtime_enabled = False

        self.assertIsNone(job._build_sampling_params(SimpleNamespace(eos_token="<eos>")))

    def test_build_sampling_params_appends_custom_stop_strings(self) -> None:
        module_name = "vllm"
        original_module = sys.modules.get(module_name)

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_module = ModuleType("vllm")
        fake_module.SamplingParams = FakeSamplingParams
        sys.modules[module_name] = fake_module
        try:
            job = EvalJob(
                config={
                    "evaluation": {"run_name": "eval_test"},
                    "vllm": {
                        "enabled": True,
                        "max_tokens": 64,
                        "temperature": 0.0,
                        "top_p": 1.0,
                    },
                    "chat_template": {
                        "stop_strings": ["</answer>"],
                    },
                },
                task=object(),
            )
            job._vllm_runtime_enabled = True

            params = job._build_sampling_params(SimpleNamespace(eos_token="<eos>"))

            self.assertEqual(
                params.kwargs,
                {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 64,
                    "include_stop_str_in_output": True,
                    "stop": ["<eos>", "</answer>"],
                },
            )
        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

    def test_generate_batch_uses_transformers_fallback_when_sampling_params_missing(self) -> None:
        job = EvalJob(
            config={
                "evaluation": {"run_name": "eval_test"},
                "vllm": {"enabled": True, "max_tokens": 2, "temperature": 0.0},
            },
            task=object(),
        )

        completions = job._generate_batch(
            _DummyModel(),
            _DummyTokenizer(),
            ["ab", "c"],
            sampling_params=None,
        )

        self.assertEqual(completions, ["XY", "Z"])

    def test_build_model_and_tokenizer_fails_when_vllm_startup_breaks(self) -> None:
        module_name = "unsloth"
        original_module = sys.modules.get(module_name)
        calls = []

        class FakeModel:
            def train(self, mode):
                self.mode = mode
                return self

        class FakeFastLanguageModel:
            @staticmethod
            def from_pretrained(**kwargs):
                calls.append(kwargs)
                if kwargs.get("fast_inference", False):
                    raise RuntimeError(
                        "<function standalone_compile> does not have the attribute 'FakeTensorMode'"
                    )
                return FakeModel(), SimpleNamespace()

            @staticmethod
            def for_inference(model):
                setattr(model, "for_inference_called", True)
                return model

        sys.modules[module_name] = SimpleNamespace(FastLanguageModel=FakeFastLanguageModel)
        try:
            job = EvalJob(
                config={
                    "evaluation": {"run_name": "eval_test"},
                    "model": {
                        "name": "Qwen/Qwen3-4B",
                        "load_in_4bit": True,
                        "fast_inference": True,
                    },
                    "vllm": {"enabled": True, "gpu_memory_utilization": 0.8},
                },
                task=object(),
            )

            with unittest.mock.patch.object(
                eval_module,
                "normalize_tokenizer_special_tokens",
            ) as normalize_mock, self.assertRaisesRegex(
                RuntimeError,
                "Eval requires vLLM and will now abort",
            ):
                job._build_model_and_tokenizer()

        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["fast_inference"])
        self.assertEqual(calls[0]["gpu_memory_utilization"], 0.8)
        self.assertTrue(job._vllm_runtime_enabled)
        normalize_mock.assert_not_called()

    def test_build_model_and_tokenizer_loads_full_model_artifact_directly(self) -> None:
        module_name = "unsloth"
        original_module = sys.modules.get(module_name)
        calls = []

        class FakeModel:
            def train(self, mode):
                self.mode = mode
                return self

        class FakeFastLanguageModel:
            @staticmethod
            def from_pretrained(**kwargs):
                calls.append(kwargs)
                return FakeModel(), SimpleNamespace()

            @staticmethod
            def for_inference(model):
                setattr(model, "for_inference_called", True)
                return model

        sys.modules[module_name] = SimpleNamespace(
            FastLanguageModel=FakeFastLanguageModel
        )
        try:
            job = EvalJob(
                config={
                    "evaluation": {"run_name": "eval_test"},
                    "model": {
                        "name": "Qwen/Qwen3-4B",
                        "load_in_4bit": True,
                        "fast_inference": True,
                        "init_artifact_type": "full_model",
                        "init_model_repo": "org/full-model",
                        "init_model_revision": "main",
                    },
                    "vllm": {"enabled": True, "gpu_memory_utilization": 0.8},
                },
                task=object(),
            )

            with unittest.mock.patch.object(
                eval_module,
                "resolve_hf_repo_revision",
                return_value="rev-full",
            ), unittest.mock.patch.object(
                eval_module,
                "download_hf_lora_adapter",
            ) as adapter_download_mock, unittest.mock.patch.object(
                eval_module,
                "normalize_tokenizer_special_tokens",
            ) as normalize_mock:
                model, _tokenizer = job._build_model_and_tokenizer()
        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model_name"], "org/full-model")
        self.assertEqual(calls[0]["revision"], "rev-full")
        self.assertTrue(calls[0]["fast_inference"])
        self.assertTrue(getattr(model, "for_inference_called", False))
        adapter_download_mock.assert_not_called()
        normalize_mock.assert_called_once()

    def test_build_model_and_tokenizer_forces_transformers_runtime_for_init_adapter(
        self,
    ) -> None:
        module_name = "unsloth"
        original_module = sys.modules.get(module_name)
        calls = []

        class FakeModel:
            def train(self, mode):
                self.mode = mode
                return self

        class FakeFastLanguageModel:
            @staticmethod
            def from_pretrained(**kwargs):
                calls.append(kwargs)
                return FakeModel(), SimpleNamespace()

            @staticmethod
            def get_peft_model(model, **kwargs):
                model.peft_kwargs = kwargs
                return model

            @staticmethod
            def for_inference(model):
                setattr(model, "for_inference_called", True)
                return model

        adapter = SimpleNamespace(
            repo_id="org/adapter",
            resolved_revision="rev-123",
            weights_in_repo="adapter_model.safetensors",
        )

        sys.modules[module_name] = SimpleNamespace(
            FastLanguageModel=FakeFastLanguageModel
        )
        try:
            job = EvalJob(
                config={
                    "evaluation": {"run_name": "eval_test"},
                    "model": {
                        "name": "Qwen/Qwen3-4B",
                        "load_in_4bit": True,
                        "fast_inference": True,
                        "init_adapter_repo": "org/adapter",
                        "init_adapter_revision": "main",
                    },
                    "vllm": {"enabled": True, "gpu_memory_utilization": 0.8},
                },
                task=object(),
            )

            with unittest.mock.patch.object(
                eval_module,
                "download_hf_lora_adapter",
                return_value=adapter,
            ), unittest.mock.patch.object(
                eval_module,
                "resolve_hf_lora_runtime_kwargs",
                return_value={
                    "r": 16,
                    "target_modules": ["up_proj", "gate_proj", "down_proj"],
                    "lora_alpha": 32,
                    "lora_dropout": 0.0,
                    "bias": "none",
                },
            ), unittest.mock.patch.object(
                eval_module,
                "strict_load_hf_lora_adapter_weights",
                return_value=7,
            ) as strict_load_mock, unittest.mock.patch.object(
                eval_module,
                "normalize_tokenizer_special_tokens",
            ) as normalize_mock:
                model, _tokenizer = job._build_model_and_tokenizer()
        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["fast_inference"])
        self.assertNotIn("gpu_memory_utilization", calls[0])
        self.assertFalse(job._vllm_runtime_enabled)
        self.assertTrue(getattr(model, "for_inference_called", False))
        strict_load_mock.assert_called_once()
        normalize_mock.assert_called_once()


class RLFallbackTests(unittest.TestCase):
    def test_build_model_and_tokenizer_fails_when_vllm_startup_breaks(self) -> None:
        module_name = "unsloth"
        original_module = sys.modules.get(module_name)
        original_standby = os.environ.get("UNSLOTH_VLLM_STANDBY")
        calls = []
        standby_values = []

        class FakeModel:
            def train(self):
                self.mode = "train"
                return self

        class FakeFastLanguageModel:
            @staticmethod
            def from_pretrained(**kwargs):
                standby_values.append(os.environ.get("UNSLOTH_VLLM_STANDBY"))
                calls.append(kwargs)
                if kwargs.get("fast_inference", False):
                    raise RuntimeError(
                        "<function standalone_compile> does not have the attribute 'FakeTensorMode'"
                    )
                return FakeModel(), SimpleNamespace()

            @staticmethod
            def get_peft_model(model, **kwargs):
                model.peft_kwargs = kwargs
                return model

        sys.modules[module_name] = SimpleNamespace(FastLanguageModel=FakeFastLanguageModel)
        try:
            os.environ.pop("UNSLOTH_VLLM_STANDBY", None)
            job = RLJob(
                config={
                    "training": {"run_name": "rl_test", "seed": 3407},
                    "model": {
                        "name": "Qwen/Qwen3-4B",
                        "load_in_4bit": True,
                        "fast_inference": True,
                    },
                    "lora": {
                        "r": 16,
                        "target_modules": ["up_proj", "gate_proj", "down_proj"],
                        "alpha": 32,
                        "dropout": 0.0,
                        "bias": "none",
                    },
                    "vllm": {"enabled": True, "gpu_memory_utilization": 0.8},
                },
                task=object(),
            )

            with unittest.mock.patch(
                "tenyson.jobs.rl.normalize_tokenizer_special_tokens"
            ) as normalize_mock, self.assertRaisesRegex(
                RuntimeError,
                "RL requires vLLM and will now abort",
            ):
                job._build_model_and_tokenizer()
        finally:
            if original_standby is None:
                os.environ.pop("UNSLOTH_VLLM_STANDBY", None)
            else:
                os.environ["UNSLOTH_VLLM_STANDBY"] = original_standby
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

        self.assertEqual(os.environ.get("UNSLOTH_VLLM_STANDBY"), original_standby)
        self.assertEqual(len(calls), 1)
        self.assertEqual(standby_values, ["1"])
        self.assertTrue(calls[0]["fast_inference"])
        self.assertEqual(calls[0]["gpu_memory_utilization"], 0.8)
        self.assertTrue(job._vllm_runtime_enabled)
        normalize_mock.assert_not_called()

    def test_build_model_and_tokenizer_respects_disabled_standby_mode(self) -> None:
        module_name = "unsloth"
        original_module = sys.modules.get(module_name)
        original_standby = os.environ.get("UNSLOTH_VLLM_STANDBY")
        standby_values = []

        class FakeModel:
            def train(self):
                self.mode = "train"
                return self

        class FakeFastLanguageModel:
            @staticmethod
            def from_pretrained(**kwargs):
                standby_values.append(os.environ.get("UNSLOTH_VLLM_STANDBY"))
                raise RuntimeError(
                    "<function standalone_compile> does not have the attribute 'FakeTensorMode'"
                )

            @staticmethod
            def get_peft_model(model, **kwargs):
                _unused = kwargs
                return model

        sys.modules[module_name] = SimpleNamespace(FastLanguageModel=FakeFastLanguageModel)
        try:
            os.environ.pop("UNSLOTH_VLLM_STANDBY", None)
            job = RLJob(
                config={
                    "training": {"run_name": "rl_test", "seed": 3407},
                    "model": {
                        "name": "Qwen/Qwen3-4B",
                        "load_in_4bit": True,
                        "fast_inference": True,
                    },
                    "lora": {
                        "r": 16,
                        "target_modules": ["up_proj", "gate_proj", "down_proj"],
                        "alpha": 32,
                        "dropout": 0.0,
                        "bias": "none",
                    },
                    "vllm": {
                        "enabled": True,
                        "gpu_memory_utilization": 0.8,
                        "standby_mode": False,
                    },
                },
                task=object(),
            )

            with unittest.mock.patch(
                "tenyson.jobs.rl.normalize_tokenizer_special_tokens"
            ), self.assertRaisesRegex(
                RuntimeError,
                "RL requires vLLM and will now abort",
            ):
                job._build_model_and_tokenizer()
        finally:
            if original_standby is None:
                os.environ.pop("UNSLOTH_VLLM_STANDBY", None)
            else:
                os.environ["UNSLOTH_VLLM_STANDBY"] = original_standby
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

        self.assertEqual(standby_values, ["0"])

    def test_destroy_process_group_runs_when_distributed_is_initialized(self) -> None:
        with patch(
            "torch.distributed.is_available",
            return_value=True,
        ), patch(
            "torch.distributed.is_initialized",
            return_value=True,
        ), patch(
            "torch.distributed.destroy_process_group",
        ) as destroy_mock:
            rl_module._destroy_torch_process_group_if_initialized()

        destroy_mock.assert_called_once_with()

    def test_destroy_process_group_skips_when_not_initialized(self) -> None:
        with patch(
            "torch.distributed.is_available",
            return_value=True,
        ), patch(
            "torch.distributed.is_initialized",
            return_value=False,
        ), patch(
            "torch.distributed.destroy_process_group",
        ) as destroy_mock:
            rl_module._destroy_torch_process_group_if_initialized()

        destroy_mock.assert_not_called()

    def test_build_model_and_tokenizer_keeps_unsloth_lora_when_loading_init_adapter(
        self,
    ) -> None:
        unsloth_module_name = "unsloth"
        original_unsloth_module = sys.modules.get(unsloth_module_name)
        unsloth_calls = []
        strict_load_calls = []

        class FakeModel:
            def load_lora(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

        class FakeFastLanguageModel:
            @staticmethod
            def from_pretrained(**kwargs):
                unsloth_calls.append(kwargs)
                return FakeModel(), SimpleNamespace()

            @staticmethod
            def get_peft_model(model, **kwargs):
                model.peft_kwargs = kwargs
                return model

        sys.modules[unsloth_module_name] = SimpleNamespace(
            FastLanguageModel=FakeFastLanguageModel
        )

        adapter = SimpleNamespace(
            repo_id="org/adapter",
            resolved_revision="rev-123",
            weights_in_repo="adapter_model.safetensors",
            config={
                "base_model_name_or_path": "Qwen/Qwen3-4B",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": ["up_proj", "gate_proj", "down_proj"],
            },
        )

        try:
            job = RLJob(
                config={
                    "training": {"run_name": "rl_test", "seed": 3407},
                    "model": {
                        "name": "unsloth/Qwen3-4B-Base",
                        "fast_inference": True,
                        "init_adapter_repo": "org/adapter",
                        "init_adapter_revision": "main",
                    },
                    "lora": {
                        "r": 16,
                        "target_modules": ["up_proj", "gate_proj", "down_proj"],
                        "alpha": 32,
                        "dropout": 0.0,
                        "bias": "none",
                    },
                    "vllm": {"enabled": True, "gpu_memory_utilization": 0.8},
                },
                task=object(),
            )

            with unittest.mock.patch.object(
                rl_module,
                "download_hf_lora_adapter",
                return_value=adapter,
            ), unittest.mock.patch.object(
                rl_module,
                "resolve_hf_lora_runtime_kwargs",
                return_value={
                    "r": 16,
                    "target_modules": ["up_proj", "gate_proj", "down_proj"],
                    "lora_alpha": 32,
                    "lora_dropout": 0.0,
                    "bias": "none",
                },
            ), unittest.mock.patch.object(
                rl_module,
                "strict_load_hf_lora_adapter_weights",
                side_effect=lambda model, adapter_arg: strict_load_calls.append(
                    adapter_arg
                )
                or 7,
            ), unittest.mock.patch.object(
                rl_module,
                "normalize_tokenizer_special_tokens",
            ) as normalize_mock:
                model, _tokenizer = job._build_model_and_tokenizer()
        finally:
            if original_unsloth_module is None:
                sys.modules.pop(unsloth_module_name, None)
            else:
                sys.modules[unsloth_module_name] = original_unsloth_module

        self.assertEqual(len(unsloth_calls), 1)
        self.assertEqual(unsloth_calls[0]["model_name"], "Qwen/Qwen3-4B")
        self.assertEqual(unsloth_calls[0]["max_seq_length"], 2048)
        self.assertFalse(unsloth_calls[0]["load_in_4bit"])
        self.assertTrue(unsloth_calls[0]["fast_inference"])
        self.assertEqual(unsloth_calls[0]["gpu_memory_utilization"], 0.8)
        self.assertEqual(
            getattr(model, "peft_kwargs", None),
            {
                "r": 16,
                "target_modules": ["up_proj", "gate_proj", "down_proj"],
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "bias": "none",
                "use_gradient_checkpointing": "unsloth",
                "random_state": 3407,
            },
        )
        self.assertTrue(callable(getattr(model, "load_lora", None)))
        self.assertEqual(strict_load_calls, [adapter])
        normalize_mock.assert_called_once()

    def test_build_model_and_tokenizer_full_mode_loads_full_model_artifact(self) -> None:
        unsloth_module_name = "unsloth"
        original_unsloth_module = sys.modules.get(unsloth_module_name)
        unsloth_calls = []

        class FakeModel:
            def __init__(self):
                self.for_training_calls = []

            def for_training(self, *args, **kwargs):
                self.for_training_calls.append((args, kwargs))

            def load_lora(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

        class FakeFastLanguageModel:
            @staticmethod
            def from_pretrained(**kwargs):
                unsloth_calls.append(kwargs)
                return FakeModel(), SimpleNamespace()

            @staticmethod
            def get_peft_model(model, **kwargs):
                model.peft_kwargs = kwargs
                return model

        sys.modules[unsloth_module_name] = SimpleNamespace(
            FastLanguageModel=FakeFastLanguageModel
        )

        try:
            job = RLJob(
                config={
                    "training": {
                        "run_name": "rl_test",
                        "seed": 3407,
                        "finetune_mode": "full",
                    },
                    "model": {
                        "name": "unsloth/Qwen3-4B-Base",
                        "fast_inference": True,
                        "load_in_4bit": False,
                        "init_artifact_type": "full_model",
                        "init_model_repo": "org/full-model",
                        "init_model_revision": "main",
                    },
                    "lora": {
                        "r": 16,
                        "target_modules": ["up_proj", "gate_proj", "down_proj"],
                        "alpha": 32,
                        "dropout": 0.0,
                        "bias": "none",
                        "gradient_checkpointing": "unsloth",
                    },
                    "vllm": {"enabled": True, "gpu_memory_utilization": 0.8},
                },
                task=object(),
            )

            with unittest.mock.patch.object(
                rl_module,
                "resolve_hf_repo_revision",
                return_value="rev-full",
            ), unittest.mock.patch.object(
                rl_module,
                "download_hf_lora_adapter",
            ) as adapter_download_mock, unittest.mock.patch.object(
                rl_module,
                "normalize_tokenizer_special_tokens",
            ) as normalize_mock:
                model, _tokenizer = job._build_model_and_tokenizer()
        finally:
            if original_unsloth_module is None:
                sys.modules.pop(unsloth_module_name, None)
            else:
                sys.modules[unsloth_module_name] = original_unsloth_module

        self.assertEqual(len(unsloth_calls), 1)
        self.assertEqual(unsloth_calls[0]["model_name"], "org/full-model")
        self.assertEqual(unsloth_calls[0]["revision"], "rev-full")
        self.assertTrue(unsloth_calls[0]["full_finetuning"])
        self.assertFalse(unsloth_calls[0]["fast_inference"])
        self.assertNotIn("gpu_memory_utilization", unsloth_calls[0])
        self.assertFalse(job.config["model"]["fast_inference"])
        self.assertFalse(job.config["vllm"]["enabled"])
        self.assertTrue(model.for_training_calls)
        self.assertFalse(hasattr(model, "peft_kwargs"))
        adapter_download_mock.assert_not_called()
        normalize_mock.assert_called_once()

    def test_build_model_and_tokenizer_full_mode_rejects_adapter_seed(self) -> None:
        job = RLJob(
            config={
                "training": {
                    "run_name": "rl_test",
                    "seed": 3407,
                    "finetune_mode": "full",
                },
                "model": {
                    "name": "unsloth/Qwen3-4B-Base",
                    "fast_inference": True,
                    "load_in_4bit": False,
                    "init_adapter_repo": "org/adapter",
                    "init_adapter_revision": "main",
                },
                "lora": {
                    "r": 16,
                    "target_modules": ["up_proj", "gate_proj", "down_proj"],
                    "alpha": 32,
                    "dropout": 0.0,
                    "bias": "none",
                },
                "vllm": {"enabled": True},
            },
            task=object(),
        )

        with self.assertRaisesRegex(ValueError, "cannot start from .*adapter"):
            job._build_model_and_tokenizer()


if __name__ == "__main__":
    unittest.main()
