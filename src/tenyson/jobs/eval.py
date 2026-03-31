import functools
import os
import time
from typing import Any, Dict, List, Sequence

from tenyson.core.hf_adapter import (
    download_hf_lora_adapter,
    resolve_hf_lora_runtime_kwargs,
    strict_load_hf_lora_adapter_weights,
)
from tenyson.core.hf_checkpoint import resolve_hf_repo_revision
from tenyson.core.model_policy import require_qwen3_model_name
from tenyson.core.plugin import TaskPlugin
from tenyson.core.execution_policy import require_gpu_provider_runtime
from tenyson.core.run_name import resolve_required_run_name
from tenyson.jobs.result import JobResult
from tenyson.jobs.tokenizer_utils import normalize_tokenizer_special_tokens


def _format_failure_reason(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        return exc.__class__.__name__
    return f"{exc.__class__.__name__}: {message}"


def _resolve_eval_fast_inference_requested(
    model_cfg: Dict[str, Any],
    vllm_cfg: Dict[str, Any],
) -> bool:
    if "fast_inference" in model_cfg:
        return bool(model_cfg.get("fast_inference", False))
    return bool(vllm_cfg.get("enabled", False))


def _resolve_eval_model_load_kwargs(
    model_cfg: Dict[str, Any],
    vllm_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    fast_inference_requested = _resolve_eval_fast_inference_requested(
        model_cfg,
        vllm_cfg,
    )
    kwargs: Dict[str, Any] = {
        "fast_inference": fast_inference_requested,
    }
    if fast_inference_requested:
        kwargs["gpu_memory_utilization"] = float(
            vllm_cfg.get("gpu_memory_utilization", 0.9)
        )
    return kwargs


def _resolve_init_artifact_type(model_cfg: Dict[str, Any]) -> str:
    artifact_type = str(
        model_cfg.get("init_artifact_type", "adapter") or "adapter"
    ).strip().lower()
    if artifact_type not in {"adapter", "full_model"}:
        raise ValueError(
            "model.init_artifact_type must be either 'adapter' or 'full_model'."
        )
    return artifact_type


def _require_eval_vllm_config(
    model_cfg: Dict[str, Any],
    vllm_cfg: Dict[str, Any],
) -> None:
    if not bool(vllm_cfg.get("enabled", False)):
        raise ValueError(
            "Eval requires vLLM. Set vllm.enabled=true for eval runs."
        )
    if not _resolve_eval_fast_inference_requested(model_cfg, vllm_cfg):
        raise ValueError(
            "Eval requires vLLM fast inference. Do not disable model.fast_inference "
            "while vllm.enabled=true."
        )


def _configure_eval_unsloth_runtime_env(vllm_cfg: Dict[str, Any]) -> None:
    if not bool(vllm_cfg.get("enabled", False)):
        return
    disable_flashinfer = vllm_cfg.get("disable_flashinfer")
    if disable_flashinfer is None:
        disable_flashinfer = True
    if bool(disable_flashinfer):
        # Unsloth expects this before importing its vLLM utilities.
        os.environ["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        _hide_flashinfer_imports()
        _force_vllm_attention_backend()


def _hide_flashinfer_imports() -> None:
    import importlib.util

    current = importlib.util.find_spec
    if getattr(current, "_tenyson_hides_flashinfer", False):
        return

    def wrapped(name: str, package: str | None = None):  # type: ignore[override]
        normalized = str(name or "")
        if normalized == "flashinfer" or normalized.startswith("flashinfer."):
            return None
        return current(name, package)

    setattr(wrapped, "_tenyson_hides_flashinfer", True)
    importlib.util.find_spec = wrapped  # type: ignore[assignment]


def _force_vllm_attention_backend() -> None:
    try:
        from vllm.engine.arg_utils import EngineArgs
    except Exception:  # noqa: BLE001
        return

    init = EngineArgs.__init__
    if getattr(init, "_tenyson_forces_flash_attn_backend", False):
        return

    @functools.wraps(init)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        backend = kwargs.get("attention_backend")
        if backend in (None, "", "FLASHINFER"):
            kwargs["attention_backend"] = "FLASH_ATTN"
        init(self, *args, **kwargs)

    setattr(wrapped, "_tenyson_forces_flash_attn_backend", True)
    EngineArgs.__init__ = wrapped


class EvalJob:
    """
    Evaluation job using Unsloth + vLLM.

    This mirrors `src/runners/evals.py` but is driven by a `TaskPlugin`.
    """

    def __init__(self, config: Dict[str, Any], task: TaskPlugin):
        self.config = config
        self.task = task
        self.run_id = resolve_required_run_name(self.config, "eval")
        self._vllm_runtime_enabled: bool | None = None

    def _build_model_and_tokenizer(self):
        model_cfg = self.config.get("model", {})
        vllm_cfg = self.config.get("vllm", {})
        _require_eval_vllm_config(model_cfg, vllm_cfg)
        _configure_eval_unsloth_runtime_env(vllm_cfg)
        from unsloth import FastLanguageModel

        init_artifact_type = _resolve_init_artifact_type(model_cfg)
        init_repo = ""
        init_revision = "main"
        init_adapter = None
        lora_runtime_kwargs: Dict[str, Any] | None = None
        if init_artifact_type == "full_model":
            init_repo = str(model_cfg.get("init_model_repo") or "").strip()
            init_revision = str(
                model_cfg.get("init_model_revision", "main") or "main"
            ).strip() or "main"
        else:
            init_repo = str(model_cfg.get("init_adapter_repo") or "").strip()
            init_revision = str(
                model_cfg.get("init_adapter_revision", "main") or "main"
            ).strip() or "main"

        if init_repo and init_artifact_type == "adapter":
            init_adapter = download_hf_lora_adapter(
                repo_id=init_repo, revision=init_revision
            )
            lora_runtime_kwargs = resolve_hf_lora_runtime_kwargs(
                init_adapter,
                expected_r=model_cfg.get("lora_r") if "lora_r" in model_cfg else None,
                expected_alpha=model_cfg.get("lora_alpha")
                if "lora_alpha" in model_cfg
                else None,
                expected_target_modules=model_cfg.get("lora_target_modules")
                if "lora_target_modules" in model_cfg
                else None,
            )
            print(
                "[EvalJob] Using init adapter "
                f"{init_repo}@{init_adapter.resolved_revision} "
                f"({init_adapter.weights_in_repo}).",
                flush=True,
            )
        elif init_repo and init_artifact_type == "full_model":
            print(
                "[EvalJob] Using init full model "
                f"{init_repo}@{init_revision}.",
                flush=True,
            )

        model_name = require_qwen3_model_name(
            model_cfg.get("name", "Qwen/Qwen3-4B")
        )
        model_revision = model_cfg.get("revision")
        if init_repo and init_artifact_type == "full_model":
            model_name = init_repo
            model_revision = resolve_hf_repo_revision(
                repo_id=init_repo,
                revision=init_revision,
            )
            print(
                "[EvalJob] Loading the init full-model artifact directly from "
                f"{init_repo}@{model_revision}.",
                flush=True,
            )
        seq_len = model_cfg.get("max_seq_length", 4096)
        max_lora_rank = (
            int(lora_runtime_kwargs["r"])
            if lora_runtime_kwargs is not None
            else int(model_cfg.get("lora_r", 16))
        )
        unsloth_load_kwargs = _resolve_eval_model_load_kwargs(model_cfg, vllm_cfg)
        requested_fast_inference = bool(
            unsloth_load_kwargs.get("fast_inference", False)
        )
        self._vllm_runtime_enabled = requested_fast_inference

        print(f"[EvalJob] Loading model {model_name}...", flush=True)
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=seq_len,
                load_in_4bit=model_cfg.get("load_in_4bit", True),
                max_lora_rank=max_lora_rank,
                revision=model_revision,
                **unsloth_load_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            if requested_fast_inference:
                raise RuntimeError(
                    "[EvalJob] Unsloth vLLM startup failed during eval model load. "
                    "Eval requires vLLM and will now abort instead of retrying "
                    f"without fast inference. {exc}"
                ) from exc
            raise

        normalize_tokenizer_special_tokens(tokenizer, padding_side="left")

        if init_adapter is not None and lora_runtime_kwargs is not None:
            model = FastLanguageModel.get_peft_model(
                model,
                r=int(lora_runtime_kwargs["r"]),
                target_modules=lora_runtime_kwargs["target_modules"],
                lora_alpha=lora_runtime_kwargs["lora_alpha"],
                lora_dropout=lora_runtime_kwargs["lora_dropout"],
                bias=lora_runtime_kwargs["bias"],
                use_gradient_checkpointing="unsloth",
            )
            loaded_tensors = strict_load_hf_lora_adapter_weights(model, init_adapter)
            print(
                "[EvalJob] Successfully loaded init adapter "
                f"from {init_repo}@{init_adapter.resolved_revision} "
                f"({loaded_tensors} tensors).",
                flush=True,
            )

        print("[EvalJob] Switching model to inference mode...", flush=True)
        model.train(False)
        model = FastLanguageModel.for_inference(model)
        return model, tokenizer

    def _build_sampling_params(self, tokenizer: Any):
        vllm_cfg = self.config.get("vllm", {})
        runtime_vllm_enabled = self._vllm_runtime_enabled
        if runtime_vllm_enabled is None:
            runtime_vllm_enabled = _resolve_eval_fast_inference_requested(
                self.config.get("model", {}),
                vllm_cfg,
            )
        if not vllm_cfg.get("enabled", False) or not runtime_vllm_enabled:
            raise RuntimeError(
                "Eval requires vLLM, but the runtime is not currently configured "
                "for vLLM fast inference."
            )

        from vllm import SamplingParams

        kwargs: Dict[str, Any] = {
            "temperature": float(vllm_cfg.get("temperature", 0.0)),
            "top_p": float(vllm_cfg.get("top_p", 1.0)),
            "max_tokens": int(vllm_cfg.get("max_tokens", 1024)),
        }
        if "top_k" in vllm_cfg:
            kwargs["top_k"] = int(vllm_cfg["top_k"])
        if vllm_cfg.get("min_p") is not None:
            kwargs["min_p"] = float(vllm_cfg["min_p"])
        if getattr(tokenizer, "eos_token", None) is not None:
            kwargs["stop"] = [tokenizer.eos_token]
        return SamplingParams(**kwargs)

    def _resolve_model_device(self, model: Any) -> Any:
        if getattr(model, "device", None) is not None:
            return model.device

        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            try:
                return next(parameters()).device
            except (StopIteration, TypeError):
                pass
        return "cpu"

    def _generate_batch(
        self,
        model: Any,
        tokenizer: Any,
        batch_prompts: Sequence[str],
        sampling_params: Any,
    ) -> List[str]:
        if sampling_params is None:
            raise RuntimeError(
                "Eval requires vLLM SamplingParams. Refusing to fall back to "
                "Transformers generation."
            )

        outputs = model.fast_generate(
            list(batch_prompts),
            sampling_params=sampling_params,
            use_tqdm=True,
        )
        return [out.outputs[0].text for out in outputs]

    def _extract_prompts(self, dataset: Any) -> List[str]:
        return [row["prompt"] for row in dataset]

    def run(self) -> JobResult:
        require_gpu_provider_runtime()
        from tenyson.core.telemetry import (
            beat_run_heartbeat,
            start_run_heartbeat,
            begin_run_attempt,
            ensure_wandb_telemetry_run,
            record_run_result,
            record_run_summary,
            resolve_required_telemetry_context,
            run_stop_requested,
            TelemetryClient,
        )

        start = time.time()
        eval_cfg = self.config.get("evaluation", {})
        run_name = self.run_id
        attempt_token = str(
            self.config.get("telemetry", {}).get("attempt_token") or ""
        ).strip() or None

        backend_ref, experiment_id = resolve_required_telemetry_context(self.config)
        client = TelemetryClient(db_url=backend_ref)
        wandb_run = ensure_wandb_telemetry_run(
            client,
            experiment_id=experiment_id,
            phase="eval",
            run_name=run_name,
            config=self.config,
            attempt_token=attempt_token,
        )
        if begin_run_attempt(
            client,
            experiment_id,
            run_name,
            phase="eval",
            attempt_token=attempt_token,
        ):
            print(
                "[EvalJob] Cleared stale manual stop request from a previous attempt.",
                flush=True,
            )
        try:
            start_run_heartbeat(
                client,
                experiment_id,
                run_name,
                "eval",
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "[EvalJob] Warning: initial heartbeat registration failed; "
                f"continuing evaluation. {exc}",
                flush=True,
            )
        try:
            model, tokenizer = self._build_model_and_tokenizer()

            print("[EvalJob] Loading eval dataset via TaskPlugin...", flush=True)
            dataset = self.task.get_eval_dataset(self.config)
            print(
                f"[EvalJob] Loaded {len(dataset)} examples for evaluation.",
                flush=True,
            )

            sampling_params = self._build_sampling_params(tokenizer)
            all_prompts: Sequence[str] = self._extract_prompts(dataset)

            batch_size = int(self.config.get("evaluation", {}).get("batch_size", 32))
            batch_size = max(1, batch_size)

            processed_prompts: List[str] = []
            processed_completions: List[str] = []
            stop_requested = False
            last_heartbeat_at = 0.0
            heartbeat_warned = False

            def _heartbeat(force: bool = False) -> None:
                nonlocal heartbeat_warned, last_heartbeat_at
                now = time.monotonic()
                if not force and (now - last_heartbeat_at) < 10.0:
                    return
                try:
                    beat_run_heartbeat(
                        client=client,
                        experiment_id=experiment_id,
                        run_id=run_name,
                        phase="eval",
                    )
                    last_heartbeat_at = now
                    heartbeat_warned = False
                except Exception as exc:  # noqa: BLE001
                    if not heartbeat_warned:
                        print(
                            "[EvalJob] Warning: heartbeat update failed; continuing "
                            f"evaluation. {exc}",
                            flush=True,
                        )
                        heartbeat_warned = True

            def _should_stop() -> bool:
                return run_stop_requested(
                    client,
                    experiment_id=experiment_id,
                    run_id=run_name,
                    phase="eval",
                    attempt_token=attempt_token,
                )

            backend = "vLLM"
            temperature = sampling_params.temperature
            print(
                f"[EvalJob] Starting batched generation with {backend} (temp={temperature})...",
                flush=True,
            )
            _heartbeat(force=True)
            for start_idx in range(0, len(all_prompts), batch_size):
                if _should_stop():
                    stop_requested = True
                    print(
                        "[EvalJob] Manual stop requested before the next eval batch; "
                        "stopping early.",
                        flush=True,
                    )
                    break
                end_idx = min(start_idx + batch_size, len(all_prompts))
                batch_prompts = list(all_prompts[start_idx:end_idx])

                batch_completions = self._generate_batch(
                    model,
                    tokenizer,
                    batch_prompts,
                    sampling_params,
                )
                processed_completions.extend(batch_completions)
                processed_prompts.extend(batch_prompts)
                _heartbeat(force=False)

                if _should_stop():
                    stop_requested = True
                    print(
                        f"[EvalJob] Manual stop requested after processing {len(processed_prompts)} prompts; stopping early.",
                        flush=True,
                    )
                    break

            print("[EvalJob] Generation complete. Computing metrics...", flush=True)

            # Only pass the processed subset through to metrics.
            processed_samples = len(processed_prompts)
            expected_samples = len(all_prompts)
            processed_dataset = dataset.select(range(processed_samples))
            results = self.task.compute_metrics(
                processed_prompts,
                processed_completions,
                processed_dataset,
                self.config,
                tokenizer,
            )
            stopped_early = stop_requested or processed_samples < expected_samples
            results.setdefault("metadata", {})
            results["metadata"]["processed_samples"] = processed_samples
            results["metadata"]["expected_samples"] = expected_samples
            if stopped_early:
                # Preserve the partial/stopped shape so downstream tooling can
                # recognise that this eval did not consume the full dataset.
                results["metadata"]["stopped_early"] = True
            if stop_requested:
                results["metadata"]["manual_stop_requested"] = True

            metrics = results.get("metrics", {})
            for key, value in metrics.items():
                print(f"[EvalJob] {key}: {value}", flush=True)

            total_time = time.time() - start
            failure_reason = None
            if stop_requested:
                failure_reason = (
                    "Manual stop requested after processing "
                    f"{processed_samples} / {expected_samples} prompts."
                )

            result = JobResult(
                run_id=run_name,
                status="stopped"
                if stop_requested
                else ("partial" if stopped_early else "success"),
                total_time_seconds=total_time,
                metrics=metrics,
                stopped_early=stopped_early,
                processed_samples=processed_samples,
                expected_samples=expected_samples,
                failure_reason=failure_reason,
                wandb_url=getattr(wandb_run, "url", None)
                if wandb_run is not None
                else None,
                attempt_token=attempt_token,
            )
            record_run_summary(
                client=client,
                experiment_id=experiment_id,
                phase="eval",
                result=result,
            )
            record_run_result(
                client=client,
                experiment_id=experiment_id,
                run_id=run_name,
                phase="eval",
                results_payload=results,
                job_result_payload=result,
            )
            if wandb_run is not None:
                try:
                    import wandb

                    wandb.log(metrics)
                except Exception:  # noqa: BLE001
                    pass
            return result
        except Exception as exc:
            result = JobResult(
                run_id=run_name,
                status="failed",
                total_time_seconds=time.time() - start,
                metrics={},
                failure_reason=_format_failure_reason(exc),
                wandb_url=getattr(wandb_run, "url", None)
                if wandb_run is not None
                else None,
                attempt_token=attempt_token,
            )
            try:
                record_run_summary(
                    client=client,
                    experiment_id=experiment_id,
                    phase="eval",
                    result=result,
                )
                record_run_result(
                    client=client,
                    experiment_id=experiment_id,
                    run_id=run_name,
                    phase="eval",
                    results_payload=result,
                    job_result_payload=result,
                )
            except Exception as record_exc:  # noqa: BLE001
                print(
                    "[EvalJob] Warning: failed to record terminal eval failure "
                    f"telemetry. {record_exc}",
                    flush=True,
                )
            return result
        finally:
            if wandb_run is not None:
                try:
                    import wandb

                    wandb.finish()
                except Exception:  # noqa: BLE001
                    pass
