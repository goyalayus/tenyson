import asyncio
import functools
import importlib
import inspect
import json
import math
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence
from uuid import uuid4

from tenyson.core.hf_adapter import (
    download_hf_lora_adapter,
    resolve_hf_lora_runtime_kwargs,
    strict_load_hf_lora_adapter_weights,
)
from tenyson.core.hf_checkpoint import (
    download_hf_resume_checkpoint,
    resolve_hf_repo_revision,
    resolve_hf_resume_revision,
)
from tenyson.core.plugin import TaskPlugin
from tenyson.core.execution_policy import require_gpu_provider_runtime
from tenyson.core.run_name import resolve_required_run_name
from tenyson.jobs.hf_repo import unique_repo_id
from tenyson.jobs.reporting_utils import normalize_report_to
from tenyson.jobs.result import JobResult
from tenyson.jobs.tokenizer_utils import normalize_tokenizer_special_tokens


def _resolve_grpo_max_completion_length(
    train_cfg: Dict[str, Any],
    vllm_cfg: Dict[str, Any],
) -> int:
    train_length = train_cfg.get("max_completion_length")
    vllm_length = vllm_cfg.get("max_tokens") if vllm_cfg.get("enabled", False) else None

    if train_length is None and vllm_length is None:
        return 2048
    if train_length is None:
        return int(vllm_length)
    if vllm_length is None:
        return int(train_length)

    resolved_train_length = int(train_length)
    resolved_vllm_length = int(vllm_length)
    if resolved_train_length != resolved_vllm_length:
        raise ValueError(
            "training.max_completion_length and vllm.max_tokens must match when "
            "vLLM is enabled because TRL GRPO uses a single completion-length setting."
        )
    return resolved_train_length


def _build_vllm_generation_kwargs(
    tokenizer: Any,
    vllm_cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not vllm_cfg.get("enabled", False):
        return None

    generation_kwargs: Dict[str, Any] = {
        "include_stop_str_in_output": True,
    }
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token is not None:
        generation_kwargs["stop"] = [eos_token]
    return generation_kwargs


def _build_vllm_sampling_params_kwargs(
    tokenizer: Any,
    vllm_cfg: Dict[str, Any],
    *,
    seed: int,
) -> Optional[Dict[str, Any]]:
    if not vllm_cfg.get("enabled", False):
        return None

    sampling_kwargs: Dict[str, Any] = {
        "top_p": float(vllm_cfg.get("top_p", 1.0)),
        "top_k": int(vllm_cfg.get("top_k", -1)),
        "seed": int(seed),
        "include_stop_str_in_output": True,
    }
    min_p = vllm_cfg.get("min_p")
    if min_p is not None:
        sampling_kwargs["min_p"] = float(min_p)
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token is not None:
        sampling_kwargs["stop"] = [eos_token]
    return sampling_kwargs


def _build_grpo_vllm_overrides(
    tokenizer: Any,
    vllm_cfg: Dict[str, Any],
    *,
    seed: int = 3407,
    prefer_explicit_sampling_params: bool = False,
) -> Dict[str, Any]:
    if not vllm_cfg.get("enabled", False):
        return {"use_vllm": False}

    mode = str(vllm_cfg.get("mode", "colocate") or "colocate").strip().lower()
    if mode not in {"colocate", "server"}:
        raise ValueError(
            "vllm.mode must be either 'colocate' or 'server' when RL vLLM is enabled."
        )

    overrides: Dict[str, Any] = {
        "use_vllm": True,
        "vllm_mode": mode,
        "vllm_gpu_memory_utilization": float(
            vllm_cfg.get("gpu_memory_utilization", 0.9)
        ),
    }

    if mode == "server":
        base_url = str(vllm_cfg.get("server_base_url") or "").strip()
        if base_url:
            overrides["vllm_server_base_url"] = base_url
        else:
            overrides["vllm_server_host"] = str(
                vllm_cfg.get("server_host", "127.0.0.1")
            ).strip() or "127.0.0.1"
            overrides["vllm_server_port"] = int(vllm_cfg.get("server_port", 8000))
        if vllm_cfg.get("server_timeout") is not None:
            overrides["vllm_server_timeout"] = float(vllm_cfg["server_timeout"])
    else:
        if vllm_cfg.get("tensor_parallel_size") is not None:
            overrides["vllm_tensor_parallel_size"] = int(
                vllm_cfg["tensor_parallel_size"]
            )
        if vllm_cfg.get("max_model_length") is not None:
            overrides["vllm_max_model_length"] = int(vllm_cfg["max_model_length"])
        if vllm_cfg.get("enable_sleep_mode") is not None:
            overrides["vllm_enable_sleep_mode"] = bool(
                vllm_cfg["enable_sleep_mode"]
            )

    if prefer_explicit_sampling_params:
        try:
            from vllm import SamplingParams
        except Exception:  # noqa: BLE001
            prefer_explicit_sampling_params = False
        else:
            sampling_kwargs = _build_vllm_sampling_params_kwargs(
                tokenizer,
                vllm_cfg,
                seed=seed,
            )
            if sampling_kwargs is not None:
                overrides["vllm_sampling_params"] = SamplingParams(**sampling_kwargs)

    if "vllm_sampling_params" not in overrides:
        overrides["top_p"] = float(vllm_cfg.get("top_p", 1.0))
        overrides["top_k"] = int(vllm_cfg.get("top_k", -1))
        min_p = vllm_cfg.get("min_p")
        overrides["min_p"] = float(min_p) if min_p is not None else None

        generation_kwargs = _build_vllm_generation_kwargs(tokenizer, vllm_cfg)
        if generation_kwargs is not None:
            overrides["generation_kwargs"] = generation_kwargs
    return overrides


def _resolve_unsloth_model_load_kwargs(
    model_cfg: Dict[str, Any],
    vllm_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    fast_inference_requested = bool(model_cfg.get("fast_inference", False))
    mode = str(vllm_cfg.get("mode", "colocate") or "colocate").strip().lower()
    if vllm_cfg.get("enabled", False) and mode == "server":
        return {"fast_inference": False}

    kwargs: Dict[str, Any] = {
        "fast_inference": fast_inference_requested,
    }
    if fast_inference_requested:
        kwargs["gpu_memory_utilization"] = float(
            vllm_cfg.get("gpu_memory_utilization", 0.9)
        )
    return kwargs


def _ensure_trl_vllm_guided_decoding_compat() -> None:
    """
    Older remote TRL builds still import GuidedDecodingParams even when paired
    with newer vLLM releases that renamed the type to StructuredOutputsParams.
    Provide a lightweight alias before importing GRPOTrainer so RL can start
    without changing the shared SFT/eval runtime dependency surface.
    """
    try:
        sampling_params = importlib.import_module("vllm.sampling_params")
    except Exception:  # noqa: BLE001
        return

    if hasattr(sampling_params, "GuidedDecodingParams"):
        return

    structured_outputs = getattr(sampling_params, "StructuredOutputsParams", None)
    if structured_outputs is not None:
        setattr(sampling_params, "GuidedDecodingParams", structured_outputs)


def _ensure_trl_vllm_sampling_params_compat() -> None:
    """
    Some TRL builds pass `truncate_prompt_tokens` into vLLM SamplingParams even
    when paired with newer vLLM releases whose SamplingParams constructor does
    not accept that keyword. Drop the unsupported kwarg so RL can reach actual
    rollout generation on the worker.
    """
    try:
        sampling_params = importlib.import_module("vllm.sampling_params")
    except Exception:  # noqa: BLE001
        return

    sampling_params_cls = getattr(sampling_params, "SamplingParams", None)
    if sampling_params_cls is None:
        return

    init = getattr(sampling_params_cls, "__init__", None)
    if init is None:
        return

    try:
        parameters = inspect.signature(init).parameters
    except (TypeError, ValueError):
        return

    if "truncate_prompt_tokens" in parameters:
        return
    if getattr(init, "_tenyson_truncate_prompt_tokens_compat", False):
        return

    @functools.wraps(init)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("truncate_prompt_tokens", None)
        init(self, *args, **kwargs)

    setattr(wrapped, "_tenyson_truncate_prompt_tokens_compat", True)
    sampling_params_cls.__init__ = wrapped


def _resolve_reward_component_name(reward_func: Callable[..., Any], index: int) -> str:
    explicit_name = getattr(reward_func, "tenyson_reward_name", None)
    if explicit_name is not None:
        explicit_name = str(explicit_name).strip()
        if explicit_name:
            return explicit_name

    func_name = str(getattr(reward_func, "__name__", "") or "").strip()
    if func_name and func_name != "wrapped" and func_name != "<lambda>":
        return func_name

    qualname = str(getattr(reward_func, "__qualname__", "") or "").strip()
    if qualname and "<lambda>" not in qualname:
        return qualname.split(".")[-1]

    return f"reward_component_{index + 1}"


def _resolve_reward_component_names(
    reward_funcs: Sequence[Callable[..., Any]],
) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for index, reward_func in enumerate(reward_funcs):
        name = _resolve_reward_component_name(reward_func, index)
        if name in seen:
            raise ValueError(
                "Reward function names must be unique for telemetry. Set "
                "`tenyson_reward_name` on task reward functions to disambiguate them."
            )
        seen.add(name)
        names.append(name)
    return names


def _normalize_reward_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    numeric_value = float(value)
    if math.isnan(numeric_value):
        return None
    return numeric_value


def _resolve_reward_logging_step(kwargs: Dict[str, Any]) -> int:
    trainer_state = kwargs.get("trainer_state")
    if trainer_state is not None:
        trainer_state_step = None
        if isinstance(trainer_state, dict):
            trainer_state_step = trainer_state.get("global_step")
        else:
            trainer_state_step = getattr(trainer_state, "global_step", None)
        if trainer_state_step is not None:
            return int(trainer_state_step)

    explicit_step = kwargs.get("global_step")
    if explicit_step is not None:
        return int(explicit_step)

    return 0


class _RewardTelemetryCollector:
    def __init__(
        self,
        *,
        client: Any,
        experiment_id: str,
        run_id: str,
        reward_component_names: Sequence[str],
        rollout_tracker: Any,
    ) -> None:
        self.client = client
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.reward_component_names = list(reward_component_names)
        self.reward_weights = [1.0 for _ in self.reward_component_names]
        self.rollout_tracker = rollout_tracker
        self._pending_batches: Dict[tuple[int, int, int], Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def set_reward_weights(self, reward_weights: Any) -> None:
        if reward_weights is None:
            self.reward_weights = [1.0 for _ in self.reward_component_names]
            return
        if hasattr(reward_weights, "tolist"):
            reward_weights = reward_weights.tolist()
        weights = [float(weight) for weight in reward_weights]
        if len(weights) != len(self.reward_component_names):
            raise ValueError(
                "Reward weight count does not match reward function count for telemetry."
            )
        self.reward_weights = weights

    def record_component(
        self,
        *,
        component_name: str,
        prompts: List[Any],
        completions: List[Any],
        rewards: Sequence[Any],
        kwargs: Dict[str, Any],
    ) -> None:
        reward_values = [_normalize_reward_value(value) for value in rewards]
        if len(reward_values) != len(prompts) or len(prompts) != len(completions):
            raise ValueError(
                "Reward telemetry expects prompts, completions, and rewards to have "
                "the same length."
            )

        batch_key = (
            _resolve_reward_logging_step(kwargs),
            id(prompts),
            id(completions),
        )
        completed_batch: Optional[Dict[str, Any]] = None
        with self._lock:
            batch = self._pending_batches.setdefault(
                batch_key,
                {
                    "step": batch_key[0],
                    "prompts": list(prompts),
                    "completions": list(completions),
                    "components": {},
                },
            )
            if component_name in batch["components"]:
                raise ValueError(
                    f"Reward telemetry received duplicate component '{component_name}' for one batch."
                )
            batch["components"][component_name] = reward_values
            if len(batch["components"]) == len(self.reward_component_names):
                completed_batch = self._pending_batches.pop(batch_key)

        if completed_batch is not None:
            self._flush_completed_batch(completed_batch)

    def _flush_completed_batch(self, batch: Dict[str, Any]) -> None:
        from tenyson.core.telemetry import Generation, Rollout

        step = int(batch["step"])
        prompts = batch["prompts"]
        completions = batch["completions"]
        component_rewards = batch["components"]
        weights = list(self.reward_weights)
        rollout_window = self.rollout_tracker.start_rollout(step)

        session = self.client.Session()
        try:
            for index, (prompt, completion) in enumerate(zip(prompts, completions)):
                weighted_components: Dict[str, Optional[float]] = {}
                total_reward = 0.0
                has_reward = False
                for component_name, weight in zip(
                    self.reward_component_names,
                    weights,
                    strict=True,
                ):
                    component_values = component_rewards[component_name]
                    component_value = component_values[index]
                    weighted_value = (
                        None
                        if component_value is None
                        else float(component_value) * float(weight)
                    )
                    weighted_components[component_name] = weighted_value
                    if weighted_value is not None:
                        total_reward += weighted_value
                        has_reward = True

                session.add(
                    Rollout(
                        id=str(uuid4()),
                        experiment_id=self.experiment_id,
                        run_id=self.run_id,
                        global_step=step,
                        rollout_step=rollout_window.rollout_step,
                        rollout_batch_id=rollout_window.rollout_batch_id,
                        prompt_text=str(prompt),
                    )
                )
                session.add(
                    Generation(
                        id=str(uuid4()),
                        experiment_id=self.experiment_id,
                        run_id=self.run_id,
                        global_step=step,
                        rollout_step=rollout_window.rollout_step,
                        rollout_batch_id=rollout_window.rollout_batch_id,
                        phase="rl",
                        prompt_text=str(prompt),
                        completion_text=str(completion),
                        reward=(total_reward if has_reward else None),
                        reward_components_json=json.dumps(
                            weighted_components,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
                )
            session.commit()
        finally:
            session.close()


def _wrap_reward_func_for_telemetry(
    base_func: Callable[..., Any],
    *,
    component_name: str,
    collector: _RewardTelemetryCollector,
) -> Callable[..., Any]:
    if asyncio.iscoroutinefunction(base_func):

        @functools.wraps(base_func)
        async def wrapped(
            prompts: List[Any], completions: List[Any], **kwargs: Any
        ) -> Any:
            rewards = await base_func(prompts, completions, **kwargs)
            collector.record_component(
                component_name=component_name,
                prompts=prompts,
                completions=completions,
                rewards=rewards,
                kwargs=kwargs,
            )
            return rewards

    else:

        @functools.wraps(base_func)
        def wrapped(prompts: List[Any], completions: List[Any], **kwargs: Any) -> Any:
            rewards = base_func(prompts, completions, **kwargs)
            collector.record_component(
                component_name=component_name,
                prompts=prompts,
                completions=completions,
                rewards=rewards,
                kwargs=kwargs,
            )
            return rewards

    setattr(wrapped, "tenyson_reward_name", component_name)
    return wrapped


class RLJob:
    """
    GRPO RL job.

    This class encapsulates the behaviour that previously lived in
    `src/runners/rl_run.py`, but is driven by a `TaskPlugin`.
    """

    def __init__(self, config: Dict[str, Any], task: TaskPlugin):
        self.config = config
        self.task = task
        self.run_id = resolve_required_run_name(self.config, "rl")

    def _build_model_and_tokenizer(self) -> Any:
        from unsloth import FastLanguageModel

        model_cfg = self.config.get("model", {})
        lora_cfg = self.config.get("lora", {})
        train_cfg = self.config.get("training", {})
        vllm_cfg = self.config.get("vllm", {})

        init_repo = str(model_cfg.get("init_adapter_repo") or "").strip()
        init_adapter = None
        lora_runtime_kwargs: Dict[str, Any] = {
            "r": lora_cfg.get("r", 16),
            "target_modules": lora_cfg.get(
                "target_modules", ["up_proj", "gate_proj", "down_proj"]
            ),
            "lora_alpha": lora_cfg.get("alpha", 32),
            "lora_dropout": lora_cfg.get("dropout", 0.0),
            "bias": lora_cfg.get("bias", "none"),
        }
        if init_repo:
            init_rev = model_cfg.get("init_adapter_revision", "main")
            init_adapter = download_hf_lora_adapter(
                repo_id=init_repo, revision=init_rev
            )
            lora_runtime_kwargs = resolve_hf_lora_runtime_kwargs(
                init_adapter,
                expected_r=lora_cfg.get("r") if "r" in lora_cfg else None,
                expected_alpha=lora_cfg.get("alpha") if "alpha" in lora_cfg else None,
                expected_dropout=lora_cfg.get("dropout")
                if "dropout" in lora_cfg
                else None,
                expected_bias=lora_cfg.get("bias") if "bias" in lora_cfg else None,
                expected_target_modules=lora_cfg.get("target_modules")
                if "target_modules" in lora_cfg
                else None,
            )
            print(
                "[RLJob] Using init adapter "
                f"{init_repo}@{init_adapter.resolved_revision} "
                f"({init_adapter.weights_in_repo}).",
                flush=True,
            )

        model_name = model_cfg.get("name", "Qwen/Qwen3-4B")
        seq_len = model_cfg.get("max_seq_length", 4096)
        load_in_4bit = model_cfg.get("load_in_4bit", True)
        unsloth_load_kwargs = _resolve_unsloth_model_load_kwargs(model_cfg, vllm_cfg)

        print("[RLJob] Loading model and tokenizer via Unsloth...", flush=True)
        if vllm_cfg.get("enabled", False):
            print(
                "[RLJob] GRPO vLLM is enabled; mirroring the Unsloth reference "
                "setup with fast inference on so the trainable model exposes "
                "its attached vLLM engine.",
                flush=True,
            )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=seq_len,
            load_in_4bit=load_in_4bit,
            max_lora_rank=int(lora_runtime_kwargs["r"]),
            **unsloth_load_kwargs,
        )

        normalize_tokenizer_special_tokens(tokenizer, padding_side="left")

        print("[RLJob] Applying LoRA...", flush=True)
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lora_runtime_kwargs["r"]),
            target_modules=lora_runtime_kwargs["target_modules"],
            lora_alpha=lora_runtime_kwargs["lora_alpha"],
            lora_dropout=lora_runtime_kwargs["lora_dropout"],
            bias=lora_runtime_kwargs["bias"],
            use_gradient_checkpointing=lora_cfg.get(
                "gradient_checkpointing", "unsloth"
            ),
            random_state=train_cfg.get("seed", 3407),
        )

        if init_adapter is not None:
            loaded_tensors = strict_load_hf_lora_adapter_weights(model, init_adapter)
            print(
                "[RLJob] Successfully loaded init adapter "
                f"from {init_repo}@{init_adapter.resolved_revision} "
                f"({loaded_tensors} tensors).",
                flush=True,
            )

        model.train()
        return model, tokenizer

    def run(self) -> JobResult:
        require_gpu_provider_runtime()
        start = time.time()
        train_cfg = self.config.get("training", {})
        vllm_cfg = self.config.get("vllm", {})
        attempt_token = str(
            self.config.get("telemetry", {}).get("attempt_token") or ""
        ).strip() or None
        run_name = self.run_id
        hf_repo_base = (train_cfg.get("hf_repo_base") or "").strip()
        if not hf_repo_base:
            raise ValueError(
                "training.hf_repo_base is required for RL runs. "
                "Tenyson stores checkpoints/adapters on Hugging Face only."
            )
        if not os.getenv("HF_TOKEN", "").strip():
            raise ValueError("HF_TOKEN environment variable is required for RL runs.")

        push_repo_id = unique_repo_id(hf_repo_base, run_name)
        if not push_repo_id:
            raise ValueError("Failed to derive a valid Hugging Face repo id.")
        hf_push_every_steps = int(
            train_cfg.get("hf_push_every_steps", train_cfg.get("save_steps", 100))
        )
        if hf_push_every_steps <= 0:
            raise ValueError(
                "training.hf_push_every_steps must be >= 1 when HF push is enabled."
            )
        output_dir = train_cfg.get("output_dir", f"./outputs/{run_name}")

        model, tokenizer = self._build_model_and_tokenizer()
        # Import helpers that pull in Transformers only after Unsloth has had a
        # chance to patch the runtime via model construction above.
        from tenyson.core.hub_push import ensure_hf_repo
        from tenyson.core.telemetry import (
            start_run_heartbeat,
            begin_run_attempt,
            ensure_wandb_telemetry_run,
            ManualStopTelemetryCallback,
            RLRolloutTracker,
            record_run_result,
            record_run_summary,
            resolve_required_telemetry_context,
            RunHeartbeatTelemetryCallback,
            TelemetryClient,
            WandBUrlTelemetryCallback,
        )

        backend_ref, experiment_id = resolve_required_telemetry_context(self.config)
        telemetry_client = TelemetryClient(db_url=backend_ref)

        ensure_wandb_telemetry_run(
            telemetry_client,
            experiment_id=experiment_id,
            phase="rl",
            run_name=run_name,
            config=self.config,
        )
        if begin_run_attempt(
            telemetry_client,
            experiment_id,
            run_name,
            phase="rl",
            attempt_token=attempt_token,
        ):
            print(
                "[RLJob] Cleared stale manual stop request from a previous attempt.",
                flush=True,
            )
        try:
            start_run_heartbeat(
                telemetry_client,
                experiment_id,
                run_name,
                "rl",
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "[RLJob] Warning: initial heartbeat registration failed; "
                f"continuing training. {exc}",
                flush=True,
            )

        ensure_hf_repo(push_repo_id)

        # Keep Unsloth import order aligned with the reference GRPO setup so its
        # runtime patches land before TRL is imported on the worker.
        _ensure_trl_vllm_guided_decoding_compat()
        _ensure_trl_vllm_sampling_params_compat()
        from trl import GRPOConfig, GRPOTrainer

        print("[RLJob] Loading RL dataset via TaskPlugin...", flush=True)
        dataset = self.task.get_rl_dataset(self.config)
        print(f"[RLJob] Loaded {len(dataset)} examples.", flush=True)

        print("[RLJob] Loading reward functions via TaskPlugin...", flush=True)
        reward_funcs = self.task.get_reward_funcs(self.config, tokenizer)

        report_to = normalize_report_to(
            train_cfg.get("report_to", "none"),
            telemetry_backend=telemetry_client.backend,
        )

        max_completion_length = _resolve_grpo_max_completion_length(
            train_cfg, vllm_cfg
        )

        cfg_kwargs: Dict[str, Any] = dict(
            output_dir=str(os.path.join(output_dir, "trainer_out")),
            max_steps=train_cfg.get("max_steps", 1000),
            per_device_train_batch_size=train_cfg.get("per_device_batch_size", 1),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
            learning_rate=float(train_cfg.get("learning_rate", 5e-6)),
            temperature=float(vllm_cfg.get("temperature", 1.0))
            if vllm_cfg.get("enabled", False)
            else 1.0,
            optim=train_cfg.get("optim", "adamw_8bit"),
            logging_steps=1,
            report_to=report_to,
            run_name=run_name,
            seed=train_cfg.get("seed", 3407),
            remove_unused_columns=False,
            num_generations=train_cfg.get("num_generations", 4),
            max_prompt_length=train_cfg.get("max_prompt_length", 2048),
            max_completion_length=max_completion_length,
            bf16=False,
            fp16=False,
            save_steps=hf_push_every_steps,
            save_strategy="steps",
            save_total_limit=train_cfg.get("save_total_limit", 2),
            push_to_hub=True,
            hub_model_id=push_repo_id,
            hub_strategy="checkpoint",
        )

        accepted = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
        use_explicit_vllm_sampling_params = "vllm_sampling_params" in accepted
        cfg_kwargs.update(
            _build_grpo_vllm_overrides(
                tokenizer,
                vllm_cfg,
                seed=int(train_cfg.get("seed", 3407)),
                prefer_explicit_sampling_params=use_explicit_vllm_sampling_params,
            )
        )
        if cfg_kwargs.get("use_vllm", False):
            if "vllm_sampling_params" in cfg_kwargs:
                print(
                    "[RLJob] Using explicit vLLM SamplingParams in GRPOConfig "
                    "to match the Unsloth GRPO reference setup.",
                    flush=True,
                )
            else:
                print(
                    "[RLJob] Installed GRPOConfig does not expose "
                    "`vllm_sampling_params`; using the legacy TRL vLLM "
                    "generation-kwargs path.",
                    flush=True,
                )
        required_hub_fields = {
            "save_strategy",
            "save_steps",
            "push_to_hub",
            "hub_model_id",
            "hub_strategy",
        }
        missing_hub_fields = required_hub_fields - accepted
        if missing_hub_fields:
            raise RuntimeError(
                "Installed GRPOConfig does not expose required Hub checkpoint args "
                f"for full-state push: {sorted(missing_hub_fields)}. "
                "Upgrade TRL/Transformers runtime on the worker."
            )
        if "save_only_model" in accepted:
            cfg_kwargs["save_only_model"] = False
        grpo_args = GRPOConfig(**{k: v for k, v in cfg_kwargs.items() if k in accepted})

        callbacks = self.task.get_rl_callbacks(self.config, tokenizer, str(output_dir))
        callbacks = list(callbacks)

        telemetry_cfg = self.config.get("telemetry", {})
        manual_stop_every = int(telemetry_cfg.get("manual_stop_check_every_n_steps", 1))
        rollout_tracker = RLRolloutTracker()
        manual_stop_callback = ManualStopTelemetryCallback(
            run_id=run_name,
            experiment_id=experiment_id,
            phase="rl",
            client=telemetry_client,
            check_every_n_steps=manual_stop_every,
        )
        callbacks = list(callbacks) + [
            manual_stop_callback,
            RunHeartbeatTelemetryCallback(
                run_id=run_name,
                experiment_id=experiment_id,
                phase="rl",
                client=telemetry_client,
            ),
        ]
        if report_to == "wandb" or (
            isinstance(report_to, list) and "wandb" in report_to
        ):
            callbacks = list(callbacks) + [
                WandBUrlTelemetryCallback(
                    run_id=run_name,
                    experiment_id=experiment_id,
                    client=telemetry_client,
                ),
            ]

        reward_telemetry_collector = None
        if reward_funcs and telemetry_client.backend == "sql":
            reward_component_names = _resolve_reward_component_names(reward_funcs)
            reward_telemetry_collector = _RewardTelemetryCollector(
                client=telemetry_client,
                experiment_id=experiment_id,
                run_id=run_name,
                reward_component_names=reward_component_names,
                rollout_tracker=rollout_tracker,
            )
            reward_funcs = [
                _wrap_reward_func_for_telemetry(
                    reward_func,
                    component_name=component_name,
                    collector=reward_telemetry_collector,
                )
                for reward_func, component_name in zip(
                    reward_funcs,
                    reward_component_names,
                    strict=True,
                )
            ]

        trainer = GRPOTrainer(
            model=model,
            args=grpo_args,
            train_dataset=dataset,
            reward_funcs=reward_funcs,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
        if reward_telemetry_collector is not None:
            reward_telemetry_collector.set_reward_weights(
                getattr(trainer, "reward_weights", None)
            )

        trainable_count = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
        print(f"[RLJob] Trainable parameters: {trainable_count}", flush=True)

        resume_path = None
        resume_ref = train_cfg.get("resume_from_checkpoint")
        if resume_ref:
            resume_ref = str(resume_ref).strip()
            try:
                resume_path = download_hf_resume_checkpoint(resume_ref)
            except Exception as e:  # noqa: BLE001
                raise ValueError(
                    "training.resume_from_checkpoint must be a valid Hugging Face "
                    "trainer checkpoint reference 'repo_id:revision'; "
                    f"got {resume_ref}: {e}"
                ) from e
            print(f"[RLJob] Resuming from checkpoint: {resume_path}", flush=True)

        print("[RLJob] Starting GRPO training...", flush=True)
        if resume_path:
            train_result = trainer.train(resume_from_checkpoint=resume_path)
        else:
            train_result = trainer.train()

        total_time = time.time() - start

        metrics: Dict[str, Any] = {}
        if hasattr(train_result, "metrics") and isinstance(train_result.metrics, dict):
            metrics.update(train_result.metrics)
        if hasattr(trainer, "state"):
            metrics.setdefault(
                "train_runtime", getattr(trainer.state, "train_runtime", None)
            )
            metrics.setdefault(
                "global_step", getattr(trainer.state, "global_step", None)
            )

        # Best-effort capture of the active WandB run URL, if any.
        wandb_url = None
        try:
            import wandb  # type: ignore[import-not-found]

            run = getattr(wandb, "run", None)
            if run is not None:
                wandb_url = getattr(run, "url", None)
        except Exception:  # noqa: BLE001
            wandb_url = None

        stop_requested = bool(getattr(manual_stop_callback, "stop_requested", False))
        stop_step = getattr(manual_stop_callback, "stop_step", None)
        if stop_requested:
            try:
                hf_revision = (
                    resolve_hf_resume_revision(push_repo_id) if push_repo_id else None
                )
            except Exception:  # noqa: BLE001
                hf_revision = None
        else:
            hf_revision = (
                resolve_hf_repo_revision(push_repo_id) if push_repo_id else None
            )

        status = "stopped" if stop_requested else "success"
        failure_reason = (
            f"Manual stop requested at step {stop_step}."
            if stop_requested and stop_step is not None
            else ("Manual stop requested." if stop_requested else None)
        )

        result = JobResult(
            run_id=run_name,
            status=status,
            total_time_seconds=total_time,
            metrics=metrics,
            stopped_early=stop_requested,
            wandb_url=wandb_url,
            hf_repo_id=push_repo_id or None,
            hf_revision=hf_revision,
            failure_reason=failure_reason,
            attempt_token=attempt_token,
        )
        record_run_summary(
            client=telemetry_client,
            experiment_id=experiment_id,
            phase="rl",
            result=result,
        )
        record_run_result(
            client=telemetry_client,
            experiment_id=experiment_id,
            run_id=run_name,
            phase="rl",
            results_payload=result,
            job_result_payload=result,
        )
        try:
            import wandb  # type: ignore[import-not-found]

            wandb.finish()
        except Exception:  # noqa: BLE001
            pass
        return result
