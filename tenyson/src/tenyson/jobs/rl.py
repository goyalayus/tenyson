import os
import time
from typing import Any, Dict, List
from uuid import uuid4

from tenyson.core.hf_adapter import (
    download_hf_lora_adapter,
    resolve_hf_lora_runtime_kwargs,
    strict_load_hf_lora_adapter_weights,
)
from tenyson.core.hf_checkpoint import (
    download_hf_resume_checkpoint,
    resolve_hf_repo_revision,
)
from tenyson.core.plugin import TaskPlugin
from tenyson.core.execution_policy import require_gpu_provider_runtime
from tenyson.core.run_name import resolve_required_run_name
from tenyson.jobs.hf_repo import unique_repo_id
from tenyson.jobs.result import JobResult


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

    def _build_vllm_sampling_params(self, tokenizer: Any):
        vllm_cfg = self.config.get("vllm", {})
        if not vllm_cfg.get("enabled", False):
            return None
        from vllm import SamplingParams

        max_tokens = int(vllm_cfg.get("max_tokens", 2048))
        return SamplingParams(
            temperature=float(vllm_cfg.get("temperature", 1.0)),
            min_p=float(vllm_cfg.get("min_p", 0.1)),
            top_p=float(vllm_cfg.get("top_p", 1.0)),
            top_k=int(vllm_cfg.get("top_k", -1)),
            seed=int(self.config.get("training", {}).get("seed", 3407)),
            max_tokens=max_tokens,
            stop=[tokenizer.eos_token]
            if getattr(tokenizer, "eos_token", None) is not None
            else None,
            include_stop_str_in_output=True,
        )

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
        fast_inference = vllm_cfg.get("enabled", False)

        print("[RLJob] Loading model and tokenizer via Unsloth...", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=seq_len,
            load_in_4bit=load_in_4bit,
            fast_inference=fast_inference,
            max_lora_rank=int(lora_runtime_kwargs["r"]),
            gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.9),
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

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
        from trl import GRPOConfig, GRPOTrainer
        from tenyson.core.hub_push import ensure_hf_repo
        from tenyson.core.telemetry import (
            GRPOEpochTelemetryCallback,
            Generation,
            ManualStopTelemetryCallback,
            record_run_result,
            record_run_summary,
            resolve_required_telemetry_context,
            Rollout,
            TelemetryClient,
            WandBUrlTelemetryCallback,
        )

        start = time.time()
        train_cfg = self.config.get("training", {})
        vllm_cfg = self.config.get("vllm", {})
        db_url, experiment_id = resolve_required_telemetry_context(self.config)
        telemetry_client = TelemetryClient(db_url=db_url)

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

        ensure_hf_repo(push_repo_id)

        model, tokenizer = self._build_model_and_tokenizer()

        print("[RLJob] Loading RL dataset via TaskPlugin...", flush=True)
        dataset = self.task.get_rl_dataset(self.config)
        print(f"[RLJob] Loaded {len(dataset)} examples.", flush=True)

        print("[RLJob] Loading reward functions via TaskPlugin...", flush=True)
        reward_funcs = self.task.get_reward_funcs(self.config, tokenizer)

        vllm_sampling_params = self._build_vllm_sampling_params(tokenizer)

        cfg_kwargs: Dict[str, Any] = dict(
            output_dir=str(os.path.join(output_dir, "trainer_out")),
            max_steps=train_cfg.get("max_steps", 1000),
            per_device_train_batch_size=train_cfg.get("per_device_batch_size", 1),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
            learning_rate=float(train_cfg.get("learning_rate", 5e-6)),
            temperature=float(vllm_cfg.get("temperature", 1.0))
            if vllm_cfg.get("enabled", False)
            else 1.0,
            use_vllm=vllm_cfg.get("enabled", False),
            vllm_sampling_params=vllm_sampling_params,
            optim=train_cfg.get("optim", "adamw_8bit"),
            logging_steps=1,
            report_to=train_cfg.get("report_to", ["none"]),
            run_name=run_name,
            seed=train_cfg.get("seed", 3407),
            remove_unused_columns=False,
            num_generations=train_cfg.get("num_generations", 4),
            max_prompt_length=train_cfg.get("max_prompt_length", 2048),
            max_completion_length=train_cfg.get("max_completion_length", 2048),
            bf16=False,
            fp16=False,
            save_steps=hf_push_every_steps,
            save_strategy="steps",
            save_total_limit=train_cfg.get("save_total_limit", 2),
            push_to_hub=True,
            hub_model_id=push_repo_id,
            hub_strategy="checkpoint",
        )

        import inspect

        accepted = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
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
        callbacks = list(callbacks) + [
            GRPOEpochTelemetryCallback(
                run_id=run_name,
                experiment_id=experiment_id,
                client=telemetry_client,
            ),
            ManualStopTelemetryCallback(
                run_id=run_name,
                experiment_id=experiment_id,
                client=telemetry_client,
                check_every_n_steps=manual_stop_every,
            ),
        ]
        report_to = train_cfg.get("report_to", ["none"])
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

        # Wrap the first reward function to log per-prompt/per-completion rewards
        # into the Generation / Rollout tables. We only wrap the first entry to
        # avoid double-logging if users supply multiple reward functions.
        if reward_funcs:

            def make_wrapped(base_func, client: TelemetryClient, run_id: str):  # type: ignore[override]
                def wrapped(
                    prompts: List[Any], completions: List[Any], **kwargs
                ) -> List[float]:
                    rewards = base_func(prompts, completions, **kwargs)
                    session = None
                    try:
                        session = client.Session()
                        # Best-effort global_step; fall back to 0 if not provided.
                        step = int(kwargs.get("global_step", 0))
                        for prompt, completion, reward in zip(
                            prompts, completions, rewards
                        ):
                            rollout = Rollout(
                                id=str(uuid4()),
                                experiment_id=experiment_id,
                                run_id=run_id,
                                global_step=step,
                                prompt_text=str(prompt),
                            )
                            session.add(rollout)
                            generation = Generation(
                                id=str(uuid4()),
                                experiment_id=experiment_id,
                                run_id=run_id,
                                global_step=step,
                                phase="rl",
                                prompt_text=str(prompt),
                                completion_text=str(completion),
                                reward=float(reward),
                            )
                            session.add(generation)
                        session.commit()
                    finally:
                        if session is not None:
                            session.close()
                    return rewards

                return wrapped

            # Replace only the first reward function with the wrapped version.
            reward_funcs = [
                make_wrapped(reward_funcs[0], telemetry_client, run_name),
                *reward_funcs[1:],
            ]

        trainer = GRPOTrainer(
            model=model,
            args=grpo_args,
            train_dataset=dataset,
            reward_funcs=reward_funcs,
            processing_class=tokenizer,
            callbacks=callbacks,
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

        hf_revision = resolve_hf_repo_revision(push_repo_id) if push_repo_id else None

        result = JobResult(
            run_id=run_name,
            status="success",
            total_time_seconds=total_time,
            metrics=metrics,
            wandb_url=wandb_url,
            hf_repo_id=push_repo_id or None,
            hf_revision=hf_revision,
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
        return result
