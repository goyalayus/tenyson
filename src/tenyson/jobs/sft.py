import copy
import inspect
import os
import time
from typing import Any, Dict

from tenyson.core.hf_checkpoint import (
    download_hf_resume_checkpoint,
    resolve_hf_repo_revision,
    resolve_hf_resume_revision,
)
from tenyson.core.hub_push import push_pretrained_snapshot_to_hub
from tenyson.core.model_policy import require_qwen3_model_name
from tenyson.core.plugin import TaskPlugin
from tenyson.core.execution_policy import require_gpu_provider_runtime
from tenyson.core.run_name import resolve_required_run_name
from tenyson.jobs.hf_repo import unique_repo_id
from tenyson.jobs.reporting_utils import normalize_report_to
from tenyson.jobs.result import JobResult
from tenyson.jobs.tokenizer_utils import normalize_tokenizer_special_tokens


def _resolve_eval_steps(
    train_cfg: Dict[str, Any],
    *,
    has_eval_dataset: bool,
) -> int | None:
    if not has_eval_dataset:
        return None

    eval_steps = int(train_cfg.get("eval_steps", 100))
    if eval_steps <= 0:
        raise ValueError(
            "training.eval_steps must be >= 1 when an SFT eval dataset is configured."
        )
    return eval_steps


def _resolve_early_stopping_settings(
    train_cfg: Dict[str, Any],
    *,
    has_eval_dataset: bool,
    save_steps: int,
    eval_steps: int | None,
) -> Dict[str, Any] | None:
    patience_value = train_cfg.get("early_stopping_patience")
    if patience_value is None:
        return None

    if not has_eval_dataset:
        raise ValueError(
            "training.early_stopping_patience requires an SFT eval dataset so "
            "eval_loss can be tracked."
        )

    patience = int(patience_value)
    if patience <= 0:
        raise ValueError("training.early_stopping_patience must be >= 1.")

    resolved_eval_steps = int(eval_steps or 0)
    if resolved_eval_steps <= 0:
        raise ValueError(
            "training.eval_steps must be >= 1 when early stopping is enabled."
        )

    resolved_save_steps = int(save_steps)
    if resolved_save_steps != resolved_eval_steps:
        raise ValueError(
            "training.hf_push_every_steps and training.eval_steps must match when "
            "early stopping is enabled so every evaluated checkpoint is saved and "
            "the best checkpoint can be restored."
        )

    return {
        "patience": patience,
        "min_delta": float(train_cfg.get("early_stopping_min_delta", 0.0)),
    }


def _enable_best_model_tracking(training_args: Any) -> None:
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "eval_loss"
    training_args.greater_is_better = False


def _reject_removed_sft_packing_setting(train_cfg: Dict[str, Any]) -> None:
    if "packing" not in train_cfg:
        return
    raise ValueError(
        "training.packing is no longer supported for SFT. Remove this field from "
        "the config; Tenyson keeps SFT packing disabled internally."
    )


def _resolve_finetune_mode(train_cfg: Dict[str, Any]) -> str:
    mode = str(train_cfg.get("finetune_mode", "lora") or "lora").strip().lower()
    if mode not in {"lora", "full"}:
        raise ValueError(
            "training.finetune_mode must be either 'lora' or 'full'."
        )
    return mode


def _require_full_finetune_model_config(model_cfg: Dict[str, Any]) -> None:
    if bool(model_cfg.get("load_in_4bit", False)):
        raise ValueError(
            "Full SFT finetuning requires model.load_in_4bit=false."
        )
    if bool(model_cfg.get("load_in_8bit", False)):
        raise ValueError(
            "Full SFT finetuning requires model.load_in_8bit=false."
        )


def _enable_unsloth_full_finetune_training_mode(
    model: Any,
    *,
    gradient_checkpointing: Any,
) -> None:
    for_training = getattr(model, "for_training", None)
    if not callable(for_training):
        return
    try:
        for_training(use_gradient_checkpointing=gradient_checkpointing)
    except TypeError:
        for_training()


def _normalize_full_finetune_sft_config(
    config: Dict[str, Any],
) -> tuple[Dict[str, Any], list[str]]:
    normalized = copy.deepcopy(config)
    train_cfg = normalized.setdefault("training", {})
    finetune_mode = str(train_cfg.get("finetune_mode", "lora") or "lora").strip().lower()
    if finetune_mode != "full":
        return normalized, []

    model_cfg = normalized.setdefault("model", {})
    messages: list[str] = []
    if bool(model_cfg.get("fast_inference", False)):
        model_cfg["fast_inference"] = False
        messages.append(
            "[SFTJob] Full finetuning is incompatible with fast inference; "
            "forcing model.fast_inference=false."
        )
    return normalized, messages


def _push_final_model_snapshot(
    *,
    repo_id: str,
    run_name: str,
    model: Any,
    tokenizer: Any,
    step: int,
    artifact_label: str = "model",
    best_checkpoint: str | None = None,
) -> None:
    checkpoint_name = os.path.basename(str(best_checkpoint or "").rstrip("/"))
    if checkpoint_name:
        reason = f"final best-model sync from {checkpoint_name}"
    else:
        reason = "final model sync"
    commit_message = f"[{run_name}] {reason} (step={int(step)})"

    try:
        push_pretrained_snapshot_to_hub(
            repo_id,
            model=model,
            tokenizer=tokenizer,
            commit_message=commit_message,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed pushing final SFT {artifact_label} snapshot to Hugging Face repo "
            f"'{repo_id}': {exc}"
        ) from exc


def _push_final_adapter_snapshot(
    *,
    repo_id: str,
    run_name: str,
    model: Any,
    tokenizer: Any,
    step: int,
    best_checkpoint: str | None = None,
) -> None:
    _push_final_model_snapshot(
        repo_id=repo_id,
        run_name=run_name,
        model=model,
        tokenizer=tokenizer,
        step=step,
        artifact_label="adapter",
        best_checkpoint=best_checkpoint,
    )


class SFTJob:
    """
    Supervised fine-tuning job.

    This class encapsulates the behaviour that previously lived in
    `src/runners/sft_run.py`, but is driven by a `TaskPlugin` instead of a
    loose task module.
    """

    def __init__(self, config: Dict[str, Any], task: TaskPlugin):
        self.config, self._runtime_normalization_messages = (
            _normalize_full_finetune_sft_config(config)
        )
        self.task = task
        self.run_id = resolve_required_run_name(self.config, "sft")

    def _build_model_and_tokenizer(self) -> Any:
        # Import locally so library import doesn't require heavy deps unless used.
        from unsloth import FastLanguageModel

        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})
        finetune_mode = _resolve_finetune_mode(train_cfg)
        full_finetuning = finetune_mode == "full"

        model_name = require_qwen3_model_name(
            model_cfg.get("name", "Qwen/Qwen3-4B")
        )
        seq_len = model_cfg.get("max_seq_length", 2048)
        if full_finetuning:
            _require_full_finetune_model_config(model_cfg)

        for message in self._runtime_normalization_messages:
            print(message, flush=True)

        print(f"[SFTJob] Loading model {model_name}...", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=seq_len,
            load_in_4bit=model_cfg.get("load_in_4bit", True),
            load_in_8bit=model_cfg.get("load_in_8bit", False),
            fast_inference=model_cfg.get("fast_inference", False),
            full_finetuning=full_finetuning,
        )

        normalize_tokenizer_special_tokens(tokenizer)

        if full_finetuning:
            print("[SFTJob] Preparing model for full finetuning...", flush=True)
            _enable_unsloth_full_finetune_training_mode(
                model,
                gradient_checkpointing=self.config.get("lora", {}).get(
                    "gradient_checkpointing",
                    "unsloth",
                ),
            )
            return model, tokenizer, seq_len

        print("[SFTJob] Applying LoRA...", flush=True)
        lora_cfg = self.config.get("lora", {})
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.get("r", 16),
            target_modules=lora_cfg.get(
                "target_modules", ["gate_proj", "up_proj", "down_proj"]
            ),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0),
            bias=lora_cfg.get("bias", "none"),
            use_gradient_checkpointing=lora_cfg.get(
                "gradient_checkpointing", "unsloth"
            ),
            random_state=train_cfg.get("seed", 3407),
            max_seq_length=seq_len,
        )

        return model, tokenizer, seq_len

    def run(self) -> JobResult:
        require_gpu_provider_runtime()
        from tenyson.core.hub_push import ensure_hf_repo
        from tenyson.core.telemetry import (
            start_run_heartbeat,
            begin_run_attempt,
            ensure_wandb_telemetry_run,
            ManualStopTelemetryCallback,
            record_run_result,
            record_run_summary,
            resolve_required_telemetry_context,
            RunHeartbeatTelemetryCallback,
            run_stop_requested,
            SFTTelemetryCallback,
            TelemetryClient,
            WandBUrlTelemetryCallback,
        )

        start = time.time()
        train_cfg = self.config.get("training", {})
        finetune_mode = _resolve_finetune_mode(train_cfg)
        run_name = self.run_id
        output_dir = train_cfg.get("output_dir", f"./outputs/{run_name}")
        attempt_token = str(
            self.config.get("telemetry", {}).get("attempt_token") or ""
        ).strip() or None
        _reject_removed_sft_packing_setting(train_cfg)
        backend_ref, experiment_id = resolve_required_telemetry_context(self.config)
        telemetry_client: Any = TelemetryClient(db_url=backend_ref)
        ensure_wandb_telemetry_run(
            telemetry_client,
            experiment_id=experiment_id,
            phase="sft",
            run_name=run_name,
            config=self.config,
            attempt_token=attempt_token,
        )
        if begin_run_attempt(
            telemetry_client,
            experiment_id,
            run_name,
            phase="sft",
            attempt_token=attempt_token,
        ):
            print(
                "[SFTJob] Cleared stale manual stop request from a previous attempt.",
                flush=True,
            )
        try:
            start_run_heartbeat(
                telemetry_client,
                experiment_id,
                run_name,
                "sft",
                attempt_token=attempt_token,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "[SFTJob] Warning: initial heartbeat registration failed; "
                f"continuing training. {exc}",
                flush=True,
            )

        def _wandb_url() -> Any:
            try:
                import wandb  # type: ignore[import-not-found]

                run = getattr(wandb, "run", None)
                if run is not None:
                    return getattr(run, "url", None)
            except Exception:  # noqa: BLE001
                return None
            return None

        def _finalize_result(result: JobResult) -> JobResult:
            record_run_summary(
                client=telemetry_client,
                experiment_id=experiment_id,
                phase="sft",
                result=result,
            )
            record_run_result(
                client=telemetry_client,
                experiment_id=experiment_id,
                run_id=run_name,
                phase="sft",
                results_payload=result,
                job_result_payload=result,
            )
            try:
                import wandb  # type: ignore[import-not-found]

                wandb.finish()
            except Exception:  # noqa: BLE001
                pass
            return result

        def _stop_requested_now() -> bool:
            try:
                return run_stop_requested(
                    telemetry_client,
                    experiment_id=experiment_id,
                    run_id=run_name,
                    phase="sft",
                    attempt_token=attempt_token,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    "[SFTJob] Warning: startup stop polling failed; continuing "
                    f"training startup. {exc}",
                    flush=True,
                )
                return False

        hf_repo_base = (train_cfg.get("hf_repo_base") or "").strip()
        if not hf_repo_base:
            raise ValueError(
                "training.hf_repo_base is required for SFT runs. "
                "Tenyson stores checkpoints/adapters on Hugging Face only."
            )
        if not os.getenv("HF_TOKEN", "").strip():
            raise ValueError("HF_TOKEN environment variable is required for SFT runs.")

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

        if _stop_requested_now():
            return _finalize_result(
                JobResult(
                    run_id=run_name,
                    status="stopped",
                    total_time_seconds=time.time() - start,
                    metrics={},
                    stopped_early=True,
                    wandb_url=_wandb_url(),
                    hf_repo_id=push_repo_id or None,
                    hf_revision=None,
                    failure_reason="Manual stop requested before SFT model load.",
                    attempt_token=attempt_token,
                )
            )

        model, tokenizer, seq_len = self._build_model_and_tokenizer()
        if _stop_requested_now():
            return _finalize_result(
                JobResult(
                    run_id=run_name,
                    status="stopped",
                    total_time_seconds=time.time() - start,
                    metrics={},
                    stopped_early=True,
                    wandb_url=_wandb_url(),
                    hf_repo_id=push_repo_id or None,
                    hf_revision=None,
                    failure_reason="Manual stop requested after SFT model load.",
                    attempt_token=attempt_token,
                )
            )
        # Import TRL/Transformers only after Unsloth has patched the runtime.
        from transformers import EarlyStoppingCallback
        from trl import SFTConfig, SFTTrainer
        import transformers
        import trl

        print(
            f"[SFTJob] Runtime versions: transformers={transformers.__version__}, trl={trl.__version__}",
            flush=True,
        )

        # Datasets via TaskPlugin hooks.
        print("[SFTJob] Loading datasets via TaskPlugin...", flush=True)
        train_dataset = self.task.get_sft_dataset(self.config, tokenizer)
        eval_dataset = self.task.get_sft_eval_dataset(self.config, tokenizer)
        eval_steps = _resolve_eval_steps(
            train_cfg,
            has_eval_dataset=eval_dataset is not None,
        )

        if train_dataset is None or len(train_dataset) == 0:
            raise ValueError("Training dataset is empty.")

        print(f"[SFTJob] Train size: {len(train_dataset)}", flush=True)
        if _stop_requested_now():
            return _finalize_result(
                JobResult(
                    run_id=run_name,
                    status="stopped",
                    total_time_seconds=time.time() - start,
                    metrics={},
                    stopped_early=True,
                    wandb_url=_wandb_url(),
                    hf_repo_id=push_repo_id or None,
                    hf_revision=None,
                    failure_reason="Manual stop requested before SFT training started.",
                    attempt_token=attempt_token,
                )
            )

        formatting_func = self.task.get_sft_formatting_func(self.config, tokenizer)
        report_to = normalize_report_to(
            train_cfg.get("report_to", "none"),
            telemetry_backend=telemetry_client.backend,
        )

        cfg_kwargs: Dict[str, Any] = dict(
            output_dir=output_dir,
            max_length=seq_len,
            max_steps=train_cfg.get("max_steps", 1000),
            per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
            learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
            warmup_steps=train_cfg.get("warmup_steps", 10),
            logging_steps=train_cfg.get("logging_steps", 1),
            save_strategy="steps",
            save_steps=hf_push_every_steps,
            save_total_limit=train_cfg.get("save_total_limit", 2),
            push_to_hub=True,
            hub_model_id=push_repo_id,
            hub_strategy="checkpoint",
            optim=train_cfg.get("optim", "adamw_8bit"),
            report_to=report_to,
            run_name=run_name,
            seed=train_cfg.get("seed", 3407),
            dataset_text_field=(
                train_cfg.get("dataset_text_field", "text")
                if not formatting_func
                else None
            ),
        )
        accepted = set(inspect.signature(SFTConfig.__init__).parameters.keys())
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
                "Installed SFTConfig does not expose required Hub checkpoint args "
                f"for full-state push: {sorted(missing_hub_fields)}. "
                "Upgrade TRL/Transformers runtime on the worker."
            )
        if "save_only_model" in accepted:
            cfg_kwargs["save_only_model"] = False
        training_args = SFTConfig(
            **{k: v for k, v in cfg_kwargs.items() if k in accepted}
        )
        print(
            "[SFTJob] SFTConfig special tokens: "
            f"eos_token={getattr(training_args, 'eos_token', None)!r}, "
            f"pad_token={getattr(training_args, 'pad_token', None)!r}.",
            flush=True,
        )

        if eval_dataset is not None:
            # Align evaluation and saving to step-based cadence.
            training_args.eval_strategy = "steps"
            training_args.eval_steps = eval_steps
            training_args.per_device_eval_batch_size = train_cfg.get(
                "per_device_eval_batch_size", 2
            )

        trainer_kwargs: Dict[str, Any] = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "processing_class": tokenizer,
        }

        if formatting_func is not None:
            trainer_kwargs["formatting_func"] = formatting_func

        collator = self.task.get_sft_data_collator(self.config, tokenizer)
        if collator is None:
            loss_on_assistant_only = train_cfg.get("loss_on_assistant_only", False)
            response_template = train_cfg.get("response_template") or ""
            if loss_on_assistant_only and response_template:
                from tenyson.jobs.sft_collator import CompletionOnlyDataCollator

                instruction_template = train_cfg.get("instruction_template")
                collator = CompletionOnlyDataCollator(
                    tokenizer=tokenizer,
                    response_template=response_template,
                    instruction_template=instruction_template,
                    max_length=seq_len,
                )
        if collator is not None:
            trainer_kwargs["data_collator"] = collator

        callbacks = []
        ensure_hf_repo(push_repo_id)

        callbacks.append(
            SFTTelemetryCallback(
                run_id=run_name,
                experiment_id=experiment_id,
                client=telemetry_client,
            )
        )
        manual_stop_callback = ManualStopTelemetryCallback(
            run_id=run_name,
            experiment_id=experiment_id,
            phase="sft",
            client=telemetry_client,
            attempt_token=attempt_token,
        )
        callbacks.append(manual_stop_callback)
        callbacks.append(
            RunHeartbeatTelemetryCallback(
                run_id=run_name,
                experiment_id=experiment_id,
                phase="sft",
                client=telemetry_client,
                attempt_token=attempt_token,
            )
        )
        if report_to == "wandb" or (
            isinstance(report_to, list) and "wandb" in report_to
        ):
            callbacks.append(
                WandBUrlTelemetryCallback(
                    run_id=run_name,
                    experiment_id=experiment_id,
                    client=telemetry_client,
                )
            )

        # Optional eval-loss early stopping.
        early_stopping_settings = _resolve_early_stopping_settings(
            train_cfg,
            has_eval_dataset=eval_dataset is not None,
            save_steps=hf_push_every_steps,
            eval_steps=eval_steps,
        )
        if early_stopping_settings is not None:
            _enable_best_model_tracking(training_args)
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_settings["patience"],
                    early_stopping_threshold=early_stopping_settings["min_delta"],
                )
            )

        if callbacks:
            trainer_kwargs["callbacks"] = callbacks

        trainer = SFTTrainer(**trainer_kwargs)

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
            print(f"[SFTJob] Resuming from checkpoint: {resume_path}", flush=True)

        print("[SFTJob] Starting training...", flush=True)
        if resume_path:
            train_result = trainer.train(resume_from_checkpoint=resume_path)
        else:
            train_result = trainer.train()

        total_time = time.time() - start

        # Basic metrics summary.
        metrics: Dict[str, Any] = {}
        if hasattr(train_result, "metrics") and isinstance(train_result.metrics, dict):
            metrics.update(train_result.metrics)
        trainer_state = getattr(trainer, "state", None)
        if trainer_state is not None:
            metrics.setdefault(
                "train_runtime", getattr(trainer_state, "train_runtime", None)
            )
            metrics.setdefault(
                "train_samples", getattr(trainer_state, "num_train_samples", None)
            )
            metrics.setdefault(
                "global_step", getattr(trainer_state, "global_step", None)
            )

        stop_requested = bool(getattr(manual_stop_callback, "stop_requested", False))
        stop_step = getattr(manual_stop_callback, "stop_step", None)
        if not stop_requested and push_repo_id:
            best_checkpoint = getattr(
                trainer_state,
                "best_model_checkpoint",
                None,
            )
            if finetune_mode == "full":
                _push_final_model_snapshot(
                    repo_id=push_repo_id,
                    run_name=run_name,
                    model=getattr(trainer, "model", model),
                    tokenizer=tokenizer,
                    step=int(getattr(trainer_state, "global_step", 0) or 0),
                    artifact_label="model",
                    best_checkpoint=best_checkpoint,
                )
            elif early_stopping_settings is not None:
                _push_final_adapter_snapshot(
                    repo_id=push_repo_id,
                    run_name=run_name,
                    model=getattr(trainer, "model", model),
                    tokenizer=tokenizer,
                    step=int(getattr(trainer_state, "global_step", 0) or 0),
                    best_checkpoint=best_checkpoint,
                )

        # Best-effort capture of the active WandB run URL, if any.
        wandb_url = _wandb_url()

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
            hf_artifact_type="full_model" if finetune_mode == "full" else "adapter",
            failure_reason=failure_reason,
            attempt_token=attempt_token,
        )
        return _finalize_result(result)
