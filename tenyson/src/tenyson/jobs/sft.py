import json
import os
import time
from typing import Any, Dict

from tenyson.core.plugin import TaskPlugin
from tenyson.jobs.hf_repo import unique_repo_id
from tenyson.jobs.result import JobResult


class SFTJob:
    """
    Supervised fine-tuning job.

    This class encapsulates the behaviour that previously lived in
    `src/runners/sft_run.py`, but is driven by a `TaskPlugin` instead of a
    loose task module.
    """

    def __init__(self, config: Dict[str, Any], task: TaskPlugin):
        self.config = config
        self.task = task
        self.run_id = self.config.get("training", {}).get("run_name", "sft_job")

    def _build_model_and_tokenizer(self) -> Any:
        # Import locally so library import doesn't require heavy deps unless used.
        from unsloth import FastLanguageModel

        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})

        model_name = model_cfg.get("name", "Qwen/Qwen3-4B")
        seq_len = model_cfg.get("max_seq_length", 2048)

        print(f"[SFTJob] Loading model {model_name}...", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=seq_len,
            load_in_4bit=model_cfg.get("load_in_4bit", True),
            load_in_8bit=model_cfg.get("load_in_8bit", False),
            fast_inference=model_cfg.get("fast_inference", False),
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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
        from transformers import EarlyStoppingCallback
        from trl import SFTConfig, SFTTrainer
        from tenyson.core.telemetry import (
            ManualStopTelemetryCallback,
            record_run_summary,
            resolve_telemetry_context,
            SFTTelemetryCallback,
            TelemetryClient,
            WandBUrlTelemetryCallback,
        )

        start = time.time()
        train_cfg = self.config.get("training", {})
        run_name = train_cfg.get("run_name", self.run_id)
        output_dir = train_cfg.get("output_dir", f"./outputs/{run_name}")
        os.makedirs(output_dir, exist_ok=True)

        # Save config for reproducibility.
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

        model, tokenizer, seq_len = self._build_model_and_tokenizer()

        # Datasets via TaskPlugin hooks.
        print("[SFTJob] Loading datasets via TaskPlugin...", flush=True)
        train_dataset = self.task.get_sft_dataset(self.config, tokenizer)
        eval_dataset = self.task.get_sft_eval_dataset(self.config, tokenizer)

        if train_dataset is None or len(train_dataset) == 0:
            raise ValueError("Training dataset is empty.")

        print(f"[SFTJob] Train size: {len(train_dataset)}", flush=True)

        formatting_func = self.task.get_sft_formatting_func(self.config, tokenizer)

        training_args = SFTConfig(
            output_dir=output_dir,
            max_length=seq_len,
            max_steps=train_cfg.get("max_steps", 1000),
            per_device_train_batch_size=train_cfg.get(
                "per_device_train_batch_size", 2
            ),
            gradient_accumulation_steps=train_cfg.get(
                "gradient_accumulation_steps", 4
            ),
            learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
            warmup_steps=train_cfg.get("warmup_steps", 10),
            logging_steps=train_cfg.get("logging_steps", 1),
            save_steps=train_cfg.get("save_steps", 100),
            save_total_limit=train_cfg.get("save_total_limit", 2),
            optim=train_cfg.get("optim", "adamw_8bit"),
            report_to=train_cfg.get("report_to", "none"),
            run_name=run_name,
            seed=train_cfg.get("seed", 3407),
            packing=train_cfg.get("packing", False),
            dataset_text_field=(
                train_cfg.get("dataset_text_field", "text")
                if not formatting_func
                else None
            ),
        )

        if eval_dataset is not None:
            # Align evaluation and saving to step-based cadence.
            training_args.eval_strategy = "steps"
            training_args.eval_steps = train_cfg.get("eval_steps", 100)
            training_args.per_device_eval_batch_size = train_cfg.get(
                "per_device_eval_batch_size", 2
            )
            training_args.save_strategy = getattr(training_args, "save_strategy", "steps")

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
                if train_cfg.get("packing", False):
                    raise ValueError(
                        "loss_on_assistant_only is not supported with packing=True. "
                        "Set training.packing to false."
                    )
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

        # Optional telemetry wiring and manual stop callback.
        callbacks = []
        db_url, experiment_id = resolve_telemetry_context(self.config)
        telemetry_client: Any = None
        if db_url:
            telemetry_client = TelemetryClient(db_url=db_url)
            callbacks.append(
                SFTTelemetryCallback(
                    run_id=run_name,
                    experiment_id=experiment_id,  # type: ignore[arg-type]
                    client=telemetry_client,
                )
            )
            callbacks.append(
                ManualStopTelemetryCallback(
                    run_id=run_name,
                    experiment_id=experiment_id,  # type: ignore[arg-type]
                    client=telemetry_client,
                )
            )
            report_to = train_cfg.get("report_to", "none")
            if report_to == "wandb" or (
                isinstance(report_to, list) and "wandb" in report_to
            ):
                callbacks.append(
                    WandBUrlTelemetryCallback(
                        run_id=run_name,
                        experiment_id=experiment_id,  # type: ignore[arg-type]
                        client=telemetry_client,
                    )
                )

        # Optional eval-loss early stopping.
        patience = train_cfg.get("early_stopping_patience")
        if eval_dataset is not None and patience is not None:
            min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
            # Configure best-model tracking on eval_loss.
            training_args.load_best_model_at_end = True
            training_args.metric_for_best_model = "eval_loss"
            training_args.greater_is_better = False
            training_args.save_strategy = "steps"

            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=int(patience),
                    early_stopping_threshold=min_delta,
                )
            )

        if callbacks:
            trainer_kwargs["callbacks"] = callbacks

        trainer = SFTTrainer(**trainer_kwargs)

        resume_path = train_cfg.get("resume_from_checkpoint")
        if resume_path:
            if not os.path.isdir(resume_path):
                # Maybe HF repo/revision: try to download.
                try:
                    from huggingface_hub import snapshot_download
                    parts = resume_path.split(":", 1)
                    repo_id = parts[0]
                    revision = parts[1] if len(parts) > 1 else "main"
                    resume_path = snapshot_download(repo_id=repo_id, revision=revision)
                except Exception as e:  # noqa: BLE001
                    raise ValueError(
                        f"resume_from_checkpoint must be a local directory or repo:revision; got {resume_path}: {e}"
                    ) from e
            print(f"[SFTJob] Resuming from checkpoint: {resume_path}", flush=True)

        print("[SFTJob] Starting training...", flush=True)
        if resume_path:
            train_result = trainer.train(resume_from_checkpoint=resume_path)
        else:
            train_result = trainer.train()

        hf_repo_base = train_cfg.get("hf_repo_base") or train_cfg.get("hf_repo_id") or ""
        push_repo_id = unique_repo_id(hf_repo_base, run_name) if hf_repo_base else ""
        if push_repo_id:
            print(f"[SFTJob] Pushing adapter to Hub: {push_repo_id}", flush=True)
            model.push_to_hub(push_repo_id)
            tokenizer.push_to_hub(push_repo_id)
        else:
            print("[SFTJob] Saving model locally...", flush=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        total_time = time.time() - start

        # Basic metrics summary.
        metrics: Dict[str, Any] = {}
        if hasattr(train_result, "metrics") and isinstance(train_result.metrics, dict):
            metrics.update(train_result.metrics)
        if hasattr(trainer, "state"):
            metrics.setdefault(
                "train_runtime", getattr(trainer.state, "train_runtime", None)
            )
            metrics.setdefault(
                "train_samples", getattr(trainer.state, "num_train_samples", None)
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

        result = JobResult(
            run_id=run_name,
            status="success",
            total_time_seconds=total_time,
            metrics=metrics,
             wandb_url=wandb_url,
            hf_repo_id=push_repo_id or None,
            hf_revision="main" if push_repo_id else None,
            local_output_dir=output_dir,
        )
        result.save(os.path.join(output_dir, "results.json"))
        if telemetry_client is not None and experiment_id is not None:
            record_run_summary(
                client=telemetry_client,
                experiment_id=experiment_id,
                phase="sft",
                result=result,
            )
        return result
