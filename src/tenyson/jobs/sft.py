import inspect
import os
import time
from typing import Any, Dict

from tenyson.core.hf_checkpoint import (
    download_hf_resume_checkpoint,
    resolve_hf_repo_revision,
    resolve_hf_resume_revision,
)
from tenyson.core.model_policy import require_qwen3_model_name
from tenyson.core.plugin import TaskPlugin
from tenyson.core.execution_policy import require_gpu_provider_runtime
from tenyson.core.run_name import resolve_required_run_name
from tenyson.jobs.hf_repo import unique_repo_id
from tenyson.jobs.reporting_utils import normalize_report_to
from tenyson.jobs.result import JobResult
from tenyson.jobs.sft_dataset import (
    build_builtin_sft_formatting_func,
    normalize_builtin_sft_dataset,
    supports_builtin_sft_schema,
)
from tenyson.jobs.tokenizer_utils import (
    ensure_assistant_mask_chat_template,
    normalize_tokenizer_special_tokens,
)


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


def _is_conversational_message_list(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    return isinstance(first, dict) and "role" in first and "content" in first


def _is_pretokenized_sft_row(row: Any) -> bool:
    return isinstance(row, dict) and "input_ids" in row


def _resolve_assistant_only_strategy(
    train_cfg: Dict[str, Any],
    *,
    train_sample: Any,
) -> Dict[str, Any]:
    loss_on_assistant_only = bool(train_cfg.get("loss_on_assistant_only", False))
    response_template = str(train_cfg.get("response_template") or "").strip()
    packing_enabled = bool(train_cfg.get("packing", False))
    supports_builtin_schema = supports_builtin_sft_schema(train_sample)

    if not loss_on_assistant_only:
        return {
            "use_native_assistant_only_loss": False,
            "use_manual_assistant_masks": False,
            "use_response_template_collator": False,
        }

    if supports_builtin_schema:
        return {
            "use_native_assistant_only_loss": False,
            "use_manual_assistant_masks": True,
            "use_response_template_collator": False,
        }

    if response_template and not packing_enabled:
        return {
            "use_native_assistant_only_loss": False,
            "use_manual_assistant_masks": False,
            "use_response_template_collator": True,
        }

    raise ValueError(
        "loss_on_assistant_only requires one of Tenyson's built-in SFT dataset "
        "schemas (`messages`, conversational `prompt`/`completion`, string "
        "`prompt`/`answer`, string `prompt`/`completion`, string "
        "`question`/`answer`, or `instruction`/`output` with optional `input`). "
        "Legacy preformatted text runs can still use training.response_template "
        "when packing is disabled."
    )


def _resolve_sft_special_tokens_kwargs(
    tokenizer: Any,
    *,
    accepted_fields: set[str],
) -> Dict[str, str]:
    kwargs: Dict[str, str] = {}
    eos_token = getattr(tokenizer, "eos_token", None)
    pad_token = getattr(tokenizer, "pad_token", None)
    if "eos_token" in accepted_fields and isinstance(eos_token, str) and eos_token:
        kwargs["eos_token"] = eos_token
    if "pad_token" in accepted_fields and isinstance(pad_token, str) and pad_token:
        kwargs["pad_token"] = pad_token
    return kwargs


def _fallback_sft_formatting_func(_example: Dict[str, Any]) -> list[str]:
    # Unsloth requires a formatting_func in some conversational SFT paths even
    # when the dataset is already tokenized and dataset preparation is skipped.
    return [""]


def _extract_conversational_sft_messages(example: Dict[str, Any]) -> list[Dict[str, Any]]:
    messages = example.get("messages")
    if _is_conversational_message_list(messages):
        return list(messages)

    prompt = example.get("prompt")
    completion = example.get("completion")
    merged_messages: list[Dict[str, Any]] = []
    if _is_conversational_message_list(prompt):
        merged_messages.extend(list(prompt))
    if _is_conversational_message_list(completion):
        merged_messages.extend(list(completion))
    if merged_messages:
        return merged_messages

    raise ValueError(
        "Packed assistant-only SFT requires conversational rows exposed through "
        "`messages` or conversational `prompt`/`completion` fields."
    )


def _tokenize_assistant_only_conversation(
    example: Dict[str, Any],
    *,
    tokenizer: Any,
) -> Dict[str, Any]:
    messages = _extract_conversational_sft_messages(example)
    processed = tokenizer.apply_chat_template(
        messages,
        tools=example.get("tools"),
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
        **example.get("chat_template_kwargs", {}),
    )

    input_ids = processed.get("input_ids")
    assistant_masks = processed.get("assistant_masks")
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if (
        isinstance(assistant_masks, list)
        and assistant_masks
        and isinstance(assistant_masks[0], list)
    ):
        assistant_masks = assistant_masks[0]

    if not isinstance(input_ids, list) or not input_ids:
        raise RuntimeError("Packed assistant-only SFT produced no input_ids.")
    if not isinstance(assistant_masks, list):
        raise RuntimeError(
            "Packed assistant-only SFT expected assistant_masks from the tokenizer, "
            "but none were returned. Check the chat template."
        )
    if 1 not in assistant_masks:
        raise RuntimeError(
            "Packed assistant-only SFT found a conversational example with no "
            "assistant-supervised tokens."
        )

    return {
        "input_ids": list(input_ids),
        "assistant_masks": list(assistant_masks),
    }


def _prepare_manual_assistant_only_dataset(
    dataset: Any,
    *,
    tokenizer: Any,
    max_length: int,
    packing: bool,
    packing_strategy: str,
    shuffle: bool,
    seed: int,
    dataset_name: str,
) -> Any:
    from datasets import Dataset, IterableDataset
    from trl.data_utils import pack_dataset
    from trl.trainer.sft_trainer import truncate_dataset

    if isinstance(dataset, IterableDataset):
        raise ValueError(
            "Packed assistant-only SFT currently requires a map-style Hugging Face "
            "Dataset, not an IterableDataset."
        )

    column_names = list(getattr(dataset, "column_names", []) or [])
    map_kwargs: Dict[str, Any] = {}
    if isinstance(dataset, Dataset):
        map_kwargs["desc"] = (
            f"Tokenizing {dataset_name} dataset for manual assistant-only SFT"
        )

    prepared = dataset.map(
        lambda row: _tokenize_assistant_only_conversation(row, tokenizer=tokenizer),
        remove_columns=column_names or None,
        **map_kwargs,
    )

    if shuffle:
        prepared = prepared.shuffle(seed=seed)

    if packing:
        pack_kwargs: Dict[str, Any] = {}
        if isinstance(prepared, Dataset):
            pack_kwargs["desc"] = (
                f"Packing {dataset_name} dataset for manual assistant-only SFT"
            )
        prepared = pack_dataset(
            prepared.select_columns(["input_ids", "assistant_masks"]),
            max_length,
            packing_strategy,
            pack_kwargs,
        )
    else:
        truncate_kwargs: Dict[str, Any] = {}
        if isinstance(prepared, Dataset):
            truncate_kwargs["desc"] = (
                f"Truncating {dataset_name} dataset for manual assistant-only SFT"
            )
        prepared = truncate_dataset(prepared, max_length, truncate_kwargs)

    if shuffle:
        prepared = prepared.shuffle(seed=seed)

    return prepared


def _push_final_adapter_snapshot(
    *,
    repo_id: str,
    run_name: str,
    model: Any,
    tokenizer: Any,
    step: int,
    best_checkpoint: str | None = None,
) -> None:
    checkpoint_name = os.path.basename(str(best_checkpoint or "").rstrip("/"))
    if checkpoint_name:
        reason = f"final best-model sync from {checkpoint_name}"
    else:
        reason = "final model sync"
    commit_message = f"[{run_name}] {reason} (step={int(step)})"

    try:
        model.push_to_hub(repo_id, commit_message=commit_message)
        if tokenizer is not None:
            tokenizer.push_to_hub(repo_id, commit_message=commit_message)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed pushing final SFT adapter snapshot to Hugging Face repo "
            f"'{repo_id}': {exc}"
        ) from exc


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
        self.run_id = resolve_required_run_name(self.config, "sft")

    def _build_model_and_tokenizer(self) -> Any:
        # Import locally so library import doesn't require heavy deps unless used.
        from unsloth import FastLanguageModel

        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})

        model_name = require_qwen3_model_name(
            model_cfg.get("name", "Qwen/Qwen3-4B")
        )
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

        normalize_tokenizer_special_tokens(tokenizer)

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
            SFTTelemetryCallback,
            TelemetryClient,
            WandBUrlTelemetryCallback,
        )

        start = time.time()
        train_cfg = self.config.get("training", {})
        run_name = self.run_id
        output_dir = train_cfg.get("output_dir", f"./outputs/{run_name}")
        attempt_token = str(
            self.config.get("telemetry", {}).get("attempt_token") or ""
        ).strip() or None
        backend_ref, experiment_id = resolve_required_telemetry_context(self.config)
        telemetry_client: Any = TelemetryClient(db_url=backend_ref)
        ensure_wandb_telemetry_run(
            telemetry_client,
            experiment_id=experiment_id,
            phase="sft",
            run_name=run_name,
            config=self.config,
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
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "[SFTJob] Warning: initial heartbeat registration failed; "
                f"continuing training. {exc}",
                flush=True,
            )

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

        model, tokenizer, seq_len = self._build_model_and_tokenizer()
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

        train_sample = train_dataset[0]
        if (
            not _is_pretokenized_sft_row(train_sample)
            and supports_builtin_sft_schema(train_sample)
        ):
            train_dataset = normalize_builtin_sft_dataset(
                train_dataset,
                config=self.config,
                dataset_name="train",
            )
            if (
                eval_dataset is not None
                and not _is_pretokenized_sft_row(eval_dataset[0])
                and supports_builtin_sft_schema(eval_dataset[0])
            ):
                eval_dataset = normalize_builtin_sft_dataset(
                    eval_dataset,
                    config=self.config,
                    dataset_name="eval",
                )
            print(
                "[SFTJob] Normalized supported SFT rows into canonical chat messages.",
                flush=True,
            )
            train_sample = train_dataset[0]
        formatting_func = None
        assistant_only_strategy = _resolve_assistant_only_strategy(
            train_cfg,
            train_sample=train_sample,
        )
        use_manual_assistant_masks = assistant_only_strategy[
            "use_manual_assistant_masks"
        ]
        if (
            formatting_func is None
            and not use_manual_assistant_masks
            and supports_builtin_sft_schema(train_sample)
        ):
            formatting_func = build_builtin_sft_formatting_func(tokenizer)
        if use_manual_assistant_masks:
            ensure_assistant_mask_chat_template(tokenizer)
            formatting_func = formatting_func or _fallback_sft_formatting_func
            packing_strategy = (
                str(train_cfg.get("packing_strategy", "bfd")).strip() or "bfd"
            )
            seed = int(train_cfg.get("seed", 3407))
            eval_packing = train_cfg.get("eval_packing")
            if eval_packing is None:
                eval_packing = bool(train_cfg.get("packing", False))
            else:
                eval_packing = bool(eval_packing)

            train_dataset = _prepare_manual_assistant_only_dataset(
                train_dataset,
                tokenizer=tokenizer,
                max_length=seq_len,
                packing=bool(train_cfg.get("packing", False)),
                packing_strategy=packing_strategy,
                shuffle=True,
                seed=seed,
                dataset_name="train",
            )
            if eval_dataset is not None:
                eval_dataset = _prepare_manual_assistant_only_dataset(
                    eval_dataset,
                    tokenizer=tokenizer,
                    max_length=seq_len,
                    packing=eval_packing,
                    packing_strategy=packing_strategy,
                    shuffle=False,
                    seed=seed,
                    dataset_name="eval",
                )
            print(
                "[SFTJob] Using manual assistant masks for packed conversational "
                f"SFT. Prepared train rows: {len(train_dataset)}"
                + (
                    f", eval rows: {len(eval_dataset)}"
                    if eval_dataset is not None
                    else ""
                ),
                flush=True,
            )
        if assistant_only_strategy["use_native_assistant_only_loss"]:
            ensure_assistant_mask_chat_template(tokenizer)
            print(
                "[SFTJob] Using TRL native assistant-only masking for "
                "conversational SFT.",
                flush=True,
            )
        report_to = normalize_report_to(
            train_cfg.get("report_to", "none"),
            telemetry_backend=telemetry_client.backend,
        )
        manual_dataset_kwargs = train_cfg.get("dataset_kwargs")
        if use_manual_assistant_masks:
            manual_dataset_kwargs = dict(manual_dataset_kwargs or {})
            manual_dataset_kwargs["skip_prepare_dataset"] = True

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
            packing=(
                False
                if use_manual_assistant_masks
                else train_cfg.get("packing", False)
            ),
            padding_free=bool(
                use_manual_assistant_masks and bool(train_cfg.get("packing", False))
            ),
            assistant_only_loss=assistant_only_strategy[
                "use_native_assistant_only_loss"
            ],
            dataset_text_field=(
                train_cfg.get("dataset_text_field", "text")
                if (
                    not formatting_func
                    and not _is_pretokenized_sft_row(train_dataset[0])
                )
                else None
            ),
            dataset_kwargs=manual_dataset_kwargs,
        )
        accepted = set(inspect.signature(SFTConfig.__init__).parameters.keys())
        cfg_kwargs.update(
            _resolve_sft_special_tokens_kwargs(
                tokenizer,
                accepted_fields=accepted,
            )
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
                "Installed SFTConfig does not expose required Hub checkpoint args "
                f"for full-state push: {sorted(missing_hub_fields)}. "
                "Upgrade TRL/Transformers runtime on the worker."
            )
        if (
            assistant_only_strategy["use_native_assistant_only_loss"]
            and "assistant_only_loss" not in accepted
        ):
            raise RuntimeError(
                "Installed SFTConfig does not expose assistant_only_loss. "
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

        collator = None
        if assistant_only_strategy["use_response_template_collator"]:
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
        )
        callbacks.append(manual_stop_callback)
        callbacks.append(
            RunHeartbeatTelemetryCallback(
                run_id=run_name,
                experiment_id=experiment_id,
                phase="sft",
                client=telemetry_client,
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
        if early_stopping_settings is not None and not stop_requested and push_repo_id:
            _push_final_adapter_snapshot(
                repo_id=push_repo_id,
                run_name=run_name,
                model=getattr(trainer, "model", model),
                tokenizer=tokenizer,
                step=int(getattr(trainer_state, "global_step", 0) or 0),
                best_checkpoint=getattr(
                    trainer_state,
                    "best_model_checkpoint",
                    None,
                ),
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
