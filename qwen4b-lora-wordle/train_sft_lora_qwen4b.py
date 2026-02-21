"""
LoRA SFT on Qwen3-4B for Wordle with completion-only loss.
Pushes adapter to Hugging Face every 20 steps with early stopping.

Uses Unsloth for 2x faster training, 70% less VRAM. LoRA applied to MLP layers only
(gate_proj, up_proj, down_proj).

Target: Lightning AI single T4 GPU.

Usage:
    python train_sft_lora_qwen4b.py

Environment variables (loaded from .env):
    HF_TOKEN - HuggingFace token for pushing adapters
    HF_REPO_ID - Target repo (e.g., username/wordle-lora-qwen3-4b)
    WANDB_API_KEY - Optional, for wandb logging
    WANDB_PROJECT - Wandb project name
"""

import argparse
import os
import re
import sys
from typing import Any, Optional

from unsloth import FastLanguageModel, FastModel
from datasets import load_dataset
from huggingface_hub import HfApi
import torch
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer

# Load .env file if exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ============ Configuration ============
DATASET_NAME = "goyalayus/wordle-reasoning-sft-prefix-keep-think"
MODEL_ID = "Qwen/Qwen3-4B"
MAX_SEQ_LENGTH = 2048
MAX_STEPS = 3000
GLOBAL_BATCH_SIZE = 64
PER_DEVICE_BATCH_SIZE = 16
LEARNING_RATE = 1e-5
PUSH_EVERY_STEPS = 20
EVAL_EVERY_STEPS = 20

# Early stopping criteria
PATIENCE_NO_IMPROVEMENT = 3  # Stop after 3 evals with no improvement
PATIENCE_INCREASE = 2  # Stop after 2 evals with increasing loss


def parse_args():
    p = argparse.ArgumentParser(description="LoRA SFT on Qwen3-4B with early stopping")
    p.add_argument(
        "--preset",
        choices=["qwen4b", "qwen06b"],
        default=None,
        help="Optional training preset that sets model + batch defaults",
    )
    p.add_argument("--model", default=MODEL_ID, help="Base model to fine-tune")
    p.add_argument("--dataset", default=DATASET_NAME, help="Dataset name")
    p.add_argument(
        "--hf-repo-id",
        default=os.environ.get("HF_REPO_ID"),
        help="HF repo for adapter checkpoints",
    )
    p.add_argument(
        "--max-steps", type=int, default=MAX_STEPS, help="Max training steps"
    )
    p.add_argument("--global-batch-size", type=int, default=GLOBAL_BATCH_SIZE)
    p.add_argument("--per-device-batch-size", type=int, default=PER_DEVICE_BATCH_SIZE)
    p.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=None,
        help="Eval batch size. Default: half of train batch, min 1",
    )
    p.add_argument("--push-every-steps", type=int, default=PUSH_EVERY_STEPS)
    p.add_argument("--eval-every-steps", type=int, default=EVAL_EVERY_STEPS)
    p.add_argument("--seq-len", type=int, default=MAX_SEQ_LENGTH)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--val-size", type=int, default=100, help="Validation set size")
    p.add_argument("--patience-no-improve", type=int, default=PATIENCE_NO_IMPROVEMENT)
    p.add_argument("--patience-increase", type=int, default=PATIENCE_INCREASE)
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    p.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "wordle-lora-qwen3-4b"),
    )
    p.add_argument("--wandb-name", default=None, help="Wandb run name")
    p.add_argument(
        "--output-root",
        default=None,
        help="Root folder for run outputs (default: <project>/outputs)",
    )
    return p.parse_args()


def model_to_slug(model_id: str) -> str:
    """Convert model id to filesystem-safe slug."""
    slug = model_id.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


class PushToHubCallback(TrainerCallback):
    """Push LoRA adapter to Hugging Face every N steps."""

    def __init__(self, repo_id: str, push_every: int):
        self.repo_id = repo_id
        self.push_every = push_every
        self.api = HfApi()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        step = int(getattr(state, "global_step", 0))
        if step <= 0 or step % self.push_every != 0:
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        revision = f"step-{step}"
        print(
            f"\n[PUSH] Pushing adapter to {self.repo_id} (revision={revision})...",
            flush=True,
        )
        try:
            model.push_to_hub(
                self.repo_id,
                revision=revision,
                commit_message=f"LoRA checkpoint step {step}",
            )
            print(
                f"[PUSH] Done: https://huggingface.co/{self.repo_id}/tree/{revision}\n",
                flush=True,
            )
        except Exception as e:
            print(f"[PUSH] Failed: {e}", flush=True)
        return control


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping based on validation loss."""

    def __init__(self, patience_no_improve: int = 3, patience_increase: int = 2):
        self.patience_no_improve = patience_no_improve
        self.patience_increase = patience_increase
        self.best_loss = float("inf")
        self.steps_no_improve = 0
        self.steps_increasing = 0
        self.last_loss = float("inf")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict = None,
        **kwargs,
    ):
        if metrics is None or "eval_loss" not in metrics:
            return control

        current_loss = metrics["eval_loss"]
        step = state.global_step

        print(f"\n[EVAL] Step {step}: eval_loss = {current_loss:.4f}")
        print(f"[EVAL] Best loss so far: {self.best_loss:.4f}")

        # Check for improvement
        if current_loss < self.best_loss:
            improvement = self.best_loss - current_loss
            self.best_loss = current_loss
            self.steps_no_improve = 0
            self.steps_increasing = 0
            print(f"[EVAL] New best! Improvement: {improvement:.4f}")
        else:
            self.steps_no_improve += 1
            print(
                f"[EVAL] No improvement. Steps without improvement: {self.steps_no_improve}/{self.patience_no_improve}"
            )

        # Check for increasing loss
        if current_loss > self.last_loss:
            self.steps_increasing += 1
            print(
                f"[EVAL] Loss increased. Consecutive increases: {self.steps_increasing}/{self.patience_increase}"
            )
        else:
            self.steps_increasing = 0

        self.last_loss = current_loss

        # Early stopping conditions
        if self.steps_no_improve >= self.patience_no_improve:
            print(
                f"\n[EARLY STOP] No improvement for {self.patience_no_improve} evaluations. Stopping."
            )
            control.should_training_stop = True

        if self.steps_increasing >= self.patience_increase:
            print(
                f"\n[EARLY STOP] Loss increased for {self.patience_increase} consecutive evaluations. Stopping."
            )
            control.should_training_stop = True

        return control


class BoolAttentionMaskCollator:
    """Drop incompatible attention masks on this stack."""

    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, examples):
        batch = self.base_collator(examples)
        batch.pop("attention_mask", None)
        return batch


class AssistantOnlyMaskCollator:
    """Mask labels outside assistant message bodies."""

    def __init__(self, base_collator, assistant_header_ids, im_end_id):
        self.base_collator = base_collator
        self.assistant_header_ids = assistant_header_ids
        self.im_end_id = im_end_id

    def _find_subsequence(self, sequence, pattern, start_index):
        stop = len(sequence) - len(pattern) + 1
        for idx in range(start_index, max(start_index, stop)):
            if sequence[idx : idx + len(pattern)] == pattern:
                return idx
        return -1

    def _assistant_token_mask(self, token_ids):
        mask = [False] * len(token_ids)
        cursor = 0

        while cursor < len(token_ids):
            header_start = self._find_subsequence(
                token_ids, self.assistant_header_ids, cursor
            )
            if header_start < 0:
                break

            content_start = header_start + len(self.assistant_header_ids)
            end_index = content_start
            while end_index < len(token_ids) and token_ids[end_index] != self.im_end_id:
                end_index += 1

            for pos in range(content_start, min(end_index, len(mask))):
                mask[pos] = True
            if end_index < len(mask) and token_ids[end_index] == self.im_end_id:
                mask[end_index] = True

            cursor = end_index + 1

        return mask

    def __call__(self, examples):
        batch = self.base_collator(examples)
        labels = batch["labels"]
        input_ids = batch["input_ids"]

        for row_idx in range(input_ids.size(0)):
            row_ids = input_ids[row_idx].tolist()
            assistant_mask = self._assistant_token_mask(row_ids)
            for token_idx, is_assistant in enumerate(assistant_mask):
                if not is_assistant:
                    labels[row_idx, token_idx] = -100

        batch.pop("attention_mask", None)
        return batch


def main():
    args = parse_args()

    # Optional model presets for safer defaults
    if args.preset == "qwen4b":
        args.model = "Qwen/Qwen3-4B"
        args.per_device_batch_size = 4
        args.global_batch_size = 32
    elif args.preset == "qwen06b":
        args.model = "Qwen/Qwen3-0.6B"
        args.per_device_batch_size = 48
        args.global_batch_size = 48

    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = max(1, args.per_device_batch_size // 2)

    model_slug = model_to_slug(args.model)
    output_root = args.output_root or os.path.join(os.path.dirname(__file__), "outputs")
    run_output_dir = os.path.join(output_root, f"lora_sft_{model_slug}")

    # Validate HF repo
    if not args.hf_repo_id:
        raise ValueError(
            "Set --hf-repo-id or HF_REPO_ID env var (e.g., username/wordle-lora-qwen3-4b)"
        )

    # Check HF token
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        raise ValueError("Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN for push to Hub")

    # Setup wandb
    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_name:
            os.environ["WANDB_NAME"] = args.wandb_name

    print(f"=" * 60)
    print(f"LoRA SFT on {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"HF Repo: {args.hf_repo_id}")
    print(f"Global Batch Size: {args.global_batch_size}")
    print(f"Per-Device Batch Size: {args.per_device_batch_size}")
    print(f"Per-Device Eval Batch Size: {args.per_device_eval_batch_size}")
    print(f"Max Seq Length: {args.seq_len}")
    print(f"Learning Rate: {args.lr}")
    print(f"Output Dir: {run_output_dir}")
    print(
        f"Early Stopping: {args.patience_no_improve} evals no improvement OR {args.patience_increase} evals increasing"
    )
    print(f"=" * 60)

    # Ensure repo exists
    api = HfApi()
    api.create_repo(repo_id=args.hf_repo_id, exist_ok=True)

    # Load model and tokenizer
    print("\n[1/6] Loading model with 4-bit quantization...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.seq_len,
        load_in_4bit=True,
        load_in_8bit=False,
        load_in_16bit=False,
        full_finetuning=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer must expose eos_token for assistant span masking.")

    # Apply LoRA
    print("[2/6] Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=args.seq_len,
    )
    model.print_trainable_parameters()

    # Load dataset
    print("[3/6] Loading dataset...")
    dataset = load_dataset(args.dataset, split="train")

    if "messages" not in dataset.column_names:
        raise ValueError("Dataset must contain a 'messages' column for conversational SFT.")
    print(f"Dataset size: {len(dataset)}")

    # Train/val split
    val_size = min(max(0, args.val_size), len(dataset) - 1)
    if val_size > 0:
        split = dataset.train_test_split(
            test_size=val_size, seed=args.seed, shuffle=True
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Train: {len(train_dataset)} | Val: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
        print("Warning: No validation set. Early stopping will not work.")

    # Build training text from conversational rows.
    print("[4/6] Preparing chat-template formatting...")

    def format_conversation(example):
        messages = example["messages"]
        if messages and isinstance(messages[0], list):
            return [
                tokenizer.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=False
                )
                for m in messages
            ]
        return [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        ]

    # Calculate gradient accumulation
    grad_accum = args.global_batch_size // args.per_device_batch_size
    if args.global_batch_size % args.per_device_batch_size != 0:
        raise ValueError(
            f"global_batch_size ({args.global_batch_size}) must be divisible by "
            f"per_device_batch_size ({args.per_device_batch_size})"
        )

    print(f"[5/6] Training configuration:")
    print(f"  - Gradient accumulation steps: {grad_accum}")
    print(
        f"  - Effective batch size: {args.per_device_batch_size} * {grad_accum} = {args.global_batch_size}"
    )
    print(f"  - Eval every: {args.eval_every_steps} steps")
    print(f"  - Push every: {args.push_every_steps} steps")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
    print(f"  - Precision: {'bf16' if use_bf16 else 'fp16'}")

    # Training arguments
    training_kwargs = dict(
        output_dir=run_output_dir,
        max_length=args.seq_len,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=1,
        save_steps=args.push_every_steps,
        save_total_limit=1,
        fp16=use_fp16,
        bf16=use_bf16,
        optim="adamw_8bit",
        report_to="wandb" if args.wandb else "none",
        run_name=args.wandb_name,
        seed=args.seed,
        remove_unused_columns=False,
    )

    if eval_dataset is not None:
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = args.eval_every_steps
        training_kwargs["load_best_model_at_end"] = (
            False  # We handle early stopping manually
        )

    training_kwargs["packing"] = False
    training_args = SFTConfig(**training_kwargs)

    # Setup callbacks
    callbacks = [
        PushToHubCallback(repo_id=args.hf_repo_id, push_every=args.push_every_steps),
    ]

    if eval_dataset is not None:
        callbacks.append(
            EarlyStoppingCallback(
                patience_no_improve=args.patience_no_improve,
                patience_increase=args.patience_increase,
            )
        )

    # Create trainer
    print("[6/6] Creating trainer...")

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=format_conversation,
        callbacks=callbacks,
    )

    if not training_kwargs.get("packing"):
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

        base_collator = DataCollatorForLanguageModeling(
            pad_token_id=tokenizer.pad_token_id,
            completion_only_loss=False,
        )
        assistant_header_ids = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        if not assistant_header_ids:
            raise ValueError("Could not tokenize assistant header for loss masking.")
        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        trainer_kwargs["data_collator"] = AssistantOnlyMaskCollator(
            base_collator=base_collator,
            assistant_header_ids=assistant_header_ids,
            im_end_id=im_end_id,
        )

    trainer = SFTTrainer(**trainer_kwargs)

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    trainer.train()

    # Final push
    print("\n" + "=" * 60)
    print("Pushing final adapter...")
    print("=" * 60)
    model.push_to_hub(
        args.hf_repo_id, revision="final", commit_message="Final LoRA checkpoint"
    )
    print(f"\nDone! Final adapter: https://huggingface.co/{args.hf_repo_id}/tree/final")
    print(f"All checkpoints: https://huggingface.co/{args.hf_repo_id}/tree/main")


if __name__ == "__main__":
    main()
