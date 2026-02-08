"""
Full fine-tuning of Qwen 0.6B on Wordle. TRL SFTTrainer, no Unsloth, no LoRA.
Pushes full model to Hugging Face every 40 steps.

Experiment: 400 steps, global batch 64, checkpoint every 40 steps to HF.
Target: Lightning AI single T4 GPU.

Loss is computed only on assistant tokens (system and user prompts are masked).
Uses custom preprocessing to build assistant-only completion_mask for Qwen format.

Usage:
    export HF_TOKEN=your_token
    export HF_REPO_ID=yourusername/wordle-full-qwen06b
    python train_sft_full_finetune.py --wandb
"""

import argparse
import os
from typing import Any

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from trl import SFTConfig, SFTTrainer

DATASET_NAME = "willcb/V3-wordle"
MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_SEQ_LENGTH = 1024
MAX_STEPS = 400
GLOBAL_BATCH_SIZE = 64
PER_DEVICE_BATCH_SIZE = 4  # Full fine-tune; T4 15GB (float16 model)
LEARNING_RATE = 1e-5
PUSH_EVERY_STEPS = 40


def parse_args():
    p = argparse.ArgumentParser(description="Full SFT on Qwen 0.6B, push to HF every 40 steps")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--dataset", default=DATASET_NAME)
    p.add_argument("--hf-repo-id", default=os.environ.get("HF_REPO_ID"))
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--global-batch-size", type=int, default=GLOBAL_BATCH_SIZE)
    p.add_argument("--per-device-batch-size", type=int, default=PER_DEVICE_BATCH_SIZE)
    p.add_argument("--push-every-steps", type=int, default=PUSH_EVERY_STEPS)
    p.add_argument("--seq-len", type=int, default=MAX_SEQ_LENGTH)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--val-size", type=int, default=100)
    p.add_argument("--eval-every-steps", type=int, default=40)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "wordle-sft"))
    p.add_argument("--wandb-name", default=None)
    return p.parse_args()


# Qwen chat template markers for assistant-only loss masking
ASSISTANT_START = "<|im_start|>assistant\n"
ASSISTANT_END = "<|im_end|>"


def _assistant_spans(text: str) -> list[tuple[int, int]]:
    """Find character spans of assistant content in Qwen-formatted text."""
    spans = []
    start_marker = ASSISTANT_START
    end_marker = ASSISTANT_END
    pos = 0
    while True:
        start = text.find(start_marker, pos)
        if start == -1:
            break
        content_start = start + len(start_marker)
        end = text.find(end_marker, content_start)
        if end == -1:
            break
        span_end = end + len(end_marker)
        spans.append((content_start, span_end))
        pos = span_end
    return spans


def tokenize_with_assistant_mask(example: dict, tokenizer, max_length: int) -> dict:
    """
    Tokenize text and build completion_mask for assistant tokens only.
    Used when Qwen chat template lacks {% generation %} support.
    """
    text = example["text"]
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors=None,
    )
    input_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]

    spans = _assistant_spans(text)
    completion_mask = [0] * len(input_ids)

    for i, (start, end) in enumerate(offset_mapping):
        if start is None or end is None:
            continue
        for span_start, span_end in spans:
            if start < span_end and end > span_start:
                completion_mask[i] = 1
                break

    return {
        "input_ids": input_ids,
        "completion_mask": completion_mask,
    }


class PushToHubCallback(TrainerCallback):
    """Push full model to Hugging Face every N steps."""

    def __init__(self, repo_id: str, push_every: int):
        self.repo_id = repo_id
        self.push_every = push_every

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = int(getattr(state, "global_step", 0))
        if step <= 0 or step % self.push_every != 0:
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        revision = f"step-{step}"
        print(f"\n[PUSH] Pushing model to {self.repo_id} (revision={revision})...", flush=True)
        try:
            model.push_to_hub(
                self.repo_id,
                revision=revision,
                commit_message=f"Full SFT checkpoint step {step}",
            )
            print(f"[PUSH] Done: https://huggingface.co/{self.repo_id}/tree/{revision}\n", flush=True)
        except Exception as e:
            print(f"[PUSH] Failed: {e}", flush=True)
        return control


def main():
    args = parse_args()

    if not args.hf_repo_id:
        raise ValueError("Set --hf-repo-id or HF_REPO_ID env var")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise ValueError("Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN")

    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    api = HfApi()
    api.create_repo(repo_id=args.hf_repo_id, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    dataset = load_dataset(args.dataset, split="train")

    def format_conversation(example):
        messages = example["prompt"] + example["completion"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)
    print(f"Dataset size: {len(dataset)}")

    def tokenize_example(example):
        return tokenize_with_assistant_mask(example, tokenizer, args.seq_len)

    dataset = dataset.map(
        tokenize_example,
        remove_columns=["text"],
        desc="Tokenizing with assistant-only mask",
    )
    print("Tokenized with assistant-only loss masking")

    val_size = min(max(0, args.val_size), len(dataset) - 1)
    if val_size > 0:
        split = dataset.train_test_split(test_size=val_size, seed=args.seed, shuffle=True)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Train: {len(train_dataset)} | Val: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None

    grad_accum = args.global_batch_size // args.per_device_batch_size
    if args.global_batch_size % args.per_device_batch_size != 0:
        raise ValueError("global_batch_size must be divisible by per_device_batch_size")

    training_kwargs = dict(
        output_dir="outputs/full_sft",
        max_length=args.seq_len,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=1,
        save_steps=args.push_every_steps,
        save_total_limit=1,
        fp16=False,  # Model in float16; fp16 mixed precision causes scaler conflict
        gradient_checkpointing=True,
        report_to="wandb" if args.wandb else "none",
        run_name=args.wandb_name,
        completion_only_loss=True,
        dataset_text_field="text",
        dataset_kwargs={"skip_prepare_dataset": True},
        seed=args.seed,
    )
    if eval_dataset is not None:
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = args.eval_every_steps

    training_args = SFTConfig(**training_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[PushToHubCallback(repo_id=args.hf_repo_id, push_every=args.push_every_steps)],
    )

    print("Starting training...")
    trainer.train()

    print("\nPushing final model...")
    model.push_to_hub(args.hf_repo_id, revision="final", commit_message="Final full SFT checkpoint")
    tokenizer.push_to_hub(args.hf_repo_id, revision="final")
    print(f"Done: https://huggingface.co/{args.hf_repo_id}/tree/final")


if __name__ == "__main__":
    main()
