# modal launcher for 4B LoRA SFT

This launcher runs the existing `train_sft_lora_qwen4b.py` script on Modal with a GPU and persistent output volume.

## 1) install and auth

Run from your local machine:

```bash
pip install modal
python3 -m modal setup
```

## 2) create secret for HF and W&B

Create one secret named `wordle-train-secrets`:

```bash
modal secret create wordle-train-secrets \
  HF_TOKEN=xxx \
  WANDB_API_KEY=xxx \
  WANDB_PROJECT=wordle-lora-qwen3-4b
```

`HF_TOKEN` is required. `WANDB_API_KEY` and `WANDB_PROJECT` are optional if you run without `--wandb`.

## 3) launch 4B run on A10G

From `qwen4b-lora-wordle/`:

```bash
modal run modal_train.py \
  --hf-repo-id goyalayus/wordle-full-qwen4b \
  --preset qwen4b \
  --max-steps 3000 \
  --wandb true \
  --wandb-name qwen4b-modal-retrain
```

## 4) pass extra trainer flags

Use `--extra-args` to forward flags to `train_sft_lora_qwen4b.py`.

Example:

```bash
modal run modal_train.py \
  --hf-repo-id goyalayus/wordle-full-qwen4b \
  --preset qwen4b \
  --extra-args "--eval-every-steps 20 --push-every-steps 20"
```

## notes

- GPU is set to `A10G`.
- Outputs are written to a Modal Volume named `wordle-lora-outputs`.
- The training logic is unchanged; this launcher only provides compute/runtime.
