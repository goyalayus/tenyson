# Tenyson Experiment Report

- Generated: `2026-04-01 12:52:36 UTC`
- Environment: `wordle`
- Experiment ID: `full_finetune_smoke_autofastoff_20260401_r10_06b`
- Telemetry backend: `wandb://ayush_g-iit-roorkee/wordle-research`
- Telemetry dashboard: [open project](https://wandb.ai/ayush_g-iit-roorkee/wordle-research)
- Stage summary: 3 success

## Stages

### 1. sft_full_smoke_06b_autofix

- Run type: `sft`
- Run name: `sft_full_smoke_06b_autofix`
- Environment run: `default`
- Status: `success`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/full_finetune_smoke_autofastoff_20260401_r10_06b-sft-sft_full_smoke_06b_autofix-802ca627-0d35c592b3)
- Hugging Face artifact: `goyalayus/wordle-lora-20260324-163252-sft_full_smoke_06b_autofix` @ `4034536a5966ac1f69a2857a80b509afdcbd424a`
- Runtime (seconds): `102.89`
- Metric `epoch`: `0.0005`
- Metric `global_step`: `2`
- Metric `total_flos`: `1997960380416.0000`
- Metric `train_loss`: `1.7964`
- Metric `train_runtime`: `53.1680`
- Metric `train_samples`: `n/a`
- Metric `train_samples_per_second`: `0.0380`
- Metric `train_steps_per_second`: `0.0380`

### 2. eval_full_after_sft_06b_autofix

- Run type: `eval`
- Run name: `eval_full_after_sft_06b_autofix`
- Environment run: `default`
- Status: `success`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/full_finetune_smoke_autofastoff_20260401_r10_06b-eval-eval_full_after_sft_06b_autofix-d30e921f-4c5bd511d6)
- Runtime (seconds): `138.17`
- Processed samples: `16 / 16`
- Metric `avg_constraint_reward`: `0.0000`
- Metric `constraint_accuracy`: `0.0000`
- Metric `dict_accuracy`: `0.0000`
- Metric `format_accuracy`: `0.0000`
- Metric `total_samples`: `16`

### 3. rl_full_from_sft_06b_autofix

- Run type: `rl`
- Run name: `rl_full_from_sft_06b_autofix`
- Environment run: `default`
- Status: `success`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/full_finetune_smoke_autofastoff_20260401_r10_06b-rl-rl_full_from_sft_06b_autofix-096ecc6a-1ee8ca86c4)
- Hugging Face artifact: `goyalayus/wordle-lora-20260324-163252-rl_full_from_sft_06b_autofix` @ `a0a26a57006331c221f4923239b1acf8627c2dba`
- Runtime (seconds): `402.62`
- Metric `global_step`: `2`
- Metric `rollout_batches`: `2`
- Metric `total_flos`: `0.0000`
- Metric `total_samples`: `4`
- Metric `train_loss`: `0.0000`
- Metric `train_runtime`: `295.8231`
- Metric `train_samples_per_second`: `0.0140`
- Metric `train_steps_per_second`: `0.0070`
