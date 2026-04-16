# Tenyson Experiment Report

- Generated: `2026-04-16 17:27:11 UTC`
- Environment: `arithmetic`
- Experiment ID: `arithmetic_20260416_165701`
- Telemetry backend: `wandb://ayush_g-iit-roorkee/wordle-research`
- Telemetry dashboard: [open project](https://wandb.ai/ayush_g-iit-roorkee/wordle-research)
- Stage summary: 2 partial, 3 success

## Stages

### 1. eval_2digit_baseline

- Run type: `eval`
- Run name: `eval_2digit_baseline`
- Environment run: `default`
- Status: `success`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/arithmetic_20260416_165701-eval-eval_2digit_baseline-12dbb096-316bba767c)
- Runtime (seconds): `192.93`
- Processed samples: `100 / 100`
- Metric `avg_abs_error`: `76.3600`
- Metric `exact_match_accuracy`: `0.6400`
- Metric `format_accuracy`: `0.9900`
- Metric `parsed_answer_rate`: `1.0000`
- Metric `total_samples`: `100`

### 2. sft_2digit_06b

- Run type: `sft`
- Run name: `sft_2digit_06b`
- Environment run: `default`
- Status: `partial`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/arithmetic_20260416_165701-sft-sft_2digit_06b-1db33417-8e7f0535b9)
- Hugging Face artifact: `goyalayus/arithmetic-2digit-sft_2digit_06b` @ `ac6d760fbf5d28586a29fc4c7517670c99bc4dc2`
- Runtime (seconds): `489.49`
- Stopped early: `true`
- Failure reason: `Manual stop requested at step 131.`
- Metric `epoch`: `1.0234`
- Metric `global_step`: `131`
- Metric `total_flos`: `661787482521600.0000`
- Metric `train_loss`: `0.1035`
- Metric `train_runtime`: `348.1074`
- Metric `train_samples`: `n/a`
- Metric `train_samples_per_second`: `22.9810`
- Metric `train_steps_per_second`: `0.7180`

### 3. eval_2digit_after_sft

- Run type: `eval`
- Run name: `eval_2digit_after_sft`
- Environment run: `default`
- Status: `success`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/arithmetic_20260416_165701-eval-eval_2digit_after_sft-9b42b8a6-507b666b73)
- Runtime (seconds): `113.02`
- Processed samples: `100 / 100`
- Metric `avg_abs_error`: `76.3600`
- Metric `exact_match_accuracy`: `0.6400`
- Metric `format_accuracy`: `0.9900`
- Metric `parsed_answer_rate`: `1.0000`
- Metric `total_samples`: `100`

### 4. rl_2digit_06b

- Run type: `rl`
- Run name: `rl_2digit_06b`
- Environment run: `default`
- Status: `partial`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/arithmetic_20260416_165701-rl-rl_2digit_06b-9182e734-f62ab566a1)
- Hugging Face artifact: `goyalayus/arithmetic-2digit-rl_2digit_06b` @ `507736f79a938fcb172dda6e75553ce4ef3558d2`
- Runtime (seconds): `896.41`
- Stopped early: `true`
- Failure reason: `Manual stop requested at step 105.`
- Metric `global_step`: `105`
- Metric `total_flos`: `0.0000`
- Metric `train_loss`: `0.0013`
- Metric `train_runtime`: `626.5259`
- Metric `train_samples_per_second`: `0.9580`
- Metric `train_steps_per_second`: `0.2390`

### 5. eval_2digit_after_rl

- Run type: `eval`
- Run name: `eval_2digit_after_rl`
- Environment run: `default`
- Status: `success`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/arithmetic_20260416_165701-eval-eval_2digit_after_rl-27da3a9f-10f7650716)
- Runtime (seconds): `243.52`
- Processed samples: `100 / 100`
- Metric `avg_abs_error`: `75.9100`
- Metric `exact_match_accuracy`: `0.6300`
- Metric `format_accuracy`: `0.9900`
- Metric `parsed_answer_rate`: `1.0000`
- Metric `total_samples`: `100`
