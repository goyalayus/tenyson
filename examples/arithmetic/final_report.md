# Tenyson Experiment Report

- Generated: `2026-04-16 19:03:06 UTC`
- Environment: `arithmetic`
- Experiment ID: `arithmetic_20260416_185426`
- Telemetry backend: `wandb://ayush_g-iit-roorkee/wordle-research`
- Telemetry dashboard: [open project](https://wandb.ai/ayush_g-iit-roorkee/wordle-research)
- Stage summary: 2 partial, 3 success

## Stages

### 1. eval_2digit_baseline

- Run type: `eval`
- Run name: `eval_2digit_baseline`
- Environment run: `default`
- Status: `success`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/arithmetic_20260416_185426-eval-eval_2digit_baseline-8bccf3bf-8ae65e6ccf)
- Runtime (seconds): `142.52`
- Processed samples: `100 / 100`
- Metric `avg_abs_error`: `75.9100`
- Metric `exact_match_accuracy`: `0.6300`
- Metric `format_accuracy`: `0.9900`
- Metric `parsed_answer_rate`: `1.0000`
- Metric `total_samples`: `100`

### 2. sft_2digit_06b

- Run type: `sft`
- Run name: `sft_2digit_06b`
- Environment run: `default`
- Status: `partial`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/arithmetic_20260416_185426-sft-sft_2digit_06b-e0e273c0-65ffc80d41)
- Hugging Face artifact: `goyalayus/arithmetic-2digit-sft_2digit_06b` @ `8960cc9cdf34ee155a4dccc97424a0a1e7b36f08`
- Runtime (seconds): `182.26`
- Stopped early: `true`
- Failure reason: `Manual stop requested at step 113.`
- Metric `epoch`: `0.8828`
- Metric `global_step`: `113`
- Metric `total_flos`: `570851909959680.0000`
- Metric `train_loss`: `0.1199`
- Metric `train_runtime`: `132.4936`
- Metric `train_samples`: `n/a`
- Metric `train_samples_per_second`: `60.3800`
- Metric `train_steps_per_second`: `1.8870`

### 3. eval_2digit_after_sft

- Run type: `eval`
- Run name: `eval_2digit_after_sft`
- Environment run: `default`
- Status: `success`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/arithmetic_20260416_185426-eval-eval_2digit_after_sft-55c7d54d-a3599259db)
- Runtime (seconds): `56.83`
- Processed samples: `100 / 100`
- Metric `avg_abs_error`: `0.0000`
- Metric `exact_match_accuracy`: `1.0000`
- Metric `format_accuracy`: `1.0000`
- Metric `parsed_answer_rate`: `1.0000`
- Metric `total_samples`: `100`

### 4. rl_2digit_06b

- Run type: `rl`
- Run name: `rl_2digit_06b`
- Environment run: `default`
- Status: `partial`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/arithmetic_20260416_185426-rl-rl_2digit_06b-bf157d16-64f78e0367)
- Hugging Face artifact: `goyalayus/arithmetic-2digit-rl_2digit_06b` @ `e8d107e342b9c9db7e53418f200947ab8c91e386`
- Runtime (seconds): `323.54`
- Stopped early: `true`
- Failure reason: `Manual stop requested at step 38.`
- Metric `global_step`: `38`
- Metric `total_flos`: `0.0000`
- Metric `train_loss`: `0.0007`
- Metric `train_runtime`: `171.9616`
- Metric `train_samples_per_second`: `3.4890`
- Metric `train_steps_per_second`: `0.8720`

### 5. eval_2digit_after_rl

- Run type: `eval`
- Run name: `eval_2digit_after_rl`
- Environment run: `default`
- Status: `success`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/arithmetic_20260416_185426-eval-eval_2digit_after_rl-890c46d1-eb891f97e2)
- Runtime (seconds): `46.93`
- Processed samples: `100 / 100`
- Metric `avg_abs_error`: `8.6400`
- Metric `exact_match_accuracy`: `0.8100`
- Metric `format_accuracy`: `1.0000`
- Metric `parsed_answer_rate`: `1.0000`
- Metric `total_samples`: `100`
