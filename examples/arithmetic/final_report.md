# Tenyson Experiment Report

- Generated: `2026-04-12 22:52:34 UTC`
- Environment: `arithmetic`
- Experiment ID: `arithmetic_20260412_223128`
- Telemetry backend: `wandb://ayush_g-iit-roorkee/wordle-research`
- Telemetry dashboard: [open project](https://wandb.ai/ayush_g-iit-roorkee/wordle-research)
- Stage summary: 1 running, 2 success

## Stages

### 1. rl_2digit_06b

- Run type: `rl`
- Run name: `rl_2digit_06b`
- Environment run: `default`
- Status: `success`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/arithmetic_20260412_223128-rl-rl_2digit_06b-6c3ae43a-ed32e0a792)
- Hugging Face artifact: `goyalayus/arithmetic-2digit-rl_2digit_06b` @ `14b1adead970884626b2494e979992161a834ef9`
- Runtime (seconds): `952.88`
- Metric `global_step`: `150`
- Metric `total_flos`: `0.0000`
- Metric `train_loss`: `0.0011`
- Metric `train_runtime`: `751.5525`
- Metric `train_samples_per_second`: `0.7980`
- Metric `train_steps_per_second`: `0.2000`

### 2. sft_2digit_06b

- Run type: `sft`
- Run name: `sft_2digit_06b`
- Environment run: `default`
- Status: `running`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/arithmetic_20260412_223128-sft-sft_2digit_06b-89a48c11-471fc95381)
- Metrics: pending terminal result

### 3. eval_2digit_after_rl

- Run type: `eval`
- Run name: `eval_2digit_after_rl`
- Environment run: `default`
- Status: `success`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/arithmetic_20260412_223128-eval-eval_2digit_after_rl-ec9d2db1-bdd338c706)
- Runtime (seconds): `210.36`
- Processed samples: `100 / 100`
- Metric `avg_abs_error`: `75.9100`
- Metric `exact_match_accuracy`: `0.6300`
- Metric `format_accuracy`: `0.9900`
- Metric `parsed_answer_rate`: `1.0000`
- Metric `total_samples`: `100`
