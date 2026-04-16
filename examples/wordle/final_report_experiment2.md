# Tenyson Experiment Report

- Generated: `2026-03-31 11:24:52 UTC`
- Environment: `wordle`
- Experiment ID: `wordle_experiment2_20260331_104723`
- Telemetry backend: `wandb://ayush_g-iit-roorkee/wordle-research`
- Telemetry dashboard: [open project](https://wandb.ai/ayush_g-iit-roorkee/wordle-research)
- Stage summary: 1 stopped, 1 success

## Stages

### 1. baseline

- Run type: `eval`
- Run name: `baseline`
- Environment run: `default`
- Status: `success`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/wordle_experiment2_20260331_104723-eval-baseline-30a57478-5cbfbc81e2)
- Runtime (seconds): `307.91`
- Processed samples: `100 / 100`
- Metric `avg_correct_answer_reward`: `0.0000`
- Metric `avg_format_reward`: `0.1960`
- Metric `avg_partial_answer_reward`: `0.2200`
- Metric `avg_total_reward`: `0.4160`
- Metric `correct_answer_accuracy`: `0.0000`
- Metric `dict_accuracy`: `0.9000`
- Metric `format_accuracy`: `0.9800`
- Metric `total_samples`: `100`

### 2. train

- Run type: `rl`
- Run name: `train`
- Environment run: `default`
- Status: `stopped`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/wordle_experiment2_20260331_104723-rl-train-9493bc29-dc7344382c)
- Hugging Face adapter: `goyalayus/wordle-lora-20260324-163252-train` @ `85e63c6a5daebb1f876e38b173b1c257c07db551`
- Runtime (seconds): `1820.83`
- Stopped early: `true`
- Failure reason: `Manual stop requested at step 202.`
- Metric `global_step`: `202`
- Metric `rollout_batches`: `202`
- Metric `total_flos`: `0.0000`
- Metric `total_samples`: `808`
- Metric `train_loss`: `0.0006`
- Metric `train_runtime`: `1638.2207`
- Metric `train_samples_per_second`: `2.4420`
- Metric `train_steps_per_second`: `0.6100`
