# Tenyson Experiment Report

- Generated: `2026-03-31 18:21:21 UTC`
- Environment: `wordle`
- Experiment ID: `full_ft_compare_a100_20260331`
- Telemetry backend: `wandb://ayush_g-iit-roorkee/wordle-research`
- Telemetry dashboard: [open project](https://wandb.ai/ayush_g-iit-roorkee/wordle-research)
- Stage summary: 1 failed, 2 success

## Stages

### 1. sft_full_smoke

- Run type: `sft`
- Run name: `sft_full_smoke`
- Environment run: `default`
- Status: `success`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/full_ft_compare_a100_20260331-sft-sft_full_smoke-3f4833af-fc798685ce)
- Hugging Face artifact: `goyalayus/wordle-lora-20260324-163252-sft_full_smoke` @ `f1095a024cb4a9957e6009239638393d8ac1ae85`
- Runtime (seconds): `204.89`
- Metric `epoch`: `0.0005`
- Metric `global_step`: `2`
- Metric `total_flos`: `16481610141696.0000`
- Metric `train_loss`: `2.1573`
- Metric `train_runtime`: `106.5717`
- Metric `train_samples`: `n/a`
- Metric `train_samples_per_second`: `0.0190`
- Metric `train_steps_per_second`: `0.0190`

### 2. eval_full_after_sft

- Run type: `eval`
- Run name: `eval_full_after_sft`
- Environment run: `default`
- Status: `success`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/full_ft_compare_a100_20260331-eval-eval_full_after_sft-6b056458-cb9cb673be)
- Runtime (seconds): `212.25`
- Processed samples: `16 / 16`
- Metric `avg_constraint_reward`: `0.0000`
- Metric `constraint_accuracy`: `0.0000`
- Metric `dict_accuracy`: `0.0000`
- Metric `format_accuracy`: `0.0000`
- Metric `total_samples`: `16`

### 3. rl_full_from_sft

- Run type: `rl`
- Run name: `rl_full_from_sft`
- Environment run: `default`
- Status: `failed`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/full_ft_compare_a100_20260331-rl-rl_full_from_sft-f57830bc-bdc99f0014)
- Runtime (seconds): `0.00`
- Failure reason: `Modal job completed but canonical run result was not available in telemetry DB: Detached Modal function call failed before telemetry wrote the canonical run result: Tenyson job failed inside Modal with code 1. ", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/src/tenyson/runner.py", line 73, in <module>
    main()
  File "/workspace/src/tenyson/runner.py", line 68, in main
    result = job.run()
             ^^^^^^^^^
  File "/workspace/src/tenyson/jobs/rl.py", line 978, in run
    model, tokenizer = self._build_model_and_tokenizer()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/src/tenyson/jobs/rl.py", line 780, in _build_model_and_tokenizer
    raise RuntimeError(
RuntimeError: [RLJob] Unsloth vLLM startup failed during RL model load. RL requires vLLM and will now abort instead of retrying without fast inference. Unsloth: `fast_inference=True` cannot be used together with `full_finetuning=True`.
Reason: fast_inference is optimized for inference-only workflows and does not currently support full fine-tuning.
Workaround: disable fast_inference, or use parameter-efficient fine-tuning (e.g. LoRA with rank r=512).`
- Metrics: n/a
