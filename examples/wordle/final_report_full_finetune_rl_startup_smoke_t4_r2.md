# Tenyson Experiment Report

- Generated: `2026-03-31 18:49:31 UTC`
- Environment: `wordle`
- Experiment ID: `full_ft_rl_startup_t4_r2_20260401`
- Telemetry backend: `wandb://ayush_g-iit-roorkee/wordle-research`
- Telemetry dashboard: [open project](https://wandb.ai/ayush_g-iit-roorkee/wordle-research)
- Stage summary: 1 failed

## Stages

### 1. rl_full_from_sft

- Run type: `rl`
- Run name: `rl_full_from_sft`
- Environment run: `default`
- Status: `failed`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/full_ft_rl_startup_t4_r2_20260401-rl-rl_full_from_sft-279b2533-11e102ebd1)
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
