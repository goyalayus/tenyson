# Tenyson Experiment Report

- Generated: `2026-03-31 18:24:55 UTC`
- Environment: `wordle`
- Experiment ID: `full_ft_compare_t4_20260331`
- Telemetry backend: `wandb://ayush_g-iit-roorkee/wordle-research`
- Telemetry dashboard: [open project](https://wandb.ai/ayush_g-iit-roorkee/wordle-research)
- Stage summary: 1 failed

## Stages

### 1. sft_full_smoke

- Run type: `sft`
- Run name: `sft_full_smoke`
- Environment run: `default`
- Status: `failed`
- W&B run: [open run](https://wandb.ai/ayush_g-iit-roorkee/wordle-research/runs/full_ft_compare_t4_20260331-sft-sft_full_smoke-2ad2b667-7dbdd50da6)
- Runtime (seconds): `0.00`
- Failure reason: `Modal job completed but canonical run result was not available in telemetry DB: Detached Modal function call failed before telemetry wrote the canonical run result: Tenyson job failed inside Modal with code 1. e 1, in <module>
  File "/usr/local/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1355, in to
    return self._apply(convert)
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/torch/nn/modules/module.py", line 942, in _apply
    param_applied = fn(param)
                    ^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1341, in convert
    return t.to(
           ^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 96.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 9.81 MiB is free. Process 1 has 14.55 GiB memory in use. Of the allocated memory 13.93 GiB is allocated by PyTorch, and 500.31 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)`
- Metrics: n/a
