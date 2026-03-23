# Config Templates

These are the base templates for new Tenyson runs.

- `sft.yaml`: starter defaults for supervised fine-tuning
- `rl.yaml`: starter defaults for GRPO RL runs
- `eval.yaml`: starter defaults for eval runs

The idea is:

- framework-wide defaults live here
- task-specific overrides live in the environment file
- experiment-specific tweaks live in `experiment.py`

Notes:

- telemetry is always W&B, so the starter templates do not expose a backend field
- the current runtime only accepts Qwen 3 family base models
- SFT tasks should return one of Tenyson's built-in simple row schemas; Tenyson handles chat formatting, assistant-only masking, and packing internally
- packed SFT also requires a flash-attention backend that safely preserves packed-sequence boundaries; Tenyson fails fast if the runtime resolves to an unsafe attention implementation

The Wordle example uses these templates and then applies its own environment run overrides on top.
