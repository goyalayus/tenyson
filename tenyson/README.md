# Tenyson

Tenyson is a multi-step research library for SFT, GRPO RL, and Eval workflows.

It provides:
- Job abstractions: `SFTJob`, `RLJob`, `EvalJob`
- A standard `JobResult` return type
- Cloud execution managers: AWS and Modal
- Telemetry backends for W&B (recommended) and SQL
- A fixed markdown experiment report with W&B run/project links

## basic usage

Run jobs in the cloud via a manager (AWS or Modal). The manager launches the job remotely using `python -m tenyson.runner`, reads canonical results from telemetry, and returns a `JobResult` instance.

Local execution is intentionally disabled: jobs must run through supported GPU providers (AWS or Modal).

```python
from pathlib import Path
import yaml

from tenyson.cloud.modal import ModalManager
from tenyson.jobs.sft import SFTJob
from tenyson.loader import load_task

with open("examples/wordle/configs/sft_config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

task = load_task(str(Path("examples/wordle/wordle_task.py").resolve()))
job = SFTJob(config=cfg, task=task)
cloud = ModalManager(
    gpu="A100",
    timeout=86400,
)
result = cloud.run(job)  # result is the remote JobResult
print(result.metrics, result.wandb_url)
```

- **AWS Spot instances**: Pass `use_spot=True` (and optionally `spot_max_price`) to `AWSManager`. On remote failure (e.g. Spot interruption), the manager does not raise; it returns a `JobResult` with `status="failed"`, `failure_reason`, `instance_id`, and `spot_interruption`, and prints a failure message in red to the terminal. The same behaviour applies to **Modal**: on exception the manager returns a failed `JobResult` and prints in red.
- **GPU runner package setup**: cloud managers install runtime dependencies with `python3 -m pip install unsloth vllm huggingface_hub pyyaml sqlalchemy 'psycopg[binary]' wandb` (we intentionally do not install `trl`/`transformers`/`datasets` directly because Unsloth pulls them transitively).
- **HF push cadence (SFT/RL)**: set `training.hf_repo_base` (required) to push full trainer checkpoints to a stable repo id `<hf_repo_base>-<run_name>`. `training.hf_push_every_steps` controls checkpoint save+push cadence.
- **Run naming contract**: `training.run_name` (SFT/RL) and `evaluation.run_name` (Eval) are mandatory and must be explicit (defaults like `sft_job`/`rl_job`/`eval_job` are rejected). Within one `run_pipeline(...)` execution, run names must be unique.
- **Checkpoint mode**: SFT/RL use Hub-managed trainer checkpoints (includes optimizer/scheduler/trainer state). Resume uses `training.resume_from_checkpoint: "repo_id:revision"`, resolves that ref to an immutable Hub commit SHA, and automatically restores from `last-checkpoint` (or latest `checkpoint-*`) in that frozen snapshot.
- **Pipeline with human-in-the-loop**: Use `tenyson.pipeline.run_pipeline(steps, cloud, on_failure="wait", ...)`. A step can be either `(label, config, JobClass, task)` for sequential execution, or `{"label": "stage_name", "parallel": [step1, step2, ...]}` to run branches concurrently. When a step/branch fails, the pipeline prints the failure in red, optionally logs to a file/webhook/telemetry, then waits for you to choose: **resume** (from the recorded immutable Hub revision), **restart** (same step from scratch), or **abort**. Works with both AWS and Modal.

## wordle research workflow (current example)

`examples/wordle/experiment.py` now runs a full research graph:

1. **SFT**
2. **Baseline mixed eval** (turns 1..5)
3. **Parallel branches**
   - **Branch A (mixed RL)**: RL on turns 1..5, then final mixed eval (turns 1..5)
   - **Branch B (curriculum RL)**: RL turn 2 -> 3 -> 4 -> 5, with stage evals:
     - after RL2: eval turn2
     - after RL3: eval turn2 + turn3 (parallel)
     - after RL4: eval turn3 + turn4 (parallel)
     - after RL5: eval turn4 + turn5 (parallel)
     - then final mixed eval (turns 1..5)
4. **Fixed report generation** into `examples/wordle/final_report.md`

The task file now exposes a single `ENVIRONMENT` definition with explicit named runs:

- `sft`: dataset + formatting hooks + SFT-specific config overrides
- `rl`: dataset + reward hooks + RL-specific config overrides
- `eval`: dataset + metric hooks + eval-specific config overrides

`examples/wordle/experiment.py` now only orchestrates stages. It selects task behavior by named run ids such as `wordle_rl_turn2`, `wordle_rl_turn3`, `wordle_eval_turn2`, and `wordle_eval_mixed`.

All runs keep canonical metrics/results in telemetry and adapters on HF for SFT/RL.

### required environment variables for `examples/wordle/experiment.py`

No cloud provider credential env vars are required by the script itself when using
Modal defaults (`A100` GPU). Ensure `modal` is authenticated in your shell.
Optional:

- `TENYSON_MODAL_GPU` (default: `A100`)
- `TENYSON_MODAL_TIMEOUT` (default: `86400`)
- `TENYSON_MODAL_PROFILE` (or `MODAL_PROFILE`)
- `TENYSON_HF_REPO_BASE` (override HF repo base for SFT/RL pushes)
- `TENYSON_WANDB_ENTITY`
- `TENYSON_WANDB_PROJECT`
- `HF_TOKEN` (required for SFT/RL jobs)
- `WANDB_API_KEY` (required for W&B telemetry)

Run from project root (`tenyson/`):

```bash
cp examples/wordle/run.env.example examples/wordle/run.env
# fill in your values in examples/wordle/run.env
source examples/wordle/run.env
python3 examples/wordle/experiment.py
```

The experiment entrypoint auto-adds `src/` to `PYTHONPATH` and auto-installs
missing local controller dependencies (`modal`, `boto3`, `psycopg[binary]`,
`sqlalchemy`, `datasets`, `pyyaml`, `huggingface_hub`, `wandb`) on first run.
Set `TENYSON_SKIP_LOCAL_BOOTSTRAP=1` to disable this behavior.
The reusable helper behind this is `tenyson.bootstrap.ensure_local_controller_environment(...)`.

## telemetry

Telemetry is mandatory. Every run must set `telemetry.experiment_id`.

Recommended backend: W&B.

```yaml
telemetry:
  backend: "wandb"
  experiment_id: "wordle_research_2026_03_01"
```

Provide the W&B destination through config or env vars:

- `telemetry.entity` / `telemetry.project`, or
- `TENYSON_WANDB_ENTITY` / `TENYSON_WANDB_PROJECT`

SQL remains available as a fallback with `telemetry.db_url`, but W&B is now the intended canonical telemetry path.

Each run also needs an **experiment id** so multiple runs can be grouped together:

- Prefer config: `telemetry.experiment_id`
- Fallback env var: `TENYSON_EXPERIMENT_ID`
- Missing backend destination or experiment_id fails fast.

- **SFT**:
  - `SFTTelemetryCallback` writes into the `sft_metrics` table.
  - `ManualStopTelemetryCallback` polls a simple `run_controls` table so you can request a graceful stop.
- **RL**:
  - `GRPOEpochTelemetryCallback` writes per-epoch loss and KL into `epoch_metrics`.
  - `ManualStopTelemetryCallback` on the GRPO trainer polls the same `run_controls` table and stops training cleanly after the current step.
  - A wrapped reward function writes per-prompt rollouts and rewards into `rollouts` and `generations` (with `phase="rl"`).
- **Eval**:
  - `EvalJob` streams batched generations and can log prompts/completions into `generations` when SQL telemetry is enabled.
  - When a manual stop is requested, eval stops between batches, computes metrics on the processed subset only, and marks `results.metadata.stopped_early = true`.
- **Canonical run payloads**:
  - W&B summaries/artifacts are the canonical per-run result rendezvous when `telemetry.backend: wandb`.
  - `run_summaries` / `run_results` remain available as the SQL canonical path when `telemetry.db_url` is used.

With telemetry enabled, you can mark a run for manual stop from another process.
Use the same `experiment_id` as the running job:

```bash
python -m tenyson.core.control \
  --run-id lora_sft_qwen3-4b \
  --experiment-id wordle_research_2026_03_01 \
  --db-url wandb://your-entity/wordle-research
```

The next step or batch will see the `stop_requested` flag in telemetry and exit the loop cleanly; summaries/results remain queryable afterward.

## eval and generations

`EvalJob` mirrors the old eval script: it loads a model plus adapter, runs batched vLLM generation, computes metrics via the environment hooks, and writes canonical payloads to telemetry.

The same `Generation` table is intended for logging eval prompts and completions when you need per-sample telemetry.

### task-specific eval filtering (`examples/wordle/wordle_task.py`)

The Wordle environment exposes explicit eval run definitions through the environment contract. Under the hood those runs map to task-level keys like:

```yaml
task:
  eval_samples: 200
  eval_seed: 42
  eval_exact_turns: [3]
```

When `eval_exact_turns` is set, eval dataset generation is restricted to those exact history lengths (Wordle turns), instead of mixed turn sampling.

## jobresult and reporting

`tenyson.jobs.result.JobResult` is the common return type from all jobs and cloud managers:

- **Fields**: `run_id`, `status`, `total_time_seconds`, `metrics`, `hf_repo_id`, `hf_revision`, `wandb_url`. When present, `hf_revision` is the exact immutable Hub commit SHA used for lineage/resume. On failure, cloud managers also set `failure_reason`, `instance_id`, and `spot_interruption`.
- **Persistence**: W&B summaries/artifacts or SQL `run_summaries` / `run_results`, depending on the configured backend.

The Wordle example (`examples/wordle/experiment.py`) shows how to:

- Orchestrate a mixed-vs-curriculum branching experiment with both sequential and parallel stages, and
- Build a fixed `final_report.md` with stage status, metrics, HF adapter lineage, and W&B run/project links using `ExperimentReport`.

## sft early stopping

`SFTJob` supports optional early stopping based on `eval_loss` when an eval dataset is configured.

Add the following keys under `training` in your SFT config:

```yaml
training:
  # existing fields ...
  eval_steps: 20
  max_steps: 3000

  # Early stopping on eval_loss
  early_stopping_patience: 2       # N evals with no improvement before stopping
  early_stopping_min_delta: 0.0    # X; minimum absolute improvement in eval_loss
```

Semantics:

- `metric_for_best_model = "eval_loss"` and `greater_is_better = false`;
- improvement means the best `eval_loss` decreases by more than `early_stopping_min_delta`;
- if `eval_loss` is constant or increasing for `early_stopping_patience` evaluations, training stops. Since checkpoints are Hub-managed, latest pushed full trainer checkpoint on HF is the recovery path.

## runner cli

Remote environments call into a single entrypoint:

```bash
python -m tenyson.runner \
  --job-type sft \
  --config path/to/config.yaml \
  --task-module examples/wordle/wordle_task.py
```

Or with a module:Class spec (e.g. for remote use by cloud managers):

```bash
python -m tenyson.runner \
  --job-type sft \
  --config path/to/config.yaml \
  --task-module examples.wordle.wordle_task:WordleTask
```

- **`--job-type`**: one of `sft`, `rl`, or `eval`.
- **`--config`**: YAML or JSON config file; the same schema used by jobs run via cloud managers.
- **`--task-module`**: Either a path to a Python file containing a single `ENVIRONMENT` definition (or compatible wrapper class), or a `module.path:ClassName` spec for remote/advanced use (e.g. `examples.wordle.wordle_task:WordleTask`).
- **`--resume-from-checkpoint`**: (SFT/RL only) `repo_id:revision` to resume training from Hugging Face.

`AWSManager` and `ModalManager` invoke this entrypoint on the remote worker. Running this entrypoint locally is blocked unless the cloud runtime context variables are set by a supported provider manager.
