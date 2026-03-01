# Tenyson

Tenyson is a multi-step research library for SFT, GRPO RL, and Eval workflows.

It provides:
- Job abstractions: `SFTJob`, `RLJob`, `EvalJob`
- A standard `JobResult` return type
- Cloud execution managers: AWS and Modal
- Structured SQL telemetry helpers
- Markdown report templating with placeholder filling and optional WandB integration

## basic usage

Run jobs in the cloud via a manager (AWS or Modal). The manager launches the job remotely using `python -m tenyson.runner`, syncs outputs back, and returns a `JobResult` instance.

```python
from pathlib import Path
import yaml

from tenyson.cloud.aws import AWSManager
from tenyson.jobs.sft import SFTJob
from tenyson.loader import load_task

with open("examples/wordle/configs/sft_config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

task = load_task(str(Path("examples/wordle/wordle_task.py").resolve()))
job = SFTJob(config=cfg, task=task)
cloud = AWSManager(
    instance_type="g5.2xlarge",
    key_name="my-key-name",
    key_path="~/.ssh/my-key.pem",
    security_group="sg-...",
)
result = cloud.run(job)  # result is the remote JobResult
print(result.metrics, result.wandb_url)
```

- **AWS Spot instances**: Pass `use_spot=True` (and optionally `spot_max_price`) to `AWSManager`. On remote failure (e.g. Spot interruption), the manager does not raise; it returns a `JobResult` with `status="failed"`, `failure_reason`, `instance_id`, and `spot_interruption`, and prints a failure message in red to the terminal. The same behaviour applies to **Modal**: on exception the manager returns a failed `JobResult` and prints in red.
- **GPU runner package setup**: cloud managers currently install runtime dependencies with `pip install unsloth vllm`.
- **Resume from checkpoint**: Set `training.resume_from_checkpoint` in your config (or pass `--resume-from-checkpoint` to `tenyson.runner`) to a checkpoint directory (or `repo:revision` for HF). SFT and RL load full checkpoints (model, optimizer, scheduler) and continue training.
- **Pipeline with human-in-the-loop**: Use `tenyson.pipeline.run_pipeline(steps, cloud, on_failure="wait", ...)`. A step can be either `(label, config, JobClass, task)` for sequential execution, or `{"label": "stage_name", "parallel": [step1, step2, ...]}` to run branches concurrently. When a step/branch fails, the pipeline prints the failure in red, optionally logs to a file/webhook/telemetry, then waits for you to choose: **resume** (from latest checkpoint), **restart** (same step from scratch), or **abort**. Works with both AWS and Modal.

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
4. **Final report generation** into `examples/wordle/final_report.md`

All runs use dedicated output directories under `./outputs/wordle_research/...` so artifacts never collide.

### required environment variables for `examples/wordle/experiment.py`

- `TENYSON_AWS_KEY_NAME`
- `TENYSON_AWS_KEY_PATH`
- `TENYSON_AWS_SECURITY_GROUP`

Optional:

- `TENYSON_AWS_REGION` (default: `us-east-1`)
- `TENYSON_AWS_INSTANCE_TYPE` (default: `g5.2xlarge`)
- `TENYSON_AWS_SUBNET`
- `TENYSON_AWS_SPOT_MAX_PRICE`
- `AWS_PROFILE`
- `TENYSON_HF_REPO_BASE` (override HF repo base for SFT/RL pushes)
- `HF_TOKEN`
- `WANDB_API_KEY`

Run from project root (`tenyson/`):

```bash
python examples/wordle/experiment.py
```

## telemetry

If you set `telemetry.db_url` in your config, jobs will log structured metrics to a SQL database (SQLite or Postgres via SQLAlchemy) and can also receive manual stop requests via the same control store.

When telemetry is enabled, each run also needs an **experiment id** so multiple runs can be grouped in one DB:

- Prefer config: `telemetry.experiment_id`
- Fallback env var: `TENYSON_EXPERIMENT_ID`
- If neither is set while `telemetry.db_url` is set, the job fails fast with a clear error.

- **SFT**:
  - `SFTTelemetryCallback` writes into the `sft_metrics` table.
  - `ManualStopTelemetryCallback` polls a simple `run_controls` table so you can request a graceful stop.
- **RL**:
  - `GRPOEpochTelemetryCallback` writes per-epoch loss and KL into `epoch_metrics`.
  - `ManualStopTelemetryCallback` on the GRPO trainer polls the same `run_controls` table and stops training cleanly after the current step.
  - A wrapped reward function writes per-prompt rollouts and rewards into `rollouts` and `generations` (with `phase="rl"`).
- **Eval**:
  - `EvalJob` streams batched generations and can log prompts/completions into `generations` (with `phase="eval"`).
  - When a manual stop is requested, eval stops between batches, computes metrics on the processed subset only, and marks `results.metadata.stopped_early = true`.

Example snippet in a YAML config:

```yaml
telemetry:
  db_url: "sqlite:///./tenyson_telemetry.db"
  experiment_id: "wordle_research_2026_03_01"
```

With telemetry enabled, you can mark a run for manual stop from another process:

```bash
python -m tenyson.core.control \
  --run-id lora_sft_qwen3-4b \
  --db-url sqlite:///./tenyson_telemetry.db
```

The next step or batch will see the `stop_requested` flag in the database and exit the loop cleanly; the job will still save or push results as usual (SFT/RL checkpoints, partial eval metrics and generations).

## eval and generations

`EvalJob` mirrors the old eval script: it loads a model plus adapter, runs batched vLLM generation, computes metrics via the `TaskPlugin`, and writes:

- A detailed `results.json` with metrics, and
- A `job_result.json` containing the `JobResult` fields for the run.

The same `Generation` table is intended for logging eval prompts and completions when you need per-sample telemetry.

### task-specific eval filtering (`examples/wordle/wordle_task.py`)

The Wordle task supports a task-level eval control key:

```yaml
task:
  eval_samples: 200
  eval_seed: 42
  eval_exact_turns: [3]
```

When `eval_exact_turns` is set, eval dataset generation is restricted to those exact history lengths (Wordle turns), instead of mixed turn sampling.

## jobresult and reporting

`tenyson.jobs.result.JobResult` is the common return type from all jobs and cloud managers:

- **Fields**: `run_id`, `status`, `total_time_seconds`, `metrics`, `hf_repo_id`, `hf_revision`, `wandb_url`, `local_output_dir`. On failure, cloud managers also set `failure_reason`, `instance_id`, and `spot_interruption`.
- **Persistence**: the `.save(path)` method serialises the result to JSON; jobs always save a `results.json` or `job_result.json` in their output directory.

`ReportBuilder` turns a markdown template into a final report with:

- Simple `{placeholder}` replacement via `fill(data)`, and
- Optional WandB helpers:
  - `attach_wandb_scalar_link(placeholder, run_url, metric_name)`
  - `attach_wandb_latest_value(placeholder, run_path, metric_name)`

The Wordle example (`examples/wordle/experiment.py`) shows how to:

- Orchestrate a mixed-vs-curriculum branching experiment with both sequential and parallel stages, and
- Build a `final_report.md` with branch/stage statuses and metric comparisons using `ReportBuilder`.

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
- if `eval_loss` is constant or increasing for `early_stopping_patience` evaluations, training stops and the best checkpoint is kept.

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
- **`--task-module`**: Either a path to a Python file containing a single `TaskPlugin` subclass (e.g. `examples/wordle/wordle_task.py`), or a `module.path:ClassName` spec for remote/advanced use (e.g. `examples.wordle.wordle_task:WordleTask`).
- **`--resume-from-checkpoint`**: (SFT/RL only) Path to a checkpoint directory or `repo_id:revision` to resume training.

`AWSManager` and `ModalManager` invoke this entrypoint on the remote worker. You can also run it directly on a machine that will execute the job (e.g. a dedicated runner node).
