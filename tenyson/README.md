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
from tenyson.cloud.aws import AWSManager
from tenyson.jobs.sft import SFTJob
from tenyson.examples.wordle.wordle_task import WordleTask
import yaml

with open("tenyson/examples/wordle/configs/sft_config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

task = WordleTask()
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
- **Resume from checkpoint**: Set `training.resume_from_checkpoint` in your config (or pass `--resume-from-checkpoint` to `tenyson.runner`) to a checkpoint directory (or `repo:revision` for HF). SFT and RL load full checkpoints (model, optimizer, scheduler) and continue training.
- **Pipeline with human-in-the-loop**: Use `tenyson.pipeline.run_pipeline(steps, cloud, on_failure="wait", ...)`. Each step is `(label, config, JobClass, task)`. When a step fails, the pipeline prints the failure in red, optionally logs to a file/webhook/telemetry, then waits for you to choose: **resume** (from latest checkpoint), **restart** (same step from scratch), or **abort**. Works with both AWS and Modal.

## telemetry

If you set `telemetry.db_url` in your config, jobs will log structured metrics to a SQL database (SQLite or Postgres via SQLAlchemy) and can also receive manual stop requests via the same control store.

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

## jobresult and reporting

`tenyson.jobs.result.JobResult` is the common return type from all jobs and cloud managers:

- **Fields**: `run_id`, `status`, `total_time_seconds`, `metrics`, `hf_repo_id`, `hf_revision`, `wandb_url`, `local_output_dir`. On failure, cloud managers also set `failure_reason`, `instance_id`, and `spot_interruption`.
- **Persistence**: the `.save(path)` method serialises the result to JSON; jobs always save a `results.json` or `job_result.json` in their output directory.

`ReportBuilder` turns a markdown template into a final report with:

- Simple `{placeholder}` replacement via `fill(data)`, and
- Optional WandB helpers:
  - `attach_wandb_scalar_link(placeholder, run_url, metric_name)`
  - `attach_wandb_latest_value(placeholder, run_path, metric_name)`

The Wordle example (`tenyson/examples/wordle/experiment.py`) shows how to:

- Chain `SFTJob` → `RLJob` → `EvalJob` through `AWSManager`, and
- Build a `final_report.md` that includes pipeline status, eval metrics, and optional WandB run links using `ReportBuilder`.

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
  --task-module tenyson/examples/wordle/wordle_task.py
```

Or with a module:Class spec (e.g. for remote use by cloud managers):

```bash
python -m tenyson.runner \
  --job-type sft \
  --config path/to/config.yaml \
  --task-module tenyson.examples.wordle.wordle_task:WordleTask
```

- **`--job-type`**: one of `sft`, `rl`, or `eval`.
- **`--config`**: YAML or JSON config file; the same schema used by jobs run via cloud managers.
- **`--task-module`**: Either a path to a Python file containing a single `TaskPlugin` subclass (e.g. `tenyson/examples/wordle/wordle_task.py`), or a `module.path:ClassName` spec for remote/advanced use.
- **`--resume-from-checkpoint`**: (SFT/RL only) Path to a checkpoint directory or `repo_id:revision` to resume training.

`AWSManager` and `ModalManager` invoke this entrypoint on the remote worker. You can also run it directly on a machine that will execute the job (e.g. a dedicated runner node).
