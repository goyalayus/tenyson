# Tenyson

Tenyson is a multi-step research library for SFT, GRPO RL, and Eval workflows.

It provides:
- Job abstractions: `SFTJob`, `RLJob`, `EvalJob`
- A standard `JobResult` return type
- Cloud execution managers: AWS and Modal
- Structured SQL telemetry helpers
- Markdown report templating with placeholder filling and optional WandB integration

## basic usage

- **Local jobs**: Instantiate a `TaskPlugin`, load a YAML config, then run a job.

```python
from tenyson.jobs.sft import SFTJob
from tenyson.examples.wordle.wordle_task import WordleTask
import yaml

with open("tenyson/examples/wordle/configs/sft_config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

task = WordleTask()
job = SFTJob(config=cfg, task=task)
result = job.run()
print(result.metrics, result.wandb_url)
```

- **Cloud runs via AWS / Modal**: Wrap the job with a cloud manager. The manager now launches the job remotely using `python -m tenyson.runner`, pulls back the `JobResult` JSON, and returns a populated `JobResult` instance.

```python
from tenyson.cloud.aws import AWSManager

cloud = AWSManager(
    instance_type="g5.2xlarge",
    key_name="my-key-name",
    key_path="~/.ssh/my-key.pem",
    security_group="sg-...",
)
result = cloud.run(job)  # result is the remote JobResult
```

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

- **Fields**: `run_id`, `status`, `total_time_seconds`, `metrics`, `hf_repo_id`, `hf_revision`, `wandb_url`, `local_output_dir`.
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
- **`--config`**: YAML or JSON config file; the same schema used by local jobs.
- **`--task-module`**: Either a path to a Python file containing a single `TaskPlugin` subclass (e.g. `tenyson/examples/wordle/wordle_task.py`), or a `module.path:ClassName` spec for remote/advanced use.

`AWSManager` and `ModalManager` both use this CLI internally for remote execution; you can also use it directly on any machine where `tenyson` is installed.
