# Tenyson Architecture

## High-level overview
Tenyson is a Python research orchestration library for running language-model adaptation workflows as explicit job stages:
- `SFTJob`: supervised fine-tuning with LoRA.
- `RLJob`: GRPO-style reinforcement learning over task-defined reward functions.
- `EvalJob`: batched generation + metric computation for a task.

The system is designed to run the same job abstractions locally or through cloud managers (AWS EC2, Modal), with standardized result objects (`JobResult`), optional SQL telemetry, and failure-aware pipeline control.

Primary user flow:
1. Author/choose a `TaskPlugin` implementation.
2. Provide config (YAML/JSON) per stage.
3. Instantiate a cloud manager (`AWSManager` or `ModalManager`).
4. Execute jobs directly or via `run_pipeline`.
5. Consume `JobResult` artifacts and generated outputs.

## Repo map
Repository root: `/home/ayush/Desktop/code/tenyson`

- `plan/`
  - Task-by-task execution plans and status logs.
- `ARCHITECTURE.md`
  - This file; authoritative architecture reference.
- `tenyson/`
  - Python project root (contains `pyproject.toml`).

Inside `tenyson/`:
- `pyproject.toml`
  - Packaging metadata and dependencies.
  - Uses src-layout: package source is under `src/`.
- `README.md`
  - Usage and workflow documentation.
- `src/tenyson/`
  - Main library package.
- `examples/wordle/`
  - Example task plugin, configs, report template, and experiment script.

Inside `tenyson/src/tenyson/`:
- `jobs/`
  - `sft.py`, `rl.py`, `eval.py`: concrete job implementations.
  - `result.py`: canonical run result dataclass.
  - `hf_repo.py`: unique HF repo naming helper.
- `cloud/`
  - `aws.py`: EC2 lifecycle + remote execution + rsync sync.
  - `modal.py`: Modal function execution path.
  - `base.py`, `manager.py`: manager abstractions/factory.
- `core/`
  - `plugin.py`: `TaskPlugin` interface contract.
  - `telemetry.py`: SQLAlchemy models + callbacks for metrics, manual-stop, run metadata/failures.
  - `notify.py`: failure logging/webhook/telemetry integration.
  - `control.py`, `ctl.py`: control-plane commands (e.g. request graceful stop).
- `runner.py`
  - CLI entrypoint used by cloud managers on remote workers.
- `loader.py`
  - Config loader and task loader (path-based and module-spec).
- `pipeline.py`
  - Sequential orchestration with failure handling and optional resume/restart/abort loop.
- `reporting/builder.py`
  - Markdown report generation with placeholder filling and optional WandB helpers.

## Components, responsibilities, and boundaries

### Task layer (`TaskPlugin`)
Defines task-specific hooks and isolates domain logic from infrastructure:
- SFT dataset + formatting/collator hooks.
- RL prompt dataset + reward function hooks.
- Eval dataset + metric computation hooks.

Boundary: infra/jobs never hardcode domain scoring logic; they call plugin hooks.

### Job layer (`SFTJob`, `RLJob`, `EvalJob`)
Implements model/trainer/eval runtime and writes reproducibility artifacts:
- Saves config snapshot into output directory.
- Produces `results.json` / `job_result.json`.
- Returns `JobResult` with status/metrics/metadata.

Boundary: jobs do model/task execution; they do not provision cloud resources.

### Cloud layer (`AWSManager`, `ModalManager`)
Responsible for remote runtime only:
- Provision/attach execution environment.
- Transfer configs/code and launch `python -m tenyson.runner` remotely.
- Retrieve outputs and reconstruct `JobResult`.

Boundary: cloud managers should not encode task semantics.

### Pipeline layer (`run_pipeline`)
Composes multiple jobs in order, updates optional report, and handles failures.

Boundary: currently sequential only; no built-in branch parallelism.

### Telemetry/control layer
Centralized SQL schema for:
- training/eval metrics,
- generation/rollout traces,
- stop requests,
- run metadata (e.g. WandB URL),
- run failures.

## Runtime architecture

### Process model
- Local orchestrator process constructs jobs and invokes cloud manager.
- Remote worker process executes `tenyson.runner` which creates one job instance and runs it.
- Optional additional control process can mark `run_controls.stop_requested` via `tenyson ctl stop`.

### Pipeline runtime behavior
- Iterates ordered step tuples `(label, config, JobClass, task)`.
- For each step, calls `cloud.run(job)` and appends result.
- On failure, optional notify hooks fire; if configured with `on_failure="wait"`, waits for user choice: resume, restart, or abort.

### AWS runtime specifics
- Starts EC2 instance (optionally spot), waits for SSH.
- Installs runtime dependencies and syncs workspace.
- Executes runner in resolved project root with `PYTHONPATH=src`.
- Syncs outputs back and optionally terminates instance.

### Modal runtime specifics
- Builds Modal image with required packages.
- Mounts local project root at `/workspace`.
- Executes runner from `/workspace` with `PYTHONPATH=src`.

## Data architecture

### File artifacts
- Job configs serialized to `config.json` under each run output dir.
- Training/eval outcomes:
  - SFT/RL: `results.json`.
  - Eval: detailed `results.json` + summary `job_result.json`.

### SQL telemetry schema (via SQLAlchemy)
- `sft_metrics`: step-level loss/eval_loss.
- `epoch_metrics`: RL epoch metrics.
- `rollouts`: RL prompt-level records.
- `generations`: completion records for RL/Eval.
- `run_controls`: manual stop flag per run.
- `run_metadata`: metadata like WandB URL.
- `run_failures`: failure audit records.

No DB migrations framework exists yet; tables are created opportunistically via `Base.metadata.create_all`.

## Integration points
- AWS (`boto3`/EC2): instance provisioning and lifecycle.
- Modal SDK: serverless GPU execution.
- Hugging Face Hub: adapter/model push and optional checkpoint resume via `snapshot_download`.
- Weights & Biases: trainer reporting and URL capture.
- SQL backends through SQLAlchemy URLs (SQLite/Postgres compatible at API level).
- Trainer/model stack: Unsloth, TRL, Transformers, vLLM.

## Request/data flow (critical paths)

### Cloud run path
1. Caller creates `Job` object with config + `TaskPlugin` instance.
2. Cloud manager serializes config and resolves task spec.
3. Remote command executes `tenyson.runner`.
4. Runner loads config and task plugin.
5. Runner instantiates specific job type and calls `job.run()`.
6. Job writes output artifacts and serialized result.
7. Cloud manager syncs artifacts back and returns `JobResult`.

### Manual stop path
1. Operator executes control command writing `run_controls.stop_requested = true`.
2. Active trainer callbacks poll `run_controls`.
3. Callback sets `control.should_training_stop = True`.
4. Training exits cleanly at callback-safe boundary; checkpoints/results persist.

## Build, test, run, release
- Packaging: setuptools (`pyproject.toml`), src-layout.
- Typical run: instantiate manager + job from Python script.
- Remote run entrypoint: `python -m tenyson.runner`.
- Current verification style in repo is script-driven; there is no dedicated formal test suite/CI pipeline in this checkout.

## Observability
- Console logs for step-by-step progress and failure surfaces.
- Optional SQL telemetry for machine-queryable metrics and traces.
- Optional WandB run URL capture for dashboard navigation.
- Optional failure notifications to JSON log files/webhooks/telemetry table.

## Security model
- Cloud credentials (AWS profile, SSH keypair, HF/WandB tokens) are injected via environment and/or cloud secret facilities.
- Remote execution currently uses SSH with host-key checks disabled for convenience (`StrictHostKeyChecking=no`), which is operationally convenient but weakens host authenticity guarantees.
- Secrets are not persisted by design in source files, but operational handling depends on user environment hygiene.

## Operational notes and failure modes
- If remote environment cannot import package in src-layout, jobs fail early; current managers explicitly set `PYTHONPATH=src` before runner execution.
- Task loading can fail when module specs are not importable on remote; managers now prefer importable `module:Class` and fall back to file path relative to synced/mounted repo.
- Spot interruption on AWS produces failed `JobResult` with interruption metadata when detectable.
- Missing optional dependencies in local dev env (e.g. `boto3`, `modal`) can block import-time checks for cloud modules.
- Pipeline remains single-threaded/sequential; parallel branch orchestration is currently external responsibility.

## Current implementation decisions from latest maintenance task
- RL runtime fixed to import `Path` in `jobs/rl.py` to prevent `NameError`.
- AWS manager now resolves whether the true project root is current directory or nested `tenyson/` directory and aligns remote/local output/result paths accordingly.
- AWS and Modal managers both normalize runner execution environment using `PYTHONPATH=src`.
- AWS and Modal managers both resolve task specs robustly for both importable plugins and file-loaded plugins.
- Wordle example no longer imports `tenyson.examples...`; it loads `wordle_task.py` directly and resolves config/report paths relative to script location.
