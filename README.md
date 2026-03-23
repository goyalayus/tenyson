# Tenyson

Tenyson is a remote-first research library for SFT, GRPO RL, and eval workflows.

The mental model is intentionally small:

- each environment lives in a single Python file
- that file exports explicit named runs
- each run declares its datasets, rubric or metrics, env metadata, and config overrides
- `experiment.py` just chains those runs into the graph you want
- the library handles remote execution, telemetry, Hugging Face adapter lineage, control flow, and a fixed report

Right now the Wordle example is the best reference for how the system is supposed to feel.

The visible starter templates live in [`config_templates/`](./config_templates), not inside the example folder. That way a new user can see the default SFT, RL, and eval knobs without digging through task-specific code first.

## What Tenyson Gives You

- Job abstractions: `SFTJob`, `RLJob`, `EvalJob`
- Cloud execution managers: Modal and AWS
- W&B-only telemetry
- Fixed markdown reporting with W&B project/run links
- Hugging Face adapter pushes for SFT and RL runs
- Stop, continue, resume, restart, and abort control flow across experiments

## The Shape Of An Environment

The environment contract lives in [`src/tenyson/core/environment.py`](./src/tenyson/core/environment.py).

An environment definition is a map of named runs. Each run is explicit and stable. For example, the Wordle task exposes names like:

- `wordle_sft_main`
- `wordle_rl_mixed`
- `wordle_rl_turn2`
- `wordle_eval_turn4`
- `wordle_eval_mixed`

Each named run carries:

- a run type: `sft`, `rl`, or `eval`
- dataset hooks
- reward hooks or metric hooks
- environment metadata
- run-specific config overrides

That is the big design point: task logic stays in the environment file, while experiment files stay focused on orchestration.

## The Shape Of An Experiment

The Wordle experiment entrypoint in [`examples/wordle/experiment.py`](./examples/wordle/experiment.py) is the intended style.

It does three jobs:

1. load the environment and base config templates
2. choose the named runs to execute
3. describe the graph: sequence, parallel branches, and follow-up evals

That file is not supposed to contain reward logic, dataset construction logic, or reporting customization. Those concerns live in the library and the environment definition.

## Basic Usage

Jobs run through a cloud manager. The manager launches `python -m tenyson.runner` remotely, waits for the canonical run result in telemetry, and returns a `JobResult`.

Local GPU execution is intentionally not the default path. Tenyson expects to run through a supported remote provider.
Modal execution is git-backed: commit and push your code first, then the worker checks out that exact commit remotely.

```python
from pathlib import Path
import yaml

from tenyson.cloud.modal import ModalManager
from tenyson.jobs.sft import SFTJob
from tenyson.loader import load_task

with open("config_templates/sft.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

task = load_task(str(Path("examples/wordle/wordle_task.py").resolve()))
cfg.setdefault("training", {})["run_name"] = "example_sft_run"
cfg.setdefault("telemetry", {})["experiment_id"] = "example_experiment"
job = SFTJob(config=cfg, task=task)
cloud = ModalManager(gpu="A100", timeout=86400)
result = cloud.run(job)
print(result.metrics, result.wandb_url)
```

For real runs you will usually also provide the W&B destination and HF repo base through env vars or shared experiment overrides, as shown in the Wordle example runner.

## Wordle Reference Workflow

The current Wordle example runs this graph:

1. SFT
2. Baseline mixed eval
3. Two branches in parallel
4. Branch A: mixed RL, then mixed final eval
5. Branch B: curriculum RL from turn 2 through turn 5, with follow-up evals after each stage
6. Final fixed markdown report

The important part is not Wordle itself. The important part is the shape:

- the base templates come from `config_templates/`
- the environment file exports the named runs
- the experiment file only composes them

## Running The Wordle Example

Run from the repo root:

```bash
cp examples/wordle/.env.example examples/wordle/.env
# fill in your values in examples/wordle/.env
python3 examples/wordle/experiment.py
```

The experiment entrypoint auto-loads `examples/wordle/.env` when present, auto-adds `src/` to `PYTHONPATH`, and can install missing controller-side dependencies on a fresh local checkout. Set `TENYSON_SKIP_LOCAL_BOOTSTRAP=1` if you want to disable dependency bootstrap behavior.

For a fresh machine, put `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` directly in `examples/wordle/.env`.
If your GitHub remote is private, also provide `TENYSON_GIT_AUTH_TOKEN` or `GITHUB_TOKEN` so Modal can clone the repo during image build.

Useful environment variables:

- `TENYSON_MODAL_GPU` default `A100`
- `TENYSON_MODAL_TIMEOUT` default `86400`
- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`
- `TENYSON_HF_REPO_BASE`
- `TENYSON_WANDB_ENTITY`
- `TENYSON_WANDB_PROJECT`
- `HF_TOKEN`
- `WANDB_API_KEY`

For Hugging Face pushes, `TENYSON_HF_REPO_BASE` must point at a namespace that the provided `HF_TOKEN` can actually write to.

## Telemetry

Telemetry is mandatory. Every run needs an experiment id.
Telemetry is always sent to W&B. You do not need to set a backend field in normal configs.

```yaml
telemetry:
  experiment_id: "wordle_research_2026_03_01"
  entity: "your-wandb-entity"
  project: "wordle-research"
```

You can provide the W&B destination either in config or through env vars:

- `telemetry.entity` / `telemetry.project`
- `TENYSON_WANDB_ENTITY` / `TENYSON_WANDB_PROJECT`

The canonical run result lives in telemetry:

- W&B summaries and artifacts

That same telemetry layer is also how stop requests are delivered to running jobs.

## Reporting

Reporting is fixed, not template-driven.

The experiment report captures the things you generally want when looking back at a run:

- stage status
- run type and named environment run
- key metrics
- Hugging Face adapter lineage for SFT and RL
- W&B run links and project link

The Wordle example writes this to `examples/wordle/final_report.md` during local runs.

## Checkpoints And Resume

SFT and RL use Hub-managed checkpoints.

- set `training.hf_repo_base` to enable pushes
- checkpoints are stored in a stable repo derived from the run name
- resume uses `training.resume_from_checkpoint: "repo_id:revision"`
- the revision is resolved to an immutable Hugging Face commit SHA before restore

That gives you a clean lineage story for stop, continue, resume, and restart flows.

If you manually stop an SFT or RL run, the pipeline can now do four different things:

- `continue`: accept the stopped checkpoint and move on to the next stage
- `resume`: restart from the latest saved HF checkpoint
- `restart`: run the stage again from scratch
- `abort`: stop the experiment

`continue` is only offered when the stopped run already has a concrete Hugging Face repo + revision, so later stages can still seed from an exact adapter lineage.

For SFT early stopping specifically, Tenyson now treats it as a strict best-model flow instead of a soft hint:

- early stopping requires an eval dataset
- `training.eval_steps` must match `training.hf_push_every_steps`
- the best checkpoint is re-synced to HF at the end so downstream stages use the actual best adapter revision

## Runner CLI

Remote workers call a single entrypoint:

```bash
python -m tenyson.runner \
  --job-type sft \
  --config path/to/config.yaml \
  --task-module examples/wordle/wordle_task.py
```

Or with a `module:Class` task spec:

```bash
python -m tenyson.runner \
  --job-type sft \
  --config path/to/config.yaml \
  --task-module examples.wordle.wordle_task:WordleTask
```

`--job-type` must be one of `sft`, `rl`, or `eval`.

## Notes

- AWS spot failures return a failed `JobResult` instead of throwing away run context
- Modal and AWS both return the same `JobResult` shape
- remote runtime setup installs the stack around `unsloth` and `vllm`
- within one experiment run, stage run names must stay unique
