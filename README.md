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

## Dataset Format Requirements

This needs to be explicit because the expected shape differs by run type.

### SFT

For SFT, the task hook should return a conversational dataset. The built-in
Wordle SFT path loads a Hugging Face dataset where each row looks like:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Important constraints:

- If you bring your own SFT dataset and want to use the default path, it must
  expose a `messages` column in this chat format. A plain `prompt` /
  `completion` table will not work unless you also provide a custom formatting
  hook or collator.
- The built-in assistant-only-loss path expects the formatted sequence to
  contain an assistant-turn marker such as
  `training.response_template: "<|im_start|>assistant\n"`.
- With `training.loss_on_assistant_only: true`, loss is computed only on the
  assistant message content. The assistant turn terminator is excluded from the
  loss mask.
- If one formatted example contains multiple assistant turns separated by more
  user turns, you must also set `training.instruction_template`, or provide a
  task-specific collator. Do not assume the default assistant-only collator can
  safely infer user/assistant boundaries in arbitrary text.
- In the current Wordle example, every SFT row is exactly one
  `system -> user -> assistant` conversation turn. The SFT dataset is not the
  random synthetic Wordle-history dataset used for RL and eval.

### RL And Eval

RL and eval can use task-generated synthetic rows instead of conversational
 SFT rows. The built-in Wordle synthetic rows look like:

```json
{
  "secret": "meant",
  "history_len": 4,
  "history_rows": [
    ["cogue", "X X X X Y"],
    ["irone", "X X X G Y"],
    ["exams", "Y X G Y X"],
    ["macaw", "G Y X X X"]
  ],
  "prompt": "..."
}
```

Important constraints:

- Mixed Wordle RL/eval runs intentionally sample random prior-history lengths.
- Curriculum Wordle runs intentionally fix the history window for each stage.
- Do not expect the SFT dataset and the RL/eval datasets to have the same row
  schema. They are different on purpose.

## Basic Usage

Jobs run through a cloud manager. The manager launches `python -m tenyson.runner` remotely, waits for the canonical run result in telemetry, and returns a `JobResult`.

Local GPU execution is intentionally not the default path. Tenyson expects to run through a supported remote provider.
Modal execution is git-backed: commit and push your code first, then the worker checks out that exact commit remotely.
There is no local source-sync or dev-mode upload path in the Modal runner.

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

For long-running experiments, prefer launching the local controller in detached
mode so it does not die with your interactive terminal session:

```bash
python3 -m tenyson.ctl launch \
  --name wordle \
  --cwd . \
  -- python3 examples/wordle/experiment.py

python3 -m tenyson.ctl status --name wordle
```

This writes controller pid/log metadata under `.tenyson_runs/controllers/`.
Use `python3 -m tenyson.ctl stop-controller --name wordle` if you need to
terminate the detached local controller process itself.

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

## Dashboard

Tenyson now ships a built-in local telemetry dashboard for live and post-run inspection across SFT, eval, and RL stages.

Run it from the repo root:

```bash
python3 -m tenyson.ui \
  --db-url wandb://<entity>/<project> \
  --experiment-id <experiment_id> \
  --open-browser
```

You can also omit `--db-url` if `TENYSON_WANDB_ENTITY` and `TENYSON_WANDB_PROJECT`
are already set in your environment.

The dashboard is intentionally zero-build:

- it serves a local web UI directly from Python
- it auto-refreshes while an experiment is still running
- it lists all runs in the experiment, not just eval
- eval runs show detailed sample rows when the task logs `detailed_results`
- every run also exposes raw canonical payloads and W&B-linked metadata for debugging

Useful flags:

- `--host` default `127.0.0.1`
- `--port` default `8787`
- `--refresh-seconds` default `10`
- `--history-limit` default `240`

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

You can also recover after the original local controller process has exited.
Pass an explicit recovery experiment id to `ExperimentSession` with
`recovery_experiment_id="..."`, then relaunch the experiment with the same stage
run names. On startup, Tenyson will reuse prior successful/partial stage
results, and for stopped SFT/RL stages it will prompt for:

- `resume`: restart from the saved trainer checkpoint
- `continue`: accept the stopped checkpoint and move to the next stage
- `restart`: rerun the stage from scratch

The Wordle example exposes this through
`TENYSON_RECOVER_EXPERIMENT_ID=<experiment_id>`.

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
