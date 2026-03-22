# Tenyson

Tenyson is a remote-first research runner for SFT, GRPO RL, and eval workflows.

The core idea is simple:

- put one environment in one Python file
- give that environment explicit named runs
- let `experiment.py` only orchestrate those runs
- let Tenyson handle cloud execution, telemetry, adapter lineage, stop/continue/resume/abort, and a fixed report

The current repo is laid out with the actual Python project inside [`tenyson/`](./tenyson). That is the directory you should think of as the package root.

## What An Environment Looks Like

An environment file exports named runs such as `wordle_sft_main`, `wordle_rl_turn3`, or `wordle_eval_mixed`.

Each named run declares the things that matter for that run:

- run type: `sft`, `rl`, or `eval`
- dataset hooks
- reward functions or eval metrics
- environment metadata
- config overrides for that run

That keeps task logic in the environment file and keeps experiment orchestration thin.

## What An Experiment Does

`experiment.py` should mostly read like a research graph:

- run SFT
- run baseline evals
- branch into mixed RL vs curriculum RL
- run follow-up evals
- emit a fixed markdown report with links back to telemetry

The Wordle example in this repo follows exactly that pattern.

## Repo Layout

- [`tenyson/src/tenyson/`](./tenyson/src/tenyson): library code
- [`tenyson/config_templates/`](./tenyson/config_templates): visible starter templates for SFT, RL, and eval runs
- [`tenyson/examples/wordle/wordle_task.py`](./tenyson/examples/wordle/wordle_task.py): example environment definition
- [`tenyson/examples/wordle/experiment.py`](./tenyson/examples/wordle/experiment.py): example experiment graph
- [`tenyson/README.md`](./tenyson/README.md): package-level docs and deeper reference

## Quick Start

```bash
cd tenyson
cp examples/wordle/run.env.example examples/wordle/run.env
# fill in HF + W&B credentials
source examples/wordle/run.env
python3 examples/wordle/experiment.py
```

A few important points:

- jobs are meant to run on remote GPUs through Modal or AWS
- remote workers install the runtime stack around Unsloth and vLLM
- telemetry is expected to be on, with W&B as the recommended backend
- SFT and RL pushes go to Hugging Face adapters so runs can be resumed cleanly

## Current Status

The current Wordle flow is the reference environment for the library shape. It exercises:

- explicit named runs inside one environment file
- thin experiment orchestration
- W&B-backed telemetry
- fixed report generation
- stop, continue, resume, restart, and abort control flow

If you want the full package-level details, start in [`tenyson/README.md`](./tenyson/README.md).
