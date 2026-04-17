# Tenyson

I love using Unsloth.

I started using it around three months ago, and it was a great experience until my research experiments got bigger and messier.

At that point I realized I was writing a lot of boilerplate again and again. I was managing versions of files, wiring datasets and rewards, tracking Hugging Face adapters, and once runs failed inside a bigger experiment, recovery got painful very quickly.

So I built Tenyson.

Tenyson is an open-source library that abstracts the boilerplate around larger Unsloth research experiments and makes them much easier to run.

The main idea is simple: you define the shape of the whole experiment in a single file called `experiment.py`.

A small bundled arithmetic example makes that pattern easier to see. Here is a trimmed version of its `experiment.py`:

```python
from functional import (
    addition_reward,
    addition_rl_dataset,
    build_addition_dataset,
    build_addition_sft_train_dataset,
    compute_addition_metrics,
)
from tenyson import bind_chat_sft_dataset, bind_eval_dataset, run_experiment


def build(exp):
    return exp.run_branches(
        {
            "baseline_branch": [
                exp.eval_stage(
                    "eval_2digit_baseline",
                    dataset=bind_eval_dataset(
                        build_addition_dataset,
                        digits=2,
                        sample_count=100,
                        seed=7,
                    ),
                    metrics=compute_addition_metrics,
                ),
            ],
            "sft_branch": [
                exp.sft_stage(...),
                exp.eval_stage(
                    "eval_2digit_after_sft",
                    adapter=exp.adapter("sft_2digit_06b"),
                    ...,
                ),
            ],
            "rl_branch": [
                exp.rl_stage(
                    "rl_2digit_06b",
                    dataset=addition_rl_dataset(...),
                    reward=addition_reward(),
                    ...,
                ),
                exp.eval_stage(
                    "eval_2digit_after_rl",
                    adapter=exp.adapter("rl_2digit_06b"),
                    ...,
                ),
            ],
        }
    )


if __name__ == "__main__":
    run_experiment(__file__, build)
```

In that example, there are three parallel branches: eval, SFT, and RL.

The important part is the shape. Tenyson reads that graph, handles the orchestration, and keeps the dependencies intact. In the SFT branch, there is an eval after SFT. In the RL branch, there is an eval after RL. So you get parallel branches and sequential stages without writing controller code by hand.

You are also not dealing with the provider and GPU plumbing in this file. In the workflow shown here, Modal is the provider, and Tenyson takes care of the remote launch side.

The other half of the setup lives in a separate file called `functional.py`. That is where the datasets, reward functions, and eval metrics for SFT, RL, and eval are defined.

Here is a small reward snippet from the same arithmetic example:

```python
def get_addition_reward_funcs() -> list[Any]:
    def reward_correct_answer(
        _prompts: Sequence[Any],
        completions: Sequence[Any],
        expected_answer: Sequence[Any],
    ) -> list[float]:
        rewards: list[float] = []
        for completion_obj, expected in zip(completions, expected_answer):
            completion_text = extract_generation_text(completion_obj)
            parsed_answer = parse_answer(completion_text)
            rewards.append(
                1.0
                if parsed_answer is not None and parsed_answer == str(expected)
                else 0.0
            )
        return rewards

    def reward_strict_format(
        _prompts: Sequence[Any],
        completions: Sequence[Any],
    ) -> list[float]:
        rewards: list[float] = []
        for completion_obj in completions:
            completion_text = extract_generation_text(completion_obj)
            rewards.append(0.1 if has_strict_answer_format(completion_text) else 0.0)
        return rewards

    setattr(reward_correct_answer, "tenyson_reward_name", "correct_answer")
    setattr(reward_strict_format, "tenyson_reward_name", "strict_format")
    return [reward_correct_answer, reward_strict_format]


@rl_reward_template
def addition_reward() -> RLRewardTemplate:
    def _build(_config: dict[str, Any], _tokenizer: Any) -> list[Any]:
        return get_addition_reward_funcs()

    return RLRewardTemplate(build=_build)
```

That is basically the pattern. You write these two files, and that is the core setup.

If you want to change things like batch size, learning rate, max steps, or generation limits, there are separate config templates for that. But the normal flow is that you do not edit those directly. They stay abstracted away, and you just override the values you care about inside `experiment.py`.

For example, you can override training values directly in `experiment.py`:

```python
overrides={
    "training": {
        "max_steps": 150,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5.0e-6,
        "num_generations": 4,
        "max_prompt_length": 256,
        "max_completion_length": 64,
        "hf_push_every_steps": 50,
    },
}
```

Once that experiment starts running, Tenyson creates a `final_report.md` next to your `experiment.py`.

When a stage starts, the report gets the W&B link. When it finishes, the report gets the stage metrics. The idea is that you always have a readable summary of the experiment as it unfolds, instead of stitching it together later from logs.

Now suppose you want to stop one particular run from one particular branch without disturbing the whole experiment. You can do that with:

```bash
python3 -m tenyson.ctl stop \
  --db-url wandb://<entity>/<project> \
  --experiment-id <experiment_id>
```

If you already know the exact run, you can pass it directly:

```bash
python3 -m tenyson.ctl stop \
  --db-url wandb://<entity>/<project> \
  --experiment-id <experiment_id> \
  --run-id rl_2digit_06b
```

Once that stage is stopped, Tenyson gives you four useful choices:

- `resume`: useful when a run stopped because of a bug or code issue, you fixed it, and now you want to continue from the latest saved Hugging Face checkpoint instead of starting from scratch
- `continue`: accept the stopped stage and move to the next stage
- `restart`: rerun that stage from scratch
- `abort`: stop the experiment there and skip the remaining stages

This matters a lot once the experiment graph gets larger. You do not want one interrupted branch to force you to manually reconstruct everything.

Now suppose you want to continue from a specific place in an experiment that has already finished. You can do that too.

To recover an older experiment, rerun the same `experiment.py` with the original experiment id:

```bash
TENYSON_RECOVER_EXPERIMENT_ID=<old_experiment_id> \
python3 examples/arithmetic/experiment.py
```

If you want to force a restart from a specific stage onward:

```bash
TENYSON_RECOVER_EXPERIMENT_ID=<old_experiment_id> \
TENYSON_RECOVER_RESTART_FROM_STAGE=<stage_id> \
python3 examples/arithmetic/experiment.py
```

Tenyson also uses W&B for telemetry, not just graphs.

We store the important debugging information there as well, especially for RL: prompts, rollout answers, rewards, and stage metadata.

To inspect all of that properly, there is a custom UI. You can start it with:

```bash
python3 -m tenyson.ui \
  --db-url wandb://<entity>/<project> \
  --experiment-id <experiment_id> \
  --open-browser
```

That UI is there so you can inspect what actually happened inside a run, instead of only looking at top-line charts.

We are still improving Tenyson. More model support, more GPU providers, and a better overall research workflow are coming. If this is useful to you, please star the repo and follow along.
