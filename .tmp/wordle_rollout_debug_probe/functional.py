from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path("/home/ayush/Desktop/code/tenyson")
SRC_DIR = REPO_ROOT / "src"

for path in (str(REPO_ROOT), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


from examples.wordle.functional import (  # noqa: E402
    _extract_completion_text,
    _FORMAT_REWARD,
    _GREEN_REWARD,
    _PERFECT_BONUS,
    _YELLOW_REWARD,
    _DEFAULT_WORD_SOURCE_URL,
    build_turn5_problem_rows,
    parse_guess,
    score_turn5_completion,
)
from datasets import Dataset  # noqa: E402
from tenyson import rl_dataset_template, rl_reward_template  # noqa: E402
from tenyson.core.stage_templates import RLDatasetTemplate, RLRewardTemplate, template_factory_ref  # noqa: E402


_FUNCTIONAL_MODULE = ".tmp.wordle_rollout_debug_probe.functional"

SEEDS = {
    "stopped_sft_turn5": {
        "repo_id": "goyalayus/wordle-lora-20260324-163252-sft_turn5",
        "revision": "2f92897b5cd3f760da3bdc526aa3fd2842e9bd82",
    }
}


@rl_dataset_template
def probe_turn5_rl_dataset(
    *,
    sample_count: int = 1,
    seed: int = 456,
    word_source: str = _DEFAULT_WORD_SOURCE_URL,
) -> RLDatasetTemplate:
    def _build(_config: dict[str, Any]) -> Dataset:
        rows = build_turn5_problem_rows(
            sample_count=sample_count,
            seed=seed,
            word_source=word_source,
        )
        return Dataset.from_list(rows)

    return RLDatasetTemplate(build=_build)


@rl_reward_template
def debug_turn5_reward() -> RLRewardTemplate:
    def _build(_config: dict[str, Any], _tokenizer: Any) -> list[Any]:
        def reward_debug(
            prompts: Sequence[Any],
            completions: Sequence[Any],
            secret: Sequence[Any],
            **_kwargs: Any,
        ) -> list[float]:
            rewards: list[float] = []
            print("\n[DEBUG] ----- RL rollout batch start -----", flush=True)
            for index, (prompt, completion, target) in enumerate(
                zip(prompts, completions, secret)
            ):
                completion_text = _extract_completion_text(completion)
                scored = score_turn5_completion(
                    completion_text,
                    str(target),
                    format_reward=_FORMAT_REWARD,
                    yellow_reward=_YELLOW_REWARD,
                    green_reward=_GREEN_REWARD,
                    perfect_bonus=_PERFECT_BONUS,
                )
                rewards.append(float(scored["reward_total"]))
                print(f"[DEBUG] sample={index}", flush=True)
                print(f"[DEBUG] secret={target}", flush=True)
                print(
                    "[DEBUG] prompt=\n"
                    f"{str(prompt)[:1500]}\n",
                    flush=True,
                )
                print(
                    "[DEBUG] completion=\n"
                    f"{completion_text[:4000]}\n",
                    flush=True,
                )
                print(
                    "[DEBUG] parsed_guess="
                    f"{parse_guess(completion_text)!r} "
                    f"format_ok={scored['format_ok']} "
                    f"feedback={scored['feedback']} "
                    f"reward_total={scored['reward_total']}",
                    flush=True,
                )
            print("[DEBUG] ----- RL rollout batch end -----\n", flush=True)
            return rewards

        setattr(reward_debug, "tenyson_reward_name", "debug_total")
        return [reward_debug]

    return RLRewardTemplate(
        build=_build,
        factory_ref=template_factory_ref(
            _FUNCTIONAL_MODULE,
            "debug_turn5_reward",
        ),
    )
