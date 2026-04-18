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


_FUNCTIONAL_MODULE = "tmp_wordle_rollout_debug_probe.functional"

SEEDS = {
    "stopped_sft_turn5": {
        "repo_id": "goyalayus/wordle-lora-20260324-163252-sft_turn5",
        "revision": "2f92897b5cd3f760da3bdc526aa3fd2842e9bd82",
    }
}

_VLLM_PROBE_INSTALLED = False
_VLLM_PROBE_LOGGED = False


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
        _install_vllm_adapter_probe()

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


def _install_vllm_adapter_probe() -> None:
    global _VLLM_PROBE_INSTALLED
    if _VLLM_PROBE_INSTALLED:
        return

    try:
        from vllm import LLM
    except Exception as exc:  # noqa: BLE001
        print(
            f"[PROBE] Skipping vLLM adapter probe install because vllm import failed: {exc}",
            flush=True,
        )
        return

    original_generate = LLM.generate

    def wrapped_generate(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _VLLM_PROBE_LOGGED
        if not _VLLM_PROBE_LOGGED:
            _VLLM_PROBE_LOGGED = True
            _log_vllm_adapter_state(self, kwargs)
        return original_generate(self, *args, **kwargs)

    LLM.generate = wrapped_generate
    _VLLM_PROBE_INSTALLED = True
    print("[PROBE] Installed vLLM generate() adapter probe.", flush=True)


def _log_vllm_adapter_state(llm: Any, kwargs: dict[str, Any]) -> None:
    print("\n[PROBE] ===== vLLM adapter inspection start =====", flush=True)

    lora_request = kwargs.get("lora_request")
    print(
        f"[PROBE] lora_request_present={lora_request is not None} "
        f"type={type(lora_request).__name__ if lora_request is not None else 'None'}",
        flush=True,
    )
    if lora_request is not None:
        for name in (
            "lora_name",
            "lora_int_id",
            "adapter_id",
            "rank",
            "path",
            "lora_path",
            "local_path",
            "lora_local_path",
        ):
            if hasattr(lora_request, name):
                print(
                    f"[PROBE] lora_request.{name}={getattr(lora_request, name)!r}",
                    flush=True,
                )
        tensor_map = _extract_lora_tensor_map(lora_request)
        print(
            f"[PROBE] lora_request_tensor_count={len(tensor_map)}",
            flush=True,
        )
        for key in sorted(tensor_map.keys())[:3]:
            print(
                f"[PROBE] lora_request_tensor[{key}]={_tensor_summary(tensor_map[key])}",
                flush=True,
            )

    model_runner = _find_first_attr_chain(
        llm,
        [
            "llm_engine.model_executor.driver_worker.model_runner",
            "llm_engine.model_executor.worker.model_runner",
            "llm_engine.model_executor.driver_worker.worker.model_runner",
            "llm_engine.model_executor.workers[0].model_runner",
        ],
    )
    print(
        f"[PROBE] model_runner_found={model_runner is not None}",
        flush=True,
    )

    lora_manager = None
    if model_runner is not None:
        lora_manager = getattr(model_runner, "lora_manager", None)
        if lora_manager is None:
            model = getattr(model_runner, "model", None)
            lora_manager = getattr(model, "lora_manager", None)
    print(
        f"[PROBE] lora_manager_found={lora_manager is not None}",
        flush=True,
    )

    adapter_manager = None
    if lora_manager is not None:
        adapter_manager = getattr(lora_manager, "_adapter_manager", None)
        if adapter_manager is None:
            adapter_manager = getattr(lora_manager, "adapter_manager", None)
    print(
        f"[PROBE] adapter_manager_found={adapter_manager is not None}",
        flush=True,
    )

    if adapter_manager is not None:
        registered = getattr(adapter_manager, "_registered_adapters", None)
        active = getattr(adapter_manager, "_active_adapters", None)
        if registered is not None:
            print(
                f"[PROBE] registered_adapter_keys={list(registered.keys())}",
                flush=True,
            )
        if active is not None:
            print(
                f"[PROBE] active_adapter_keys={list(active.keys())}",
                flush=True,
            )
        punica_mapping = getattr(adapter_manager, "punica_wrapper_mapping", None)
        if isinstance(punica_mapping, dict):
            print(
                f"[PROBE] punica_wrapper_keys={list(punica_mapping.keys())}",
                flush=True,
            )
            for key, wrapper in punica_mapping.items():
                token_indices = getattr(wrapper, "_token_lora_indices", None)
                if token_indices is not None:
                    print(
                        f"[PROBE] punica[{key}]_token_lora_indices={_tensor_summary(token_indices)}",
                        flush=True,
                    )

    print("[PROBE] ===== vLLM adapter inspection end =====\n", flush=True)


def _find_first_attr_chain(root: Any, chains: Sequence[str]) -> Any:
    for chain in chains:
        try:
            value = _resolve_attr_chain(root, chain)
        except Exception:  # noqa: BLE001
            continue
        if value is not None:
            return value
    return None


def _resolve_attr_chain(root: Any, chain: str) -> Any:
    current = root
    for piece in chain.split("."):
        if piece.endswith("]") and "[" in piece:
            name, raw_index = piece[:-1].split("[", 1)
            current = getattr(current, name)
            current = current[int(raw_index)]
        else:
            current = getattr(current, piece)
    return current


def _extract_lora_tensor_map(lora_request: Any) -> dict[str, Any]:
    for name in ("lora_tensors", "tensors", "adapter_tensors"):
        value = getattr(lora_request, name, None)
        if isinstance(value, dict):
            return value
    return {}


def _tensor_summary(value: Any) -> str:
    try:
        shape = tuple(value.shape)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        shape = None
    try:
        norm = float(value.norm().item())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        norm = None
    try:
        max_abs = float(value.abs().max().item())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        max_abs = None
    return f"shape={shape} norm={norm} max_abs={max_abs}"
