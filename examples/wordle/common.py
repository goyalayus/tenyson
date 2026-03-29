from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

from tenyson.bootstrap import is_truthy
from tenyson.core.experiment_runtime import (
    LocalExperimentContext,
    env_float,
    env_int,
    load_local_task,
)
from tenyson.experiment import AdapterRef


WORDLE_TASK_FILENAME = "wordle_task.py"
WORDLE_REPORT_ENV_VAR = "TENYSON_WORDLE_REPORT_PATH"
WORDLE_REPORT_FILENAME = "final_report.md"
WORDLE_EXPERIMENT2_REPORT_ENV_VAR = "TENYSON_WORDLE_EXPERIMENT2_REPORT_PATH"
WORDLE_EXPERIMENT2_REPORT_FILENAME = "final_report_experiment2.md"
WORDLE_EXPERIMENT2_PREFIX = "wordle_experiment2_"

WORDLE_EXPERIMENT2_DEFAULT_SFT_REPO = (
    "goyalayus/wordle-lora-20260324-163252-sft_main"
)
WORDLE_EXPERIMENT2_DEFAULT_SFT_REVISION = (
    "30a33278640fcc5bcce216adce59984bfb8f7698"
)

_WORDLE_REPORT_STAGE_ORDER = [
    "sft_main",
    "eval_baseline_mixed",
    "mixed_rl",
    "mixed_final_eval",
    "curr_rl_t2",
    "curr_eval_after_t2_turn2",
    "curr_rl_t3",
    "curr_eval_after_t3_turn2",
    "curr_eval_after_t3_turn3",
    "curr_rl_t4",
    "curr_eval_after_t4_turn3",
    "curr_eval_after_t4_turn4",
    "curr_rl_t5",
    "curr_eval_after_t5_turn4",
    "curr_eval_after_t5_turn5",
    "curr_final_eval",
]


def load_wordle_task(context: LocalExperimentContext) -> Any:
    return load_local_task(context=context, task_filename=WORDLE_TASK_FILENAME)


def wordle_report_stage_order() -> list[str]:
    return list(_WORDLE_REPORT_STAGE_ORDER)


def wordle_recovery_restart_stages() -> list[str]:
    restart_from = str(
        os.getenv("TENYSON_WORDLE_RECOVER_RESTART_FROM_STAGE", "")
    ).strip()
    if not restart_from:
        return []
    ordered_stages = wordle_report_stage_order()
    if restart_from not in ordered_stages:
        raise ValueError(
            "TENYSON_WORDLE_RECOVER_RESTART_FROM_STAGE must be one of "
            f"{ordered_stages}, got {restart_from!r}."
        )
    return ordered_stages[ordered_stages.index(restart_from) :]


def wordle_smoke_overrides(
    *,
    include_sft: bool,
    label: str,
) -> dict[str, dict] | None:
    if not is_truthy(os.getenv("TENYSON_WORDLE_SMOKE", "false")):
        return None

    rl_steps = max(1, env_int("TENYSON_WORDLE_SMOKE_RL_STEPS", 20))
    rl_samples = max(1, env_int("TENYSON_WORDLE_SMOKE_RL_SAMPLES", 64))
    eval_samples = max(1, env_int("TENYSON_WORDLE_SMOKE_EVAL_SAMPLES", 25))
    rl_push_steps = max(
        1,
        min(rl_steps, env_int("TENYSON_WORDLE_SMOKE_RL_PUSH_STEPS", 10)),
    )
    rl_vllm_gpu_util = min(
        0.95,
        max(0.1, env_float("TENYSON_WORDLE_SMOKE_RL_VLLM_GPU_UTIL", 0.5)),
    )

    overrides: dict[str, dict] = {
        "rl": {
            "training": {
                "max_steps": rl_steps,
                "hf_push_every_steps": rl_push_steps,
                "save_total_limit": 1,
            },
            "vllm": {
                "gpu_memory_utilization": rl_vllm_gpu_util,
            },
            "task": {
                "synthetic_samples": rl_samples,
            },
        },
        "eval": {
            "task": {
                "eval_samples": eval_samples,
            }
        },
    }

    if include_sft:
        sft_steps = max(1, env_int("TENYSON_WORDLE_SMOKE_SFT_STEPS", 30))
        sft_train_samples = max(
            1, env_int("TENYSON_WORDLE_SMOKE_SFT_TRAIN_SAMPLES", 128)
        )
        sft_val_size = max(1, env_int("TENYSON_WORDLE_SMOKE_SFT_VAL_SIZE", 32))
        sft_save_steps = max(
            1,
            min(sft_steps, env_int("TENYSON_WORDLE_SMOKE_SFT_SAVE_STEPS", 10)),
        )
        sft_eval_steps = max(
            1,
            min(sft_steps, env_int("TENYSON_WORDLE_SMOKE_SFT_EVAL_STEPS", 10)),
        )
        overrides["sft"] = {
            "task": {
                "sft_train_samples": sft_train_samples,
            },
            "training": {
                "max_steps": sft_steps,
                "val_size": sft_val_size,
                "save_steps": sft_save_steps,
                "eval_steps": sft_eval_steps,
                "hf_push_every_steps": sft_save_steps,
                "save_total_limit": 1,
                "logging_steps": 1,
            },
        }
        print(
            f"[{label}] Smoke mode enabled "
            f"(sft_steps={sft_steps}, sft_train_samples={sft_train_samples}, "
            f"rl_steps={rl_steps}, rl_samples={rl_samples}, "
            f"eval_samples={eval_samples}, rl_vllm_gpu_util={rl_vllm_gpu_util}).",
            flush=True,
        )
        return overrides

    print(
        f"[{label}] Smoke mode enabled "
        f"(rl_steps={rl_steps}, rl_samples={rl_samples}, "
        f"eval_samples={eval_samples}, rl_vllm_gpu_util={rl_vllm_gpu_util}).",
        flush=True,
    )
    return overrides


def configure_wordle_smoke_identity(
    *,
    context: LocalExperimentContext,
    label: str,
    report_env_var: str = WORDLE_REPORT_ENV_VAR,
) -> None:
    if not is_truthy(os.getenv("TENYSON_WORDLE_SMOKE", "false")):
        return

    explicit_smoke_experiment_id = str(
        os.getenv("TENYSON_WORDLE_SMOKE_EXPERIMENT_ID", "")
    ).strip()
    explicit_smoke_repo_base = str(
        os.getenv("TENYSON_WORDLE_SMOKE_HF_REPO_BASE", "")
    ).strip()
    explicit_smoke_report_path = str(
        os.getenv("TENYSON_WORDLE_SMOKE_REPORT_PATH", "")
    ).strip()

    experiment_id = str(os.getenv("TENYSON_EXPERIMENT_ID", "")).strip()
    if explicit_smoke_experiment_id:
        experiment_id = explicit_smoke_experiment_id
        os.environ["TENYSON_EXPERIMENT_ID"] = experiment_id
    elif "TENYSON_EXPERIMENT_ID" in context.loaded_env or not experiment_id:
        experiment_id = f"wordle_smoke_{_smoke_timestamp()}"
        os.environ["TENYSON_EXPERIMENT_ID"] = experiment_id

    hf_repo_base = str(os.getenv("TENYSON_HF_REPO_BASE", "")).strip()
    if explicit_smoke_repo_base:
        hf_repo_base = explicit_smoke_repo_base
        os.environ["TENYSON_HF_REPO_BASE"] = hf_repo_base
    elif hf_repo_base and "TENYSON_HF_REPO_BASE" in context.loaded_env:
        hf_repo_base = _append_hf_repo_suffix(hf_repo_base, "smoke")
        os.environ["TENYSON_HF_REPO_BASE"] = hf_repo_base

    if explicit_smoke_report_path:
        os.environ[report_env_var] = explicit_smoke_report_path
    elif report_env_var in context.loaded_env or not str(
        os.getenv(report_env_var, "")
    ).strip():
        report_path = context.file("smoke_reports") / f"{experiment_id}.md"
        os.environ[report_env_var] = str(report_path)

    print(
        f"[{label}] Smoke identity "
        f"(experiment_id={os.getenv('TENYSON_EXPERIMENT_ID', '')}, "
        f"hf_repo_base={os.getenv('TENYSON_HF_REPO_BASE', 'n/a')}, "
        f"report_path={os.getenv(report_env_var, '')}).",
        flush=True,
    )


def resolve_wordle_experiment2_seed_adapter(*, label: str) -> AdapterRef:
    repo_id = str(
        os.getenv(
            "TENYSON_WORDLE_EXPERIMENT2_SFT_REPO",
            WORDLE_EXPERIMENT2_DEFAULT_SFT_REPO,
        )
    ).strip()
    revision = str(
        os.getenv(
            "TENYSON_WORDLE_EXPERIMENT2_SFT_REVISION",
            WORDLE_EXPERIMENT2_DEFAULT_SFT_REVISION,
        )
    ).strip()
    if not repo_id or not revision:
        raise ValueError(
            "Experiment 2 needs a concrete SFT adapter seed. Set "
            "TENYSON_WORDLE_EXPERIMENT2_SFT_REPO and "
            "TENYSON_WORDLE_EXPERIMENT2_SFT_REVISION if you want to override the "
            "default seed."
        )
    print(f"[{label}] Using SFT seed adapter {repo_id}@{revision}.", flush=True)
    return AdapterRef(repo_id=repo_id, revision=revision)


def _smoke_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _append_hf_repo_suffix(repo_base: str, suffix: str) -> str:
    text = str(repo_base or "").strip().rstrip("/")
    if not text:
        return ""

    namespace = ""
    repo_name = text
    if "/" in text:
        namespace, repo_name = text.split("/", 1)

    suffix_text = f"-{suffix}"
    if not repo_name.endswith(suffix_text):
        repo_name = f"{repo_name}{suffix_text}"
    repo_name = repo_name[:96].rstrip("-")
    if namespace:
        return f"{namespace}/{repo_name}"
    return repo_name
