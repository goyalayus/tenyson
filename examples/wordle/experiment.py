import os
from pathlib import Path
import sys

# src-layout convenience for running this file directly from a fresh checkout.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tenyson.bootstrap import ensure_local_controller_environment, load_env_file
from tenyson.cloud.modal import ModalManager
from tenyson.core.run_config import shared_overrides_from_env
from tenyson.experiment import ConfigTemplates, ExperimentAborted, ExperimentSession
from tenyson.loader import load_task
from tenyson.reporting.fixed import ExperimentReport

ensure_local_controller_environment(anchor_file=__file__)
load_env_file(Path(__file__).with_name(".env"), override=True)


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    return int(raw or str(default))


def _float_env(name: str, default: float) -> float:
    raw = str(os.getenv(name, str(default))).strip()
    return float(raw or str(default))


def _wordle_smoke_overrides() -> dict[str, dict] | None:
    if not _is_truthy(os.getenv("TENYSON_WORDLE_SMOKE", "false")):
        return None

    sft_steps = max(1, _int_env("TENYSON_WORDLE_SMOKE_SFT_STEPS", 30))
    rl_steps = max(1, _int_env("TENYSON_WORDLE_SMOKE_RL_STEPS", 20))
    rl_samples = max(1, _int_env("TENYSON_WORDLE_SMOKE_RL_SAMPLES", 64))
    eval_samples = max(1, _int_env("TENYSON_WORDLE_SMOKE_EVAL_SAMPLES", 25))
    sft_val_size = max(1, _int_env("TENYSON_WORDLE_SMOKE_SFT_VAL_SIZE", 32))
    sft_save_steps = max(1, min(sft_steps, _int_env("TENYSON_WORDLE_SMOKE_SFT_SAVE_STEPS", 10)))
    sft_eval_steps = max(1, min(sft_steps, _int_env("TENYSON_WORDLE_SMOKE_SFT_EVAL_STEPS", 10)))
    rl_push_steps = max(1, min(rl_steps, _int_env("TENYSON_WORDLE_SMOKE_RL_PUSH_STEPS", 10)))
    rl_vllm_gpu_util = min(
        0.95,
        max(0.1, _float_env("TENYSON_WORDLE_SMOKE_RL_VLLM_GPU_UTIL", 0.5)),
    )

    print(
        "[wordle experiment] Smoke mode enabled "
        f"(sft_steps={sft_steps}, rl_steps={rl_steps}, rl_samples={rl_samples}, "
        f"eval_samples={eval_samples}, rl_vllm_gpu_util={rl_vllm_gpu_util}).",
        flush=True,
    )

    return {
        "sft": {
            "training": {
                "max_steps": sft_steps,
                "val_size": sft_val_size,
                "save_steps": sft_save_steps,
                "eval_steps": sft_eval_steps,
                "hf_push_every_steps": sft_save_steps,
                "save_total_limit": 1,
                "logging_steps": 1,
            }
        },
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


def _report_output_path(base_dir: Path) -> Path:
    configured = str(os.getenv("TENYSON_WORDLE_REPORT_PATH", "")).strip()
    if configured:
        return Path(configured)
    return base_dir / "final_report.md"


def main() -> None:
    base_dir = Path(__file__).parent
    task = load_task(str(base_dir / "wordle_task.py"))
    report = ExperimentReport(output_path=_report_output_path(base_dir))
    smoke_overrides = _wordle_smoke_overrides() or {}
    sft_overrides = smoke_overrides.get("sft")
    rl_overrides = smoke_overrides.get("rl")
    eval_overrides = smoke_overrides.get("eval")

    on_failure = os.getenv("TENYSON_ON_FAILURE", "abort").strip().lower() or "abort"
    modal_gpu = os.getenv("TENYSON_MODAL_GPU", "A100").strip() or "A100"
    modal_timeout = int(os.getenv("TENYSON_MODAL_TIMEOUT", "86400"))
    recovery_experiment_id = (
        str(os.getenv("TENYSON_RECOVER_EXPERIMENT_ID", "")).strip() or None
    )
    if recovery_experiment_id:
        print(
            "[wordle experiment] Recovery enabled for "
            f"experiment_id={recovery_experiment_id}.",
            flush=True,
        )

    session = ExperimentSession(
        task=task,
        templates=ConfigTemplates.from_directory(_REPO_ROOT / "config_templates"),
        cloud_factory=ModalManager.factory_from_env(
            auto_terminate=True,
            gpu=modal_gpu,
            timeout=modal_timeout,
        ),
        on_failure=on_failure,
        shared_overrides=shared_overrides_from_env(),
        parallel=True,
        report=report,
        report_metric_precision=4,
        report_wandb_text="run",
        recovery_experiment_id=recovery_experiment_id,
    )
    primary_branch = session.branch(cloud=session.create_cloud())

    try:
        primary_branch.run(
            primary_branch.sft(
                "sft_main",
                run="wordle_sft_main",
                overrides=sft_overrides,
            )
        )
        sft_adapter = primary_branch.require_adapter("sft_main")
        primary_branch.run(
            primary_branch.eval(
                "eval_baseline_mixed",
                run="wordle_eval_mixed",
                adapter=sft_adapter,
                overrides=eval_overrides,
            )
        )
        session.run_branches(
            {
                "mixed": lambda branch: (
                    branch.run(
                        branch.rl(
                            "mixed_rl",
                            run="wordle_rl_mixed",
                            adapter=sft_adapter,
                            overrides=rl_overrides,
                        )
                    ),
                    branch.run(
                        branch.eval(
                            "mixed_final_eval",
                            run="wordle_eval_mixed",
                            adapter=branch.require_adapter("mixed_rl"),
                            overrides=eval_overrides,
                        )
                    ),
                ),
                "curriculum": lambda branch: (
                    branch.run(
                        branch.rl(
                            "curr_rl_t2",
                            run="wordle_rl_turn2",
                            adapter=sft_adapter,
                            overrides=rl_overrides,
                        )
                    ),
                    branch.run(
                        branch.eval(
                            "curr_eval_after_t2_turn2",
                            run="wordle_eval_turn2",
                            adapter=branch.require_adapter("curr_rl_t2"),
                            overrides=eval_overrides,
                        )
                    ),
                    branch.run(
                        branch.rl(
                            "curr_rl_t3",
                            run="wordle_rl_turn3",
                            adapter=branch.require_adapter("curr_rl_t2"),
                            overrides=rl_overrides,
                        )
                    ),
                    branch.run_parallel(
                        "curr_eval_after_t3",
                        [
                            branch.eval(
                                "curr_eval_after_t3_turn2",
                                run="wordle_eval_turn2",
                                adapter=branch.require_adapter("curr_rl_t3"),
                                overrides=eval_overrides,
                            ),
                            branch.eval(
                                "curr_eval_after_t3_turn3",
                                run="wordle_eval_turn3",
                                adapter=branch.require_adapter("curr_rl_t3"),
                                overrides=eval_overrides,
                            ),
                        ],
                    ),
                    branch.run(
                        branch.rl(
                            "curr_rl_t4",
                            run="wordle_rl_turn4",
                            adapter=branch.require_adapter("curr_rl_t3"),
                            overrides=rl_overrides,
                        )
                    ),
                    branch.run_parallel(
                        "curr_eval_after_t4",
                        [
                            branch.eval(
                                "curr_eval_after_t4_turn3",
                                run="wordle_eval_turn3",
                                adapter=branch.require_adapter("curr_rl_t4"),
                                overrides=eval_overrides,
                            ),
                            branch.eval(
                                "curr_eval_after_t4_turn4",
                                run="wordle_eval_turn4",
                                adapter=branch.require_adapter("curr_rl_t4"),
                                overrides=eval_overrides,
                            ),
                        ],
                    ),
                    branch.run(
                        branch.rl(
                            "curr_rl_t5",
                            run="wordle_rl_turn5",
                            adapter=branch.require_adapter("curr_rl_t4"),
                            overrides=rl_overrides,
                        )
                    ),
                    branch.run_parallel(
                        "curr_eval_after_t5",
                        [
                            branch.eval(
                                "curr_eval_after_t5_turn4",
                                run="wordle_eval_turn4",
                                adapter=branch.require_adapter("curr_rl_t5"),
                                overrides=eval_overrides,
                            ),
                            branch.eval(
                                "curr_eval_after_t5_turn5",
                                run="wordle_eval_turn5",
                                adapter=branch.require_adapter("curr_rl_t5"),
                                overrides=eval_overrides,
                            ),
                        ],
                    ),
                    branch.run(
                        branch.eval(
                            "curr_final_eval",
                            run="wordle_eval_mixed",
                            adapter=branch.require_adapter("curr_rl_t5"),
                            overrides=eval_overrides,
                        )
                    ),
                ),
            }
        )
    except ExperimentAborted as exc:
        print(exc)
    finally:
        session.close()


if __name__ == "__main__":
    main()
