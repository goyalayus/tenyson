import importlib.util
import os
from pathlib import Path
import subprocess
import sys


_LOCAL_BOOTSTRAP_PACKAGES = {
    "boto3": "boto3",
    "datasets": "datasets",
    "huggingface_hub": "huggingface_hub",
    "modal": "modal",
    "psycopg": "psycopg[binary]",
    "sqlalchemy": "sqlalchemy",
    "wandb": "wandb",
    "yaml": "pyyaml",
}


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _bootstrap_local_environment() -> None:
    """
    Ensure this script can run from a fresh checkout without manual local setup.

    - Adds the local src/ directory to PYTHONPATH for src-layout imports.
    - Installs missing controller-side dependencies (Modal + telemetry + task loading).
    """
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    if _is_truthy(os.getenv("TENYSON_SKIP_LOCAL_BOOTSTRAP", "false")):
        return

    missing_packages = [
        package
        for module_name, package in _LOCAL_BOOTSTRAP_PACKAGES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if not missing_packages:
        return

    print(
        "[TENYSON] Installing missing local dependencies: "
        + ", ".join(missing_packages),
        flush=True,
    )
    install_cmd = [sys.executable, "-m", "pip", "install", *missing_packages]
    result = subprocess.run(install_cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Automatic local dependency bootstrap failed. "
            "Re-run after installing the missing packages manually."
        )


_bootstrap_local_environment()

from tenyson.cloud.modal import ModalManager
from tenyson.experiment import ConfigTemplates, ExperimentAborted, ExperimentSession
from tenyson.loader import load_task
from tenyson.reporting.fixed import ExperimentReport


def _build_shared_overrides() -> dict | None:
    hf_repo_base = str(os.getenv("TENYSON_HF_REPO_BASE") or "").strip()
    experiment_id = str(os.getenv("TENYSON_EXPERIMENT_ID") or "").strip()

    shared_overrides = {}
    if hf_repo_base:
        shared_overrides.setdefault("training", {})["hf_repo_base"] = hf_repo_base
    if experiment_id:
        shared_overrides.setdefault("telemetry", {})["experiment_id"] = experiment_id
    return shared_overrides or None


def _run_mixed_branch(branch, *, sft_adapter) -> None:
    branch.run(
        branch.rl(
            "mixed_rl",
            run="wordle_rl_mixed",
            run_name="wordle_rl_mixed",
            adapter=sft_adapter,
        )
    )
    branch.run(
        branch.eval(
            "mixed_final_eval",
            run="wordle_eval_mixed",
            run_name="wordle_mixed_final_eval",
            adapter=branch.require_adapter("mixed_rl"),
        )
    )


def _run_curriculum_branch(branch, *, sft_adapter) -> None:
    branch.run(
        branch.rl(
            "curr_rl_t2",
            run="wordle_rl_turn2",
            run_name="wordle_curriculum_rl_t2",
            adapter=sft_adapter,
        )
    )
    branch.run(
        branch.eval(
            "curr_eval_after_t2_turn2",
            run="wordle_eval_turn2",
            run_name="wordle_curr_eval_after_t2_turn2",
            adapter=branch.require_adapter("curr_rl_t2"),
        )
    )

    branch.run(
        branch.rl(
            "curr_rl_t3",
            run="wordle_rl_turn3",
            run_name="wordle_curriculum_rl_t3",
            adapter=branch.require_adapter("curr_rl_t2"),
        )
    )
    branch.run_parallel(
        "curr_eval_after_t3",
        [
            branch.eval(
                "curr_eval_after_t3_turn2",
                run="wordle_eval_turn2",
                run_name="wordle_curr_eval_after_t3_turn2",
                adapter=branch.require_adapter("curr_rl_t3"),
            ),
            branch.eval(
                "curr_eval_after_t3_turn3",
                run="wordle_eval_turn3",
                run_name="wordle_curr_eval_after_t3_turn3",
                adapter=branch.require_adapter("curr_rl_t3"),
            ),
        ],
    )

    branch.run(
        branch.rl(
            "curr_rl_t4",
            run="wordle_rl_turn4",
            run_name="wordle_curriculum_rl_t4",
            adapter=branch.require_adapter("curr_rl_t3"),
        )
    )
    branch.run_parallel(
        "curr_eval_after_t4",
        [
            branch.eval(
                "curr_eval_after_t4_turn3",
                run="wordle_eval_turn3",
                run_name="wordle_curr_eval_after_t4_turn3",
                adapter=branch.require_adapter("curr_rl_t4"),
            ),
            branch.eval(
                "curr_eval_after_t4_turn4",
                run="wordle_eval_turn4",
                run_name="wordle_curr_eval_after_t4_turn4",
                adapter=branch.require_adapter("curr_rl_t4"),
            ),
        ],
    )

    branch.run(
        branch.rl(
            "curr_rl_t5",
            run="wordle_rl_turn5",
            run_name="wordle_curriculum_rl_t5",
            adapter=branch.require_adapter("curr_rl_t4"),
        )
    )
    branch.run_parallel(
        "curr_eval_after_t5",
        [
            branch.eval(
                "curr_eval_after_t5_turn4",
                run="wordle_eval_turn4",
                run_name="wordle_curr_eval_after_t5_turn4",
                adapter=branch.require_adapter("curr_rl_t5"),
            ),
            branch.eval(
                "curr_eval_after_t5_turn5",
                run="wordle_eval_turn5",
                run_name="wordle_curr_eval_after_t5_turn5",
                adapter=branch.require_adapter("curr_rl_t5"),
            ),
        ],
    )

    branch.run(
        branch.eval(
            "curr_final_eval",
            run="wordle_eval_mixed",
            run_name="wordle_curriculum_final_eval",
            adapter=branch.require_adapter("curr_rl_t5"),
        )
    )


def main() -> None:
    base_dir = Path(__file__).parent
    task = load_task(str(base_dir / "wordle_task.py"))
    report = ExperimentReport(output_path=base_dir / "final_report.md")

    on_failure = os.getenv("TENYSON_ON_FAILURE", "abort").strip().lower() or "abort"
    modal_gpu = os.getenv("TENYSON_MODAL_GPU", "A100").strip() or "A100"
    modal_timeout = int(os.getenv("TENYSON_MODAL_TIMEOUT", "86400"))
    modal_profile = os.getenv("TENYSON_MODAL_PROFILE") or os.getenv("MODAL_PROFILE")
    disable_parallel = os.getenv(
        "TENYSON_DISABLE_PARALLEL", "false"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    session = ExperimentSession(
        task=task,
        templates=ConfigTemplates.from_directory(base_dir / "configs"),
        cloud_factory=ModalManager.factory_from_env(
            auto_terminate=True,
            gpu=modal_gpu,
            timeout=modal_timeout,
            profile=modal_profile,
        ),
        on_failure=on_failure,
        shared_overrides=_build_shared_overrides(),
        parallel=not disable_parallel,
        report=report,
        report_metric_precision=4,
        report_wandb_text="run",
    )
    primary_branch = session.branch(cloud=session.create_cloud())

    try:
        primary_branch.run(
            primary_branch.sft(
                "sft_main",
                run="wordle_sft_main",
                run_name="wordle_sft_main",
            )
        )
        sft_adapter = primary_branch.require_adapter("sft_main")
        primary_branch.run(
            primary_branch.eval(
                "eval_baseline_mixed",
                run="wordle_eval_mixed",
                run_name="wordle_eval_baseline_mixed",
                adapter=sft_adapter,
            )
        )
        session.run_branches(
            {
                "mixed": lambda branch: _run_mixed_branch(
                    branch,
                    sft_adapter=sft_adapter,
                ),
                "curriculum": lambda branch: _run_curriculum_branch(
                    branch,
                    sft_adapter=sft_adapter,
                ),
            }
        )
    except ExperimentAborted as exc:
        print(exc)
    finally:
        session.close()


if __name__ == "__main__":
    main()
