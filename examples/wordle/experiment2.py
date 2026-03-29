from pathlib import Path
import sys

# src-layout convenience for running this file directly from a fresh checkout.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from common import (
    WORDLE_EXPERIMENT2_PREFIX,
    WORDLE_EXPERIMENT2_REPORT_ENV_VAR,
    WORDLE_EXPERIMENT2_REPORT_FILENAME,
    load_wordle_task,
    resolve_wordle_experiment2_seed_adapter,
    wordle_smoke_overrides,
)
from tenyson.core.experiment_runtime import (
    bootstrap_local_experiment,
    build_experiment_report,
    configure_experiment_identity,
    create_modal_experiment_session,
    install_sigterm_handler,
    resolve_recovery_experiment_id,
)
from tenyson.experiment import ExperimentAborted


def main() -> None:
    context = bootstrap_local_experiment(__file__)
    install_sigterm_handler(label="wordle experiment2")
    configure_experiment_identity(
        context=context,
        default_experiment_prefix=WORDLE_EXPERIMENT2_PREFIX,
        report_env_var=WORDLE_EXPERIMENT2_REPORT_ENV_VAR,
        default_report_filename=WORDLE_EXPERIMENT2_REPORT_FILENAME,
        regenerate_when_loaded_experiment_id=True,
    )

    task = load_wordle_task(context)
    report = build_experiment_report(
        context=context,
        report_env_var=WORDLE_EXPERIMENT2_REPORT_ENV_VAR,
        default_filename=WORDLE_EXPERIMENT2_REPORT_FILENAME,
    )
    smoke_overrides = (
        wordle_smoke_overrides(include_sft=False, label="wordle experiment2") or {}
    )
    rl_overrides = smoke_overrides.get("rl")
    eval_overrides = smoke_overrides.get("eval")

    recovery_experiment_id = resolve_recovery_experiment_id()
    if recovery_experiment_id:
        print(
            "[wordle experiment2] Recovery enabled for "
            f"experiment_id={recovery_experiment_id}.",
            flush=True,
        )

    session = create_modal_experiment_session(
        context=context,
        task=task,
        report=report,
        recovery_experiment_id=recovery_experiment_id,
    )
    branch = session.branch()
    seed_adapter = resolve_wordle_experiment2_seed_adapter(label="wordle experiment2")

    try:
        branch.run(
            branch.eval(
                "baseline_eval_turn6",
                run="wordle_eval_turn6_prime",
                adapter=seed_adapter,
                overrides=eval_overrides,
            )
        )
        branch.run(
            branch.rl(
                "turn6_rl",
                run="wordle_rl_turn6_prime",
                adapter=seed_adapter,
                overrides=rl_overrides,
            )
        )
        branch.run(
            branch.eval(
                "final_eval_turn6",
                run="wordle_eval_turn6_prime",
                adapter=branch.require_adapter("turn6_rl"),
                overrides=eval_overrides,
            )
        )
    except ExperimentAborted as exc:
        print(exc)
    except KeyboardInterrupt:
        print("[wordle experiment2] Interrupted.", flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
