from pathlib import Path
import os
import sys
import threading
import time

# src-layout convenience for running this file directly from a fresh checkout.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from common import (
    WORDLE_REPORT_ENV_VAR,
    WORDLE_REPORT_FILENAME,
    configure_wordle_smoke_identity,
    load_wordle_task,
    wordle_recovery_restart_stages,
    wordle_report_stage_order,
    wordle_smoke_overrides,
)
from tenyson.core.control import list_live_runs
from tenyson.core.experiment_runtime import (
    bootstrap_local_experiment,
    build_experiment_report,
    create_modal_experiment_session,
    install_sigterm_handler,
    resolve_recovery_experiment_id,
)
from tenyson.experiment import ExperimentAborted
from tenyson.reporting.fixed import ExperimentReport


_FORCED_STOP_REQUESTED = False


def _mark_forced_stop(_signum: int) -> None:
    global _FORCED_STOP_REQUESTED
    _FORCED_STOP_REQUESTED = True


def _report_backend_ref(report: ExperimentReport) -> str:
    existing = str(getattr(report, "telemetry_backend_ref", "") or "").strip()
    if existing:
        return existing
    entity = str(os.getenv("TENYSON_WANDB_ENTITY", "")).strip()
    project = str(os.getenv("TENYSON_WANDB_PROJECT", "")).strip()
    if entity and project:
        return f"wandb://{entity}/{project}"
    return ""


def _rebuild_report_from_telemetry(report: ExperimentReport, task: object) -> None:
    experiment_id = str(
        getattr(report, "experiment_id", None)
        or os.getenv("TENYSON_EXPERIMENT_ID", "")
        or ""
    ).strip()
    backend_ref = _report_backend_ref(report)
    if not experiment_id or not backend_ref:
        return

    environment_name = None
    if hasattr(task, "get_environment_name"):
        try:
            environment_name = task.get_environment_name()
        except Exception:  # noqa: BLE001
            environment_name = None

    try:
        report.rebuild_from_telemetry(
            backend_ref=backend_ref,
            experiment_id=experiment_id,
            environment_name=environment_name,
            run_name_allowlist=wordle_report_stage_order(),
            prefer_terminal_results=True,
        )
        print(
            "[wordle experiment] Rebuilt final report from telemetry at "
            f"{report.output_path}.",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            "[wordle experiment] Warning: failed to rebuild final report from "
            f"telemetry: {exc}",
            flush=True,
        )


def _best_effort_rebuild_report_from_telemetry(
    report: ExperimentReport,
    task: object,
    *,
    timeout_seconds: float,
) -> None:
    outcome: dict[str, BaseException | None] = {"error": None}

    def _target() -> None:
        try:
            _rebuild_report_from_telemetry(report, task)
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    worker = threading.Thread(
        target=_target,
        daemon=True,
        name="wordle-report-rebuild",
    )
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        print(
            "[wordle experiment] Report rebuild timed out during forced stop; keeping the latest local report.",
            flush=True,
        )
        return
    error = outcome["error"]
    if isinstance(error, KeyboardInterrupt):
        print(
            "[wordle experiment] Report rebuild interrupted; keeping the latest local report.",
            flush=True,
        )
        return
    if error is not None:
        raise error


def _wait_for_wordle_live_runs_to_finish(
    report: ExperimentReport,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 2.0,
) -> None:
    experiment_id = str(
        getattr(report, "experiment_id", None)
        or os.getenv("TENYSON_EXPERIMENT_ID", "")
        or ""
    ).strip()
    backend_ref = _report_backend_ref(report)
    if not experiment_id or not backend_ref:
        return

    tracked_run_ids = set(wordle_report_stage_order())
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    max_age_seconds = max(30, int(timeout_seconds) + 30)

    while time.monotonic() < deadline:
        try:
            live_rows = list_live_runs(
                db_url=backend_ref,
                experiment_id=experiment_id,
                max_age_seconds=max_age_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "[wordle experiment] Warning: failed to poll live runs during "
                f"forced-stop shutdown: {exc}",
                flush=True,
            )
            return

        tracked_live_rows = [
            row for row in live_rows if str(row.run_id or "").strip() in tracked_run_ids
        ]
        if not tracked_live_rows:
            return
        time.sleep(max(0.1, float(poll_interval_seconds)))

    print(
        "[wordle experiment] Timed out waiting for live runs to finish during forced stop.",
        flush=True,
    )


def main() -> None:
    context = bootstrap_local_experiment(__file__)
    install_sigterm_handler(
        label="wordle experiment",
        on_signal=_mark_forced_stop,
    )
    configure_wordle_smoke_identity(
        context=context,
        label="wordle experiment",
    )

    task = load_wordle_task(context)
    report = build_experiment_report(
        context=context,
        report_env_var=WORDLE_REPORT_ENV_VAR,
        default_filename=WORDLE_REPORT_FILENAME,
    )
    smoke_overrides = wordle_smoke_overrides(
        include_sft=True,
        label="wordle experiment",
    ) or {}
    sft_overrides = smoke_overrides.get("sft")
    rl_overrides = smoke_overrides.get("rl")
    eval_overrides = smoke_overrides.get("eval")

    recovery_experiment_id = resolve_recovery_experiment_id()
    if recovery_experiment_id:
        print(
            "[wordle experiment] Recovery enabled for "
            f"experiment_id={recovery_experiment_id}.",
            flush=True,
        )
    recovery_restart_stages = wordle_recovery_restart_stages()
    if recovery_restart_stages:
        print(
            "[wordle experiment] Recovery will restart stages: "
            f"{', '.join(recovery_restart_stages)}.",
            flush=True,
        )

    session = create_modal_experiment_session(
        context=context,
        task=task,
        report=report,
        recovery_experiment_id=recovery_experiment_id,
        recovery_restart_stages=recovery_restart_stages,
    )
    primary_branch = session.branch()

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
    except KeyboardInterrupt:
        print("[wordle experiment] Interrupted.", flush=True)
    finally:
        session.close()
        if _FORCED_STOP_REQUESTED:
            _wait_for_wordle_live_runs_to_finish(
                report,
                timeout_seconds=60.0,
            )
            _best_effort_rebuild_report_from_telemetry(
                report,
                task,
                timeout_seconds=5.0,
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        try:
            _rebuild_report_from_telemetry(report, task)
        except KeyboardInterrupt:
            print(
                "[wordle experiment] Report rebuild interrupted; keeping the latest local report.",
                flush=True,
            )


if __name__ == "__main__":
    main()
