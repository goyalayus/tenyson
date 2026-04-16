from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from tenyson.core.control import list_live_runs
from tenyson.core.experiment_runtime import (
    DEFAULT_FUNCTIONAL_FILENAME,
    DEFAULT_RECOVERY_RESTART_STAGE_ENV_VAR,
    DEFAULT_REPORT_ENV_VAR,
    LocalExperimentContext,
    bootstrap_local_experiment,
    build_experiment_report,
    configure_experiment_identity,
    create_modal_experiment_session,
    default_experiment_prefix,
    default_report_filename,
    install_sigterm_handler,
    resolve_recovery_experiment_id,
    resolve_recovery_restart_stage_name,
    resolve_recovery_restart_stages,
)
from tenyson.core.functional import FunctionalManifest, load_functional_manifest
from tenyson.experiment import (
    ExperimentAborted,
    ExperimentBranch,
    StageSpec,
    _DeferredStageArtifactRef,
)
from tenyson.reporting.fixed import ExperimentReport


BuildFn = Callable[["ExperimentController"], Any]
BranchBuildFn = Callable[["ExperimentBranchController"], Any]
BranchPlan = BranchBuildFn | Sequence[StageSpec]
PrepareFn = Callable[[LocalExperimentContext, FunctionalManifest], Any]


class ExperimentController:
    def __init__(
        self,
        *,
        runtime: "_ExperimentRuntime",
        branch: ExperimentBranch,
    ) -> None:
        self._runtime = runtime
        self._branch = branch

    def seed(self, alias: str):
        return self._runtime.manifest.resolve_seed(alias)

    def artifact(self, stage_id: str):
        require_artifact = getattr(self._branch, "require_artifact", None)
        if callable(require_artifact):
            try:
                return require_artifact(stage_id)
            except KeyError:
                return _DeferredStageArtifactRef(stage_id=str(stage_id))
        try:
            return self._branch.require_adapter(stage_id)
        except KeyError:
            return _DeferredStageArtifactRef(stage_id=str(stage_id))

    def require_artifact(self, stage_id: str):
        return self.artifact(stage_id)

    def adapter(self, stage_id: str):
        require_adapter = getattr(self._branch, "require_adapter", None)
        if callable(require_adapter):
            try:
                return require_adapter(stage_id)
            except KeyError:
                return _DeferredStageArtifactRef(stage_id=str(stage_id))
        return self.artifact(stage_id)

    def require_adapter(self, stage_id: str):
        return self.adapter(stage_id)

    def result(self, stage_id: str):
        return self._branch.result(stage_id)

    def sft(self, stage_id: str, **kwargs: Any):
        stage = self.sft_stage(stage_id, **kwargs)
        return self._branch.run(stage)

    def rl(self, stage_id: str, **kwargs: Any):
        stage = self.rl_stage(stage_id, **kwargs)
        return self._branch.run(stage)

    def eval(self, stage_id: str, **kwargs: Any):
        stage = self.eval_stage(stage_id, **kwargs)
        return self._branch.run(stage)

    def sft_stage(self, stage_id: str, **kwargs: Any) -> StageSpec:
        return self._build_stage("sft", stage_id, **kwargs)

    def rl_stage(self, stage_id: str, **kwargs: Any) -> StageSpec:
        return self._build_stage("rl", stage_id, **kwargs)

    def eval_stage(self, stage_id: str, **kwargs: Any) -> StageSpec:
        return self._build_stage("eval", stage_id, **kwargs)

    def run_parallel(
        self,
        label: str,
        stages: Sequence[StageSpec],
    ):
        self._runtime.register_stage_specs(stages)
        return self._branch.run_parallel(label, stages)

    def run_branches(
        self,
        branches: Mapping[str, BranchPlan],
    ):
        return self._runtime.run_branches(branches)

    def _build_stage(self, run_type: str, stage_id: str, **kwargs: Any) -> StageSpec:
        params = dict(kwargs)
        run_name = params.get("run")
        if run_name is not None:
            params["run"] = self._runtime.resolve_run(run_name)
        stage = getattr(self._branch, run_type)(stage_id, **params)
        self._runtime.register_stage_specs([stage])
        return stage


class ExperimentBranchController(ExperimentController):
    pass


class _ExperimentRuntime:
    def __init__(
        self,
        *,
        manifest: FunctionalManifest,
        session: Any,
        report: ExperimentReport,
        label: str,
    ) -> None:
        self.manifest = manifest
        self.session = session
        self.report = report
        self.label = label
        self._stage_ids: list[str] = []
        self._stage_lock = threading.Lock()

    def controller(self) -> ExperimentController:
        return ExperimentController(
            runtime=self,
            branch=self.session.branch(),
        )

    def resolve_run(self, run_name: str) -> str:
        return self.manifest.resolve_run(run_name)

    def register_stage_specs(self, stages: Sequence[StageSpec]) -> None:
        with self._stage_lock:
            for stage in stages:
                if stage.id not in self._stage_ids:
                    self._stage_ids.append(stage.id)

    def stage_ids(self) -> list[str]:
        with self._stage_lock:
            return list(self._stage_ids)

    def run_branches(
        self,
        branches: Mapping[str, BranchPlan],
    ):
        prepared: dict[str, Callable[[ExperimentBranch], Any] | Sequence[StageSpec]] = {}
        for label, plan in branches.items():
            prepared[label] = self._prepare_branch_plan(plan)
        return self.session.run_branches(prepared)

    def _prepare_branch_plan(
        self,
        plan: BranchPlan,
    ) -> Callable[[ExperimentBranch], Any] | Sequence[StageSpec]:
        if callable(plan):
            return self._wrap_branch_builder(plan)
        return plan

    def _wrap_branch_builder(
        self,
        builder: BranchBuildFn,
    ) -> Callable[[ExperimentBranch], Any]:
        def _runner(branch: ExperimentBranch) -> Any:
            controller = ExperimentBranchController(runtime=self, branch=branch)
            return builder(controller)

        return _runner


@dataclass(frozen=True)
class _PlannedStage:
    id: str


@dataclass(frozen=True)
class _PlannedValue:
    kind: str
    ref: str


class _PlanningController:
    def __init__(
        self,
        *,
        runtime: "_PlanningRuntime",
    ) -> None:
        self._runtime = runtime

    def seed(self, alias: str):
        return self._runtime.manifest.resolve_seed(alias)

    def artifact(self, stage_id: str):
        return _DeferredStageArtifactRef(stage_id=str(stage_id))

    def require_artifact(self, stage_id: str):
        return self.artifact(stage_id)

    def adapter(self, stage_id: str):
        return _DeferredStageArtifactRef(stage_id=str(stage_id))

    def require_adapter(self, stage_id: str):
        return self.adapter(stage_id)

    def result(self, stage_id: str):
        return _PlannedValue(kind="result", ref=stage_id)

    def sft(self, stage_id: str, **kwargs: Any):
        return self.run(self.sft_stage(stage_id, **kwargs))

    def rl(self, stage_id: str, **kwargs: Any):
        return self.run(self.rl_stage(stage_id, **kwargs))

    def eval(self, stage_id: str, **kwargs: Any):
        return self.run(self.eval_stage(stage_id, **kwargs))

    def sft_stage(self, stage_id: str, **kwargs: Any) -> _PlannedStage:
        return self._build_stage(stage_id, **kwargs)

    def rl_stage(self, stage_id: str, **kwargs: Any) -> _PlannedStage:
        return self._build_stage(stage_id, **kwargs)

    def eval_stage(self, stage_id: str, **kwargs: Any) -> _PlannedStage:
        return self._build_stage(stage_id, **kwargs)

    def run(self, stage: _PlannedStage):
        return _PlannedValue(kind="run", ref=_planned_stage_id(stage))

    def run_parallel(
        self,
        label: str,
        stages: Sequence[object],
    ):
        del label
        return {
            _planned_stage_id(stage): _PlannedValue(
                kind="run",
                ref=_planned_stage_id(stage),
            )
            for stage in stages
        }

    def run_branches(
        self,
        branches: Mapping[str, BranchPlan | Sequence[object]],
    ):
        return self._runtime.run_branches(branches)

    def _build_stage(self, stage_id: str, **kwargs: Any) -> _PlannedStage:
        run_name = kwargs.get("run")
        if run_name is not None:
            self._runtime.resolve_run(run_name)
        self._runtime.register_stage_id(stage_id)
        return _PlannedStage(id=stage_id)


class _PlanningBranchController(_PlanningController):
    pass


class _PlanningRuntime:
    def __init__(self, *, manifest: FunctionalManifest) -> None:
        self.manifest = manifest
        self._stage_ids: list[str] = []

    def controller(self) -> _PlanningController:
        return _PlanningController(runtime=self)

    def resolve_run(self, run_name: str) -> str:
        return self.manifest.resolve_run(run_name)

    def register_stage_id(self, stage_id: str) -> None:
        stage_key = str(stage_id).strip()
        if stage_key and stage_key not in self._stage_ids:
            self._stage_ids.append(stage_key)

    def stage_ids(self) -> list[str]:
        return list(self._stage_ids)

    def run_branches(
        self,
        branches: Mapping[str, BranchPlan | Sequence[object]],
    ):
        results: dict[str, Any] = {}
        for label, plan in branches.items():
            if callable(plan):
                results[label] = plan(_PlanningBranchController(runtime=self))
                continue
            results[label] = {
                _planned_stage_id(stage): _PlannedValue(
                    kind="run",
                    ref=_planned_stage_id(stage),
                )
                for stage in plan
            }
        return results


def run_experiment(
    anchor_file: str | Path,
    build: BuildFn,
    *,
    functional_filename: str = DEFAULT_FUNCTIONAL_FILENAME,
    report_env_var: str = DEFAULT_REPORT_ENV_VAR,
    prepare: Optional[PrepareFn] = None,
    recovery_restart_stage_env_var: str = DEFAULT_RECOVERY_RESTART_STAGE_ENV_VAR,
    recovery_restart_stage_fallback_env_vars: Sequence[str] = (),
) -> Any:
    context = bootstrap_local_experiment(anchor_file)
    functional_path = context.file(functional_filename)
    if not functional_path.is_file():
        raise FileNotFoundError(
            f"Expected experiment manifest at {functional_path}, found none."
        )

    manifest = load_functional_manifest(functional_path)
    if prepare is not None:
        prepare(context, manifest)
    label = _experiment_label(manifest.task, context.anchor_file)
    forced_stop_requested = {"value": False}
    recovery_restart_stage_name = resolve_recovery_restart_stage_name(
        env_var=recovery_restart_stage_env_var,
        fallback_env_vars=recovery_restart_stage_fallback_env_vars,
    )
    recovery_restart_stages: list[str] = []
    if recovery_restart_stage_name:
        recovery_restart_stages = resolve_recovery_restart_stages(
            ordered_stage_ids=_plan_stage_ids(build=build, manifest=manifest),
            env_var=recovery_restart_stage_env_var,
            fallback_env_vars=recovery_restart_stage_fallback_env_vars,
        )

    def _mark_forced_stop(_signum: int) -> None:
        forced_stop_requested["value"] = True

    install_sigterm_handler(label=label, on_signal=_mark_forced_stop)
    configure_experiment_identity(
        context=context,
        default_experiment_prefix=default_experiment_prefix(
            environment_name=str(
                getattr(manifest.task, "get_environment_name", lambda: None)()
                or "experiment"
            ).strip()
            or "experiment",
            anchor_file=context.anchor_file,
        ),
        report_env_var=report_env_var,
        default_report_filename=default_report_filename(context.anchor_file),
        regenerate_when_loaded_experiment_id=True,
    )

    report = build_experiment_report(
        context=context,
        report_env_var=report_env_var,
    )
    recovery_experiment_id = resolve_recovery_experiment_id()
    if recovery_experiment_id:
        print(
            f"[{label}] Recovery enabled for experiment_id={recovery_experiment_id}.",
            flush=True,
        )
    if recovery_restart_stages:
        print(
            f"[{label}] Recovery will restart stages: {', '.join(recovery_restart_stages)}.",
            flush=True,
        )

    session = create_modal_experiment_session(
        context=context,
        task=manifest.task,
        report=report,
        recovery_experiment_id=recovery_experiment_id,
        recovery_restart_stages=recovery_restart_stages,
    )
    runtime = _ExperimentRuntime(
        manifest=manifest,
        session=session,
        report=report,
        label=label,
    )

    try:
        return build(runtime.controller())
    except ExperimentAborted as exc:
        print(exc)
        return None
    except KeyboardInterrupt:
        print(f"[{label}] Interrupted.", flush=True)
        return None
    finally:
        session.close()
        stage_ids = runtime.stage_ids()
        if forced_stop_requested["value"]:
            _wait_for_live_runs_to_finish(
                report,
                tracked_stage_ids=stage_ids,
                timeout_seconds=60.0,
            )
            _best_effort_rebuild_report_from_telemetry(
                report,
                manifest.task,
                tracked_stage_ids=stage_ids,
                timeout_seconds=5.0,
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        try:
            _rebuild_report_from_telemetry(
                report,
                manifest.task,
                tracked_stage_ids=stage_ids,
            )
        except KeyboardInterrupt:
            print(
                f"[{label}] Report rebuild interrupted; keeping the latest local report.",
                flush=True,
            )


def _plan_stage_ids(
    *,
    build: BuildFn,
    manifest: FunctionalManifest,
) -> list[str]:
    planning_runtime = _PlanningRuntime(manifest=manifest)
    build(planning_runtime.controller())
    return planning_runtime.stage_ids()


def _planned_stage_id(stage: object) -> str:
    stage_id = str(getattr(stage, "id", "") or getattr(stage, "ref", "")).strip()
    if not stage_id:
        raise ValueError(
            "Planning controller expected a stage-like object with an id or ref."
        )
    return stage_id


def _experiment_label(task: Any, anchor_file: Path) -> str:
    environment_name = str(getattr(task, "get_environment_name", lambda: None)() or "tenyson").strip()
    stem = str(anchor_file.stem or "experiment").strip().replace("_", " ")
    return f"{environment_name} {stem}".strip()


def _report_backend_ref(report: ExperimentReport) -> str:
    existing = str(getattr(report, "telemetry_backend_ref", "") or "").strip()
    if existing:
        return existing
    entity = str(os.getenv("TENYSON_WANDB_ENTITY", "")).strip()
    project = str(os.getenv("TENYSON_WANDB_PROJECT", "")).strip()
    if entity and project:
        return f"wandb://{entity}/{project}"
    return ""


def _rebuild_report_from_telemetry(
    report: ExperimentReport,
    task: Any,
    *,
    tracked_stage_ids: Sequence[str],
) -> None:
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

    report.rebuild_from_telemetry(
        backend_ref=backend_ref,
        experiment_id=experiment_id,
        environment_name=environment_name,
        run_name_allowlist=list(tracked_stage_ids) or None,
        prefer_terminal_results=True,
    )
    print(
        "[TENYSON] Rebuilt final report from telemetry at "
        f"{report.output_path}.",
        flush=True,
    )


def _best_effort_rebuild_report_from_telemetry(
    report: ExperimentReport,
    task: Any,
    *,
    tracked_stage_ids: Sequence[str],
    timeout_seconds: float,
) -> None:
    outcome: dict[str, BaseException | None] = {"error": None}

    def _target() -> None:
        try:
            _rebuild_report_from_telemetry(
                report,
                task,
                tracked_stage_ids=tracked_stage_ids,
            )
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    worker = threading.Thread(
        target=_target,
        daemon=True,
        name="tenyson-report-rebuild",
    )
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        print(
            "[TENYSON] Report rebuild timed out during forced stop; keeping the latest local report.",
            flush=True,
        )
        return
    error = outcome["error"]
    if isinstance(error, KeyboardInterrupt):
        print(
            "[TENYSON] Report rebuild interrupted; keeping the latest local report.",
            flush=True,
        )
        return
    if error is not None:
        raise error


def _wait_for_live_runs_to_finish(
    report: ExperimentReport,
    *,
    tracked_stage_ids: Sequence[str],
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

    tracked_run_ids = {str(stage_id).strip() for stage_id in tracked_stage_ids if str(stage_id).strip()}
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
                "[TENYSON] Warning: failed to poll live runs during forced-stop shutdown: "
                f"{exc}",
                flush=True,
            )
            return

        if tracked_run_ids:
            tracked_live_rows = [
                row
                for row in live_rows
                if str(getattr(row, "run_id", "") or "").strip() in tracked_run_ids
            ]
        else:
            tracked_live_rows = list(live_rows)
        if not tracked_live_rows:
            return
        time.sleep(max(0.1, float(poll_interval_seconds)))

    print(
        "[TENYSON] Timed out waiting for live runs to finish during forced stop.",
        flush=True,
    )
