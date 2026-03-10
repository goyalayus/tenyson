from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from tenyson.cloud.base import _red_print
from tenyson.core.control import request_stop
from tenyson.core.telemetry import resolve_telemetry_context
from tenyson.jobs.eval import EvalJob
from tenyson.jobs.result import JobResult
from tenyson.jobs.rl import RLJob
from tenyson.jobs.sft import SFTJob
from tenyson.loader import load_config
from tenyson.pipeline import run_pipeline


_DEFAULT_ABORT_MESSAGE = (
    "[TENYSON] Experiment aborted. Skipping remaining stages and final outputs."
)


def _deep_merge(base: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _is_retryable_failure(result: JobResult) -> bool:
    return str(getattr(result, "status", "") or "").lower() not in {
        "success",
        "partial",
    }


class ExperimentAborted(RuntimeError):
    pass


@dataclass(frozen=True)
class AdapterRef:
    repo_id: str
    revision: str


@dataclass(frozen=True)
class StageSpec:
    id: str
    config: Dict[str, Any]
    job_class: type
    task: Any

    def as_pipeline_step(self) -> Tuple[str, Dict[str, Any], type, Any]:
        return (self.id, self.config, self.job_class, self.task)


class ConfigTemplates:
    def __init__(self, templates: Mapping[str, Dict[str, Any]]) -> None:
        self._templates = {name: deepcopy(config) for name, config in templates.items()}

    @classmethod
    def from_paths(cls, **named_paths: str | Path) -> "ConfigTemplates":
        templates = {name: load_config(str(path)) for name, path in named_paths.items()}
        return cls(templates)

    @classmethod
    def from_directory(
        cls,
        config_dir: str | Path,
        *,
        sft: str = "sft_config.yaml",
        rl: str = "rl_config.yaml",
        eval: str = "eval_config.yaml",
    ) -> "ConfigTemplates":
        config_root = Path(config_dir)
        return cls.from_paths(
            sft=config_root / sft,
            rl=config_root / rl,
            eval=config_root / eval,
        )

    def clone(self, name: str) -> Dict[str, Any]:
        if name not in self._templates:
            raise KeyError(f"Unknown config template: {name}")
        return deepcopy(self._templates[name])


@dataclass(frozen=True)
class _ActiveRun:
    run_id: str
    db_url: Optional[str]
    experiment_id: Optional[str]
    label: str


class _ExperimentAbortController:
    def __init__(self) -> None:
        import threading

        self._event = threading.Event()
        self._lock = threading.Lock()
        self._active_runs: Dict[str, _ActiveRun] = {}

    def is_set(self) -> bool:
        return self._event.is_set()

    def register_stage(self, stage: StageSpec) -> Optional[str]:
        run_id = _resolve_stage_run_name(stage)
        if not run_id:
            return None
        db_url, experiment_id = resolve_telemetry_context(stage.config)
        with self._lock:
            self._active_runs[run_id] = _ActiveRun(
                run_id=run_id,
                db_url=db_url,
                experiment_id=experiment_id,
                label=stage.id,
            )
        return run_id

    def unregister_run(self, run_id: Optional[str]) -> None:
        if not run_id:
            return
        with self._lock:
            self._active_runs.pop(run_id, None)

    def request_abort(
        self,
        *,
        source_run_id: Optional[str],
        source_label: str,
    ) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            active_runs = [
                active_run
                for run_id, active_run in self._active_runs.items()
                if run_id != str(source_run_id or "")
            ]

        _red_print(
            f'[TENYSON] Abort selected in stage "{source_label}". '
            f"Stopping experiment and signalling {len(active_runs)} active sibling run(s)."
        )
        for active_run in active_runs:
            if not active_run.db_url or not active_run.experiment_id:
                continue
            try:
                request_stop(
                    db_url=active_run.db_url,
                    run_id=active_run.run_id,
                    experiment_id=active_run.experiment_id,
                    create_if_missing=True,
                )
                _red_print(
                    f'[TENYSON] Requested stop for sibling stage "{active_run.label}" '
                    f"(run_id={active_run.run_id})."
                )
            except Exception as exc:  # noqa: BLE001
                _red_print(
                    f"[TENYSON] Warning: failed to request stop for sibling stage "
                    f'"{active_run.label}" (run_id={active_run.run_id}): {exc}'
                )


def _resolve_stage_run_name(stage: StageSpec) -> str:
    run_id = stage.config.get("training", {}).get("run_name") or stage.config.get(
        "evaluation", {}
    ).get("run_name")
    run_id = str(run_id or "").strip()
    if not run_id:
        raise ValueError(f'Stage "{stage.id}" is missing a run_name in its config.')
    return run_id


class ExperimentBranch:
    def __init__(
        self, session: "ExperimentSession", *, cloud: Optional[Any] = None
    ) -> None:
        self.session = session
        self._cloud = cloud
        self._results: Dict[str, JobResult] = {}

    def sft(self, stage_id: str, **kwargs: Any) -> StageSpec:
        return self.session.sft(stage_id, **kwargs)

    def rl(self, stage_id: str, **kwargs: Any) -> StageSpec:
        return self.session.rl(stage_id, **kwargs)

    def eval(self, stage_id: str, **kwargs: Any) -> StageSpec:
        return self.session.eval(stage_id, **kwargs)

    def run(self, stage: StageSpec) -> JobResult:
        result = self.session.run_stage(stage, cloud=self._resolve_cloud())
        self._store_result(stage.id, result)
        self.session.raise_if_aborted()
        return result

    def run_parallel(
        self,
        label: str,
        stages: Sequence[StageSpec],
    ) -> Dict[str, JobResult]:
        if not self.session.parallel:
            results: Dict[str, JobResult] = {}
            for stage in stages:
                result = self.session.run_stage(stage, cloud=self._resolve_cloud())
                results[stage.id] = result
                self._store_result(stage.id, result)
            self.session.raise_if_aborted()
            return results

        results = self.session.run_parallel(label, stages, cloud=self._resolve_cloud())
        for stage in stages:
            self._store_result(stage.id, results[stage.id])
        self.session.raise_if_aborted()
        return results

    def result(self, stage_id: str) -> JobResult:
        if stage_id not in self._results:
            raise KeyError(f'No result recorded for stage "{stage_id}".')
        return self._results[stage_id]

    def require_adapter(self, stage_id: str) -> AdapterRef:
        return self.session.require_adapter(self.result(stage_id), stage_id)

    def results(self) -> Dict[str, JobResult]:
        return dict(self._results)

    def _resolve_cloud(self) -> Any:
        if self._cloud is None:
            self._cloud = self.session.create_cloud()
        return self._cloud

    def _store_result(self, stage_id: str, result: JobResult) -> None:
        if stage_id in self._results:
            raise ValueError(f'Duplicate branch result for stage "{stage_id}".')
        self._results[stage_id] = result


class ExperimentSession:
    def __init__(
        self,
        *,
        task: Any,
        templates: ConfigTemplates,
        cloud_factory: Optional[Callable[[], Any]] = None,
        on_failure: str = "wait",
        shared_overrides: Optional[Mapping[str, Any]] = None,
        abort_message: str = _DEFAULT_ABORT_MESSAGE,
        parallel: bool = True,
    ) -> None:
        self.task = task
        self.templates = templates
        self.cloud_factory = cloud_factory
        self.on_failure = on_failure
        self.shared_overrides = deepcopy(shared_overrides) if shared_overrides else None
        self.abort_message = abort_message
        self.parallel = parallel
        self._abort_controller = _ExperimentAbortController()

    @property
    def aborted(self) -> bool:
        return self._abort_controller.is_set()

    def create_cloud(self) -> Any:
        if self.cloud_factory is None:
            raise RuntimeError("This experiment session has no cloud_factory.")
        return self.cloud_factory()

    def branch(self, *, cloud: Optional[Any] = None) -> ExperimentBranch:
        self.raise_if_aborted()
        return ExperimentBranch(self, cloud=cloud)

    def raise_if_aborted(self) -> None:
        if self.aborted:
            raise ExperimentAborted(self.abort_message)

    def run_branches(
        self,
        branches: Mapping[str, Callable[[ExperimentBranch], None]],
    ) -> Dict[str, Dict[str, JobResult]]:
        self.raise_if_aborted()
        if not branches:
            return {}

        def _run_branch(
            runner: Callable[[ExperimentBranch], None],
        ) -> Dict[str, JobResult]:
            branch = self.branch()
            runner(branch)
            return branch.results()

        if not self.parallel:
            completed: Dict[str, Dict[str, JobResult]] = {}
            for label, runner in branches.items():
                completed[label] = _run_branch(runner)
            return completed

        completed: Dict[str, Dict[str, JobResult]] = {}
        with ThreadPoolExecutor(max_workers=len(branches)) as executor:
            future_to_label = {
                executor.submit(_run_branch, runner): label
                for label, runner in branches.items()
            }
            for future in as_completed(future_to_label):
                label = future_to_label[future]
                completed[label] = future.result()
        return {label: completed[label] for label in branches}

    @staticmethod
    def combine_results(
        *result_groups: Mapping[str, JobResult],
    ) -> Dict[str, JobResult]:
        combined: Dict[str, JobResult] = {}
        for group in result_groups:
            for stage_id, result in group.items():
                if stage_id in combined:
                    raise ValueError(
                        f'Duplicate stage id "{stage_id}" while combining experiment results.'
                    )
                combined[stage_id] = result
        return combined

    def sft(
        self,
        stage_id: str,
        *,
        base: str = "sft",
        run_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> StageSpec:
        return self._build_stage(
            stage_id=stage_id,
            base=base,
            job_class=SFTJob,
            run_name=run_name,
            output_dir=output_dir,
            overrides=overrides,
            adapter=None,
            run_section="training",
        )

    def rl(
        self,
        stage_id: str,
        *,
        adapter: AdapterRef,
        base: str = "rl",
        run_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> StageSpec:
        return self._build_stage(
            stage_id=stage_id,
            base=base,
            job_class=RLJob,
            run_name=run_name,
            output_dir=output_dir,
            overrides=overrides,
            adapter=adapter,
            run_section="training",
        )

    def eval(
        self,
        stage_id: str,
        *,
        adapter: AdapterRef,
        base: str = "eval",
        run_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> StageSpec:
        return self._build_stage(
            stage_id=stage_id,
            base=base,
            job_class=EvalJob,
            run_name=run_name,
            output_dir=output_dir,
            overrides=overrides,
            adapter=adapter,
            run_section="evaluation",
        )

    def run_stage(self, stage: StageSpec, *, cloud: Optional[Any] = None) -> JobResult:
        self.raise_if_aborted()
        run_id = _resolve_stage_run_name(stage)
        active_run_id = self._abort_controller.register_stage(stage)
        try:
            cloud_instance = cloud if cloud is not None else self.create_cloud()
            results = run_pipeline(
                [stage.as_pipeline_step()],
                cloud_instance,
                on_failure=self.on_failure,
            )
            matched_result = None
            for result in reversed(results):
                if result.run_id == run_id:
                    matched_result = result
                    break
            if matched_result is None:
                returned_run_ids = [result.run_id for result in results]
                raise RuntimeError(
                    f'Stage "{stage.id}" expected run_id "{run_id}" but no exact result was returned. '
                    f"Returned run_ids: {returned_run_ids!r}"
                )
        finally:
            self._abort_controller.unregister_run(active_run_id)

        if _is_retryable_failure(matched_result):
            self._abort_controller.request_abort(
                source_run_id=run_id,
                source_label=stage.id,
            )
        return matched_result

    def run_parallel(
        self,
        label: str,
        stages: Sequence[StageSpec],
        *,
        cloud: Optional[Any] = None,
    ) -> Dict[str, JobResult]:
        self.raise_if_aborted()
        if not stages:
            return {}

        active_run_ids = [
            active_run_id
            for active_run_id in (
                self._abort_controller.register_stage(stage) for stage in stages
            )
            if active_run_id
        ]
        try:
            cloud_instance = cloud if cloud is not None else self.create_cloud()
            results = run_pipeline(
                [
                    {
                        "label": label,
                        "parallel": [stage.as_pipeline_step() for stage in stages],
                    }
                ],
                cloud_instance,
                on_failure=self.on_failure,
            )
        finally:
            for active_run_id in active_run_ids:
                self._abort_controller.unregister_run(active_run_id)

        output: Dict[str, JobResult] = {}
        for stage in stages:
            run_id = _resolve_stage_run_name(stage)
            matched_result = None
            for result in reversed(results):
                if result.run_id == run_id:
                    matched_result = result
                    break
            if matched_result is None:
                returned_run_ids = [result.run_id for result in results]
                raise RuntimeError(
                    f'Parallel stage "{label}" expected run_id "{run_id}" for stage "{stage.id}" '
                    f"but no exact result was returned. Returned run_ids: {returned_run_ids!r}"
                )
            output[stage.id] = matched_result

        for stage in stages:
            result = output[stage.id]
            if _is_retryable_failure(result):
                self._abort_controller.request_abort(
                    source_run_id=result.run_id,
                    source_label=stage.id,
                )
                break
        return output

    def require_adapter(self, result: JobResult, stage_id: str) -> AdapterRef:
        repo = getattr(result, "hf_repo_id", None)
        if not repo:
            raise RuntimeError(
                f"{stage_id} finished without hf_repo_id. "
                "Set training.hf_repo_base to a valid Hugging Face namespace/repo prefix."
            )
        revision = getattr(result, "hf_revision", None)
        if not revision:
            raise RuntimeError(
                f"{stage_id} finished without hf_revision. "
                "This run cannot seed later stages safely because its exact adapter revision is unknown."
            )
        return AdapterRef(repo_id=str(repo), revision=str(revision))

    def _build_stage(
        self,
        *,
        stage_id: str,
        base: str,
        job_class: type,
        run_name: Optional[str],
        output_dir: Optional[str],
        overrides: Optional[Mapping[str, Any]],
        adapter: Optional[AdapterRef],
        run_section: str,
    ) -> StageSpec:
        config = self.templates.clone(base)
        if self.shared_overrides:
            _deep_merge(config, self.shared_overrides)
        if overrides:
            _deep_merge(config, overrides)

        section_cfg = config.setdefault(run_section, {})
        section_cfg["run_name"] = run_name or stage_id
        if output_dir is not None:
            section_cfg["output_dir"] = output_dir

        if adapter is not None:
            model_cfg = config.setdefault("model", {})
            model_cfg["init_adapter_repo"] = adapter.repo_id
            model_cfg["init_adapter_revision"] = adapter.revision

        return StageSpec(
            id=stage_id,
            config=config,
            job_class=job_class,
            task=self.task,
        )
