from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
import json
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from tenyson.core import wandb_store
from tenyson.core.environment import resolve_bound_environment_run
from tenyson.jobs.result import JobResult


@dataclass
class StageReportEntry:
    stage_id: str
    run_type: str
    run_name: str
    variant: Optional[str] = None
    status: str = "pending"
    wandb_url: Optional[str] = None
    metrics: Dict[str, str] = field(default_factory=dict)
    total_time_seconds: Optional[str] = None
    hf_repo_id: Optional[str] = None
    hf_revision: Optional[str] = None
    failure_reason: Optional[str] = None
    processed_samples: Optional[int] = None
    expected_samples: Optional[int] = None
    stopped_early: bool = False


@dataclass(frozen=True)
class _TelemetryStageCandidate:
    phase: str
    run_name: str
    attempt_token: Optional[str]
    status: str
    is_active: bool
    heartbeat_at: Optional[datetime]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    wandb_url: Optional[str]
    environment_run: Optional[str]
    has_job_result: bool
    failure_reason: Optional[str]
    job_result_payload: Dict[str, Any] = field(default_factory=dict)
    summary_metrics: Dict[str, Any] = field(default_factory=dict)


_PHASE_ORDER = {
    "sft": 0,
    "eval": 1,
    "rl": 2,
}


class ExperimentReport:
    """
    Fixed-format markdown report for experiment runs.

    The rendered fields are intentionally not user-customizable. The report always
    shows experiment context plus per-stage status, run links, HF adapter details,
    and metrics.
    """

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_dir = self.output_path.parent
        self.environment_name: Optional[str] = None
        self.experiment_id: Optional[str] = None
        self.telemetry_backend_ref: Optional[str] = None
        self.telemetry_project_url: Optional[str] = None
        self._stages: Dict[str, StageReportEntry] = {}
        self._stage_order: list[str] = []
        self._lock = RLock()

    def set_context(
        self,
        *,
        environment_name: Optional[str] = None,
        experiment_id: Optional[str] = None,
        telemetry_backend_ref: Optional[str] = None,
        telemetry_project_url: Optional[str] = None,
    ) -> None:
        with self._lock:
            if environment_name:
                self.environment_name = str(environment_name)
            if experiment_id:
                self.experiment_id = str(experiment_id)
            if telemetry_backend_ref:
                self.telemetry_backend_ref = str(telemetry_backend_ref)
            if telemetry_project_url:
                self.telemetry_project_url = str(telemetry_project_url)
            self.generate()

    def register_stage(
        self,
        *,
        stage_id: str,
        run_type: str,
        run_name: str,
        variant: Optional[str] = None,
    ) -> None:
        with self._lock:
            entry = self._stages.get(stage_id)
            if entry is None:
                entry = StageReportEntry(
                    stage_id=stage_id,
                    run_type=str(run_type),
                    run_name=str(run_name),
                    variant=str(variant) if variant else None,
                )
                self._stages[stage_id] = entry
                self._stage_order.append(stage_id)
            else:
                entry.run_type = str(run_type)
                entry.run_name = str(run_name)
                entry.variant = str(variant) if variant else None
            self.generate()

    def mark_stage_running(self, stage_id: str) -> None:
        with self._lock:
            entry = self._require_stage(stage_id)
            entry.status = "running"
            self.generate()

    def update_wandb_link(
        self,
        stage_id: str,
        run_url: Optional[str],
        *,
        text: str = "run",
        missing: str = "n/a",
    ) -> None:
        del text, missing
        with self._lock:
            entry = self._require_stage(stage_id)
            entry.wandb_url = str(run_url) if run_url else None
            self.generate()

    def update_result(
        self,
        stage_id: str,
        result: JobResult,
        *,
        metric_precision: Optional[int] = 4,
        wandb_text: str = "run",
        missing: str = "n/a",
    ) -> None:
        del wandb_text
        with self._lock:
            entry = self._require_stage(stage_id)
            entry.status = str(getattr(result, "status", "") or missing)
            entry.wandb_url = getattr(result, "wandb_url", None) or entry.wandb_url
            entry.metrics = {
                metric_name: self._format_value(
                    metric_value,
                    precision=metric_precision,
                    missing=missing,
                )
                for metric_name, metric_value in (getattr(result, "metrics", {}) or {}).items()
            }
            total_time = getattr(result, "total_time_seconds", None)
            entry.total_time_seconds = self._format_value(
                total_time,
                precision=2,
                missing=missing,
            )
            entry.hf_repo_id = getattr(result, "hf_repo_id", None)
            entry.hf_revision = getattr(result, "hf_revision", None)
            entry.failure_reason = getattr(result, "failure_reason", None)
            entry.processed_samples = getattr(result, "processed_samples", None)
            entry.expected_samples = getattr(result, "expected_samples", None)
            entry.stopped_early = bool(getattr(result, "stopped_early", False))
            self.generate()

    def generate(self) -> None:
        with self._lock:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(self._render(), encoding="utf-8")

    def rebuild_from_telemetry(
        self,
        *,
        backend_ref: Optional[str] = None,
        experiment_id: Optional[str] = None,
        environment_name: Optional[str] = None,
        metric_precision: Optional[int] = 4,
        missing: str = "n/a",
        max_live_age_seconds: int = 90,
        run_name_allowlist: Optional[Sequence[str]] = None,
        prefer_terminal_results: bool = False,
    ) -> bool:
        resolved_backend_ref = str(
            backend_ref or self.telemetry_backend_ref or ""
        ).strip()
        resolved_experiment_id = str(
            experiment_id or self.experiment_id or ""
        ).strip()
        if not resolved_backend_ref or not resolved_experiment_id:
            return False

        allowed_names = None
        run_order: Dict[str, int] = {}
        if run_name_allowlist is not None:
            allowed_names = {
                str(run_name).strip()
                for run_name in run_name_allowlist
                if str(run_name).strip()
            }
            run_order = {
                str(run_name).strip(): index
                for index, run_name in enumerate(run_name_allowlist)
                if str(run_name).strip()
            }

        runs = _project_runs_for_backend(resolved_backend_ref)
        candidates_by_key: Dict[tuple[str, str], list[_TelemetryStageCandidate]] = {}
        for run in runs:
            candidate = _candidate_from_run(
                run,
                experiment_id=resolved_experiment_id,
            )
            if candidate is None:
                continue
            if allowed_names is not None and candidate.run_name not in allowed_names:
                continue
            key = (candidate.phase, candidate.run_name)
            candidates_by_key.setdefault(key, []).append(candidate)

        selected_entries: list[tuple[int, datetime, int, str, StageReportEntry]] = []
        for (phase, run_name), candidates in candidates_by_key.items():
            selected = _select_candidate(
                candidates,
                max_live_age_seconds=max_live_age_seconds,
                prefer_terminal_results=prefer_terminal_results,
            )
            order_time = _stage_first_seen_at(candidates) or _candidate_sort_time(selected)
            entry = self._entry_from_telemetry_candidate(
                phase=phase,
                run_name=run_name,
                candidate=selected,
                backend_ref=resolved_backend_ref,
                experiment_id=resolved_experiment_id,
                metric_precision=metric_precision,
                missing=missing,
                max_live_age_seconds=max_live_age_seconds,
            )
            selected_entries.append(
                (
                    run_order.get(run_name, len(run_order)),
                    order_time,
                    _PHASE_ORDER.get(phase, 99),
                    run_name,
                    entry,
                )
            )

        selected_entries.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            )
        )

        project_url = None
        try:
            project_url = wandb_store.parse_backend_ref(resolved_backend_ref).project_url
        except Exception:  # noqa: BLE001
            project_url = None

        with self._lock:
            if environment_name:
                self.environment_name = str(environment_name)
            if resolved_experiment_id:
                self.experiment_id = resolved_experiment_id
            if resolved_backend_ref:
                self.telemetry_backend_ref = resolved_backend_ref
            if project_url:
                self.telemetry_project_url = project_url
            self._stages = {}
            self._stage_order = []
            for _run_index, _order_time, _phase_order, _run_name, entry in selected_entries:
                self._stages[entry.stage_id] = entry
                self._stage_order.append(entry.stage_id)
            self.generate()
        return True

    def _require_stage(self, stage_id: str) -> StageReportEntry:
        entry = self._stages.get(stage_id)
        if entry is None:
            entry = StageReportEntry(
                stage_id=stage_id,
                run_type="unknown",
                run_name=stage_id,
            )
            self._stages[stage_id] = entry
            self._stage_order.append(stage_id)
        return entry

    def _entry_from_telemetry_candidate(
        self,
        *,
        phase: str,
        run_name: str,
        candidate: _TelemetryStageCandidate,
        backend_ref: str,
        experiment_id: str,
        metric_precision: Optional[int],
        missing: str,
        max_live_age_seconds: int,
    ) -> StageReportEntry:
        stage_id = run_name
        variant = candidate.environment_run
        is_fresh_active = _candidate_is_fresh_active(
            candidate,
            max_live_age_seconds=max_live_age_seconds,
        )
        if is_fresh_active and not candidate.has_job_result:
            return StageReportEntry(
                stage_id=stage_id,
                run_type=phase,
                run_name=run_name,
                variant=variant,
                status=str(candidate.status or "running"),
                wandb_url=candidate.wandb_url,
                failure_reason=candidate.failure_reason,
            )

        results_payload: Dict[str, Any] = {}
        job_result_payload: Dict[str, Any] = {}
        if candidate.has_job_result:
            job_result_payload = dict(candidate.job_result_payload)
            result_pair = wandb_store.fetch_run_result(
                backend_ref,
                experiment_id=experiment_id,
                phase=phase,
                run_name=run_name,
                attempt_token=candidate.attempt_token,
            )
            if result_pair is not None:
                results_payload, fetched_job_result_payload = result_pair
                if fetched_job_result_payload:
                    job_result_payload = dict(fetched_job_result_payload)

        if job_result_payload:
            merged_payload = dict(job_result_payload)
            merged_metrics = dict(candidate.summary_metrics)
            merged_metrics.update(dict(merged_payload.get("metrics") or {}))
            results_metrics = results_payload.get("metrics")
            if isinstance(results_metrics, Mapping):
                for metric_name, metric_value in results_metrics.items():
                    merged_metrics.setdefault(str(metric_name), metric_value)
            merged_payload["metrics"] = merged_metrics
            if not merged_payload.get("wandb_url") and candidate.wandb_url:
                merged_payload["wandb_url"] = candidate.wandb_url
            result = JobResult.from_dict(merged_payload)
            return self._entry_from_result(
                stage_id=stage_id,
                run_type=phase,
                run_name=run_name,
                variant=variant,
                result=result,
                metric_precision=metric_precision,
                missing=missing,
            )

        return StageReportEntry(
            stage_id=stage_id,
            run_type=phase,
            run_name=run_name,
            variant=variant,
            status=str(candidate.status or "unknown"),
            wandb_url=candidate.wandb_url,
            failure_reason=candidate.failure_reason,
        )

    def _entry_from_result(
        self,
        *,
        stage_id: str,
        run_type: str,
        run_name: str,
        variant: Optional[str],
        result: JobResult,
        metric_precision: Optional[int],
        missing: str,
    ) -> StageReportEntry:
        return StageReportEntry(
            stage_id=stage_id,
            run_type=str(run_type),
            run_name=str(run_name),
            variant=str(variant) if variant else None,
            status=str(getattr(result, "status", "") or missing),
            wandb_url=getattr(result, "wandb_url", None),
            metrics={
                metric_name: self._format_value(
                    metric_value,
                    precision=metric_precision,
                    missing=missing,
                )
                for metric_name, metric_value in (getattr(result, "metrics", {}) or {}).items()
            },
            total_time_seconds=self._format_value(
                getattr(result, "total_time_seconds", None),
                precision=2,
                missing=missing,
            ),
            hf_repo_id=getattr(result, "hf_repo_id", None),
            hf_revision=getattr(result, "hf_revision", None),
            failure_reason=getattr(result, "failure_reason", None),
            processed_samples=getattr(result, "processed_samples", None),
            expected_samples=getattr(result, "expected_samples", None),
            stopped_early=bool(getattr(result, "stopped_early", False)),
        )

    def _render(self) -> str:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines = [
            "# Tenyson Experiment Report",
            "",
            f"- Generated: `{generated_at}`",
            f"- Environment: `{self.environment_name or 'unknown'}`",
            f"- Experiment ID: `{self.experiment_id or 'unknown'}`",
            f"- Telemetry backend: `{self.telemetry_backend_ref or 'n/a'}`",
            (
                f"- Telemetry dashboard: [open project]({self.telemetry_project_url})"
                if self.telemetry_project_url
                else "- Telemetry dashboard: n/a"
            ),
            f"- Stage summary: {self._stage_summary()}",
            "",
        ]
        if not self._stage_order:
            lines.extend(
                [
                    "## Stages",
                    "",
                    "No stages have started yet.",
                    "",
                ]
            )
            return "\n".join(lines)

        lines.extend(["## Stages", ""])
        for index, stage_id in enumerate(self._stage_order, start=1):
            entry = self._stages[stage_id]
            lines.extend(self._render_stage(index, entry))
        return "\n".join(lines).rstrip() + "\n"

    def _render_stage(self, index: int, entry: StageReportEntry) -> list[str]:
        lines = [
            f"### {index}. {entry.stage_id}",
            "",
            f"- Run type: `{entry.run_type}`",
            f"- Run name: `{entry.run_name}`",
            f"- Environment run: `{entry.variant or 'default'}`",
            f"- Status: `{entry.status}`",
            (
                f"- W&B run: [open run]({entry.wandb_url})"
                if entry.wandb_url
                else "- W&B run: n/a"
            ),
        ]
        if entry.hf_repo_id:
            if entry.hf_revision:
                lines.append(
                    f"- Hugging Face adapter: `{entry.hf_repo_id}` @ `{entry.hf_revision}`"
                )
            else:
                lines.append(f"- Hugging Face adapter: `{entry.hf_repo_id}`")
        if entry.total_time_seconds is not None:
            lines.append(f"- Runtime (seconds): `{entry.total_time_seconds}`")
        if entry.expected_samples is not None:
            sample_text = str(entry.processed_samples or 0)
            if entry.expected_samples is not None:
                sample_text = f"{sample_text} / {entry.expected_samples}"
            lines.append(f"- Processed samples: `{sample_text}`")
        if entry.stopped_early:
            lines.append("- Stopped early: `true`")
        if entry.failure_reason:
            lines.append(f"- Failure reason: `{entry.failure_reason}`")
        if entry.metrics:
            for metric_name in sorted(entry.metrics):
                lines.append(f"- Metric `{metric_name}`: `{entry.metrics[metric_name]}`")
        else:
            lines.append("- Metrics: n/a")
        lines.append("")
        return lines

    def _stage_summary(self) -> str:
        counts: Dict[str, int] = {}
        for stage_id in self._stage_order:
            status = str(self._stages[stage_id].status or "unknown")
            counts[status] = counts.get(status, 0) + 1
        if not counts:
            return "0 stages"
        parts = [f"{count} {status}" for status, count in sorted(counts.items())]
        return ", ".join(parts)

    @staticmethod
    def _format_value(
        value: Any,
        *,
        precision: Optional[int] = None,
        missing: str = "n/a",
    ) -> str:
        if value is None:
            return missing
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, int) and precision is not None:
            return str(value)
        if isinstance(value, float) and precision is not None:
            return f"{float(value):.{precision}f}"
        return str(value)


def _project_runs_for_backend(backend_ref: str) -> list[Any]:
    target = wandb_store.parse_backend_ref(backend_ref)
    import wandb

    api = wandb.Api()
    return list(api.runs(path=f"{target.entity}/{target.project}"))


def _candidate_from_run(
    run: Any,
    *,
    experiment_id: str,
) -> Optional[_TelemetryStageCandidate]:
    summary_experiment_id = str(
        wandb_store._summary_get(run, wandb_store.SUMMARY_EXPERIMENT_ID)
        or getattr(run, "group", "")
        or ""
    ).strip()
    if summary_experiment_id != str(experiment_id):
        return None

    summary_run_name = str(
        wandb_store._summary_get(run, wandb_store.SUMMARY_RUN_NAME) or ""
    ).strip()
    actual_run_name = str(getattr(run, "name", "") or "").strip()
    if summary_run_name and actual_run_name and summary_run_name != actual_run_name:
        return None

    run_name = summary_run_name or actual_run_name
    if not run_name:
        return None

    phase = str(
        wandb_store._summary_get(run, wandb_store.SUMMARY_PHASE)
        or getattr(run, "job_type", "")
        or ""
    ).strip()
    if not phase:
        return None

    job_result_payload = _maybe_json_dict(
        wandb_store._summary_get(run, wandb_store.SUMMARY_JOB_RESULT_JSON)
    )
    summary_metrics = _maybe_json_dict(
        wandb_store._summary_get(run, wandb_store.SUMMARY_METRICS_JSON)
    )
    status = str(
        wandb_store._summary_get(run, wandb_store.SUMMARY_STATUS)
        or job_result_payload.get("status")
        or "unknown"
    ).strip()
    wandb_url = str(
        wandb_store._summary_get(run, wandb_store.SUMMARY_WANDB_URL)
        or getattr(run, "url", "")
        or ""
    ).strip() or None
    config = getattr(run, "config", {}) or {}
    environment_run = None
    if isinstance(config, Mapping):
        environment_run = resolve_bound_environment_run(config)

    return _TelemetryStageCandidate(
        phase=phase,
        run_name=run_name,
        attempt_token=str(
            wandb_store._summary_get(run, wandb_store.SUMMARY_ATTEMPT_TOKEN) or ""
        ).strip() or None,
        status=status or "unknown",
        is_active=bool(wandb_store._summary_get(run, wandb_store.SUMMARY_IS_ACTIVE)),
        heartbeat_at=wandb_store._parse_datetime(
            wandb_store._summary_get(run, wandb_store.SUMMARY_HEARTBEAT_AT)
        ),
        created_at=wandb_store._parse_datetime(getattr(run, "created_at", None)),
        updated_at=wandb_store._parse_datetime(getattr(run, "updated_at", None)),
        wandb_url=wandb_url,
        environment_run=environment_run,
        has_job_result=bool(job_result_payload),
        failure_reason=str(
            wandb_store._summary_get(run, wandb_store.SUMMARY_FAILURE_REASON)
            or job_result_payload.get("failure_reason")
            or ""
        ).strip()
        or None,
        job_result_payload=job_result_payload,
        summary_metrics=summary_metrics,
    )


def _select_candidate(
    candidates: Iterable[_TelemetryStageCandidate],
    *,
    max_live_age_seconds: int,
    prefer_terminal_results: bool = False,
) -> _TelemetryStageCandidate:
    rows = list(candidates)
    completed = [candidate for candidate in rows if candidate.has_job_result]
    if prefer_terminal_results and completed:
        return max(completed, key=_candidate_sort_key)

    fresh_active = [
        candidate
        for candidate in rows
        if _candidate_is_fresh_active(
            candidate,
            max_live_age_seconds=max_live_age_seconds,
        )
    ]
    if fresh_active:
        return max(fresh_active, key=_candidate_sort_key)

    if completed:
        return max(completed, key=_candidate_sort_key)

    return max(rows, key=_candidate_sort_key)


def _candidate_is_fresh_active(
    candidate: _TelemetryStageCandidate,
    *,
    max_live_age_seconds: int,
) -> bool:
    if not candidate.is_active:
        return False
    if candidate.heartbeat_at is None:
        return False
    age_seconds = (datetime.now(timezone.utc) - candidate.heartbeat_at).total_seconds()
    return age_seconds <= max(1, int(max_live_age_seconds))


def _candidate_sort_time(candidate: _TelemetryStageCandidate) -> datetime:
    return (
        candidate.heartbeat_at
        or candidate.updated_at
        or candidate.created_at
        or datetime.fromtimestamp(0, tz=timezone.utc)
    )


def _candidate_sort_key(candidate: _TelemetryStageCandidate) -> tuple[float, float]:
    primary = _candidate_sort_time(candidate)
    secondary = (
        candidate.updated_at
        or candidate.created_at
        or primary
    )
    return (
        primary.timestamp(),
        secondary.timestamp(),
    )


def _stage_first_seen_at(
    candidates: Iterable[_TelemetryStageCandidate],
) -> Optional[datetime]:
    seen_times = [
        candidate.created_at or candidate.updated_at or candidate.heartbeat_at
        for candidate in candidates
        if candidate.created_at or candidate.updated_at or candidate.heartbeat_at
    ]
    if not seen_times:
        return None
    return min(seen_times)


def _maybe_json_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001
        return {}
    if isinstance(parsed, dict):
        return dict(parsed)
    return {}
