from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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

    def set_context(
        self,
        *,
        environment_name: Optional[str] = None,
        experiment_id: Optional[str] = None,
        telemetry_backend_ref: Optional[str] = None,
        telemetry_project_url: Optional[str] = None,
    ) -> None:
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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(self._render(), encoding="utf-8")

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
