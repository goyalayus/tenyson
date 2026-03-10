from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from tenyson.jobs.result import JobResult


class ReportBuilder:
    """
    Simple markdown templating plus optional WandB metric embedding.
    """

    def __init__(self, template_path: str, output_path: str):
        self.template_path = Path(template_path)
        self.output_path = Path(output_path)
        self.output_dir = self.output_path.parent
        self.template = self.template_path.read_text(encoding="utf-8")
        self.content = self.template
        self.values: Dict[str, str] = {}

    def _render(self) -> str:
        content = self.template
        for key, value in self.values.items():
            content = content.replace("{" + key + "}", value)
        self.content = content
        return content

    def _store_values(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            self.values[key] = str(value)

    @staticmethod
    def _format_value(
        value: Any,
        *,
        precision: Optional[int] = None,
        missing: str = "n/a",
    ) -> str:
        if value is None:
            return missing
        if precision is not None and isinstance(value, (int, float)):
            return f"{float(value):.{precision}f}"
        return str(value)

    @staticmethod
    def _format_wandb_link(
        run_url: Optional[str],
        *,
        text: str = "run",
        missing: str = "n/a",
    ) -> str:
        if not run_url:
            return missing
        return f"[{text}]({run_url})"

    @classmethod
    def result_placeholder_data(
        cls,
        label: str,
        result: JobResult,
        *,
        metric_precision: Optional[int] = None,
        wandb_text: str = "run",
        missing: str = "n/a",
    ) -> Dict[str, str]:
        data: Dict[str, str] = {
            f"{label}_status": cls._format_value(result.status, missing=missing),
            f"{label}_wandb_link": cls._format_wandb_link(
                result.wandb_url,
                text=wandb_text,
                missing=missing,
            ),
        }
        for metric_name, metric_value in result.metrics.items():
            data[f"{label}_{metric_name}"] = cls._format_value(
                metric_value,
                precision=metric_precision,
                missing=missing,
            )
        return data

    @classmethod
    def metric_delta_value(
        cls,
        left: Optional[JobResult],
        right: Optional[JobResult],
        metric_name: str,
        *,
        precision: int = 4,
        missing: str = "n/a",
    ) -> str:
        if left is None or right is None:
            return missing
        left_value = left.metrics.get(metric_name)
        right_value = right.metrics.get(metric_name)
        if not isinstance(left_value, (int, float)) or not isinstance(
            right_value, (int, float)
        ):
            return missing
        return cls._format_value(
            float(left_value) - float(right_value),
            precision=precision,
            missing=missing,
        )

    def fill(self, data: Dict[str, Any]) -> None:
        self._store_values(data)
        self._render()

    def fill_result(
        self,
        label: str,
        result: JobResult,
        *,
        metric_precision: Optional[int] = 4,
        wandb_text: str = "run",
        missing: str = "n/a",
    ) -> None:
        self.fill(
            self.result_placeholder_data(
                label,
                result,
                metric_precision=metric_precision,
                wandb_text=wandb_text,
                missing=missing,
            )
        )

    def fill_results(
        self,
        results: Mapping[str, JobResult],
        *,
        metric_precision: Optional[int] = 4,
        wandb_text: str = "run",
        missing: str = "n/a",
    ) -> None:
        data: Dict[str, str] = {}
        for label, result in results.items():
            data.update(
                self.result_placeholder_data(
                    label,
                    result,
                    metric_precision=metric_precision,
                    wandb_text=wandb_text,
                    missing=missing,
                )
            )
        self.fill(data)

    def fill_metric_delta(
        self,
        placeholder: str,
        left: Optional[JobResult],
        right: Optional[JobResult],
        metric_name: str,
        *,
        precision: int = 4,
        missing: str = "n/a",
    ) -> None:
        self.fill(
            {
                placeholder: self.metric_delta_value(
                    left,
                    right,
                    metric_name,
                    precision=precision,
                    missing=missing,
                )
            }
        )

    def update(self, data: Dict[str, Any]) -> None:
        """
        Incremental update: merge the latest values and re-render the report from the
        original template so retries can overwrite earlier failed values.
        """
        self._store_values(data)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(self._render(), encoding="utf-8")

    def attach_wandb_scalar_link(
        self,
        placeholder: str,
        run_url: str,
        metric_name: str,
    ) -> None:
        """
        Replace a placeholder with a markdown link to a WandB run + metric.
        """
        link = f"[{metric_name}]({run_url})"
        self.values[placeholder] = link
        self._render()

    def attach_wandb_latest_value(
        self,
        placeholder: str,
        run_path: str,
        metric_name: str,
        api: Optional[Any] = None,
    ) -> None:
        """
        Replace a placeholder with the latest scalar value for a metric
        from a WandB run specified as 'entity/project/run_id'.
        """
        if api is None:
            import wandb

            api = wandb.Api()
        run = api.run(run_path)
        history = run.history(keys=[metric_name], pandas=False)
        latest = None
        for row in history:
            if metric_name in row:
                latest = row[metric_name]
        if latest is not None:
            self.values[placeholder] = str(latest)
            self._render()

    def generate(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(self._render(), encoding="utf-8")
