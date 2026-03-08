from pathlib import Path
from typing import Any, Dict, Optional

import wandb


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

    def fill(self, data: Dict[str, Any]) -> None:
        self._store_values(data)
        self._render()

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
        api: Optional[wandb.Api] = None,
    ) -> None:
        """
        Replace a placeholder with the latest scalar value for a metric
        from a WandB run specified as 'entity/project/run_id'.
        """
        api = api or wandb.Api()
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
