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
        self.content = self.template_path.read_text(encoding="utf-8")

    def fill(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            self.content = self.content.replace("{" + key + "}", str(value))

    def update(self, data: Dict[str, Any]) -> None:
        """
        Incremental update: read current report from disk (or use in-memory content),
        replace only the placeholders for keys in `data`, then write back.
        Use after an initial fill() + generate() to update the report as steps complete.
        """
        if self.output_path.exists():
            self.content = self.output_path.read_text(encoding="utf-8")
        for key, value in data.items():
            self.content = self.content.replace("{" + key + "}", str(value))
        self.output_dir = self.output_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(self.content, encoding="utf-8")

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
        self.content = self.content.replace("{" + placeholder + "}", link)

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
            self.content = self.content.replace("{" + placeholder + "}", str(latest))

    def generate(self) -> None:
        self.output_dir = self.output_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(self.content, encoding="utf-8")
