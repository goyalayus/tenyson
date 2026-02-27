from pathlib import Path
from typing import Any, Dict


class ReportBuilder:
    def __init__(self, template_path: str, output_path: str):
        self.template_path = Path(template_path)
        self.output_path = Path(output_path)
        self.content = self.template_path.read_text(encoding="utf-8")

    def fill(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            self.content = self.content.replace("{" + key + "}", str(value))

    def generate(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(self.content, encoding="utf-8")
