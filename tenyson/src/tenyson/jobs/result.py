import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class JobResult:
    run_id: str
    status: str
    total_time_seconds: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    hf_repo_id: Optional[str] = None
    hf_revision: Optional[str] = None
    wandb_url: Optional[str] = None
    local_output_dir: Optional[str] = None

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)
