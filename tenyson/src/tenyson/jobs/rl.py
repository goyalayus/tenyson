import os
import time
from typing import Any, Dict

from tenyson.core.plugin import TaskPlugin
from tenyson.jobs.result import JobResult


class RLJob:
    def __init__(self, config: Dict[str, Any], task: TaskPlugin):
        self.config = config
        self.task = task
        self.run_id = self.config.get("training", {}).get("run_name", "rl_job")

    def run(self) -> JobResult:
        start = time.time()
        output_dir = self.config.get("training", {}).get("output_dir", f"./outputs/{self.run_id}")
        os.makedirs(output_dir, exist_ok=True)
        result = JobResult(
            run_id=self.run_id,
            status="success",
            total_time_seconds=time.time() - start,
            local_output_dir=output_dir,
        )
        result.save(os.path.join(output_dir, "results.json"))
        return result
