from typing import Any

from tenyson.cloud.base import BaseCloudManager
from tenyson.jobs.result import JobResult


class AWSManager(BaseCloudManager):
    def __init__(self, instance_type: str = "g5.2xlarge", auto_terminate: bool = True, **kwargs):
        super().__init__(auto_terminate=auto_terminate)
        self.instance_type = instance_type
        self.kwargs = kwargs

    def run(self, job: Any) -> JobResult:
        # Placeholder manager: local execution passthrough until full remote runner is wired.
        return job.run()
