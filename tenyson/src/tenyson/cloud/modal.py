from typing import Any

from tenyson.cloud.base import BaseCloudManager
from tenyson.jobs.result import JobResult


class ModalManager(BaseCloudManager):
    def __init__(self, gpu: str = "A100", auto_terminate: bool = True, **kwargs):
        super().__init__(auto_terminate=auto_terminate)
        self.gpu = gpu
        self.kwargs = kwargs

    def run(self, job: Any) -> JobResult:
        # Placeholder manager: local execution passthrough until full remote runner is wired.
        return job.run()
