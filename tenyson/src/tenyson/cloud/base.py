from abc import ABC, abstractmethod
from typing import Any

from tenyson.jobs.result import JobResult


class JobFailedError(Exception):
    def __init__(self, message: str, exit_code: int = 1, log_path: str = ""):
        super().__init__(message)
        self.exit_code = exit_code
        self.log_path = log_path


class BaseCloudManager(ABC):
    def __init__(self, auto_terminate: bool = True):
        self.auto_terminate = auto_terminate

    @abstractmethod
    def run(self, job: Any) -> JobResult:
        raise NotImplementedError
