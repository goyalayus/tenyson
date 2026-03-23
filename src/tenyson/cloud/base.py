from abc import ABC, abstractmethod
import sys
from typing import Any

from tenyson.jobs.result import JobResult


def _red_print(message: str) -> None:
    """Print message to stderr in red if stderr is a TTY; otherwise plain."""
    if sys.stderr.isatty():
        sys.stderr.write(f"\033[91m{message}\033[0m\n")
    else:
        sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


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
