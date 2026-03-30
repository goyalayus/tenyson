from __future__ import annotations

import time
from typing import Any, Optional

from huggingface_hub import create_repo
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import RequestException
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState


def ensure_hf_repo(
    repo_id: str,
    *,
    max_attempts: int = 5,
    initial_backoff_seconds: float = 2.0,
) -> None:
    """Create the target Hub repository if it does not exist."""
    if int(max_attempts) < 1:
        raise ValueError("max_attempts must be >= 1.")

    last_error: Exception | None = None
    for attempt in range(1, int(max_attempts) + 1):
        try:
            create_repo(repo_id=repo_id, exist_ok=True)
            return
        except HfHubHTTPError as exc:
            if not _is_retryable_hf_http_error(exc) or attempt >= int(max_attempts):
                raise
            last_error = exc
        except RequestException as exc:
            if attempt >= int(max_attempts):
                raise
            last_error = exc
        time.sleep(_retry_backoff_seconds(attempt, initial_backoff_seconds))

    if last_error is not None:
        raise last_error


def _is_retryable_hf_http_error(exc: HfHubHTTPError) -> bool:
    response = getattr(exc, "response", None)
    status_code = int(getattr(response, "status_code", 0) or 0)
    return status_code in {408, 429} or status_code >= 500


def _retry_backoff_seconds(attempt: int, initial_backoff_seconds: float) -> float:
    base_delay = max(0.1, float(initial_backoff_seconds))
    return base_delay * (2 ** max(0, int(attempt) - 1))


class PeriodicHubPushCallback(TrainerCallback):
    """
    Periodically pushes current model/tokenizer weights to a fixed Hub repo.

    The same repository path is reused across pushes so each push updates the
    latest adapter lineage instead of creating a new repo.
    """

    def __init__(
        self,
        repo_id: str,
        run_name: str,
        push_every_steps: int,
        tokenizer: Optional[Any] = None,
    ):
        if int(push_every_steps) <= 0:
            raise ValueError("push_every_steps must be >= 1.")
        self.repo_id = repo_id
        self.run_name = run_name
        self.push_every_steps = int(push_every_steps)
        self.tokenizer = tokenizer
        self._last_pushed_step = -1

    def _push(self, model: Any, step: int, reason: str) -> None:
        if step <= 0 or step == self._last_pushed_step:
            return
        commit_message = f"[{self.run_name}] {reason} (step={step})"
        try:
            model.push_to_hub(self.repo_id, commit_message=commit_message)
            if self.tokenizer is not None:
                self.tokenizer.push_to_hub(self.repo_id, commit_message=commit_message)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed pushing to Hugging Face Hub repo '{self.repo_id}' at step {step}: {exc}"
            ) from exc
        self._last_pushed_step = step

    def on_step_end(
        self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs
    ):
        step = int(state.global_step)
        if step > 0 and step % self.push_every_steps == 0:
            current_model = model if model is not None else kwargs.get("model")
            if current_model is None:
                raise RuntimeError(
                    "PeriodicHubPushCallback could not access model during on_step_end."
                )
            self._push(current_model, step, reason="periodic push")
        return control

    def on_train_end(
        self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs
    ):
        step = int(state.global_step)
        current_model = model if model is not None else kwargs.get("model")
        if current_model is None:
            raise RuntimeError(
                "PeriodicHubPushCallback could not access model during on_train_end."
            )
        self._push(current_model, step, reason="final push")
        return control
