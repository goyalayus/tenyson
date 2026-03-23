from __future__ import annotations

from typing import Any, Optional

from huggingface_hub import create_repo
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState


def ensure_hf_repo(repo_id: str) -> None:
    """Create the target Hub repository if it does not exist."""
    create_repo(repo_id=repo_id, exist_ok=True)


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
