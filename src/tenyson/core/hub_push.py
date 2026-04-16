from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any, Mapping, Optional

from huggingface_hub import CommitOperationDelete, create_repo, HfApi
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import RequestException
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from tenyson.jobs.hf_repo import unique_repo_id


_MODEL_ARTIFACT_PATTERNS: tuple[str, ...] = (
    "adapter_config.json",
    "adapter_model*.bin",
    "adapter_model*.index.json",
    "adapter_model*.safetensors",
    "config.json",
    "generation_config.json",
    "model*.bin",
    "model*.index.json",
    "model*.safetensors",
    "pytorch_model*.bin",
    "pytorch_model*.index.json",
    "training_args.bin",
)

_TOKENIZER_ARTIFACT_PATTERNS: tuple[str, ...] = (
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)

_HF_TOKEN_CACHE_PATHS: tuple[Path, ...] = (
    Path.home() / ".cache" / "huggingface" / "token",
    Path.home() / ".huggingface" / "token",
)


def resolve_hf_token() -> str:
    """Resolve a Hugging Face token from env first, then the local cache."""

    env_candidates = (
        os.getenv("HF_TOKEN"),
        os.getenv("HUGGING_FACE_HUB_TOKEN"),
        os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )
    for candidate in env_candidates:
        token = str(candidate or "").strip()
        if token:
            return token

    for token_path in _HF_TOKEN_CACHE_PATHS:
        try:
            token = token_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        if token:
            return token

    return ""


def resolve_hf_push_repo_id(
    training_cfg: Mapping[str, Any],
    *,
    run_name: str,
) -> str:
    """Resolve the exact repo id a training run should push to."""

    explicit_repo_id = str(training_cfg.get("hf_repo_id") or "").strip().rstrip("/")
    if explicit_repo_id:
        return explicit_repo_id

    hf_repo_base = str(training_cfg.get("hf_repo_base") or "").strip().rstrip("/")
    if not hf_repo_base:
        return ""

    return unique_repo_id(hf_repo_base, run_name)


def ensure_hf_repo(
    repo_id: str,
    *,
    token: str | None = None,
    max_attempts: int = 5,
    initial_backoff_seconds: float = 2.0,
) -> None:
    """Ensure the target Hub repo exists and the token can create/push to it."""
    if int(max_attempts) < 1:
        raise ValueError("max_attempts must be >= 1.")

    last_error: Exception | None = None
    for attempt in range(1, int(max_attempts) + 1):
        try:
            create_repo(
                repo_id=repo_id,
                token=token,
                exist_ok=True,
            )
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


def preflight_cloud_hf_push(
    config: Mapping[str, Any],
    *,
    run_name: str,
    job_type: str,
) -> str | None:
    """Validate Hub push config locally before a cloud training launch."""

    normalized_job_type = str(job_type or "").strip().lower()
    if normalized_job_type not in {"sft", "rl"}:
        return None

    training_cfg = config.get("training", {})
    if training_cfg is None:
        training_cfg = {}
    if not isinstance(training_cfg, Mapping):
        raise TypeError("training must be a mapping when provided.")

    repo_id = resolve_hf_push_repo_id(
        training_cfg,
        run_name=run_name,
    )
    if not repo_id:
        raise ValueError(
            f"training.hf_repo_id or training.hf_repo_base is required for "
            f"{normalized_job_type.upper()} cloud runs. Set one of them before "
            "launching."
        )

    hf_token = resolve_hf_token()
    if not hf_token:
        raise ValueError(
            f"A Hugging Face token is required before launching "
            f"{normalized_job_type.upper()} so Tenyson can push checkpoints to "
            f"'{repo_id}'. Export HF_TOKEN or log in locally with Hugging Face "
            "so the cached token can be reused."
        )

    try:
        ensure_hf_repo(repo_id, token=hf_token)
    except HfHubHTTPError as exc:
        response = getattr(exc, "response", None)
        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code in {401, 403}:
            raise ValueError(
                f"{normalized_job_type.upper()} cloud launch preflight failed: "
                f"HF_TOKEN is invalid or does not have write access to '{repo_id}'."
            ) from exc
        raise RuntimeError(
            f"{normalized_job_type.upper()} cloud launch preflight could not verify "
            f"Hugging Face repo '{repo_id}': {exc}"
        ) from exc
    except RequestException as exc:
        raise RuntimeError(
            f"{normalized_job_type.upper()} cloud launch preflight could not reach "
            f"Hugging Face while verifying '{repo_id}': {exc}"
        ) from exc

    return repo_id


def _is_retryable_hf_http_error(exc: HfHubHTTPError) -> bool:
    response = getattr(exc, "response", None)
    status_code = int(getattr(response, "status_code", 0) or 0)
    return status_code in {408, 429} or status_code >= 500


def _retry_backoff_seconds(attempt: int, initial_backoff_seconds: float) -> float:
    base_delay = max(0.1, float(initial_backoff_seconds))
    return base_delay * (2 ** max(0, int(attempt) - 1))


def _matches_any_pattern(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _resolve_stale_root_artifact_files(
    *,
    remote_files: list[str],
    local_files: list[str],
    include_tokenizer: bool,
) -> list[str]:
    local_file_set = set(local_files)
    stale_files: list[str] = []
    tokenizer_patterns = (
        _TOKENIZER_ARTIFACT_PATTERNS if include_tokenizer else tuple()
    )

    for path_in_repo in remote_files:
        if "/" in path_in_repo.strip("/"):
            continue
        if path_in_repo in local_file_set:
            continue

        if _matches_any_pattern(path_in_repo, _MODEL_ARTIFACT_PATTERNS):
            stale_files.append(path_in_repo)
            continue

        if tokenizer_patterns and _matches_any_pattern(
            path_in_repo,
            tokenizer_patterns,
        ):
            stale_files.append(path_in_repo)

    return sorted(stale_files)


def _delete_stale_root_artifact_files(
    api: HfApi,
    *,
    repo_id: str,
    stale_files: list[str],
    commit_message: str,
) -> None:
    if not stale_files:
        return

    api.create_commit(
        repo_id=repo_id,
        operations=[
            CommitOperationDelete(path_in_repo=path_in_repo)
            for path_in_repo in stale_files
        ],
        commit_message=commit_message,
    )


def push_pretrained_snapshot_to_hub(
    repo_id: str,
    *,
    model: Any,
    tokenizer: Optional[Any],
    commit_message: str,
) -> None:
    hf_token = resolve_hf_token() or None
    ensure_hf_repo(repo_id, token=hf_token)

    with TemporaryDirectory(prefix="tenyson-hf-push-") as tmpdir:
        snapshot_dir = Path(tmpdir)
        model.save_pretrained(snapshot_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(snapshot_dir)
        local_files = sorted(
            path.relative_to(snapshot_dir).as_posix()
            for path in snapshot_dir.rglob("*")
            if path.is_file()
        )
        api = HfApi(token=hf_token)
        stale_files = _resolve_stale_root_artifact_files(
            remote_files=list(api.list_repo_files(repo_id=repo_id)),
            local_files=local_files,
            include_tokenizer=tokenizer is not None,
        )
        _delete_stale_root_artifact_files(
            api,
            repo_id=repo_id,
            stale_files=stale_files,
            commit_message=f"{commit_message} [cleanup stale files]",
        )
        api.upload_folder(
            repo_id=repo_id,
            folder_path=snapshot_dir,
            commit_message=commit_message,
        )


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
            push_pretrained_snapshot_to_hub(
                self.repo_id,
                model=model,
                tokenizer=self.tokenizer,
                commit_message=commit_message,
            )
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
