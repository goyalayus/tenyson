from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Tuple

from huggingface_hub import HfApi, snapshot_download


def _checkpoint_step(checkpoint_dir: Path) -> int:
    match = re.match(r"^checkpoint-(\d+)$", checkpoint_dir.name)
    return int(match.group(1)) if match else -1


def _is_trainer_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / "trainer_state.json").is_file()


def _parse_resume_ref(resume_ref: str) -> Tuple[str, str]:
    ref = str(resume_ref or "").strip()
    if ":" not in ref:
        raise ValueError("resume_from_checkpoint must be of form 'repo_id:revision'.")

    repo_id, revision = ref.split(":", 1)
    repo_id = repo_id.strip()
    revision = revision.strip()
    if not repo_id or not revision:
        raise ValueError(
            "Both repo_id and revision are required in 'repo_id:revision'."
        )
    return repo_id, revision


def resolve_hf_repo_revision(repo_id: str, revision: str = "main") -> str:
    repo_id = str(repo_id or "").strip()
    revision = str(revision or "").strip() or "main"
    if not repo_id:
        raise ValueError("repo_id is required to resolve a Hugging Face revision.")

    info = HfApi().model_info(repo_id=repo_id, revision=revision)
    resolved_revision = str(getattr(info, "sha", "") or "").strip()
    if not resolved_revision:
        raise ValueError(
            f"Unable to resolve an immutable commit SHA for Hugging Face repo '{repo_id}' at revision '{revision}'."
        )
    return resolved_revision


def _repo_has_trainer_checkpoint(repo_files: Iterable[str]) -> bool:
    for repo_file in repo_files:
        normalized = str(repo_file or "").strip("/")
        if normalized == "trainer_state.json":
            return True
        if normalized == "last-checkpoint/trainer_state.json":
            return True
        if re.match(r"^checkpoint-\d+/trainer_state\.json$", normalized):
            return True
    return False


def resolve_hf_resume_revision(repo_id: str, revision: str = "main") -> str:
    resolved_revision = resolve_hf_repo_revision(repo_id=repo_id, revision=revision)
    repo_files = HfApi().list_repo_files(repo_id=repo_id, revision=resolved_revision)
    if not _repo_has_trainer_checkpoint(repo_files):
        raise ValueError(
            "No trainer checkpoint found in Hugging Face repo revision. Expected one of: "
            "'last-checkpoint/trainer_state.json', 'checkpoint-*/trainer_state.json', "
            "or root 'trainer_state.json'."
        )
    return resolved_revision


def _resolve_downloaded_checkpoint(snapshot_root: Path) -> str:
    last_checkpoint = snapshot_root / "last-checkpoint"
    if _is_trainer_checkpoint(last_checkpoint):
        return str(last_checkpoint)

    checkpoint_dirs = [
        path
        for path in snapshot_root.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-")
    ]
    checkpoint_dirs = [path for path in checkpoint_dirs if _is_trainer_checkpoint(path)]
    if checkpoint_dirs:
        checkpoint_dirs.sort(key=_checkpoint_step, reverse=True)
        return str(checkpoint_dirs[0])

    if _is_trainer_checkpoint(snapshot_root):
        return str(snapshot_root)

    raise ValueError(
        "No trainer checkpoint found in Hugging Face snapshot. Expected one of: "
        "'last-checkpoint', 'checkpoint-*', or root trainer_state.json."
    )


def download_hf_resume_checkpoint(resume_ref: str) -> str:
    """
    Resolve and download a Hugging Face trainer checkpoint from "repo_id:revision".

    Preference order after snapshot download:
    1) "<snapshot>/last-checkpoint"
    2) "<snapshot>/checkpoint-*" (largest step)
    3) "<snapshot>" itself if it is already a trainer checkpoint
    """
    repo_id, revision = _parse_resume_ref(resume_ref)
    resolved_revision = resolve_hf_resume_revision(repo_id=repo_id, revision=revision)
    snapshot_root = Path(snapshot_download(repo_id=repo_id, revision=resolved_revision))
    return _resolve_downloaded_checkpoint(snapshot_root)
