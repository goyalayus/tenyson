from __future__ import annotations

import re
from pathlib import Path

from huggingface_hub import snapshot_download


def _checkpoint_step(checkpoint_dir: Path) -> int:
    match = re.match(r"^checkpoint-(\d+)$", checkpoint_dir.name)
    return int(match.group(1)) if match else -1


def _is_trainer_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / "trainer_state.json").is_file()


def download_hf_resume_checkpoint(resume_ref: str) -> str:
    """
    Resolve and download a Hugging Face trainer checkpoint from "repo_id:revision".

    Preference order after snapshot download:
    1) "<snapshot>/last-checkpoint"
    2) "<snapshot>/checkpoint-*" (largest step)
    3) "<snapshot>" itself if it is already a trainer checkpoint
    """
    ref = str(resume_ref or "").strip()
    if ":" not in ref:
        raise ValueError(
            "resume_from_checkpoint must be of form 'repo_id:revision'."
        )

    repo_id, revision = ref.split(":", 1)
    repo_id = repo_id.strip()
    revision = revision.strip()
    if not repo_id or not revision:
        raise ValueError(
            "Both repo_id and revision are required in 'repo_id:revision'."
        )

    snapshot_root = Path(snapshot_download(repo_id=repo_id, revision=revision))

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
