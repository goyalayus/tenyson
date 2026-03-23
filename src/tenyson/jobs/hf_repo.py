"""Helpers for deriving stable Hugging Face repo ids from a base and run name."""

import re


def sanitize_run_name(run_name: str) -> str:
    """Make run_name safe for use in a Hugging Face repo name (lowercase, no spaces/slashes)."""
    s = run_name.lower().strip()
    s = re.sub(r"[\s/]+", "-", s)
    s = re.sub(r"[^a-z0-9._-]", "", s)
    return s or "run"


def unique_repo_id(base: str, run_name: str) -> str:
    """
    Derive a stable repo id for pushing: base + sanitized run_name.

    This keeps a single canonical repository per run_name lineage so repeated pushes
    (including periodic checkpoint pushes) update the same target.
    """
    base = (base or "").strip().rstrip("/")
    if not base:
        return ""
    safe_name = sanitize_run_name(run_name)
    return f"{base}-{safe_name}"
