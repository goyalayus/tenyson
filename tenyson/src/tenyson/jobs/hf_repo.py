"""Helpers for deriving unique Hugging Face repo ids from a base and run name."""

import re
from uuid import uuid4


def sanitize_run_name(run_name: str) -> str:
    """Make run_name safe for use in a Hugging Face repo name (lowercase, no spaces/slashes)."""
    s = run_name.lower().strip()
    s = re.sub(r"[\s/]+", "-", s)
    s = re.sub(r"[^a-z0-9._-]", "", s)
    return s or "run"


def unique_repo_id(base: str, run_name: str) -> str:
    """Derive a unique repo id for pushing: base + sanitized run_name + short uid."""
    base = (base or "").strip().rstrip("/")
    if not base:
        return ""
    safe_name = sanitize_run_name(run_name)
    short_uid = uuid4().hex[:8]
    return f"{base}-{safe_name}-{short_uid}"
