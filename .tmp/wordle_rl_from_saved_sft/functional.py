from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path("/home/ayush/Desktop/code/tenyson")
SRC_DIR = REPO_ROOT / "src"

for path in (str(REPO_ROOT), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


from examples.wordle.functional import *  # noqa: F401,F403,E402


SEEDS = {
    "stopped_sft_turn5": {
        "repo_id": "goyalayus/wordle-lora-20260324-163252-sft_turn5",
        "revision": "2f92897b5cd3f760da3bdc526aa3fd2842e9bd82",
    }
}
