#!/usr/bin/env python3
import sys
from datetime import datetime

from train_grpo_mixed_qwen4b_unsloth import main as mixed_main


def _has_flag(argv: list[str], flag: str) -> bool:
    return flag in argv


def _inject_defaults(argv: list[str]) -> list[str]:
    out = list(argv)

    # Turn-2-only training means exactly one prior turn in history.
    if not _has_flag(out, "--min-history-turns"):
        out.extend(["--min-history-turns", "1"])
    if not _has_flag(out, "--max-history-turns"):
        out.extend(["--max-history-turns", "1"])

    # Keep run names distinct from mixed 2-5 runs unless caller sets one.
    if not _has_flag(out, "--run-name"):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out.extend(["--run-name", f"unsloth_grpo_turn2_{now}"])

    return out


if __name__ == "__main__":
    # Reuse the existing mixed script end-to-end:
    # - same model / SFT init defaults
    # - same reward logic
    # - same training/library wiring
    injected_argv = _inject_defaults(sys.argv[1:])
    sys.argv = [sys.argv[0]] + injected_argv
    mixed_main()
