# Tenyson

Multi-step research library for SFT, GRPO RL, and evaluation workflows (Unsloth, TRL, vLLM).

- **Library and docs**: [tenyson/README.md](tenyson/README.md)
- **Example**: Wordle pipeline in `tenyson/examples/wordle/`.

## Setup

1. Copy `.env.example` to `.env` and add your secrets (WandB, HF, etc.). Do not commit `.env`.
2. Install the Tenyson package: `pip install -e tenyson/`

## Push to GitHub (one-time)

If the remote `tenyson` repo does not exist yet:

1. On GitHub: [Create a new repository](https://github.com/new) named **tenyson** (empty, no README).
2. From this repo: `git push -u origin main`
