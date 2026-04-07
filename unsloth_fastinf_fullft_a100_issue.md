Title:
`fast_inference=True` fails with `full_finetuning=True` on A100 during `FastLanguageModel.from_pretrained(...)`

Body:

This looks related to earlier issue #3577 and merged PR #3768.
From what I can tell, #3768 clarified the error message for this case, but did not add support for the combination itself.

I kept the install side intentionally minimal:

```bash
python -m pip install --upgrade vllm unsloth
```

Observed behavior:

Calling `FastLanguageModel.from_pretrained(...)` with:

- `fast_inference=True`
- `full_finetuning=True`

fails immediately with:

```text
NotImplementedError: Unsloth: `fast_inference=True` cannot be used together with `full_finetuning=True`.
Reason: fast_inference is optimized for inference-only workflows and does not currently support full fine-tuning.
```

I am filing this to confirm the current released behavior on a clean repro after #3768.

Environment:

- GPU: A100
- Python: 3.12
- install command: `python -m pip install --upgrade vllm unsloth`

Minimal repro:

This is the exact standalone script I used:

```python
#!/usr/bin/env python3
import importlib
import importlib.metadata as metadata
import os
from pathlib import Path
import subprocess
import sys
import traceback

SCRIPT_PATH = Path(__file__).resolve()
REEXEC_MARKER = "--skip-install"

def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def install_runtime_stack():
    run([sys.executable, "-m", "pip", "install", "--upgrade", "vllm", "unsloth"])

def installed_versions():
    versions = {}
    for package_name in ("unsloth", "vllm", "transformers", "trl", "torch"):
        try:
            versions[package_name] = metadata.version(package_name)
        except Exception as exc:
            versions[package_name] = f"MISSING:{exc.__class__.__name__}"
    return versions

def run_repro():
    print("VERSIONS", installed_versions(), flush=True)
    importlib.invalidate_caches()
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-4B",
        max_seq_length=1024,
        load_in_4bit=False,
        load_in_8bit=False,
        fast_inference=True,
        full_finetuning=True,
        gpu_memory_utilization=0.5,
        trust_remote_code=True,
    )
    print("LOAD_OK", type(model).__name__, type(tokenizer).__name__, flush=True)

def main():
    if REEXEC_MARKER not in sys.argv:
        install_runtime_stack()
        os.execv(sys.executable, [sys.executable, str(SCRIPT_PATH), REEXEC_MARKER])

    try:
        run_repro()
        return 0
    except Exception:
        traceback.print_exc()
        return 1

raise SystemExit(main())
```

Question:

Is this combination still intentionally unsupported in current releases after #3768, or is there a supported path planned for:

- full finetuning
- with vLLM / `fast_inference=True`

on A100?
