import json
import os
from pathlib import Path
import sys
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tenyson.bootstrap import ensure_local_controller_environment, load_env_file
from tenyson.cloud.modal import ModalManager
from tenyson.core.run_config import shared_overrides_from_env
from tenyson.experiment import AdapterRef, ConfigTemplates, ExperimentSession
from tenyson.loader import load_task


WORDLE_DIR = REPO_ROOT / "examples" / "wordle"
DEFAULT_ADAPTER = AdapterRef(
    repo_id="goyalayus/wordle-lora-20260324-163252-sft_main",
    revision="30a33278640fcc5bcce216adce59984bfb8f7698",
)


def main() -> None:
    ensure_local_controller_environment(anchor_file=__file__)
    explicit_experiment_id = os.getenv("TENYSON_EXPERIMENT_ID", "").strip()
    load_env_file(WORDLE_DIR / ".env", override=True)

    experiment_id = explicit_experiment_id or f"wordle_rl_smoke_{uuid4().hex[:10]}"
    os.environ["TENYSON_EXPERIMENT_ID"] = experiment_id

    modal_gpu = os.getenv("TENYSON_MODAL_GPU", "A100").strip() or "A100"
    modal_timeout = int(os.getenv("TENYSON_MODAL_TIMEOUT", "86400"))

    task = load_task(str(WORDLE_DIR / "wordle_task.py"))
    session = ExperimentSession(
        task=task,
        templates=ConfigTemplates.from_directory(REPO_ROOT / "config_templates"),
        cloud_factory=ModalManager.factory_from_env(
            auto_terminate=True,
            gpu=modal_gpu,
            timeout=modal_timeout,
        ),
        on_failure="wait",
        shared_overrides=shared_overrides_from_env(),
        parallel=False,
        report=None,
    )

    stage = session.rl(
        "modal_smoke_rl",
        run="wordle_rl_mixed",
        adapter=DEFAULT_ADAPTER,
        overrides={
            "training": {
                "max_steps": 1,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "num_generations": 2,
                "max_prompt_length": 256,
                "max_completion_length": 128,
                "hf_push_every_steps": 1,
                "save_total_limit": 1,
            },
            "vllm": {
                "gpu_memory_utilization": 0.45,
                "max_tokens": 128,
            },
            "task": {
                "synthetic_samples": 8,
            },
        },
    )
    stage.config.setdefault("telemetry", {})["attempt_token"] = uuid4().hex

    cloud = session.create_cloud()
    job = stage.job_class(stage.config, stage.task)
    result = cloud.run(job)
    payload = {
        "experiment_id": experiment_id,
        "run_id": result.run_id,
        "status": result.status,
        "wandb_url": result.wandb_url,
        "failure_reason": result.failure_reason,
        "hf_repo_id": result.hf_repo_id,
        "hf_revision": result.hf_revision,
    }
    print(json.dumps(payload, indent=2))

    if str(result.status).lower() != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
