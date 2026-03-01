from typing import Any, Dict, List

from tenyson.cloud.base import BaseCloudManager, _red_print
from tenyson.jobs.result import JobResult


class ModalManager(BaseCloudManager):
    """
    Modal-based cloud manager.

    This mirrors the behaviour of `src/run_modal.py`, but delegates execution
    to `python -m tenyson.runner` inside a Modal function.
    """

    def __init__(
        self,
        gpu: str = "A100",
        timeout: int = 86400,
        profile: str | None = None,
        auto_terminate: bool = True,
    ):
        super().__init__(auto_terminate=auto_terminate)
        self.gpu = gpu
        self.timeout = timeout
        self.profile = profile

    def run(self, job: Any) -> JobResult:
        try:
            import modal
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "Modal SDK not found. Please install: pip install modal"
            ) from exc

        app = modal.App("tenyson-job-runner")

        image = (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install("git")
            .pip_install(
                "torch==2.5.1",
                "vllm==0.7.3",
                "huggingface_hub",
                "hf_transfer",
                "wandb",
                "python-dotenv",
                "datasets",
                "pyyaml",
            )
            .run_commands(
                "pip install unsloth unsloth-zoo",
                "pip install git+https://github.com/huggingface/trl.git",
            )
            .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
        )

        secrets: List[Any] = []
        try:
            secrets.append(modal.Secret.from_name("huggingface-secret"))
        except modal.exception.NotFoundError:
            pass
        try:
            secrets.append(modal.Secret.from_name("wandb-secret"))
        except modal.exception.NotFoundError:
            pass

        repo_root = modal.Mount.from_local_dir(
            ".", remote_path="/workspace"
        )

        @app.function(
            image=image,
            gpu=None,
            timeout=self.timeout,
            secrets=secrets,
            mounts=[repo_root],
        )
        def run_remote(job_type: str, config_rel_path: str, task_spec: str) -> None:
            import os
            import sys

            os.chdir("/workspace")
            cmd = [
                sys.executable,
                "-m",
                "tenyson.runner",
                "--job-type",
                job_type,
                "--config",
                config_rel_path,
                "--task-module",
                task_spec,
            ]
            print(f"[ModalManager] Running on Modal: {' '.join(cmd)}", flush=True)
            import subprocess

            result = subprocess.run(cmd, env=os.environ.copy(), check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Tenyson job failed inside Modal with code {result.returncode}"
                )

        # Prepare job metadata.
        from pathlib import Path
        import json

        job_type = "sft"
        from tenyson.jobs.sft import SFTJob as _S
        from tenyson.jobs.rl import RLJob as _R
        from tenyson.jobs.eval import EvalJob as _E

        if isinstance(job, _R):
            job_type = "rl"
        elif isinstance(job, _E):
            job_type = "eval"

        task = job.task
        module_path = task.__class__.__module__
        class_name = task.__class__.__name__
        task_spec = f"{module_path}:{class_name}"

        # Write config into the repo root so it is visible under /workspace.
        cfg_dir = Path(".") / ".tenyson_modal_configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        config_rel_path = cfg_dir / f"{job_type}_job_config.json"
        with open(config_rel_path, "w", encoding="utf-8") as f:
            json.dump(job.config, f, indent=2)

        if self.profile:
            import os

            os.environ["MODAL_PROFILE"] = self.profile

        gpu_map: Dict[str, Any] = {
            "A10G": modal.gpu.A10G(),
            "A100": modal.gpu.A100(),
            "A100-80GB": modal.gpu.A100(size="80GB"),
            "H100": modal.gpu.H100(),
        }
        gpu_request = gpu_map.get(self.gpu, modal.gpu.A100())

        # Run synchronously; on failure return failed JobResult instead of raising.
        try:
            run_remote.with_options(gpu=gpu_request, timeout=self.timeout).remote(
                job_type, str(config_rel_path), task_spec
            )
        except Exception as exc:  # noqa: BLE001
            train_cfg = job.config.get("training", {})
            eval_cfg = job.config.get("evaluation", {})
            run_name = train_cfg.get("run_name") or eval_cfg.get("run_name") or job.run_id
            result = JobResult(
                run_id=run_name,
                status="failed",
                total_time_seconds=0.0,
                failure_reason=str(exc),
                instance_id=None,
                spot_interruption=None,
                local_output_dir=None,
            )
            _red_print(f"[TENYSON] Step failed (Modal): {exc}")
            return result

        # The repo is mounted read-write into the Modal container, so outputs
        # produced under /workspace/outputs are visible locally under ./outputs.
        # Try to load the JobResult written by the remote job.
        try:
            from tenyson.jobs.sft import SFTJob as _S
            from tenyson.jobs.rl import RLJob as _R
            from tenyson.jobs.eval import EvalJob as _E

            if isinstance(job, _E):
                eval_cfg = job.config.get("evaluation", {})
                run_name = eval_cfg.get("run_name", job.run_id)
                output_dir = eval_cfg.get(
                    "output_dir", f"./outputs/{run_name}"
                )
                result_filename = "job_result.json"
            else:
                train_cfg = job.config.get("training", {})
                run_name = train_cfg.get("run_name", job.run_id)
                output_dir = train_cfg.get(
                    "output_dir", f"./outputs/{run_name}"
                )
                result_filename = "results.json"

            rel_output_dir = str(output_dir).lstrip("./")
            repo_root = Path(".")
            local_result_path = repo_root / rel_output_dir / result_filename
            print(f"[ModalManager] Loading JobResult from {local_result_path}")
            with open(local_result_path, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)
            return JobResult.from_dict(data)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"ModalManager: failed to load remote JobResult after run: {exc}"
            ) from exc
