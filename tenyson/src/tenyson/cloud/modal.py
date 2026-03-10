import importlib
import inspect
import os
import sys
from typing import Any, Callable, Dict, List

from tenyson.cloud.base import BaseCloudManager, _red_print
from tenyson.cloud.rds_access import prepare_modal_rds_access
from tenyson.cloud.runtime_deps import runtime_pip_install_command
from tenyson.core.hf_checkpoint import resolve_hf_resume_revision
from tenyson.core.run_name import resolve_required_run_name
from tenyson.core.run_config import materialize_run_config
from tenyson.jobs.hf_repo import unique_repo_id
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

    @classmethod
    def from_env(cls, **overrides: Any) -> "ModalManager":
        timeout_value = os.getenv("TENYSON_MODAL_TIMEOUT", "86400").strip()
        try:
            timeout = int(timeout_value)
        except ValueError as exc:
            raise ValueError(
                "TENYSON_MODAL_TIMEOUT must be an integer number of seconds."
            ) from exc

        config: Dict[str, Any] = {
            "gpu": os.getenv("TENYSON_MODAL_GPU", "A100").strip() or "A100",
            "timeout": timeout,
            "profile": (
                os.getenv("TENYSON_MODAL_PROFILE") or os.getenv("MODAL_PROFILE") or None
            ),
            "auto_terminate": True,
        }
        config.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
        return cls(**config)

    @classmethod
    def factory_from_env(cls, **overrides: Any) -> Callable[[], "ModalManager"]:
        return lambda: cls.from_env(**overrides)

    def _resolve_modal_gpu_request(self) -> str:
        gpu_name = str(self.gpu or "").strip().upper()
        gpu_map = {
            "A10G": "A10G",
            "A100": "A100-40GB",
            "A100-80GB": "A100-80GB",
            "H100": "H100",
        }
        return gpu_map.get(gpu_name, "A100-40GB")

    def _resolve_local_project_root(self) -> str:
        """
        Resolve the local project root that contains src-layout package files.
        Supports invoking from either project root or one directory above.
        """
        root = os.path.abspath(".")
        candidates = [root, os.path.join(root, "tenyson")]
        for candidate in candidates:
            if os.path.isfile(
                os.path.join(candidate, "pyproject.toml")
            ) and os.path.isfile(
                os.path.join(candidate, "src", "tenyson", "runner.py")
            ):
                return os.path.abspath(candidate)
        return os.path.abspath(root)

    def _resolve_task_spec(self, task: Any, repo_root: str) -> str:
        """
        Prefer module:Class for importable task modules; fall back to a task file path
        relative to the mounted repo root for file-loaded plugins.
        """
        module_path = task.__class__.__module__
        class_name = task.__class__.__name__
        try:
            importlib.import_module(module_path)
            return f"{module_path}:{class_name}"
        except Exception:  # noqa: BLE001
            pass

        task_file = inspect.getsourcefile(task.__class__)
        if task_file:
            abs_repo = os.path.abspath(repo_root)
            abs_task = os.path.abspath(task_file)
            if abs_task.startswith(abs_repo + os.sep) or abs_task == abs_repo:
                return os.path.relpath(abs_task, abs_repo)
        return f"{module_path}:{class_name}"

    def run(self, job: Any) -> JobResult:
        try:
            import modal
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "Modal SDK not found. Please install: pip install modal"
            ) from exc

        app = modal.App("tenyson-job-runner")
        local_python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        image = (
            modal.Image.debian_slim(python_version=local_python_version)
            .run_commands(runtime_pip_install_command())
            .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
        )

        secrets: List[Any] = []
        runtime_secret_env: Dict[str, str] = {}
        hf_token = str(os.getenv("HF_TOKEN") or "").strip()
        wandb_api_key = str(os.getenv("WANDB_API_KEY") or "").strip()
        if hf_token:
            runtime_secret_env["HF_TOKEN"] = hf_token
        if wandb_api_key:
            runtime_secret_env["WANDB_API_KEY"] = wandb_api_key
        if runtime_secret_env:
            secrets.append(modal.Secret.from_dict(runtime_secret_env))

        local_project_root = self._resolve_local_project_root()
        image = image.add_local_dir(local_project_root, "/workspace")
        gpu_request = self._resolve_modal_gpu_request()

        @app.function(
            image=image,
            gpu=gpu_request,
            timeout=self.timeout,
            secrets=secrets,
            serialized=True,
        )
        def run_remote(job_type: str, config_rel_path: str, task_spec: str) -> None:
            import os
            import sys

            os.chdir("/workspace")
            os.environ["PYTHONPATH"] = f"src:{os.environ.get('PYTHONPATH', '')}"
            os.environ["TENYSON_EXECUTION_MODE"] = "cloud"
            os.environ["TENYSON_GPU_PROVIDER"] = "modal"
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

            result = subprocess.run(
                cmd,
                env=os.environ.copy(),
                check=False,
                text=True,
                capture_output=True,
            )
            if result.stdout:
                print(result.stdout, end="", flush=True)
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr, flush=True)
            if result.returncode != 0:
                details = (result.stderr or result.stdout or "").strip()
                if len(details) > 1000:
                    details = details[-1000:]
                raise RuntimeError(
                    "Tenyson job failed inside Modal with code "
                    f"{result.returncode}. {details}"
                )

        from pathlib import Path
        from tenyson.core.telemetry import (
            TelemetryClient,
            record_run_result,
            record_run_summary,
            resolve_required_telemetry_context,
            wait_for_run_result,
        )

        job_type = "sft"
        from tenyson.jobs.sft import SFTJob as _S
        from tenyson.jobs.rl import RLJob as _R
        from tenyson.jobs.eval import EvalJob as _E

        if isinstance(job, _R):
            job_type = "rl"
        elif isinstance(job, _E):
            job_type = "eval"

        task = job.task
        task_spec = self._resolve_task_spec(task, local_project_root)

        run_name = resolve_required_run_name(job.config, job_type)
        db_url, experiment_id = resolve_required_telemetry_context(job.config)
        cleanup_modal_rds_ingress = prepare_modal_rds_access()
        telemetry_client = TelemetryClient(db_url=db_url)

        def _resolve_failed_resume_target() -> tuple[str | None, str | None]:
            if job_type not in ("sft", "rl"):
                return None, None
            train_cfg = job.config.get("training", {})
            hf_repo_base = str(train_cfg.get("hf_repo_base") or "").strip()
            if not hf_repo_base:
                return None, None
            repo_id = unique_repo_id(hf_repo_base, run_name)
            if not repo_id:
                return None, None
            try:
                return repo_id, resolve_hf_resume_revision(repo_id)
            except Exception:  # noqa: BLE001
                return None, None

        config_path = materialize_run_config(
            config=job.config,
            project_root=Path(local_project_root),
            job_type=job_type,
            run_name=run_name,
        )
        config_rel_path = os.path.relpath(str(config_path), local_project_root)

        if self.profile:
            os.environ["MODAL_PROFILE"] = self.profile

        try:
            # Run synchronously; on failure return failed JobResult instead of raising.
            try:
                with modal.enable_output():
                    with app.run():
                        run_remote.remote(job_type, config_rel_path, task_spec)
            except Exception as exc:  # noqa: BLE001
                hf_repo_id, hf_revision = _resolve_failed_resume_target()
                result = JobResult(
                    run_id=run_name,
                    status="failed",
                    total_time_seconds=0.0,
                    hf_repo_id=hf_repo_id,
                    hf_revision=hf_revision,
                    failure_reason=str(exc),
                    instance_id=None,
                    spot_interruption=None,
                )
                record_run_summary(
                    client=telemetry_client,
                    experiment_id=experiment_id,
                    phase=job_type,
                    result=result,
                )
                record_run_result(
                    client=telemetry_client,
                    experiment_id=experiment_id,
                    run_id=run_name,
                    phase=job_type,
                    results_payload=result,
                    job_result_payload=result,
                )
                _red_print(f"[TENYSON] Step failed (Modal): {exc}")
                return result

            try:
                _results_payload, job_result_payload = wait_for_run_result(
                    client=telemetry_client,
                    experiment_id=experiment_id,
                    run_id=run_name,
                    phase=job_type,
                )
                return JobResult.from_dict(job_result_payload)
            except Exception as exc:  # noqa: BLE001
                failure_reason = (
                    "Modal job completed but canonical run result was not available in "
                    f"telemetry DB: {exc}"
                )
                hf_repo_id, hf_revision = _resolve_failed_resume_target()
                result = JobResult(
                    run_id=run_name,
                    status="failed",
                    total_time_seconds=0.0,
                    hf_repo_id=hf_repo_id,
                    hf_revision=hf_revision,
                    failure_reason=failure_reason,
                    instance_id=None,
                    spot_interruption=None,
                )
                record_run_summary(
                    client=telemetry_client,
                    experiment_id=experiment_id,
                    phase=job_type,
                    result=result,
                )
                record_run_result(
                    client=telemetry_client,
                    experiment_id=experiment_id,
                    run_id=run_name,
                    phase=job_type,
                    results_payload=result,
                    job_result_payload=result,
                )
                _red_print(f"[TENYSON] Step failed (Modal): {failure_reason}")
                return result
        finally:
            cleanup_modal_rds_ingress()
