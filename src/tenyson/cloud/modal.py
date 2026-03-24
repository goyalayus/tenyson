import importlib
import inspect
import os
from collections import deque
from dataclasses import dataclass
import shlex
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable, Dict, List
import yaml

from tenyson.cloud.base import BaseCloudManager, _red_print
from tenyson.cloud.rds_access import prepare_modal_rds_access
from tenyson.cloud.runtime_deps import runtime_pip_install_command
from tenyson.core.hf_checkpoint import resolve_hf_resume_revision
from tenyson.core.run_name import resolve_required_run_name
from tenyson.jobs.hf_repo import unique_repo_id
from tenyson.jobs.result import JobResult


def _is_truthy_env(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _drain_subprocess_stream(
    stream: Any,
    *,
    writer: Any,
    recent_lines: deque[str],
) -> None:
    if stream is None:
        return
    try:
        for line in iter(stream.readline, ""):
            writer.write(line)
            writer.flush()
            recent_lines.append(line.rstrip("\n"))
    finally:
        stream.close()


def _run_subprocess_with_streaming_logs(
    cmd: List[str],
    *,
    env: Dict[str, str],
    cwd: str | None = None,
    close_fds: bool = True,
    recent_line_limit: int = 200,
) -> tuple[int, str]:
    """
    Run a subprocess while forwarding stdout/stderr live and keeping a short tail
    for failure summaries.
    """
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        close_fds=close_fds,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )

    stdout_tail: deque[str] = deque(maxlen=max(1, int(recent_line_limit)))
    stderr_tail: deque[str] = deque(maxlen=max(1, int(recent_line_limit)))
    workers = [
        threading.Thread(
            target=_drain_subprocess_stream,
            kwargs={
                "stream": process.stdout,
                "writer": sys.stdout,
                "recent_lines": stdout_tail,
            },
            daemon=True,
        ),
        threading.Thread(
            target=_drain_subprocess_stream,
            kwargs={
                "stream": process.stderr,
                "writer": sys.stderr,
                "recent_lines": stderr_tail,
            },
            daemon=True,
        ),
    ]
    for worker in workers:
        worker.start()

    returncode = process.wait()
    for worker in workers:
        worker.join()

    detail_lines = list(stderr_tail) or list(stdout_tail)
    details = "\n".join(detail_lines).strip()
    return returncode, details


def _modal_run_remote(job_type: str, config_payload: Dict[str, Any], task_spec: str) -> None:
    import tempfile
    import yaml

    os.chdir("/workspace")
    os.environ["TENYSON_EXECUTION_MODE"] = "cloud"
    os.environ["TENYSON_GPU_PROVIDER"] = "modal"
    config_path = os.path.join(
        tempfile.gettempdir(),
        f"tenyson-{job_type}-config.yaml",
    )
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_payload, handle, sort_keys=False)
    cmd = [
        sys.executable,
        "-m",
        "tenyson.runner",
        "--job-type",
        job_type,
        "--config",
        config_path,
        "--task-module",
        task_spec,
    ]
    print(f"[ModalManager] Running on Modal: {' '.join(cmd)}", flush=True)
    result_code, details = _run_subprocess_with_streaming_logs(
        cmd,
        env=os.environ.copy(),
    )
    if result_code != 0:
        if len(details) > 1000:
            details = details[-1000:]
        raise RuntimeError(
            "Tenyson job failed inside Modal with code "
            f"{result_code}. {details}"
        )


def _bind_modal_run_remote(app: Any, options: Dict[str, Any]) -> Any:
    return app.function(**options)(_modal_run_remote)


def _write_temp_config_payload(job_type: str, config_payload: Dict[str, Any]) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".yaml",
        prefix=f"tenyson-{job_type}-",
        delete=False,
    ) as handle:
        yaml.safe_dump(config_payload, handle, sort_keys=False)
        return handle.name


def _build_modal_launcher_command(
    *,
    python_executable: str,
    job_type: str,
    config_path: str,
    task_spec: str,
    gpu: str,
    timeout: int,
    serialized: bool,
) -> List[str]:
    return [
        python_executable,
        "-m",
        "tenyson.cloud.modal_launcher",
        "--job-type",
        job_type,
        "--config",
        config_path,
        "--task-spec",
        task_spec,
        "--gpu",
        gpu,
        "--timeout",
        str(timeout),
        "--serialized",
        "true" if serialized else "false",
    ]


@dataclass(frozen=True)
class GitRepoSource:
    clone_url: str
    commit: str


def _build_clone_repo_command() -> str:
    script = """
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

repo_url = os.environ["TENYSON_GIT_REPO_URL"]
repo_commit = os.environ["TENYSON_GIT_COMMIT"]
git_token = str(os.environ.get("TENYSON_GIT_AUTH_TOKEN") or "").strip()
if git_token and repo_url.startswith(("https://", "http://")):
    scheme, rest = repo_url.split("://", 1)
    repo_url = f"{scheme}://x-access-token:{quote(git_token, safe='')}@{rest}"

workspace = Path("/workspace")
if workspace.exists():
    shutil.rmtree(workspace)

subprocess.run(["git", "clone", repo_url, str(workspace)], check=True)
subprocess.run(
    ["git", "-C", str(workspace), "checkout", "--detach", repo_commit],
    check=True,
)
""".strip()
    return "python3 -c " + shlex.quote(f"exec({script!r})")


def _normalize_git_clone_url(raw_url: str) -> str:
    url = str(raw_url or "").strip()
    if url.startswith("git@") and ":" in url:
        user_host, repo_path = url.split(":", 1)
        host = user_host.split("@", 1)[1]
        return f"https://{host}/{repo_path}"
    if url.startswith("ssh://git@"):
        without_scheme = url[len("ssh://git@") :]
        if "/" in without_scheme:
            host, repo_path = without_scheme.split("/", 1)
            return f"https://{host}/{repo_path}"
    return url


def _run_git_command(repo_root: str, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"Git command failed ({' '.join(args)}): {details}")
    return (result.stdout or "").strip()


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
        auto_terminate: bool = True,
        serialized: bool = False,
    ):
        super().__init__(auto_terminate=auto_terminate)
        self.gpu = gpu
        self.timeout = timeout
        self.serialized = serialized

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
            "auto_terminate": True,
            "serialized": _is_truthy_env(
                os.getenv("TENYSON_MODAL_SERIALIZED", "false")
            ),
        }
        config.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
        return cls(**config)

    def _build_run_remote_function_options(
        self,
        *,
        image: Any,
        gpu_request: str,
        secrets: List[Any],
    ) -> Dict[str, Any]:
        return {
            "image": image,
            "gpu": gpu_request,
            "timeout": self.timeout,
            "secrets": secrets,
            # Allow true branch-level concurrency by default. This can still be
            # re-enabled explicitly for debugging with TENYSON_MODAL_SERIALIZED=1.
            "serialized": self.serialized,
        }

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
        override = str(os.getenv("TENYSON_LOCAL_PROJECT_ROOT") or "").strip()
        if override:
            return os.path.abspath(override)
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
        relative to the checked-out repo root for file-loaded plugins.
        """
        module_path = task.__class__.__module__
        class_name = task.__class__.__name__

        task_file = None
        adapter_task_file = getattr(task, "__tenyson_source_path__", None)
        if adapter_task_file:
            task_file = adapter_task_file
        module = sys.modules.get(module_path)
        if task_file is None and module is not None:
            task_file = getattr(module, "__file__", None)
        if task_file is None:
            try:
                task_file = inspect.getsourcefile(task.__class__)
            except (OSError, TypeError):
                task_file = None
        if task_file:
            abs_repo = os.path.abspath(repo_root)
            abs_task = os.path.abspath(task_file)
            if abs_task.startswith(abs_repo + os.sep) or abs_task == abs_repo:
                return os.path.relpath(abs_task, abs_repo)

        try:
            importlib.import_module(module_path)
            return f"{module_path}:{class_name}"
        except Exception:  # noqa: BLE001
            pass
        return f"{module_path}:{class_name}"

    def _resolve_git_source(self, repo_root: str) -> GitRepoSource:
        remote_name = str(os.getenv("TENYSON_GIT_REMOTE", "origin")).strip() or "origin"
        override_url = str(os.getenv("TENYSON_GIT_REPO_URL") or "").strip()
        raw_url = override_url or _run_git_command(repo_root, "remote", "get-url", remote_name)
        clone_url = _normalize_git_clone_url(raw_url)
        if not clone_url.startswith(("https://", "http://")):
            raise ValueError(
                "Modal git-backed execution requires an HTTP(S) clone URL. "
                "Set TENYSON_GIT_REPO_URL if your git remote uses an unsupported format."
            )

        tracked_status = _run_git_command(
            repo_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
        if tracked_status:
            raise RuntimeError(
                "Modal execution is git-backed only. Commit tracked changes before launching "
                "so the remote worker can run the exact pushed code."
            )

        commit = str(os.getenv("TENYSON_GIT_COMMIT") or "").strip() or _run_git_command(
            repo_root, "rev-parse", "HEAD"
        )
        return GitRepoSource(clone_url=clone_url, commit=commit)

    def _run_modal_job(
        self,
        *,
        job_type: str,
        config_payload: Dict[str, Any],
        task_spec: str,
    ) -> None:
        try:
            import modal
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "Modal SDK not found. Please install: pip install modal"
            ) from exc

        app = modal.App("tenyson-job-runner")
        local_python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        local_project_root = self._resolve_local_project_root()
        git_source = self._resolve_git_source(local_project_root)

        build_secret_env: Dict[str, str] = {}
        git_auth_token = str(
            os.getenv("TENYSON_GIT_AUTH_TOKEN") or os.getenv("GITHUB_TOKEN") or ""
        ).strip()
        if git_auth_token:
            build_secret_env["TENYSON_GIT_AUTH_TOKEN"] = git_auth_token
        build_secrets = (
            [modal.Secret.from_dict(build_secret_env)] if build_secret_env else None
        )

        image = (
            modal.Image.debian_slim(python_version=local_python_version)
            .run_commands("apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*")
            .run_commands(runtime_pip_install_command())
            .run_commands(
                _build_clone_repo_command(),
                env={
                    "TENYSON_GIT_REPO_URL": git_source.clone_url,
                    "TENYSON_GIT_COMMIT": git_source.commit,
                },
                secrets=build_secrets,
            )
            .env(
                {
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    "PYTHONPATH": "/workspace/src",
                }
            )
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

        gpu_request = self._resolve_modal_gpu_request()
        run_remote = _bind_modal_run_remote(
            app,
            self._build_run_remote_function_options(
                image=image,
                gpu_request=gpu_request,
                secrets=secrets,
            ),
        )
        with app.run():
            run_remote.remote(job_type, config_payload, task_spec)

    def _run_modal_job_via_launcher(
        self,
        *,
        job_type: str,
        config_payload: Dict[str, Any],
        task_spec: str,
        local_project_root: str,
    ) -> None:
        config_path = _write_temp_config_payload(job_type, config_payload)
        returncode = -1
        details = ""
        try:
            cmd = _build_modal_launcher_command(
                python_executable=sys.executable,
                job_type=job_type,
                config_path=config_path,
                task_spec=task_spec,
                gpu=self.gpu,
                timeout=self.timeout,
                serialized=self.serialized,
            )
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["TENYSON_LOCAL_PROJECT_ROOT"] = local_project_root
            existing_pythonpath = str(env.get("PYTHONPATH") or "").strip()
            src_path = os.path.join(local_project_root, "src")
            env["PYTHONPATH"] = (
                src_path
                if not existing_pythonpath
                else os.pathsep.join([src_path, existing_pythonpath])
            )
            returncode, details = _run_subprocess_with_streaming_logs(
                cmd,
                cwd=None,
                close_fds=False,
                env=env,
            )
        finally:
            try:
                os.unlink(config_path)
            except FileNotFoundError:
                pass
        if returncode != 0:
            if len(details) > 1000:
                details = details[-1000:]
            raise RuntimeError(
                "Modal launcher subprocess failed with code "
                f"{returncode}. {details}"
            )

    def run(self, job: Any) -> JobResult:
        local_project_root = self._resolve_local_project_root()

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
        backend_ref, experiment_id = resolve_required_telemetry_context(job.config)
        attempt_token = str(
            job.config.get("telemetry", {}).get("attempt_token") or ""
        ).strip() or None
        cleanup_modal_rds_ingress = (
            prepare_modal_rds_access()
            if not str(backend_ref).startswith("wandb://")
            else (lambda: None)
        )
        telemetry_client = TelemetryClient(db_url=backend_ref)

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

        try:
            # Run synchronously; on failure return failed JobResult instead of raising.
            try:
                self._run_modal_job_via_launcher(
                    job_type=job_type,
                    config_payload=job.config,
                    task_spec=task_spec,
                    local_project_root=local_project_root,
                )
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
                    attempt_token=attempt_token,
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
                    attempt_token=attempt_token,
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
                    attempt_token=attempt_token,
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
