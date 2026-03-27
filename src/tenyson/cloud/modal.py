import contextlib
import importlib
import inspect
import os
from collections import deque
from dataclasses import dataclass
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List
import yaml

from tenyson.cloud.base import BaseCloudManager, _red_print
from tenyson.cloud.runtime_deps import runtime_pip_install_command
from tenyson.core.control import request_stop
from tenyson.core.hf_checkpoint import resolve_hf_resume_revision
from tenyson.core import wandb_store
from tenyson.core.run_name import resolve_required_run_name
from tenyson.jobs.hf_repo import unique_repo_id
from tenyson.jobs.result import JobResult


def _is_truthy_env(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_modal_python_version(value: Any) -> str:
    raw = str(value or "").strip()
    parts = raw.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(
            "Modal Python version must look like '3.11' or '3.12' or '3.13'. "
            f"Got {raw!r}."
        )
    major, minor = (int(part) for part in parts)
    return f"{major}.{minor}"


def _resolve_modal_python_version(explicit: Any | None = None) -> str:
    if explicit is not None and str(explicit).strip():
        return _normalize_modal_python_version(explicit)

    env_value = str(os.getenv("TENYSON_MODAL_PYTHON_VERSION") or "").strip()
    if env_value:
        return _normalize_modal_python_version(env_value)

    major = int(sys.version_info.major)
    minor = int(sys.version_info.minor)
    if major == 3 and minor <= 10:
        # vLLM 0.18.0 currently trips over Python <= 3.10 during Unsloth's
        # fast-inference startup path, so prefer a newer worker runtime by
        # default while still allowing an explicit override.
        return "3.12"
    return f"{major}.{minor}"


def _finish_local_failed_run_record(
    *,
    experiment_id: str,
    phase: str,
    run_name: str,
    attempt_token: str | None,
) -> None:
    try:
        current_run = wandb_store.active_run()
        if current_run is None:
            return
        expected_id = wandb_store.build_run_id(
            experiment_id,
            phase,
            run_name,
            attempt_token=attempt_token,
        )
        if getattr(current_run, "id", None) != expected_id:
            return
        import wandb

        wandb.finish()
    except Exception:  # noqa: BLE001
        return


_DETACHED_MODAL_APP_RE = re.compile(
    r"Spawned (?:detached )?Modal app (?P<app_id>ap-[A-Za-z0-9]+) "
    r"with function call (?P<function_call_id>fc-[A-Za-z0-9]+)\."
)
_MODAL_CLOSE_GRACE_SECONDS = 90.0
_MODAL_APP_STOP_TIMEOUT_SECONDS = 30.0
_MODAL_TELEMETRY_START_TIMEOUT_SECONDS = 180.0
_MODAL_RUN_RESULT_TIMEOUT_BUFFER_SECONDS = 300.0


@dataclass(frozen=True)
class ActiveModalLaunch:
    app_id: str
    function_call_id: str
    backend_ref: str
    experiment_id: str
    phase: str
    run_name: str
    attempt_token: str | None


def _drain_subprocess_stream(
    stream: Any,
    *,
    writer: Any,
    recent_lines: deque[str],
    on_line: Callable[[str], None] | None = None,
) -> None:
    if stream is None:
        return
    try:
        for line in iter(stream.readline, ""):
            writer.write(line)
            writer.flush()
            stripped = line.rstrip("\n")
            recent_lines.append(stripped)
            if on_line is not None:
                on_line(stripped)
    except (OSError, ValueError):
        # The parent can close the pipe after the child exits to avoid hanging
        # forever on inherited file descriptors from grandchildren.
        return
    finally:
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass


def _run_subprocess_with_streaming_logs(
    cmd: List[str],
    *,
    env: Dict[str, str],
    cwd: str | None = None,
    close_fds: bool = True,
    recent_line_limit: int = 200,
    on_line: Callable[[str], None] | None = None,
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
                "on_line": on_line,
            },
            daemon=True,
        ),
        threading.Thread(
            target=_drain_subprocess_stream,
            kwargs={
                "stream": process.stderr,
                "writer": sys.stderr,
                "recent_lines": stderr_tail,
                "on_line": on_line,
            },
            daemon=True,
        ),
    ]
    for worker in workers:
        worker.start()

    try:
        returncode = process.wait()
    except BaseException:
        with contextlib.suppress(Exception):
            process.terminate()
        with contextlib.suppress(Exception):
            process.wait(timeout=5.0)
        with contextlib.suppress(Exception):
            process.kill()
        raise
    finally:
        for stream in (process.stdout, process.stderr):
            if stream is None:
                continue
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        for worker in workers:
            worker.join(timeout=5.0)

    detail_lines = list(stderr_tail) or list(stdout_tail)
    details = "\n".join(detail_lines).strip()
    return returncode, details


def _modal_run_remote(job_type: str, config_payload: Dict[str, Any], task_spec: str) -> None:
    import tempfile
    import yaml

    os.chdir("/workspace")
    os.environ["TENYSON_EXECUTION_MODE"] = "cloud"
    os.environ["TENYSON_GPU_PROVIDER"] = "modal"
    vllm_cfg = (
        config_payload.get("vllm", {})
        if isinstance(config_payload, dict)
        else {}
    )
    vllm_enabled = bool(vllm_cfg.get("enabled", False))
    disable_flashinfer = vllm_cfg.get("disable_flashinfer")
    if disable_flashinfer is None:
        disable_flashinfer = True
    if vllm_enabled and bool(disable_flashinfer):
        # Set this before spawning the Python runner so Unsloth sees it before
        # any vLLM-related imports.
        os.environ["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        print(
            "[ModalManager] Runtime env: "
            f"UNSLOTH_VLLM_NO_FLASHINFER=1 "
            f"VLLM_USE_FLASHINFER_SAMPLER=0",
            flush=True,
        )
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


def _wait_for_modal_function_call(
    function_call_id: str,
    *,
    poll_timeout_seconds: float = 30.0,
    overall_timeout_seconds: float | None = None,
) -> None:
    import modal

    function_call = modal.FunctionCall.from_id(function_call_id)
    poll_timeout_seconds = max(0.1, float(poll_timeout_seconds))
    deadline = (
        None
        if overall_timeout_seconds is None
        else time.monotonic() + max(0.1, float(overall_timeout_seconds))
    )

    while True:
        timeout = poll_timeout_seconds
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    "Timed out waiting for Modal function call "
                    f"{function_call_id} to complete."
                )
            timeout = min(timeout, remaining)
        try:
            function_call.get(timeout=timeout)
            return
        except TimeoutError:
            continue


def _wait_for_run_start_or_terminal_result(
    *,
    client: Any,
    experiment_id: str,
    run_id: str,
    phase: str,
    timeout_seconds: float,
    poll_interval_seconds: float = 2.0,
    attempt_token: str | None = None,
) -> Dict[str, Any] | None:
    from tenyson.core.telemetry import get_run_result, list_live_run_heartbeats

    deadline = time.time() + max(1.0, float(timeout_seconds))
    max_age_seconds = max(90, int(timeout_seconds) + 30)
    while True:
        row = get_run_result(
            client=client,
            experiment_id=experiment_id,
            run_id=run_id,
            phase=phase,
            attempt_token=attempt_token,
            include_results_payload=False,
        )
        if row is not None:
            _results_payload, job_result_payload = row
            del _results_payload
            return job_result_payload

        for live_run in list_live_run_heartbeats(
            client,
            experiment_id=experiment_id,
            max_age_seconds=max_age_seconds,
        ):
            if live_run.run_id != run_id:
                continue
            if live_run.phase != phase:
                continue
            if live_run.attempt_token != attempt_token:
                continue
            return None

        if time.time() >= deadline:
            break
        time.sleep(max(0.1, float(poll_interval_seconds)))

    raise TimeoutError(
        "Timed out waiting for detached Modal launch to publish a live "
        f"heartbeat or terminal result (experiment_id={experiment_id}, "
        f"run_id={run_id}, phase={phase})."
    )


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
        python_version: str | None = None,
    ):
        super().__init__(auto_terminate=auto_terminate)
        self.gpu = gpu
        self.timeout = timeout
        self.serialized = serialized
        self.python_version = _resolve_modal_python_version(python_version)
        self._active_launches: dict[str, ActiveModalLaunch] = {}
        self._active_launch_lock = threading.Lock()

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
            "T4": "T4",
            "A10G": "A10G",
            "A100": "A100-40GB",
            "A100-80GB": "A100-80GB",
            "H100": "H100",
        }
        return gpu_map.get(gpu_name, "A100-40GB")

    def _resolve_runtime_dependency_profile(self) -> str:
        gpu_name = str(self.gpu or "").strip().upper()
        if gpu_name == "T4":
            return "modal_t4_colab_compat"
        return "default"

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

    def _remember_active_launch(self, launch: ActiveModalLaunch) -> None:
        with self._active_launch_lock:
            self._active_launches[launch.app_id] = launch

    def _forget_active_launch(self, app_id: str | None) -> None:
        if not app_id:
            return
        with self._active_launch_lock:
            self._active_launches.pop(str(app_id), None)

    def _snapshot_active_launches(self) -> list[ActiveModalLaunch]:
        with self._active_launch_lock:
            return list(self._active_launches.values())

    def _forget_active_launch_for_run(
        self,
        *,
        experiment_id: str,
        phase: str,
        run_name: str,
        attempt_token: str | None,
    ) -> None:
        app_id_to_forget = None
        with self._active_launch_lock:
            for app_id, launch in self._active_launches.items():
                if launch.experiment_id != experiment_id:
                    continue
                if launch.phase != phase:
                    continue
                if launch.run_name != run_name:
                    continue
                if launch.attempt_token != attempt_token:
                    continue
                app_id_to_forget = app_id
                break
        self._forget_active_launch(app_id_to_forget)

    def close(self) -> None:
        launches = self._snapshot_active_launches()
        for launch in launches:
            stop_requested = False
            try:
                stop_requested = request_stop(
                    db_url=launch.backend_ref,
                    run_id=launch.run_name,
                    experiment_id=launch.experiment_id,
                    phase=launch.phase,
                    attempt_token=launch.attempt_token,
                    create_if_missing=True,
                )
            except Exception as exc:  # noqa: BLE001
                _red_print(
                    "[TENYSON] Warning: failed to request graceful stop for "
                    f'Modal app "{launch.app_id}" ({launch.run_name}): {exc}'
                )
            else:
                if stop_requested:
                    try:
                        from tenyson.core.telemetry import (
                            TelemetryClient,
                            wait_for_run_result,
                        )

                        wait_for_run_result(
                            client=TelemetryClient(db_url=launch.backend_ref),
                            experiment_id=launch.experiment_id,
                            run_id=launch.run_name,
                            phase=launch.phase,
                            timeout_seconds=_MODAL_CLOSE_GRACE_SECONDS,
                            poll_interval_seconds=2.0,
                            attempt_token=launch.attempt_token,
                            include_results_payload=False,
                        )
                        self._forget_active_launch(launch.app_id)
                        continue
                    except TimeoutError:
                        pass
                    except Exception as exc:  # noqa: BLE001
                        _red_print(
                            "[TENYSON] Warning: graceful telemetry wait failed for "
                            f'Modal app "{launch.app_id}" ({launch.run_name}): {exc}'
                        )

            if not self.auto_terminate:
                continue
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "modal",
                        "app",
                        "stop",
                        launch.app_id,
                    ],
                    check=False,
                    timeout=_MODAL_APP_STOP_TIMEOUT_SECONDS,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=os.environ.copy(),
                )
            except Exception as exc:  # noqa: BLE001
                _red_print(
                    "[TENYSON] Warning: failed to stop detached Modal app "
                    f'"{launch.app_id}" ({launch.run_name}): {exc}'
                )
            finally:
                self._forget_active_launch(launch.app_id)

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
            modal.Image.debian_slim(python_version=self.python_version)
            .run_commands("apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*")
            .run_commands(
                runtime_pip_install_command(
                    profile=self._resolve_runtime_dependency_profile()
                )
            )
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
        function_call_id = None
        app_id = None
        with modal.enable_output():
            # Launch the worker inside a detached app, then exit the app context
            # cleanly so the remote job is no longer coupled to local polling.
            with app.run(detach=True):
                function_call = run_remote.spawn(job_type, config_payload, task_spec)
                function_call_id = str(function_call.object_id)
                app_id = str(app.app_id)
                print(
                    "[ModalManager] Spawned detached Modal app "
                    f"{app_id} with function call {function_call_id}.",
                    flush=True,
                )
                if not function_call_id:
                    raise RuntimeError(
                        "Modal job launch did not return a function call id."
                    )
        print(
            "[ModalManager] Detached Modal job launched. Terminal status will "
            "now be tracked via telemetry.",
            flush=True,
        )

    def _run_modal_job_via_launcher(
        self,
        *,
        job_type: str,
        config_payload: Dict[str, Any],
        task_spec: str,
        local_project_root: str,
        backend_ref: str,
        experiment_id: str,
        run_name: str,
        attempt_token: str | None,
    ) -> None:
        config_path = _write_temp_config_payload(job_type, config_payload)
        returncode = -1
        details = ""
        active_launch: ActiveModalLaunch | None = None
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

            def _capture_launch(line: str) -> None:
                nonlocal active_launch
                if active_launch is not None:
                    return
                match = _DETACHED_MODAL_APP_RE.search(str(line or ""))
                if match is None:
                    return
                active_launch = ActiveModalLaunch(
                    app_id=match.group("app_id"),
                    function_call_id=match.group("function_call_id"),
                    backend_ref=backend_ref,
                    experiment_id=experiment_id,
                    phase=job_type,
                    run_name=run_name,
                    attempt_token=attempt_token,
                )
                self._remember_active_launch(active_launch)

            returncode, details = _run_subprocess_with_streaming_logs(
                cmd,
                cwd=None,
                close_fds=True,
                env=env,
                on_line=_capture_launch,
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
            get_run_result,
            list_live_run_heartbeats,
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
        telemetry_client = TelemetryClient(db_url=backend_ref)
        run_result_timeout_seconds = int(self.timeout) + int(
            _MODAL_RUN_RESULT_TIMEOUT_BUFFER_SECONDS
        )

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

        def _looks_like_post_run_teardown_crash(error_text: str) -> bool:
            normalized = str(error_text or "")
            crash_markers = (
                "code -11",
                "code -6",
                "segmentation fault",
                "libc10_cuda",
                "mempool::~mempool",
                "processgroupnccl",
            )
            lowered = normalized.lower()
            return any(marker in lowered for marker in crash_markers)

        def _has_active_launch_for_run() -> bool:
            for launch in self._snapshot_active_launches():
                if launch.experiment_id != experiment_id:
                    continue
                if launch.phase != job_type:
                    continue
                if launch.run_name != run_name:
                    continue
                if launch.attempt_token != attempt_token:
                    continue
                return True
            return False

        def _recover_terminal_result_after_launcher_failure() -> JobResult | None:
            try:
                _results_payload, job_result_payload = wait_for_run_result(
                    client=telemetry_client,
                    experiment_id=experiment_id,
                    run_id=run_name,
                    phase=job_type,
                    timeout_seconds=20,
                    poll_interval_seconds=1.0,
                    attempt_token=attempt_token,
                    include_results_payload=False,
                )
                del _results_payload
            except Exception:  # noqa: BLE001
                return None
            try:
                return JobResult.from_dict(job_result_payload)
            except Exception:  # noqa: BLE001
                return None

        def _recover_started_run_after_launcher_failure() -> tuple[JobResult | None, bool]:
            try:
                job_result_payload = _wait_for_run_start_or_terminal_result(
                    client=telemetry_client,
                    experiment_id=experiment_id,
                    run_id=run_name,
                    phase=job_type,
                    timeout_seconds=min(
                        float(self.timeout),
                        _MODAL_TELEMETRY_START_TIMEOUT_SECONDS,
                    ),
                    poll_interval_seconds=2.0,
                    attempt_token=attempt_token,
                )
            except Exception:
                return None, False
            if job_result_payload is None:
                return None, True
            try:
                return JobResult.from_dict(job_result_payload), False
            except Exception:  # noqa: BLE001
                return None, False

        def _wait_for_launch_telemetry_signal() -> JobResult | None:
            try:
                job_result_payload = _wait_for_run_start_or_terminal_result(
                    client=telemetry_client,
                    experiment_id=experiment_id,
                    run_id=run_name,
                    phase=job_type,
                    timeout_seconds=min(
                        float(self.timeout),
                        _MODAL_TELEMETRY_START_TIMEOUT_SECONDS,
                    ),
                    poll_interval_seconds=2.0,
                    attempt_token=attempt_token,
                )
            except Exception:
                return None
            if job_result_payload is None:
                return None
            try:
                return JobResult.from_dict(job_result_payload)
            except Exception:  # noqa: BLE001
                return None

        try:
            # Run synchronously; on failure return failed JobResult instead of raising.
            launch_handed_off_to_telemetry = False
            try:
                self._run_modal_job_via_launcher(
                    job_type=job_type,
                    config_payload=job.config,
                    task_spec=task_spec,
                    local_project_root=local_project_root,
                    backend_ref=backend_ref,
                    experiment_id=experiment_id,
                    run_name=run_name,
                    attempt_token=attempt_token,
                )
            except Exception as exc:  # noqa: BLE001
                should_try_recover = (
                    attempt_token is not None
                    or _looks_like_post_run_teardown_crash(str(exc))
                )
                if should_try_recover:
                    recovered_result = _recover_terminal_result_after_launcher_failure()
                    if recovered_result is not None:
                        self._forget_active_launch_for_run(
                            experiment_id=experiment_id,
                            phase=job_type,
                            run_name=run_name,
                            attempt_token=attempt_token,
                        )
                        _red_print(
                            "[TENYSON] Modal launcher failed, but telemetry "
                            "already has a terminal job result. Using recovered "
                            "telemetry result."
                        )
                        return recovered_result
                    if _has_active_launch_for_run():
                        recovered_launch_result, launch_handed_off_to_telemetry = (
                            _recover_started_run_after_launcher_failure()
                        )
                        if recovered_launch_result is not None:
                            self._forget_active_launch_for_run(
                                experiment_id=experiment_id,
                                phase=job_type,
                                run_name=run_name,
                                attempt_token=attempt_token,
                            )
                            _red_print(
                                "[TENYSON] Modal launcher failed, but telemetry "
                                "already handed this run off as live. Using "
                                "telemetry result."
                            )
                            return recovered_launch_result
                        if launch_handed_off_to_telemetry:
                            _red_print(
                                "[TENYSON] Modal launcher exited after the detached "
                                "job had already started. Continuing to track the "
                                "run via telemetry."
                            )

                if not launch_handed_off_to_telemetry:
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
                    _finish_local_failed_run_record(
                        experiment_id=experiment_id,
                        phase=job_type,
                        run_name=run_name,
                        attempt_token=attempt_token,
                    )
                    _red_print(f"[TENYSON] Step failed (Modal): {exc}")
                    return result

            if not launch_handed_off_to_telemetry:
                launch_terminal_result = _wait_for_launch_telemetry_signal()
                if launch_terminal_result is not None:
                    self._forget_active_launch_for_run(
                        experiment_id=experiment_id,
                        phase=job_type,
                        run_name=run_name,
                        attempt_token=attempt_token,
                    )
                    return launch_terminal_result

                live_match_seen = any(
                    live_run.run_id == run_name
                    and live_run.phase == job_type
                    and live_run.attempt_token == attempt_token
                    for live_run in list_live_run_heartbeats(
                        telemetry_client,
                        experiment_id=experiment_id,
                        max_age_seconds=max(
                            90,
                            int(_MODAL_TELEMETRY_START_TIMEOUT_SECONDS) + 30,
                        ),
                    )
                )
                if not live_match_seen and get_run_result(
                    client=telemetry_client,
                    experiment_id=experiment_id,
                    run_id=run_name,
                    phase=job_type,
                    attempt_token=attempt_token,
                    include_results_payload=False,
                ) is None:
                    failure_reason = (
                        "Detached Modal job never published a live telemetry heartbeat "
                        "or terminal result after launch."
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
                    _finish_local_failed_run_record(
                        experiment_id=experiment_id,
                        phase=job_type,
                        run_name=run_name,
                        attempt_token=attempt_token,
                    )
                    _red_print(f"[TENYSON] Step failed (Modal): {failure_reason}")
                    return result

            try:
                _results_payload, job_result_payload = wait_for_run_result(
                    client=telemetry_client,
                    experiment_id=experiment_id,
                    run_id=run_name,
                    phase=job_type,
                    timeout_seconds=run_result_timeout_seconds,
                    poll_interval_seconds=2.0,
                    attempt_token=attempt_token,
                    include_results_payload=False,
                )
                self._forget_active_launch_for_run(
                    experiment_id=experiment_id,
                    phase=job_type,
                    run_name=run_name,
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
                _finish_local_failed_run_record(
                    experiment_id=experiment_id,
                    phase=job_type,
                    run_name=run_name,
                    attempt_token=attempt_token,
                )
                _red_print(f"[TENYSON] Step failed (Modal): {failure_reason}")
                return result
        finally:
            pass
