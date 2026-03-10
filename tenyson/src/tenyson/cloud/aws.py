import importlib
import inspect
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict

try:
    import boto3  # type: ignore[import-not-found]
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - optional dependency guard
    boto3 = None

    class ClientError(Exception):
        pass


from tenyson.cloud.base import BaseCloudManager, JobFailedError, _red_print
from tenyson.cloud.runtime_deps import runtime_pip_install_command
from tenyson.core.hf_checkpoint import resolve_hf_resume_revision
from tenyson.core.run_name import resolve_required_run_name
from tenyson.core.run_config import materialize_run_config
from tenyson.jobs.hf_repo import unique_repo_id
from tenyson.jobs.result import JobResult


class AWSManager(BaseCloudManager):
    """
    EC2-based cloud manager.

    This is a library-friendly refactor of `src/run_aws.py`. It syncs the
    current repository to an EC2 instance, then invokes `python -m tenyson.runner`
    remotely to execute the job.
    """

    def __init__(
        self,
        instance_type: str = "g5.2xlarge",
        region: str = "us-east-1",
        key_name: str | None = None,
        key_path: str | None = None,
        security_group: str | None = None,
        subnet: str | None = None,
        profile: str | None = None,
        ami: str | None = None,
        auto_terminate: bool = True,
        use_spot: bool = False,
        spot_max_price: str | None = None,
        skip_runtime_install: bool = False,
    ):
        super().__init__(auto_terminate=auto_terminate)
        self.instance_type = instance_type
        self.region = region
        self.key_name = key_name
        self.key_path = key_path
        self.security_group = security_group
        self.subnet = subnet
        self.profile = profile
        self.ami = ami
        self.use_spot = use_spot
        self.spot_max_price = spot_max_price
        self.skip_runtime_install = skip_runtime_install

    @classmethod
    def from_env(cls, **overrides: Any) -> "AWSManager":
        skip_runtime_install = os.getenv(
            "TENYSON_AWS_SKIP_RUNTIME_INSTALL", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        config: Dict[str, Any] = {
            "instance_type": os.getenv("TENYSON_AWS_INSTANCE_TYPE", "g5.2xlarge"),
            "region": os.getenv("TENYSON_AWS_REGION", "us-east-1"),
            "key_name": os.getenv("TENYSON_AWS_KEY_NAME"),
            "key_path": os.getenv("TENYSON_AWS_KEY_PATH"),
            "security_group": os.getenv("TENYSON_AWS_SECURITY_GROUP"),
            "subnet": os.getenv("TENYSON_AWS_SUBNET") or None,
            "profile": os.getenv("AWS_PROFILE") or None,
            "ami": os.getenv("TENYSON_AWS_AMI") or None,
            "auto_terminate": True,
            "use_spot": False,
            "spot_max_price": os.getenv("TENYSON_AWS_SPOT_MAX_PRICE") or None,
            "skip_runtime_install": skip_runtime_install,
        }
        config.update(
            {key: value for key, value in overrides.items() if value is not None}
        )

        required_env = {
            "TENYSON_AWS_KEY_NAME": config["key_name"],
            "TENYSON_AWS_KEY_PATH": config["key_path"],
            "TENYSON_AWS_SECURITY_GROUP": config["security_group"],
        }
        missing = [name for name, value in required_env.items() if not value]
        if missing:
            raise ValueError(
                "Missing required AWS environment variables for this experiment: "
                + ", ".join(missing)
            )
        return cls(**config)

    @classmethod
    def factory_from_env(cls, **overrides: Any) -> Callable[[], "AWSManager"]:
        return lambda: cls.from_env(**overrides)

    # ---- Helpers -----------------------------------------------------

    def _get_session(self):
        if boto3 is None:
            raise ImportError(
                "boto3 is required for AWSManager. Install boto3 to run AWS experiments."
            )
        return boto3.Session(profile_name=self.profile, region_name=self.region)

    def _get_latest_dlami(self, ec2_client) -> str:
        response = ec2_client.describe_images(
            Filters=[
                {
                    "Name": "name",
                    "Values": [
                        "Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *"
                    ],
                },
                {"Name": "state", "Values": ["available"]},
            ],
            Owners=["amazon"],
        )
        images = response["Images"]
        if not images:
            raise ValueError("Could not find a valid Ubuntu Deep Learning AMI.")
        images.sort(key=lambda x: x["CreationDate"], reverse=True)
        return images[0]["ImageId"]

    def _run_ssh_command(
        self, host: str, key_path: str, user: str, command: str, stream: bool = True
    ):
        ssh_cmd = [
            "ssh",
            "-i",
            key_path,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            f"{user}@{host}",
            command,
        ]
        if stream:
            result = subprocess.run(ssh_cmd)
            return result.returncode == 0
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout

    def _remote_activate_command(self) -> str:
        activation_attempts = [
            "source /opt/conda/bin/activate pytorch",
            'source "$HOME/anaconda3/bin/activate" pytorch',
            'source "$HOME/miniconda3/bin/activate" pytorch',
            '(command -v conda >/dev/null 2>&1 && eval "$(conda shell.bash hook)" && conda activate pytorch)',
        ]
        return "( " + " || ".join(activation_attempts) + " || true )"

    def _rsync_to_host(
        self, host: str, key_path: str, user: str, local_dir: str, remote_dir: str
    ):
        rsync_cmd = [
            "rsync",
            "-avz",
            "--exclude",
            ".git",
            "--exclude",
            "outputs",
            "--exclude",
            "__pycache__",
            "-e",
            f"ssh -i {key_path} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
            f"{local_dir}/",
            f"{user}@{host}:{remote_dir}/",
        ]
        result = subprocess.run(rsync_cmd)
        return result.returncode == 0

    def _resolve_local_project_root(self, repo_root: str) -> Path:
        """
        Resolve the local project root that contains src-layout package files.
        Supports invoking from either project root or one directory above.
        """
        root = Path(repo_root).resolve()
        candidates = [root, root / "tenyson"]
        for candidate in candidates:
            if (candidate / "pyproject.toml").is_file() and (
                candidate / "src" / "tenyson" / "runner.py"
            ).is_file():
                return candidate
        return root

    def _resolve_task_spec(self, task: Any, repo_root: str) -> str:
        """
        Prefer module:Class for importable task modules; fall back to a task file path
        relative to the synced repo root for file-loaded plugins.
        """
        module_path = task.__class__.__module__
        class_name = task.__class__.__name__

        task_file = None
        module = sys.modules.get(module_path)
        if module is not None:
            task_file = getattr(module, "__file__", None)
        if task_file is None:
            try:
                task_file = inspect.getsourcefile(task.__class__)
            except (OSError, TypeError):
                task_file = None
        if task_file:
            abs_repo = Path(repo_root).resolve()
            abs_task = Path(task_file).resolve()
            try:
                rel = abs_task.relative_to(abs_repo)
            except ValueError:
                rel = None
            if rel is not None:
                return str(rel)

        try:
            importlib.import_module(module_path)
            return f"{module_path}:{class_name}"
        except Exception:  # noqa: BLE001
            pass

        return f"{module_path}:{class_name}"

    # ---- Public API --------------------------------------------------

    def run(self, job: Any) -> JobResult:
        if not (self.key_name and self.key_path and self.security_group):
            raise ValueError(
                "AWSManager requires key_name, key_path, and security_group for cloud execution. "
                "Provide all three to run on EC2."
            )

        session = self._get_session()
        ec2 = session.client("ec2")
        ec2_resource = session.resource("ec2")

        ami_id = self.ami or self._get_latest_dlami(ec2)
        print(f"[AWSManager] Using AMI: {ami_id} in {self.region}")

        print(f"[AWSManager] Launching EC2 instance: {self.instance_type}...")
        run_args: Dict[str, Any] = {
            "ImageId": ami_id,
            "InstanceType": self.instance_type,
            "KeyName": self.key_name,
            "SecurityGroupIds": [self.security_group],
            "MinCount": 1,
            "MaxCount": 1,
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {"VolumeSize": 250, "VolumeType": "gp3"},
                }
            ],
        }
        if self.subnet:
            run_args["SubnetId"] = self.subnet
        if self.use_spot:
            run_args["InstanceMarketOptions"] = {"MarketType": "spot"}
            if self.spot_max_price is not None:
                run_args["InstanceMarketOptions"]["SpotOptions"] = {
                    "MaxPrice": self.spot_max_price
                }

        instances = ec2_resource.create_instances(**run_args)
        instance = instances[0]

        print(f"[AWSManager] Instance {instance.id} launched. Waiting for it to run...")
        instance.wait_until_running()
        instance.reload()
        public_ip = instance.public_ip_address or instance.private_ip_address
        print(f"[AWSManager] Instance is running. IP: {public_ip}")

        user = "ubuntu"
        job_type = "sft"
        from tenyson.jobs.sft import SFTJob as _S
        from tenyson.jobs.rl import RLJob as _R
        from tenyson.jobs.eval import EvalJob as _E
        from tenyson.core.telemetry import (
            TelemetryClient,
            record_run_result,
            record_run_summary,
            resolve_required_telemetry_context,
            wait_for_run_result,
        )

        if isinstance(job, _R):
            job_type = "rl"
        elif isinstance(job, _E):
            job_type = "eval"

        run_name = resolve_required_run_name(job.config, job_type)
        db_url, experiment_id = resolve_required_telemetry_context(job.config)
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

        def _finalize_failure(
            failure_reason: str,
            *,
            spot_interruption: bool = False,
            include_resume_target: bool = False,
        ) -> JobResult:
            hf_repo_id = None
            hf_revision = None
            if include_resume_target:
                hf_repo_id, hf_revision = _resolve_failed_resume_target()
            result = JobResult(
                run_id=run_name,
                status="failed",
                total_time_seconds=0.0,
                hf_repo_id=hf_repo_id,
                hf_revision=hf_revision,
                failure_reason=failure_reason,
                instance_id=instance.id,
                spot_interruption=spot_interruption,
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
            return result

        print("[AWSManager] Waiting for SSH to become available...")
        max_retries = 30
        for _ in range(max_retries):
            ok, _out = self._run_ssh_command(
                public_ip, self.key_path, user, "echo 'SSH is up'", stream=False
            )
            if ok:
                break
            time.sleep(10)
        else:
            if self.auto_terminate:
                instance.terminate()
            raise JobFailedError("SSH did not become available in time.")

        print("[AWSManager] SSH is up. Preparing environment...")
        setup_cmds = [
            self._remote_activate_command(),
            "mkdir -p ~/workspace",
        ]
        if not self.skip_runtime_install:
            setup_cmds.insert(1, runtime_pip_install_command())
        setup_cmd = " && ".join(setup_cmds)
        setup_success = self._run_ssh_command(
            public_ip,
            self.key_path,
            user,
            f"bash -lc '{setup_cmd}'",
        )
        if not setup_success:
            if self.auto_terminate:
                print(f"[AWSManager] Terminating instance {instance.id}...")
                instance.terminate()
            failure_reason = (
                "Failed to prepare the remote Python environment on the EC2 instance."
            )
            result = _finalize_failure(failure_reason=failure_reason)
            _red_print(
                f"[TENYSON] Step failed (AWS): {failure_reason} (instance_id={instance.id})"
            )
            return result

        print("[AWSManager] Syncing codebase to instance...")
        repo_root = os.getcwd()
        local_project_root = self._resolve_local_project_root(repo_root)
        local_project_rel = os.path.relpath(
            local_project_root, Path(repo_root).resolve()
        )
        if local_project_rel == ".":
            remote_project_root = "~/workspace"
        else:
            remote_project_root = f"~/workspace/{local_project_rel}"

        config_path = materialize_run_config(
            config=job.config,
            project_root=Path(local_project_root),
            job_type=job_type,
            run_name=run_name,
        )
        config_rel_path = os.path.relpath(str(config_path), str(local_project_root))

        sync_ok = self._rsync_to_host(
            public_ip, self.key_path, user, repo_root, "~/workspace"
        )
        if not sync_ok:
            if self.auto_terminate:
                print(f"[AWSManager] Terminating instance {instance.id}...")
                instance.terminate()
            failure_reason = (
                "Failed to sync code/config to instance via rsync. "
                "Remote run was not started."
            )
            result = _finalize_failure(failure_reason=failure_reason)
            _red_print(
                f"[TENYSON] Step failed (AWS): {failure_reason} (instance_id={instance.id})"
            )
            return result

        # HF / W&B env vars.
        env_exports = [
            "export TENYSON_EXECUTION_MODE=cloud",
            "export TENYSON_GPU_PROVIDER=aws",
        ]
        for var in ["HF_TOKEN", "WANDB_API_KEY"]:
            val = os.environ.get(var)
            if val:
                env_exports.append(f"export {var}={val}")

        task = job.task
        try:
            task_spec = self._resolve_task_spec(task, repo_root)
        except Exception:
            if self.auto_terminate:
                print(
                    "[AWSManager] Terminating instance "
                    f"{instance.id} after local task resolution failure..."
                )
                instance.terminate()
            raise

        env_chain = " && ".join(env_exports) if env_exports else ""
        remote_cmd = (
            self._remote_activate_command()
            + (f" && {env_chain}" if env_chain else "")
            + f" && cd {remote_project_root} && "
            + "export PYTHONPATH=src:${PYTHONPATH} && "
            + "python3 -m tenyson.runner "
            + f"--job-type {job_type} "
            + f"--config {config_rel_path} "
            + f"--task-module {task_spec}"
        )

        print(f"[AWSManager] Executing remote job: {remote_cmd}")
        success = self._run_ssh_command(
            public_ip, self.key_path, user, f"bash -lc '{remote_cmd}'"
        )
        if not success:
            # Get instance state reason (e.g. Spot interruption).
            failure_reason = "Remote job failed."
            spot_interruption = False
            try:
                instance.reload()
                state_reason = getattr(instance, "state_reason", None)
                if state_reason and isinstance(state_reason, dict):
                    code = state_reason.get("Code") or ""
                    msg = state_reason.get("Message") or ""
                    if "Spot" in code or "Spot" in msg:
                        spot_interruption = True
                        failure_reason = (
                            f"Spot instance interrupted: {code} {msg}".strip()
                        )
                    else:
                        failure_reason = code or msg or failure_reason
            except Exception:  # noqa: S110
                pass
            if self.auto_terminate:
                print(f"[AWSManager] Terminating instance {instance.id}...")
                instance.terminate()
            result = _finalize_failure(
                failure_reason=failure_reason,
                spot_interruption=spot_interruption,
                include_resume_target=True,
            )
            _red_print(
                f"[TENYSON] Step failed (AWS): {failure_reason} (instance_id={instance.id})"
            )
            return result

        if self.auto_terminate:
            print(f"[AWSManager] Terminating instance {instance.id}...")
            instance.terminate()

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
                "Remote run completed but canonical run result was not available in "
                f"telemetry DB: {exc}"
            )
            result = _finalize_failure(
                failure_reason=failure_reason,
                include_resume_target=True,
            )
            _red_print(
                f"[TENYSON] Step failed (AWS): {failure_reason} (instance_id={instance.id})"
            )
            return result
