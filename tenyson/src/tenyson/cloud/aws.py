import importlib
import inspect
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

from tenyson.cloud.base import BaseCloudManager, JobFailedError, _red_print
from tenyson.core.run_config import materialize_run_config
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

    # ---- Helpers -----------------------------------------------------

    def _get_session(self):
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
        try:
            importlib.import_module(module_path)
            return f"{module_path}:{class_name}"
        except Exception:  # noqa: BLE001
            pass

        task_file = inspect.getsourcefile(task.__class__)
        if task_file:
            abs_repo = Path(repo_root).resolve()
            abs_task = Path(task_file).resolve()
            try:
                rel = abs_task.relative_to(abs_repo)
            except ValueError:
                rel = None
            if rel is not None:
                return str(rel)

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
            "source activate pytorch",
            "pip install unsloth vllm",
            "mkdir -p ~/workspace",
        ]
        setup_cmd = " && ".join(setup_cmds)
        self._run_ssh_command(public_ip, self.key_path, user, f"bash -c '{setup_cmd}'")

        print("[AWSManager] Syncing codebase to instance...")
        repo_root = os.getcwd()
        local_project_root = self._resolve_local_project_root(repo_root)
        local_project_rel = os.path.relpath(local_project_root, Path(repo_root).resolve())
        if local_project_rel == ".":
            remote_project_root = "~/workspace"
        else:
            remote_project_root = f"~/workspace/{local_project_rel}"

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

        train_cfg = job.config.get("training", {})
        eval_cfg = job.config.get("evaluation", {})
        run_name = train_cfg.get("run_name") or eval_cfg.get("run_name") or job.run_id
        db_url, experiment_id = resolve_required_telemetry_context(job.config)
        telemetry_client = TelemetryClient(db_url=db_url)

        def _finalize_failure(
            failure_reason: str,
            *,
            spot_interruption: bool = False,
        ) -> JobResult:
            result = JobResult(
                run_id=run_name,
                status="failed",
                total_time_seconds=0.0,
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

        config_path = materialize_run_config(
            config=job.config,
            project_root=Path(local_project_root),
            job_type=job_type,
            run_name=run_name,
        )
        config_rel_path = os.path.relpath(str(config_path), str(local_project_root))

        sync_ok = self._rsync_to_host(public_ip, self.key_path, user, repo_root, "~/workspace")
        if not sync_ok:
            if self.auto_terminate:
                print(f"[AWSManager] Terminating instance {instance.id}...")
                instance.terminate()
            failure_reason = (
                "Failed to sync code/config to instance via rsync. "
                "Remote run was not started."
            )
            result = _finalize_failure(failure_reason=failure_reason)
            _red_print(f"[TENYSON] Step failed (AWS): {failure_reason} (instance_id={instance.id})")
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
        task_spec = self._resolve_task_spec(task, repo_root)

        env_chain = " && ".join(env_exports) if env_exports else ""
        remote_cmd = (
            "source activate pytorch"
            + (f" && {env_chain}" if env_chain else "")
            + f" && cd {remote_project_root} && "
            + "export PYTHONPATH=src:${PYTHONPATH} && "
            + "python -m tenyson.runner "
            + f"--job-type {job_type} "
            + f"--config {config_rel_path} "
            + f"--task-module {task_spec}"
        )

        print(f"[AWSManager] Executing remote job: {remote_cmd}")
        success = self._run_ssh_command(
            public_ip, self.key_path, user, f"bash -c '{remote_cmd}'"
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
                        failure_reason = f"Spot instance interrupted: {code} {msg}".strip()
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
            )
            _red_print(f"[TENYSON] Step failed (AWS): {failure_reason} (instance_id={instance.id})")
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
            result = _finalize_failure(failure_reason=failure_reason)
            _red_print(f"[TENYSON] Step failed (AWS): {failure_reason} (instance_id={instance.id})")
            return result
