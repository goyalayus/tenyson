import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

from tenyson.cloud.base import BaseCloudManager, JobFailedError, _red_print
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

    def _rsync_from_host(
        self, host: str, key_path: str, user: str, remote_dir: str, local_dir: str
    ):
        rsync_cmd = [
            "rsync",
            "-avz",
            "--exclude",
            "__pycache__",
            "-e",
            f"ssh -i {key_path} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
            f"{user}@{host}:{remote_dir}/",
            f"{local_dir}/",
        ]
        result = subprocess.run(rsync_cmd)
        return result.returncode == 0

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
            "pip install unsloth unsloth-zoo",
            "pip install vllm==0.7.3",
            "pip install git+https://github.com/huggingface/trl.git",
            "pip install wandb datasets pyyaml",
            "mkdir -p ~/workspace",
        ]
        setup_cmd = " && ".join(setup_cmds)
        self._run_ssh_command(public_ip, self.key_path, user, f"bash -c '{setup_cmd}'")

        print("[AWSManager] Syncing codebase to instance...")
        repo_root = os.getcwd()
        self._rsync_to_host(public_ip, self.key_path, user, repo_root, "~/workspace")

        # Prepare remote config for this job.
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "job_config.yaml"
            with open(cfg_path, "w", encoding="utf-8") as f:
                import yaml as _yaml

                _yaml.safe_dump(job.config, f)

            # Re-rsync just the config file into workspace.
            self._rsync_to_host(
                public_ip, self.key_path, user, str(cfg_path.parent), "~/workspace"
            )

        # If resuming, sync the checkpoint (or outputs) to the instance so the path is valid.
        resume_path = job.config.get("training", {}).get("resume_from_checkpoint")
        if resume_path and os.path.isdir(resume_path):
            # Sync entire outputs tree so checkpoint path exists on remote.
            outputs_src = os.path.join(repo_root, "outputs")
            if os.path.isdir(outputs_src):
                self._rsync_to_host(
                    public_ip, self.key_path, user, outputs_src, "~/workspace/outputs"
                )

        # HF / W&B env vars.
        env_exports = []
        for var in ["HF_TOKEN", "WANDB_API_KEY"]:
            val = os.environ.get(var)
            if val:
                env_exports.append(f"export {var}={val}")

        task = job.task
        module_path = task.__class__.__module__
        class_name = task.__class__.__name__
        task_spec = f"{module_path}:{class_name}"

        job_type = "sft"
        from tenyson.jobs.sft import SFTJob as _S
        from tenyson.jobs.rl import RLJob as _R
        from tenyson.jobs.eval import EvalJob as _E

        if isinstance(job, _R):
            job_type = "rl"
        elif isinstance(job, _E):
            job_type = "eval"

        env_chain = " && ".join(env_exports) if env_exports else ""
        remote_cmd = (
            "source activate pytorch"
            + (f" && {env_chain}" if env_chain else "")
            + " && cd ~/workspace && "
            + "python -m tenyson.runner "
            + f"--job-type {job_type} "
            + "--config job_config.yaml "
            + f"--task-module {task_spec}"
        )

        print(f"[AWSManager] Executing remote job: {remote_cmd}")
        success = self._run_ssh_command(
            public_ip, self.key_path, user, f"bash -c '{remote_cmd}'"
        )
        if not success:
            # Best-effort rsync to recover checkpoints and logs.
            print("[AWSManager] Remote job failed. Syncing outputs back (best-effort)...")
            outputs_local_root = os.path.join(repo_root, "outputs")
            os.makedirs(outputs_local_root, exist_ok=True)
            self._rsync_from_host(
                public_ip,
                self.key_path,
                user,
                "~/workspace/outputs",
                outputs_local_root,
            )
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
            train_cfg = job.config.get("training", {})
            eval_cfg = job.config.get("evaluation", {})
            run_name = train_cfg.get("run_name") or eval_cfg.get("run_name") or job.run_id
            result = JobResult(
                run_id=run_name,
                status="failed",
                total_time_seconds=0.0,
                failure_reason=failure_reason,
                instance_id=instance.id,
                spot_interruption=spot_interruption,
                local_output_dir=os.path.join(repo_root, "outputs"),
            )
            _red_print(f"[TENYSON] Step failed (AWS): {failure_reason} (instance_id={instance.id})")
            return result

        # Sync remote outputs back to the local repo.
        print("[AWSManager] Syncing remote outputs back to local machine...")
        outputs_local_root = os.path.join(repo_root, "outputs")
        os.makedirs(outputs_local_root, exist_ok=True)
        self._rsync_from_host(
            public_ip,
            self.key_path,
            user,
            "~/workspace/outputs",
            outputs_local_root,
        )

        if self.auto_terminate:
            print(f"[AWSManager] Terminating instance {instance.id}...")
            instance.terminate()

        # Try to load the JobResult that was written remotely.
        try:
            from tenyson.jobs.sft import SFTJob as _S
            from tenyson.jobs.rl import RLJob as _R
            from tenyson.jobs.eval import EvalJob as _E

            if isinstance(job, _E):
                # EvalJob writes a dedicated job_result.json.
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

            rel_output_dir = output_dir.lstrip("./")
            local_result_path = os.path.join(repo_root, rel_output_dir, result_filename)
            print(f"[AWSManager] Loading JobResult from {local_result_path}")
            with open(local_result_path, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)
            return JobResult.from_dict(data)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"AWSManager: failed to load remote JobResult after sync: {exc}"
            ) from exc
