import os


_ALLOWED_GPU_PROVIDERS = {"aws", "modal"}


def require_gpu_provider_runtime() -> str:
    """
    Enforce cloud-only execution.

    Jobs must be launched by a supported GPU cloud provider manager, which sets:
    - TENYSON_EXECUTION_MODE=cloud
    - TENYSON_GPU_PROVIDER in {"aws", "modal"}
    """
    execution_mode = os.getenv("TENYSON_EXECUTION_MODE", "").strip().lower()
    provider = os.getenv("TENYSON_GPU_PROVIDER", "").strip().lower()

    if execution_mode != "cloud" or provider not in _ALLOWED_GPU_PROVIDERS:
        raise RuntimeError(
            "Local execution is disabled. Run jobs through a supported GPU cloud "
            "manager (AWSManager or ModalManager)."
        )
    return provider
