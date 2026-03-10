from tenyson.cloud.aws import AWSManager
from tenyson.cloud.modal import ModalManager


def CloudManager(provider: str = "modal", **kwargs):
    provider = provider.lower().strip()
    if provider == "aws":
        return AWSManager(**kwargs)
    if provider == "modal":
        return ModalManager(**kwargs)
    raise ValueError(f"Unsupported provider: {provider}")
