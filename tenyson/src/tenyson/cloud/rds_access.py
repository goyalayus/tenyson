from __future__ import annotations

import atexit
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Sequence
from urllib.error import URLError
from urllib.request import urlopen

try:
    import boto3  # type: ignore[import-not-found]
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - optional dependency guard
    boto3 = None

    class ClientError(Exception):
        pass


_POSTGRES_PORT = 5432
_LOCAL_RULE_DESCRIPTION = "Local runner"
_MODAL_RULE_DESCRIPTION = "Modal telemetry"


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PostgresIngressUpdate:
    stale_local_cidrs: tuple[str, ...]
    needs_local_rule: bool
    needs_modal_rule: bool


def compute_postgres_ingress_update(
    ip_permissions: Sequence[Dict[str, Any]],
    *,
    current_local_cidr: str,
    modal_cidr: str,
) -> PostgresIngressUpdate:
    stale_local_cidrs: list[str] = []
    has_current_local_rule = False
    has_modal_rule = False

    for permission in ip_permissions:
        if permission.get("IpProtocol") != "tcp":
            continue
        if permission.get("FromPort") != _POSTGRES_PORT:
            continue
        if permission.get("ToPort") != _POSTGRES_PORT:
            continue
        for ip_range in permission.get("IpRanges", []):
            cidr = str(ip_range.get("CidrIp") or "").strip()
            description = str(ip_range.get("Description") or "").strip()
            if cidr == modal_cidr:
                has_modal_rule = True
            if description == _LOCAL_RULE_DESCRIPTION:
                if cidr == current_local_cidr:
                    has_current_local_rule = True
                elif cidr:
                    stale_local_cidrs.append(cidr)

    return PostgresIngressUpdate(
        stale_local_cidrs=tuple(stale_local_cidrs),
        needs_local_rule=not has_current_local_rule,
        needs_modal_rule=not has_modal_rule,
    )


def _current_public_ipv4() -> str:
    with urlopen("https://checkip.amazonaws.com", timeout=5) as response:
        value = response.read().decode("utf-8").strip()
    octets = value.split(".")
    if len(octets) != 4 or not all(part.isdigit() for part in octets):
        raise ValueError(f"Unexpected public IP response: {value!r}")
    return value


def _aws_session() -> Any:
    if boto3 is None:
        raise RuntimeError(
            "boto3 is required to manage TENYSON_RDS_SECURITY_GROUP ingress for Modal runs."
        )
    profile = os.getenv("AWS_PROFILE") or None
    region = (
        os.getenv("TENYSON_AWS_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    return boto3.Session(profile_name=profile, region_name=region)


def _revoke_ip_ranges(ec2_client: Any, *, security_group_id: str, cidrs: Iterable[str]) -> None:
    cidr_list = [cidr for cidr in cidrs if cidr]
    if not cidr_list:
        return
    ec2_client.revoke_security_group_ingress(
        GroupId=security_group_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": _POSTGRES_PORT,
                "ToPort": _POSTGRES_PORT,
                "IpRanges": [{"CidrIp": cidr} for cidr in cidr_list],
            }
        ],
    )


def _authorize_ip_range(
    ec2_client: Any,
    *,
    security_group_id: str,
    cidr: str,
    description: str,
) -> None:
    ec2_client.authorize_security_group_ingress(
        GroupId=security_group_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": _POSTGRES_PORT,
                "ToPort": _POSTGRES_PORT,
                "IpRanges": [{"CidrIp": cidr, "Description": description}],
            }
        ],
    )


def prepare_modal_rds_access() -> Callable[[], None]:
    """
    Keep local telemetry monitoring reachable and open the configured RDS
    security group to Modal workers for the duration of the local controller.
    """

    security_group_id = str(os.getenv("TENYSON_RDS_SECURITY_GROUP") or "").strip()
    if not security_group_id:
        return lambda: None

    modal_cidr = (
        str(os.getenv("TENYSON_MODAL_RDS_CIDR") or "").strip() or "0.0.0.0/0"
    )
    sync_local_ip = _is_truthy(os.getenv("TENYSON_SYNC_RDS_LOCAL_IP", "true"))

    session = _aws_session()
    ec2 = session.client("ec2")
    response = ec2.describe_security_groups(GroupIds=[security_group_id])
    security_group = response["SecurityGroups"][0]

    current_local_cidr = "0.0.0.0/32"
    if sync_local_ip:
        try:
            current_local_cidr = f"{_current_public_ipv4()}/32"
        except (URLError, TimeoutError, ValueError) as exc:
            raise RuntimeError(
                "Unable to determine the current public IPv4 for local telemetry access."
            ) from exc

    update = compute_postgres_ingress_update(
        security_group.get("IpPermissions", []),
        current_local_cidr=current_local_cidr,
        modal_cidr=modal_cidr,
    )

    if update.stale_local_cidrs:
        _revoke_ip_ranges(
            ec2,
            security_group_id=security_group_id,
            cidrs=update.stale_local_cidrs,
        )

    if sync_local_ip and update.needs_local_rule:
        _authorize_ip_range(
            ec2,
            security_group_id=security_group_id,
            cidr=current_local_cidr,
            description=_LOCAL_RULE_DESCRIPTION,
        )

    added_modal_rule = False
    if update.needs_modal_rule:
        _authorize_ip_range(
            ec2,
            security_group_id=security_group_id,
            cidr=modal_cidr,
            description=_MODAL_RULE_DESCRIPTION,
        )
        added_modal_rule = True

    def _cleanup() -> None:
        if not added_modal_rule:
            return
        try:
            _revoke_ip_ranges(
                ec2,
                security_group_id=security_group_id,
                cidrs=[modal_cidr],
            )
        except ClientError:
            pass

    atexit.register(_cleanup)
    return _cleanup
