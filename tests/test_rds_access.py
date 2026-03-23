import os
import unittest
from unittest.mock import MagicMock, patch

import tenyson.cloud.rds_access as rds_access_module
from tenyson.cloud.rds_access import compute_postgres_ingress_update


class RDSAccessTests(unittest.TestCase):
    def test_detects_stale_local_and_missing_modal_rules(self) -> None:
        permissions = [
            {
                "IpProtocol": "tcp",
                "FromPort": 5432,
                "ToPort": 5432,
                "IpRanges": [
                    {"CidrIp": "122.171.20.251/32", "Description": "Local runner"},
                ],
            }
        ]

        update = compute_postgres_ingress_update(
            permissions,
            current_local_cidr="61.95.133.58/32",
            modal_cidr="0.0.0.0/0",
        )

        self.assertEqual(update.stale_local_cidrs, ("122.171.20.251/32",))
        self.assertTrue(update.needs_local_rule)
        self.assertTrue(update.needs_modal_rule)

    def test_no_changes_when_local_and_modal_already_present(self) -> None:
        permissions = [
            {
                "IpProtocol": "tcp",
                "FromPort": 5432,
                "ToPort": 5432,
                "IpRanges": [
                    {"CidrIp": "61.95.133.58/32", "Description": "Local runner"},
                    {"CidrIp": "0.0.0.0/0", "Description": "Modal telemetry"},
                ],
            }
        ]

        update = compute_postgres_ingress_update(
            permissions,
            current_local_cidr="61.95.133.58/32",
            modal_cidr="0.0.0.0/0",
        )

        self.assertEqual(update.stale_local_cidrs, ())
        self.assertFalse(update.needs_local_rule)
        self.assertFalse(update.needs_modal_rule)

    def test_ignores_non_postgres_permissions(self) -> None:
        permissions = [
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [
                    {"CidrIp": "122.171.20.251/32", "Description": "Local runner"},
                ],
            }
        ]

        update = compute_postgres_ingress_update(
            permissions,
            current_local_cidr="61.95.133.58/32",
            modal_cidr="0.0.0.0/0",
        )

        self.assertEqual(update.stale_local_cidrs, ())
        self.assertTrue(update.needs_local_rule)
        self.assertTrue(update.needs_modal_rule)

    def test_authorize_ip_range_duplicate_rule_returns_false(self) -> None:
        ec2 = MagicMock()
        try:
            duplicate_error = rds_access_module.ClientError(
                {
                    "Error": {
                        "Code": "InvalidPermission.Duplicate",
                        "Message": "duplicate rule",
                    }
                },
                "AuthorizeSecurityGroupIngress",
            )
        except TypeError:
            duplicate_error = rds_access_module.ClientError("duplicate rule")
            duplicate_error.response = {
                "Error": {"Code": "InvalidPermission.Duplicate"}
            }
        ec2.authorize_security_group_ingress.side_effect = duplicate_error

        added = rds_access_module._authorize_ip_range(
            ec2,
            security_group_id="sg-1234",
            cidr="0.0.0.0/0",
            description="Modal telemetry",
        )

        self.assertFalse(added)

    def test_prepare_modal_rds_access_skips_cleanup_for_duplicate_modal_rule(self) -> None:
        ec2 = MagicMock()
        ec2.describe_security_groups.return_value = {
            "SecurityGroups": [{"IpPermissions": []}]
        }
        session = MagicMock()
        session.client.return_value = ec2

        with patch.dict(
            os.environ,
            {
                "TENYSON_RDS_SECURITY_GROUP": "sg-1234",
                "TENYSON_MODAL_RDS_CIDR": "0.0.0.0/0",
                "TENYSON_SYNC_RDS_LOCAL_IP": "false",
            },
            clear=False,
        ), patch.object(
            rds_access_module,
            "_aws_session",
            return_value=session,
        ), patch.object(
            rds_access_module,
            "_authorize_ip_range",
            return_value=False,
        ) as authorize_mock:
            cleanup = rds_access_module.prepare_modal_rds_access()
            cleanup()

        authorize_mock.assert_called_once()
        ec2.revoke_security_group_ingress.assert_not_called()


if __name__ == "__main__":
    unittest.main()
