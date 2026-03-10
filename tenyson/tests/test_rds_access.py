import unittest

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


if __name__ == "__main__":
    unittest.main()
