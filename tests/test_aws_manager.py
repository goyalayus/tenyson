import os
import unittest
from unittest.mock import patch

from tenyson.cloud.aws import AWSManager


class AWSManagerEnvTests(unittest.TestCase):
    def test_from_env_uses_expected_env_defaults(self) -> None:
        env = {
            "TENYSON_AWS_KEY_NAME": "test-key",
            "TENYSON_AWS_KEY_PATH": "/tmp/test-key.pem",
            "TENYSON_AWS_SECURITY_GROUP": "sg-123",
            "TENYSON_AWS_INSTANCE_TYPE": "g5.4xlarge",
            "TENYSON_AWS_REGION": "us-west-2",
            "TENYSON_AWS_SUBNET": "subnet-123",
            "TENYSON_AWS_SPOT_MAX_PRICE": "2.50",
            "AWS_PROFILE": "research",
        }
        with patch.dict(os.environ, env, clear=True):
            manager = AWSManager.from_env(use_spot=True)

        self.assertEqual(manager.instance_type, "g5.4xlarge")
        self.assertEqual(manager.region, "us-west-2")
        self.assertEqual(manager.key_name, "test-key")
        self.assertEqual(manager.key_path, "/tmp/test-key.pem")
        self.assertEqual(manager.security_group, "sg-123")
        self.assertEqual(manager.subnet, "subnet-123")
        self.assertEqual(manager.profile, "research")
        self.assertTrue(manager.auto_terminate)
        self.assertTrue(manager.use_spot)
        self.assertEqual(manager.spot_max_price, "2.50")

    def test_from_env_requires_key_material(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TENYSON_AWS_KEY_NAME"):
                AWSManager.from_env()


if __name__ == "__main__":
    unittest.main()
