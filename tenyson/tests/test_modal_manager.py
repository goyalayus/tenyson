import os
import unittest
from unittest.mock import patch

from tenyson.cloud.manager import CloudManager
from tenyson.cloud.modal import ModalManager


class ModalManagerEnvTests(unittest.TestCase):
    def test_from_env_uses_expected_env_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            manager = ModalManager.from_env()

        self.assertEqual(manager.gpu, "A100")
        self.assertEqual(manager.timeout, 86400)
        self.assertIsNone(manager.profile)
        self.assertTrue(manager.auto_terminate)

    def test_from_env_reads_modal_specific_settings(self) -> None:
        env = {
            "TENYSON_MODAL_GPU": "A100-80GB",
            "TENYSON_MODAL_TIMEOUT": "7200",
            "TENYSON_MODAL_PROFILE": "research",
        }
        with patch.dict(os.environ, env, clear=True):
            manager = ModalManager.from_env()

        self.assertEqual(manager.gpu, "A100-80GB")
        self.assertEqual(manager.timeout, 7200)
        self.assertEqual(manager.profile, "research")

    def test_from_env_falls_back_to_modal_profile(self) -> None:
        env = {
            "TENYSON_MODAL_TIMEOUT": "3600",
            "MODAL_PROFILE": "team-profile",
        }
        with patch.dict(os.environ, env, clear=True):
            manager = ModalManager.from_env()

        self.assertEqual(manager.profile, "team-profile")

    def test_from_env_rejects_invalid_timeout(self) -> None:
        with patch.dict(
            os.environ, {"TENYSON_MODAL_TIMEOUT": "not-an-int"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "TENYSON_MODAL_TIMEOUT"):
                ModalManager.from_env()


class CloudManagerDefaultTests(unittest.TestCase):
    def test_cloud_manager_defaults_to_modal(self) -> None:
        manager = CloudManager()
        self.assertIsInstance(manager, ModalManager)
        self.assertEqual(manager.gpu, "A100")


if __name__ == "__main__":
    unittest.main()
