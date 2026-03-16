import os
import unittest
from unittest.mock import patch

from tenyson.bootstrap import (
    ensure_local_controller_environment,
    is_truthy,
    missing_controller_packages,
    resolve_project_root,
)
from tenyson.core.run_config import shared_overrides_from_env


class BootstrapTests(unittest.TestCase):
    def test_is_truthy(self) -> None:
        self.assertTrue(is_truthy("1"))
        self.assertTrue(is_truthy("TRUE"))
        self.assertTrue(is_truthy(" yes "))
        self.assertFalse(is_truthy("0"))
        self.assertFalse(is_truthy("false"))
        self.assertFalse(is_truthy(None))

    def test_resolve_project_root_from_repo_file(self) -> None:
        project_root = resolve_project_root(__file__)
        self.assertTrue((project_root / "pyproject.toml").is_file())
        self.assertTrue((project_root / "src" / "tenyson").is_dir())

    def test_missing_controller_packages_uses_mapping(self) -> None:
        missing = missing_controller_packages(
            {
                "json": "json",
                "definitely_missing_module_xyz": "fake-pkg",
            }
        )
        self.assertIn("fake-pkg", missing)
        self.assertNotIn("json", missing)

    def test_ensure_local_controller_environment_respects_skip_env(self) -> None:
        with patch.dict(os.environ, {"TENYSON_SKIP_LOCAL_BOOTSTRAP": "true"}, clear=False):
            installed = ensure_local_controller_environment(anchor_file=__file__)
        self.assertEqual(list(installed), [])


class SharedOverridesTests(unittest.TestCase):
    def test_shared_overrides_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TENYSON_HF_REPO_BASE": "org/repo",
                "TENYSON_EXPERIMENT_ID": "exp_123",
            },
            clear=False,
        ):
            overrides = shared_overrides_from_env()

        self.assertEqual(
            overrides,
            {
                "training": {"hf_repo_base": "org/repo"},
                "telemetry": {"experiment_id": "exp_123"},
            },
        )

    def test_shared_overrides_from_env_returns_none_when_empty(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TENYSON_HF_REPO_BASE": "",
                "TENYSON_EXPERIMENT_ID": "",
            },
            clear=False,
        ):
            overrides = shared_overrides_from_env()

        self.assertIsNone(overrides)


if __name__ == "__main__":
    unittest.main()
