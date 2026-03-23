from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import tenyson.core.hf_checkpoint as hf_checkpoint_module


class HFCheckpointTests(unittest.TestCase):
    def test_parse_resume_ref_requires_repo_and_revision(self) -> None:
        self.assertEqual(
            hf_checkpoint_module._parse_resume_ref("org/repo:main"),
            ("org/repo", "main"),
        )
        with self.assertRaisesRegex(ValueError, "repo_id:revision"):
            hf_checkpoint_module._parse_resume_ref("org/repo")
        with self.assertRaisesRegex(ValueError, "Both repo_id and revision"):
            hf_checkpoint_module._parse_resume_ref("org/repo:")

    def test_resolve_downloaded_checkpoint_prefers_last_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "last-checkpoint").mkdir()
            (root / "last-checkpoint" / "trainer_state.json").write_text("{}")
            (root / "checkpoint-20").mkdir()
            (root / "checkpoint-20" / "trainer_state.json").write_text("{}")

            resolved = hf_checkpoint_module._resolve_downloaded_checkpoint(root)

        self.assertEqual(resolved, str(root / "last-checkpoint"))

    def test_resolve_downloaded_checkpoint_chooses_highest_numbered_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for step in (2, 10, 4):
                checkpoint_dir = root / f"checkpoint-{step}"
                checkpoint_dir.mkdir()
                (checkpoint_dir / "trainer_state.json").write_text("{}")

            resolved = hf_checkpoint_module._resolve_downloaded_checkpoint(root)

        self.assertEqual(resolved, str(root / "checkpoint-10"))

    def test_resolve_hf_resume_revision_requires_checkpoint_files(self) -> None:
        with patch.object(
            hf_checkpoint_module,
            "resolve_hf_repo_revision",
            return_value="abc123",
        ), patch.object(hf_checkpoint_module, "HfApi") as hf_api_cls:
            hf_api_cls.return_value.list_repo_files.return_value = ["config.json"]
            with self.assertRaisesRegex(ValueError, "No trainer checkpoint found"):
                hf_checkpoint_module.resolve_hf_resume_revision("org/repo")

    def test_download_hf_resume_checkpoint_returns_best_downloaded_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "checkpoint-3").mkdir()
            (root / "checkpoint-3" / "trainer_state.json").write_text("{}")
            with patch.object(
                hf_checkpoint_module,
                "resolve_hf_resume_revision",
                return_value="abc123",
            ), patch.object(
                hf_checkpoint_module,
                "snapshot_download",
                return_value=str(root),
            ):
                resolved = hf_checkpoint_module.download_hf_resume_checkpoint(
                    "org/repo:main"
                )

        self.assertEqual(resolved, str(root / "checkpoint-3"))


if __name__ == "__main__":
    unittest.main()
