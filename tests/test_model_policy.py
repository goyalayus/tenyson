import unittest

from tenyson.core.model_policy import require_qwen3_model_name


class ModelPolicyTests(unittest.TestCase):
    def test_accepts_qwen3_model_names(self) -> None:
        self.assertEqual(
            require_qwen3_model_name("Qwen/Qwen3-4B"),
            "Qwen/Qwen3-4B",
        )
        self.assertEqual(
            require_qwen3_model_name("unsloth/qwen3-0.6b-unsloth-bnb-4bit"),
            "unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        )

    def test_rejects_non_qwen3_model_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Qwen 3 family"):
            require_qwen3_model_name("meta-llama/Llama-3.1-8B")


if __name__ == "__main__":
    unittest.main()
