import unittest

from datasets import Dataset

from tenyson.jobs.sft_dataset import (
    build_builtin_sft_messages,
    normalize_builtin_sft_dataset,
    supports_builtin_sft_schema,
)


class BuiltinSFTDatasetTests(unittest.TestCase):
    def test_supports_string_prompt_answer_schema(self) -> None:
        self.assertTrue(
            supports_builtin_sft_schema(
                {
                    "prompt": "Sort these letters.",
                    "answer": "abc",
                }
            )
        )

    def test_supports_instruction_output_schema(self) -> None:
        self.assertTrue(
            supports_builtin_sft_schema(
                {
                    "instruction": "Solve the task.",
                    "input": "letters: cba",
                    "output": "abc",
                }
            )
        )

    def test_build_builtin_messages_from_prompt_answer_and_system(self) -> None:
        messages = build_builtin_sft_messages(
            {
                "system": "You are helpful.",
                "prompt": "Hello",
                "answer": "Hi there",
            }
        )

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        )

    def test_build_builtin_messages_from_instruction_schema_uses_default_system(self) -> None:
        messages = build_builtin_sft_messages(
            {
                "instruction": "Sort the letters.",
                "input": "c b a",
                "output": "abc",
            },
            default_system_prompt="You are a sorting assistant.",
        )

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "You are a sorting assistant."},
                {
                    "role": "user",
                    "content": "Instruction:\nSort the letters.\n\nInput:\nc b a",
                },
                {"role": "assistant", "content": "abc"},
            ],
        )

    def test_normalize_builtin_dataset_adds_canonical_messages_column(self) -> None:
        dataset = Dataset.from_list(
            [
                {
                    "prompt": "Hello",
                    "answer": "Hi there",
                }
            ]
        )

        normalized = normalize_builtin_sft_dataset(
            dataset,
            config={"task": {"sft_system_prompt": "You are helpful."}},
            dataset_name="train",
        )

        self.assertIn("messages", normalized.column_names)
        self.assertEqual(
            normalized[0]["messages"],
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        )

    def test_normalize_builtin_dataset_leaves_unsupported_rows_unchanged(self) -> None:
        dataset = Dataset.from_list([{"text": "already formatted"}])

        normalized = normalize_builtin_sft_dataset(
            dataset,
            config={},
            dataset_name="train",
        )

        self.assertEqual(normalized.column_names, ["text"])
        self.assertEqual(normalized[0]["text"], "already formatted")

    def test_build_builtin_messages_rejects_invalid_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported SFT row schema"):
            build_builtin_sft_messages({"foo": "bar"})


if __name__ == "__main__":
    unittest.main()
