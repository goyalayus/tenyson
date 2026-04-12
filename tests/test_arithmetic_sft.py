import unittest

from tenyson import bind_chat_sft_dataset

from examples.arithmetic.functional import (
    build_addition_dataset,
    build_addition_sft_train_dataset,
)


class ArithmeticSFTTests(unittest.TestCase):
    def test_sft_dataset_holds_benchmark_problems_out(self) -> None:
        benchmark_dataset = build_addition_dataset(
            digits=2,
            sample_count=100,
            seed=7,
        )
        benchmark_problems = {
            (int(row["left"]), int(row["right"]))
            for row in benchmark_dataset
        }

        template = bind_chat_sft_dataset(
            build_addition_sft_train_dataset,
            digits=2,
            train_sample_count=256,
            train_seed=123,
            benchmark_sample_count=100,
            benchmark_seed=7,
        )
        train_dataset = template.train({}, None)
        train_problems = {
            (int(row["left"]), int(row["right"]))
            for row in train_dataset
        }

        self.assertTrue(train_problems.isdisjoint(benchmark_problems))

    def test_sft_dataset_uses_full_answer_tags_in_assistant_messages(self) -> None:
        template = bind_chat_sft_dataset(
            build_addition_sft_train_dataset,
            digits=2,
            train_sample_count=8,
            train_seed=123,
            benchmark_sample_count=4,
            benchmark_seed=7,
        )
        train_dataset = template.train({}, None)

        first_row = train_dataset[0]
        messages = first_row["messages"]
        expected_answer = str(first_row["expected_answer"])

        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], f"<answer>{expected_answer}</answer>")


if __name__ == "__main__":
    unittest.main()
