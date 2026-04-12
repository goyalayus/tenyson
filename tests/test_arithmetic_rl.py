import unittest

from examples.arithmetic.functional import (
    addition_reward,
    addition_rl_dataset,
    build_addition_dataset,
)


class ArithmeticRLTests(unittest.TestCase):
    def test_rl_dataset_holds_benchmark_problems_out(self) -> None:
        benchmark_dataset = build_addition_dataset(
            digits=2,
            sample_count=100,
            seed=7,
        )
        benchmark_problems = {
            (int(row["left"]), int(row["right"]))
            for row in benchmark_dataset
        }

        template = addition_rl_dataset(
            digits=2,
            sample_count=256,
            seed=456,
            benchmark_sample_count=100,
            benchmark_seed=7,
        )
        train_dataset = template.build({})
        train_problems = {
            (int(row["left"]), int(row["right"]))
            for row in train_dataset
        }

        self.assertTrue(train_problems.isdisjoint(benchmark_problems))

    def test_addition_reward_uses_only_exact_answer_and_strict_format(self) -> None:
        template = addition_reward()
        reward_correct_answer, reward_strict_format = template.build({}, None)

        completions = [
            "78</answer>",
            "78",
            "<answer>78</answer>",
            "99</answer>",
        ]
        expected_answer = ["78", "78", "78", "78"]

        correct_rewards = reward_correct_answer(
            [],
            completions,
            expected_answer=expected_answer,
        )
        format_rewards = reward_strict_format(
            [],
            completions,
        )

        self.assertEqual(correct_rewards, [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(format_rewards, [0.1, 0.0, 0.0, 0.1])


if __name__ == "__main__":
    unittest.main()
