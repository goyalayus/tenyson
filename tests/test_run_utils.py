import unittest

from tenyson.core.run_name import (
    infer_job_type,
    resolve_required_run_name,
    resolve_required_run_name_for_job_class,
)
from tenyson.jobs.hf_repo import sanitize_run_name, unique_repo_id


class SFTJob:
    pass


class EvalJob:
    pass


class CustomJob:
    pass


class RunUtilsTests(unittest.TestCase):
    def test_resolve_required_run_name_rejects_default_training_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "default is not allowed"):
            resolve_required_run_name(
                {"training": {"run_name": "sft_job"}},
                "sft",
            )

    def test_resolve_required_run_name_for_eval_reads_evaluation_section(self) -> None:
        job_type, run_name = resolve_required_run_name_for_job_class(
            {"evaluation": {"run_name": "wordle_eval_main"}},
            EvalJob,
        )

        self.assertEqual(job_type, "eval")
        self.assertEqual(run_name, "wordle_eval_main")

    def test_infer_job_type_falls_back_to_eval_when_config_has_evaluation(self) -> None:
        self.assertEqual(
            infer_job_type(CustomJob, {"evaluation": {"run_name": "eval_run"}}),
            "eval",
        )

    def test_unique_repo_id_sanitizes_run_name(self) -> None:
        self.assertEqual(sanitize_run_name(" Wordle / Run #1 "), "wordle-run-1")
        self.assertEqual(
            unique_repo_id("goyalayus/wordle-lora", " Wordle / Run #1 "),
            "goyalayus/wordle-lora-wordle-run-1",
        )


if __name__ == "__main__":
    unittest.main()
