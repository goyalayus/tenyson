import unittest
from unittest.mock import patch

import tenyson.core.telemetry as telemetry


class WaitForRunResultTests(unittest.TestCase):
    def test_wait_for_run_result_can_skip_results_payload(self) -> None:
        calls = []

        def fake_get_run_result(**kwargs):
            calls.append(kwargs)
            return {}, {"status": "success", "run_id": "eval_baseline_mixed"}

        with patch.object(telemetry, "get_run_result", side_effect=fake_get_run_result):
            result = telemetry.wait_for_run_result(
                client=object(),
                experiment_id="wordle_exp",
                run_id="eval_baseline_mixed",
                phase="eval",
                attempt_token="attempt-123",
                include_results_payload=False,
            )

        self.assertEqual(result, ({}, {"status": "success", "run_id": "eval_baseline_mixed"}))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["include_results_payload"], False)


if __name__ == "__main__":
    unittest.main()
