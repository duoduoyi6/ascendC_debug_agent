from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "cannbench_hidden_followup.py"


def _load():
    spec = importlib.util.spec_from_file_location("cannbench_hidden_followup", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _standard_job(score: float = 53.25) -> dict:
    return {
        "id": "job_standard",
        "status": "succeeded",
        "case_set": "standard",
        "submission_id": "sub_example",
        "selected_operators": ["quant_matmul"],
        "result_score": score,
    }


class CANNBenchHiddenFollowupTests(unittest.TestCase):
    def test_score_must_be_strictly_greater_than_50(self) -> None:
        module = _load()
        equal = module.assess_hidden_eligibility(_standard_job(50.0))
        above = module.assess_hidden_eligibility(_standard_job(50.0001))
        self.assertFalse(equal["eligible"])
        self.assertTrue(above["eligible"])
        self.assertEqual(above["threshold_rule"], "score > 50")

    def test_nonterminal_and_hidden_source_jobs_are_rejected(self) -> None:
        module = _load()
        job = _standard_job()
        job["status"] = "correctness"
        job["case_set"] = "hidden"
        result = module.assess_hidden_eligibility(job)
        self.assertFalse(result["eligible"])
        self.assertIn("standard job is not terminal", result["reasons"])
        self.assertIn("source job is not a standard-case job", result["reasons"])

    def test_hidden_job_must_match_submission_and_operator_set(self) -> None:
        module = _load()
        hidden = {
            "id": "job_hidden",
            "status": "succeeded",
            "case_set": "hidden",
            "submission_id": "sub_example",
            "selected_operators": ["quant_matmul"],
        }
        passed = module.validate_hidden_binding(_standard_job(), hidden)
        self.assertTrue(passed["passed"])
        hidden["submission_id"] = "sub_other"
        failed = module.validate_hidden_binding(_standard_job(), hidden)
        self.assertFalse(failed["passed"])

    def test_request_response_job_id_is_validated(self) -> None:
        module = _load()
        self.assertEqual(
            module.extract_requested_job_id({"job": {"id": "job_ab12"}}),
            "job_ab12",
        )
        with self.assertRaises(module.FollowupError):
            module.extract_requested_job_id({"job": {"id": "invalid"}})

    def test_summary_preserves_failure_fields(self) -> None:
        module = _load()
        hidden = {
            "id": "job_hidden",
            "status": "correctness_failed",
            "case_set": "hidden",
            "submission_id": "sub_example",
            "selected_operators": ["quant_matmul"],
            "results": {
                "passed_cases": 7,
                "total_cases": 8,
                "operators": [{
                    "operator": "quant_matmul",
                    "passed_cases": 7,
                    "total_cases": 8,
                    "failed_cases": [3],
                    "score_error_code": "case_failures",
                }],
            },
        }
        summary = module.build_summary(hidden)
        self.assertEqual(summary["passed_cases"], 7)
        self.assertEqual(summary["operators"][0]["failed_cases"], [3])
        self.assertEqual(
            summary["operators"][0]["score_error_code"], "case_failures")


if __name__ == "__main__":
    unittest.main()
