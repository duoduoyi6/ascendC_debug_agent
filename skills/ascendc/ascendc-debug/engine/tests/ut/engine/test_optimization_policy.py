from __future__ import annotations

import unittest

from engine.optimization_policy import assess_optimization_candidate, optimization_entry_gate


def baseline():
    return {
        "objective_success": True,
        "official_full_pass": True,
        "anti_cheat": "CLEAN",
        "target_compile_pass": True,
        "submission_ready": True,
        "source_sha256": "baseline",
        "case_ids_sha256": "cases",
        "environment_sha256": "env",
        "timing_samples_us": [100.0, 101.0, 99.0, 100.5, 99.5],
        "per_case_us": {"c1": 100.0, "c2": 200.0},
    }


class OptimizationPolicyTest(unittest.TestCase):
    def test_entry_requires_frozen_repeated_baseline(self):
        item = baseline()
        item["timing_samples_us"] = [100.0]
        result = optimization_entry_gate(item)
        self.assertFalse(result["eligible"])
        self.assertIn("repeated_timing", result["failed"])

    def test_accepts_pareto_improvement(self):
        candidate = {
            **baseline(),
            "source_sha256": "candidate",
            "timing_samples_us": [90.0, 90.5, 89.5, 90.0, 90.2],
            "per_case_us": {"c1": 90.0, "c2": 180.0},
        }
        result = assess_optimization_candidate(baseline(), candidate)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["decision"], "accept_pareto_improvement")

    def test_rolls_back_material_case_regression(self):
        candidate = {
            **baseline(),
            "source_sha256": "candidate",
            "timing_samples_us": [90.0, 90.5, 89.5],
            "per_case_us": {"c1": 105.0, "c2": 170.0},
        }
        result = assess_optimization_candidate(baseline(), candidate)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["decision"], "rollback_candidate")

    def test_rejects_correctness_regression_before_timing(self):
        candidate = {
            **baseline(),
            "source_sha256": "candidate",
            "official_full_pass": False,
            "timing_samples_us": [50.0, 50.0, 50.0],
        }
        result = assess_optimization_candidate(baseline(), candidate)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["decision"], "reject_before_performance_comparison")


if __name__ == "__main__":
    unittest.main()
