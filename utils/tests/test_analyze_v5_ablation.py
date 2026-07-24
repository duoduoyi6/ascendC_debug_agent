from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "analyze_v5_ablation.py"


def _load():
    spec = importlib.util.spec_from_file_location("v5_analysis", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V5AnalysisTests(unittest.TestCase):
    def test_append_only_results_avoid_latest_file_double_count(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp)
            archive = task / "precision_tuning" / "claude_results"
            archive.mkdir(parents=True)
            payload = {
                "num_turns": 12,
                "total_cost_usd": 1.25,
                "modelUsage": {
                    "qwen3.8-max-preview": {
                        "inputTokens": 100,
                        "outputTokens": 20,
                        "cacheCreationInputTokens": 30,
                        "cacheReadInputTokens": 40,
                        "costUSD": 99,
                    }
                },
            }
            (archive / "attempt0_session.json").write_text(json.dumps(payload))
            (task / "_claude_result_attempt0.json").write_text(json.dumps(payload))

            paths = module.claude_result_paths(task)
            result = module.summarize_claude_cost(paths)

            self.assertEqual(len(paths), 1)
            self.assertEqual(result["turns"], 12)
            self.assertEqual(result["total_tokens"], 190)
            self.assertEqual(result["cost_usd"], 1.25)
            self.assertEqual(result["models"], ["qwen3.8-max-preview"])

    def test_mcnemar_and_bootstrap_are_paired(self) -> None:
        module = _load()
        full = [True, True, False, False]
        treatment = [True, False, True, False]
        exact = module.mcnemar_exact(full, treatment)
        bootstrap = module.paired_bootstrap_ci(
            full, treatment, samples=1000, seed=7)
        self.assertEqual(exact["full_only"], 1)
        self.assertEqual(exact["treatment_only"], 1)
        self.assertEqual(exact["two_sided_exact_p"], 1.0)
        self.assertEqual(bootstrap["difference"], 0.0)

    def test_long_failure_requires_two_burden_dimensions(self) -> None:
        module = _load()
        rows = []
        for arm in module.ARMS:
            rows.extend([
                {
                    "arm": arm,
                    "posthoc_clean_success": False,
                    "turns": 10,
                    "total_tokens": 100,
                    "cost_usd": 1.0,
                },
                {
                    "arm": arm,
                    "posthoc_clean_success": False,
                    "turns": 100,
                    "total_tokens": 1000,
                    "cost_usd": 1.0,
                },
            ])
        module.classify_long_failures(rows)
        for arm in module.ARMS:
            arm_rows = [row for row in rows if row["arm"] == arm]
            self.assertFalse(arm_rows[0]["long_failure"])
            self.assertTrue(arm_rows[1]["long_failure"])


if __name__ == "__main__":
    unittest.main()
