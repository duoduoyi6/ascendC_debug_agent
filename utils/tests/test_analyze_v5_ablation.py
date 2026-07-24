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
    @staticmethod
    def _write_events(task: Path, rows: list[dict]) -> None:
        path = task / ".debug_events" / "events.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

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

    def test_observability_uses_real_direction_and_rollback_records(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp)
            tuning = task / "precision_tuning"
            tuning.mkdir()
            (task / "debug_status.json").write_text("{}")
            (task / "run_summary.json").write_text("{}")
            (tuning / "tuning_directions.json").write_text("{}")
            (tuning / "probe_policy_attempt0.json").write_text("{}")
            (tuning / "knowledge_search_log.json").write_text("{}")
            (tuning / "forensics_report_0.json").write_text("{}")
            archive = tuning / "claude_results"
            archive.mkdir()
            (archive / "attempt0.json").write_text("{}")
            current_best = tuning / "history" / "current_best"
            current_best.mkdir(parents=True)
            (current_best / "metric.json").write_text("{}")
            (current_best / "manifest.json").write_text("{}")
            self._write_events(task, [
                {
                    "type": "action_started",
                    "action": {"step": step},
                }
                for step in (
                    "baseline_checkpoint", "forensics", "knowledge_search",
                    "diagnose_and_fix", "validate", "checkpoint_and_rollback",
                )
            ] + [
                {
                    "type": "action_completed",
                    "action": {"step": "baseline_checkpoint"},
                    "result": {
                        "success": True,
                        "checkpoint": {"success": True, "updated": True},
                    },
                },
                {
                    "type": "action_completed",
                    "action": {"step": "checkpoint_and_rollback"},
                    "result": {
                        "success": True,
                        "checkpoint": {"success": True, "updated": False},
                        "rolled_back": True,
                    },
                },
            ])

            result = module._observability(
                task, {"attempts_used": 1}, "full")

            self.assertTrue(result["complete"])
            self.assertTrue(result["checks"]["direction"]["observed"])
            self.assertTrue(result["checks"]["checkpoint"]["observed"])
            self.assertTrue(result["checks"]["rollback"]["required"])
            self.assertTrue(result["checks"]["rollback"]["observed"])

    def test_observability_marks_disabled_or_untriggered_evidence_complete(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp)
            tuning = task / "precision_tuning"
            archive = tuning / "claude_results"
            archive.mkdir(parents=True)
            (archive / "attempt0.json").write_text("{}")
            (task / "debug_status.json").write_text("{}")
            (task / "run_summary.json").write_text("{}")
            (tuning / "diagnosis_summary_attempt_0.json").write_text("{}")
            self._write_events(task, [
                {
                    "type": "action_started",
                    "action": {"step": "diagnose_and_fix"},
                },
                {
                    "type": "action_started",
                    "action": {"step": "validate"},
                },
            ])

            result = module._observability(
                task, {"attempts_used": 1}, "no_diagnostic_evidence")

            self.assertTrue(result["complete"])
            for name in ("probe", "kb", "forensics"):
                self.assertFalse(result["checks"][name]["required"])
                self.assertEqual(
                    result["checks"][name]["reason"], "disabled_by_arm")
            self.assertFalse(result["checks"]["rollback"]["observed"])
            self.assertTrue(result["checks"]["rollback"]["complete"])

    def test_observability_flags_missing_direction_after_validate_started(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp)
            (task / "debug_status.json").write_text("{}")
            (task / "run_summary.json").write_text("{}")
            self._write_events(task, [{
                "type": "action_started",
                "action": {"step": "validate"},
            }])

            result = module._observability(
                task, {"attempts_used": 1}, "full")

            self.assertFalse(result["complete"])
            self.assertTrue(result["checks"]["direction"]["required"])
            self.assertFalse(result["checks"]["direction"]["observed"])


if __name__ == "__main__":
    unittest.main()
