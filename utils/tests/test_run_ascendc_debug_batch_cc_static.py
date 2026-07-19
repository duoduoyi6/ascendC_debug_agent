"""Static guardrails for run_ascendc_debug_batch_cc.sh.

These tests cover shell-script behavior that is hard to unit test without
Docker/NPU access, but has caused real experiment ambiguity:

* cleanup must not use broad tokens such as "engine" that can match sibling
  workers in the same container;
* SIGTERM/SIGKILL exits must be reported separately from hard timeouts.
"""

from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "run_ascendc_debug_batch_cc.sh"


class TestCleanupSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_engine_cleanup_does_not_use_broad_engine_token(self):
        self.assertNotIn(
            'cleanup_task_processes "$container" "$task_dir" "engine"',
            self.text,
        )
        self.assertIn(
            'cleanup_task_processes "$container" "$task_dir" ""',
            self.text,
        )

    def test_cleanup_skips_unsafe_tokens(self):
        self.assertIn("is_safe_token()", self.text)
        self.assertIn("skip unsafe broad token=$token", self.text)
        self.assertIn('""|engine|python|python3|claude|bash|sh|timeout|docker|make|cmake|gmake|ninja)', self.text)
        self.assertIn('kill_by_pattern TERM "$safe_token"', self.text)
        self.assertNotIn('kill_by_pattern TERM "$token"', self.text)


class TestExitClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_signal_exits_are_not_folded_into_timeout(self):
        self.assertIn('elif [[ "$status" -eq 124 ]]', self.text)
        self.assertIn("engine_timeout", self.text)
        self.assertIn('elif [[ "$status" -eq 143 ]]', self.text)
        self.assertIn("terminated_by_sigterm", self.text)
        self.assertIn('elif [[ "$status" -eq 137 ]]', self.text)
        self.assertIn("killed_by_sigkill", self.text)
        self.assertIn("SIGTERM_CNT=", self.text)
        self.assertIn("SIGKILL_CNT=", self.text)
        self.assertNotIn("超时(engine)", self.text)


class TestTurnBudgetDefaults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_turn_budget_defaults_match_smoke_recommendation(self):
        self.assertIn('MAX_TURNS="240"', self.text)
        self.assertIn('SOFT_TASK_TURNS="480"', self.text)
        self.assertIn('MAX_TASK_TURNS="600"', self.text)
        self.assertNotIn('MAX_TURNS="180"', self.text)
        self.assertNotIn('MAX_TASK_TURNS="720"', self.text)
        self.assertIn('--soft-task-turns "$soft_task_turns"', self.text)


class TestMixedProviderScheduling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_queue_carries_task_provider_and_env(self):
        self.assertIn("provider_assignments.json", self.text)
        self.assertIn('data.get("key_config") or data.get("args", {}).get("key_config", "")', self.text)
        self.assertIn("IFS=$'\\t' read -r task_dir provider_name task_claude_env", self.text)
        self.assertIn('run_engine_turn "$container" "$npu" "$task_dir" "$op_name" "$wlog" "$task_claude_env" "$provider_name"', self.text)

    def test_provider_error_is_task_local_in_mixed_mode(self):
        self.assertIn('if [[ "$MIXED_PROVIDER_MODE" == "1" ]]', self.text)
        self.assertIn("record_provider_failure", self.text)
        self.assertIn("other providers continue", self.text)


if __name__ == "__main__":
    unittest.main()
