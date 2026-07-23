"""Guard task reset against carrying stale experiment artifacts."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "supervise_ascendc_debug_batch_cc_quota.py"


def _load_supervisor():
    spec = importlib.util.spec_from_file_location("quota_supervisor", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ResetTargetTests(unittest.TestCase):
    def test_reset_target_drops_stale_outputs(self) -> None:
        supervisor = _load_supervisor()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "src" / "level1" / "001_Foo"
            task = base / "out" / "tasks" / "level1" / "001_Foo"
            (source / "kernel").mkdir(parents=True)
            (source / "precision_tuning").mkdir()
            (source / ".debug_events").mkdir()
            (source / "kernel" / "foo.cpp").write_text("// kernel\n", encoding="utf-8")
            (source / "model.py").write_text("# model\n", encoding="utf-8")
            (source / "precision_tuning" / "validation_result_attempt_1.json").write_text("{}", encoding="utf-8")
            (source / ".debug_events" / "events.jsonl").write_text("{}\n", encoding="utf-8")
            (source / "debug_status.json").write_text("{}", encoding="utf-8")
            (source / "_claude_result_attempt0.json").write_text("{}", encoding="utf-8")

            supervisor.reset_target(supervisor.Target(source=source, task=task))

            self.assertTrue((task / "kernel" / "foo.cpp").exists())
            self.assertTrue((task / "model.py").exists())
            self.assertFalse((task / "precision_tuning").exists())
            self.assertFalse((task / ".debug_events").exists())
            self.assertFalse((task / "debug_status.json").exists())
            self.assertFalse((task / "_claude_result_attempt0.json").exists())

    def test_archive_retry_evidence_preserves_signal_audit_before_reset(self) -> None:
        supervisor = _load_supervisor()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "src" / "level1" / "001_Foo"
            task = base / "out" / "tasks" / "level1" / "001_Foo"
            (source / "kernel").mkdir(parents=True)
            (source / "kernel" / "foo.cpp").write_text("// kernel\n", encoding="utf-8")
            (task / "precision_tuning" / "signal_audit").mkdir(parents=True)
            (task / ".debug_events").mkdir()
            (task / "precision_tuning" / "signal_audit" / "rc143.txt").write_text(
                "host_cmd_pid=123\n",
                encoding="utf-8",
            )
            (task / "precision_tuning" / "forensics_report_0.json").write_text(
                '{"status":"completed"}\n',
                encoding="utf-8",
            )
            (task / ".debug_events" / "events.jsonl").write_text("{}\n", encoding="utf-8")
            (task / ".verify_logs").mkdir()
            (task / ".verify_logs" / "phase8_attempt0.stdout").write_text(
                "Status: FAIL\n", encoding="utf-8")
            (task / ".verify_status").mkdir()
            (task / ".verify_status" / "latest.json").write_text(
                '{"failure_type":"provider_api_error"}\n', encoding="utf-8")
            (task / "debug_status.json").write_text('{"session_outcome":"crashed"}', encoding="utf-8")
            (task / "_claude_result_attempt0.json").write_text(
                '{"num_turns":12,"total_cost_usd":1.5}\n', encoding="utf-8")

            target = supervisor.Target(source=source, task=task)
            archive = supervisor.archive_retry_evidence(target, "unit_test")

            self.assertIsNotNone(archive)
            assert archive is not None
            self.assertTrue((archive / "precision_tuning" / "signal_audit" / "rc143.txt").exists())
            self.assertTrue(
                (archive / "precision_tuning" / "forensics_report_0.json").exists())
            self.assertTrue((archive / ".debug_events" / "events.jsonl").exists())
            self.assertTrue(
                (archive / ".verify_logs" / "phase8_attempt0.stdout").exists())
            self.assertTrue((archive / ".verify_status" / "latest.json").exists())
            self.assertTrue((archive / "_claude_result_attempt0.json").exists())
            self.assertTrue((archive / "debug_status.json").exists())
            self.assertTrue((archive / "archive_manifest.json").exists())
            manifest = json.loads(
                (archive / "archive_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(
                manifest["cost_accounting"]["excluded_from_primary_cost"])


class SupervisorConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_kb_read_only_is_recorded_and_forwarded(self) -> None:
        self.assertIn('parser.add_argument("--kb-read-only", action="store_true")', self.text)
        self.assertIn('"kb_read_only": args.kb_read_only', self.text)
        self.assertIn('cmd.append("--kb-read-only")', self.text)


if __name__ == "__main__":
    unittest.main()
