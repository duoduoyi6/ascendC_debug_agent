"""test_exit_artifacts.py — 退出产物从 events 重建 (debug_status.json + debug_trace.md)。

验收点 (REWRITE_PLAN §6.5): events → debug_status/trace 重建正确 (含 timeout/crashed 路径)。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine import transition as tr
from engine.exit_artifacts import (
    build_debug_status,
    build_debug_trace,
    write_exit_artifacts,
)
from engine.types import Abort, Continue, Done


class TestDebugStatus(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_success_status(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "precision_failed"))
        tr.record_decision(self.task_dir, Done(session_outcome="success", reason="ok"))
        s = build_debug_status(self.task_dir)
        self.assertEqual(s["schema_version"], 1)
        self.assertEqual(s["session_outcome"], "success")
        self.assertEqual(s["session_branch"], "1-P")
        self.assertEqual(s["attempts_used"], 1)
        self.assertEqual(s["entry_failure_type"], "precision_failed")
        self.assertEqual(s["final_failure_type"], "precision_failed")
        self.assertIsNotNone(s["started_at"])
        self.assertIsNotNone(s["ended_at"])

    def test_all_ten_keys_present(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="build_failed")
        tr.record_decision(self.task_dir, Done(session_outcome="failed", reason="x"))
        s = build_debug_status(self.task_dir)
        expected = {"schema_version", "session_outcome", "session_branch",
                    "started_at", "ended_at", "attempts_used", "entry_failure_type",
                    "final_failure_type", "final_verify_status_path", "notes"}
        self.assertEqual(set(s), expected)

    def test_no_terminal_event_is_crashed(self) -> None:
        # 异常中断 (无终态事件) → crashed。
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        s = build_debug_status(self.task_dir)
        self.assertEqual(s["session_outcome"], "crashed")
        self.assertIsNone(s["ended_at"])

    def test_skipped_no_attempt(self) -> None:
        # skipped_* : 无 attempt → final_verify_status_path 为 None，final_ft=entry。
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="import_failed")
        tr.record_decision(self.task_dir, Done(session_outcome="skipped_env_issue",
                                               reason="env"))
        s = build_debug_status(self.task_dir)
        self.assertEqual(s["session_outcome"], "skipped_env_issue")
        self.assertEqual(s["attempts_used"], 0)
        self.assertIsNone(s["final_verify_status_path"])
        self.assertEqual(s["final_failure_type"], "import_failed")

    def test_abort_session_outcome_from_details(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_decision(self.task_dir, Abort(category="missing_failure_type",
                                                reason="no ft",
                                                details={"session_outcome": "crashed"}))
        s = build_debug_status(self.task_dir)
        self.assertEqual(s["session_outcome"], "crashed")

    def test_timeout_status(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "precision_failed"))
        tr.record_decision(self.task_dir, Done(session_outcome="timeout", reason="wall-clock"))
        s = build_debug_status(self.task_dir)
        self.assertEqual(s["session_outcome"], "timeout")

    def test_final_failure_type_tracks_drift(self) -> None:
        # 漂移: entry=precision, 后漂到 build → final_failure_type=build, branch 仍 1-P。
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "precision_failed"))
        tr.record_attempt_started(self.task_dir, Continue(1, "build_failed"))
        tr.record_decision(self.task_dir, Done(session_outcome="failed", reason="x"))
        s = build_debug_status(self.task_dir)
        self.assertEqual(s["entry_failure_type"], "precision_failed")
        self.assertEqual(s["final_failure_type"], "build_failed")
        self.assertEqual(s["session_branch"], "1-P")  # 入口分支，不随漂移更新


class TestDebugTrace(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_four_sections_present(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "precision_failed"))
        tr.record_decision(self.task_dir, Done(session_outcome="success", reason="ok"))
        trace = build_debug_trace(self.task_dir)
        self.assertIn("## 1. 调用入口快照", trace)
        self.assertIn("## 2. 迭代历史", trace)
        self.assertIn("## 3. 最终 Verdict", trace)
        self.assertIn("## 4. 产物清单", trace)
        self.assertIn("session_outcome: success", trace)

    def test_trace_records_each_attempt(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "precision_failed"))
        tr.record_attempt_started(self.task_dir, Continue(1, "build_failed"))
        tr.record_decision(self.task_dir, Done(session_outcome="failed", reason="x"))
        trace = build_debug_trace(self.task_dir)
        self.assertIn("### Attempt 0", trace)
        self.assertIn("### Attempt 1", trace)

    def test_write_exit_artifacts_creates_files(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_decision(self.task_dir, Done(session_outcome="success", reason="ok"))
        status_path, trace_path = write_exit_artifacts(self.task_dir)
        self.assertTrue(status_path.exists())
        self.assertTrue(trace_path.exists())
        # status 是合法 JSON。
        json.loads(status_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
