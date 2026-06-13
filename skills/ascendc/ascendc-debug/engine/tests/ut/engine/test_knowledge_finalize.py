"""test_knowledge_finalize.py — KB 入库编排 (修复问题 6)。

验收: finalize 条件门控 (outcome==success && cheat_history clean && 候选存在) +
action 透传 (new/merge/abandon) + 子进程失败/缺脚本降级为 skip + 默认 kb_path=None 禁用。
用 fake _run 替换 subprocess.run，不真跑 precision_knowledge.py。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine.knowledge_finalize import finalize_knowledge


class _FakeRun:
    """记录 dump 子进程调用；可编程 returncode/stderr。不真跑脚本。"""

    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)

        class _Proc:
            pass
        p = _Proc()
        p.returncode = self.returncode
        p.stderr = self.stderr
        p.stdout = ""
        return p


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


class TestKnowledgeFinalize(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)
        self.tuning = self.task_dir / "precision_tuning"
        # 默认放一个合法候选 (个别用例覆盖/删除)。
        _write(self.tuning / "candidate_kb_entry.json",
               {"title": "t", "type": "FIX_PRECISION_TAIL"})

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _call(self, *, outcome="success", kb_path="kb.json", run=None):
        run = run or _FakeRun()
        res = finalize_knowledge(self.task_dir, kb_path=kb_path,
                                 session_outcome=outcome, op_name="add", _run=run)
        return res, run

    def test_no_kb_path_disabled(self) -> None:
        res, run = self._call(kb_path=None)
        self.assertFalse(res["finalized"])
        self.assertEqual(len(run.calls), 0)

    def test_non_success_skips(self) -> None:
        res, run = self._call(outcome="failed")
        self.assertFalse(res["finalized"])
        self.assertEqual(len(run.calls), 0)

    def test_missing_candidate_skips(self) -> None:
        (self.tuning / "candidate_kb_entry.json").unlink()
        res, run = self._call()
        self.assertFalse(res["finalized"])
        self.assertEqual(len(run.calls), 0)

    def test_cheat_violation_blocks(self) -> None:
        _write(self.tuning / "cheat_history.json",
               {"cheating_attempts": [{"attempt": 1, "severity": "violation"}]})
        res, run = self._call()
        self.assertFalse(res["finalized"])
        self.assertEqual(len(run.calls), 0)

    def test_ast_warning_blocks(self) -> None:
        # AST validator 异常 (severity=warning, ast_status=unknown) 也阻止入库
        # (文档 6.5: unknown 不等价 pass → success_unverified，保守不入库)。
        _write(self.tuning / "cheat_history.json",
               {"cheating_attempts": [{"attempt": 1, "severity": "warning",
                                       "cheat_type": "AST_VALIDATOR_ERROR"}]})
        res, run = self._call()
        self.assertFalse(res["finalized"])
        self.assertEqual(len(run.calls), 0)

    def test_success_clean_runs_dump(self) -> None:
        res, run = self._call()
        self.assertTrue(res["finalized"])
        self.assertEqual(len(run.calls), 1)
        cmd = run.calls[0]
        self.assertIn("dump", cmd)
        self.assertIn("--action", cmd)
        self.assertIn("new", cmd)

    def test_abandon_candidate_skips(self) -> None:
        _write(self.tuning / "candidate_kb_entry.json",
               {"title": "t", "type": "FIX_PRECISION_TAIL", "action": "abandon"})
        res, run = self._call()
        self.assertFalse(res["finalized"])
        self.assertEqual(len(run.calls), 0)

    def test_merge_passes_target_title(self) -> None:
        _write(self.tuning / "candidate_kb_entry.json",
               {"title": "t", "type": "FIX_PRECISION_TAIL", "action": "merge",
                "merge_target_title": "existing-entry"})
        res, run = self._call()
        self.assertTrue(res["finalized"])
        cmd = run.calls[0]
        self.assertIn("--merge-target-title", cmd)
        self.assertIn("existing-entry", cmd)

    def test_dump_failure_skips(self) -> None:
        res, run = self._call(run=_FakeRun(returncode=1, stderr="boom"))
        self.assertFalse(res["finalized"])
        self.assertEqual(len(run.calls), 1)  # 调了但失败
        self.assertIn("boom", res["reason"])

    def test_result_written_to_disk(self) -> None:
        self._call()
        out = self.task_dir / "precision_tuning" / "kb_finalize_result.json"
        self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
