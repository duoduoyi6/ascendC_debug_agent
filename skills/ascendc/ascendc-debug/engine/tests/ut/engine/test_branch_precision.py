"""test_branch_precision.py — 12a: precision_audit 方向评估缺失兜底。

验收点 (6.11 文档 Tier 2 项 12a):
  - audit 产出缺失 (文件/marker/section 不存在) → _check_direction_assessment 返回
    "missing"，stagnant 分支映射 CONTINUE (stop_reason_code=stagnant_audit_missing)，
    不据缺失误判 STOP 提前掐断诊断。
  - audit 存在但答案模糊 (非 是/否) → "unknown" → 沿用同方向 STOP (原行为不变)。
  - audit 明确 否/是 → "continue"/"stop" (原行为不变)。

gates 在 scripts/ 下 (非 engine 包)，测试自插 scripts 到 sys.path (镜像
test_common_anticheat.py 的 bootstrap)，独立于 PYTHONPATH。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from gates.branch_precision import _LegacyPrecisionChecker  # noqa: E402

# stagnant 分支触发所需: mismatch 连续未改善，且未达 fp16/max_attempts 早停。
_STAGNANT_FORENSICS = {
    "history_trend": {
        "mismatch_improving": False,
        "trend": [
            {"mismatch_ratio": 0.05},
            {"mismatch_ratio": 0.05},
            {"mismatch_ratio": 0.05},
        ],
    }
}


def _audit_path(task: Path, attempt: int) -> Path:
    return task / "precision_tuning" / f"precision_audit_{attempt}.md"


class TestDirectionAssessmentState(unittest.TestCase):
    """_check_direction_assessment 四态判定。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task = Path(self._tmp.name) / "005_FakeOp"
        (self.task / "precision_tuning").mkdir(parents=True)
        self.checker = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=2)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_file_absent_is_missing(self) -> None:
        self.assertEqual(self.checker._check_direction_assessment(), "missing")

    def test_no_marker_is_missing(self) -> None:
        _audit_path(self.task, 2).write_text("无方向 section 的正文\n", encoding="utf-8")
        self.assertEqual(self.checker._check_direction_assessment(), "missing")

    def test_empty_section_is_missing(self) -> None:
        _audit_path(self.task, 2).write_text(
            "[DIRECTION_ASSESSMENT]\n[NEXT]\n", encoding="utf-8")
        self.assertEqual(self.checker._check_direction_assessment(), "missing")

    def test_ambiguous_answer_is_unknown(self) -> None:
        _audit_path(self.task, 2).write_text(
            "[DIRECTION_ASSESSMENT]\n本轮是否延续上一轮方向: 不确定，待定\n",
            encoding="utf-8")
        self.assertEqual(self.checker._check_direction_assessment(), "unknown")

    def test_answer_no_is_continue(self) -> None:
        _audit_path(self.task, 2).write_text(
            "[DIRECTION_ASSESSMENT]\n本轮是否延续上一轮方向: 否 换新方向\n",
            encoding="utf-8")
        self.assertEqual(self.checker._check_direction_assessment(), "continue")

    def test_answer_yes_is_stop(self) -> None:
        _audit_path(self.task, 2).write_text(
            "[DIRECTION_ASSESSMENT]\n本轮是否延续上一轮方向: 是 继续深挖\n",
            encoding="utf-8")
        self.assertEqual(self.checker._check_direction_assessment(), "stop")


class TestStagnantLoopSignalMapping(unittest.TestCase):
    """stagnant 分支 direction 态 → loop_signal 映射 (12a 核心)。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task = Path(self._tmp.name) / "005_FakeOp"
        (self.task / "precision_tuning").mkdir(parents=True)
        # attempt=2: 未达 MAX_ATTEMPTS(5) 早停; 无上一轮 forensics 文件 → fp16 不触发。
        self.checker = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=2)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _signal(self) -> tuple:
        return self.checker._compute_loop_signal(
            passed=False, match_rate=80.0, forensics_data=_STAGNANT_FORENSICS)

    def test_missing_audit_continues(self) -> None:
        # 不写 audit → missing → 兜底 CONTINUE。
        sig, _reason, code = self._signal()
        self.assertEqual(sig, "CONTINUE")
        self.assertEqual(code, "stagnant_audit_missing")

    def test_unknown_audit_stops(self) -> None:
        _audit_path(self.task, 2).write_text(
            "[DIRECTION_ASSESSMENT]\n本轮是否延续上一轮方向: 模糊回答\n",
            encoding="utf-8")
        sig, _reason, code = self._signal()
        self.assertEqual(sig, "STOP")
        self.assertEqual(code, "stagnant_same_direction")

    def test_continue_audit_continues(self) -> None:
        _audit_path(self.task, 2).write_text(
            "[DIRECTION_ASSESSMENT]\n本轮是否延续上一轮方向: 否\n", encoding="utf-8")
        sig, _reason, code = self._signal()
        self.assertEqual(sig, "CONTINUE")
        self.assertEqual(code, "stagnant_new_direction")

    def test_stop_audit_stops(self) -> None:
        _audit_path(self.task, 2).write_text(
            "[DIRECTION_ASSESSMENT]\n本轮是否延续上一轮方向: 是\n", encoding="utf-8")
        sig, _reason, code = self._signal()
        self.assertEqual(sig, "STOP")
        self.assertEqual(code, "stagnant_same_direction")


if __name__ == "__main__":
    unittest.main()
