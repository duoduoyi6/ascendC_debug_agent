"""test_best_rollback.py — current_best 保存 + 匹配率回滚 (问题 7)。

验收点 (消融实验问题反馈.md 问题 7 决策 1-3):
  - 指标: 排序键 (case_pass_rate, match_rate)，match_rate 仅 tie-break。
  - 粒度: 整个 kernel/ 目录被覆盖存/回滚。
  - 时机: 单轮下降不回滚 (NMS 50→40→100)；连续 2 轮无改善才回滚。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine import best_rollback as br
from engine.runner import run_debug_session


def _mk_task(tmp: Path) -> Path:
    task = tmp / "task"
    (task / "kernel").mkdir(parents=True)
    (task / "kernel" / "k.cpp").write_text("v0", encoding="utf-8")
    (task / "precision_tuning").mkdir()
    return task


def _write_val(task: Path, attempt: int, case: float, match: float, ok: bool) -> None:
    pt = task / "precision_tuning"
    pt.mkdir(parents=True, exist_ok=True)
    (pt / f"validation_result_attempt_{attempt}.json").write_text(
        json.dumps({"attempt": attempt, "case_pass_rate": case,
                    "match_rate": f"{match:.2f}", "correctness_passed": ok}),
        encoding="utf-8")


class TestSaveBest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.task = _mk_task(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_first_attempt_saves_best(self):
        _write_val(self.task, 0, 50.0, 57.62, False)
        self.assertTrue(br.save_current_best(self.task, 0))
        self.assertTrue((self.task / "precision_tuning/history/current_best/src/k.cpp").exists())

    def test_higher_case_updates_best(self):
        _write_val(self.task, 0, 50.0, 90.0, False)
        br.save_current_best(self.task, 0)
        _write_val(self.task, 1, 70.0, 10.0, False)  # case 更高，match 更低
        self.assertTrue(br.save_current_best(self.task, 1), "case% 提升应更新 best")
        self.assertEqual(br.read_best_metric(self.task)["attempt"], 1)

    def test_tie_case_match_breaks(self):
        _write_val(self.task, 0, 50.0, 57.62, False)
        br.save_current_best(self.task, 0)
        _write_val(self.task, 1, 50.0, 64.60, False)  # case 相同，match 更高
        self.assertTrue(br.save_current_best(self.task, 1), "case 同分时 match 更高应更新")

    def test_no_improvement_keeps_best(self):
        _write_val(self.task, 0, 50.0, 64.60, False)
        br.save_current_best(self.task, 0)
        _write_val(self.task, 1, 40.0, 54.30, False)  # 更差
        self.assertFalse(br.save_current_best(self.task, 1), "变差不应覆盖 best")
        self.assertEqual(br.read_best_metric(self.task)["attempt"], 0)


class TestRollbackTiming(unittest.TestCase):
    """决策 3: 单轮下降不回滚，连续 2 轮无改善才回滚。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.task = _mk_task(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_nms_single_dip_no_rollback(self):
        """NMS 实测: case% 50→50→40，第2轮单轮下降不回滚。"""
        _write_val(self.task, 0, 50.0, 57.62, False); br.save_current_best(self.task, 0)
        _write_val(self.task, 1, 50.0, 64.60, False); br.save_current_best(self.task, 1)
        _write_val(self.task, 2, 40.0, 54.30, False); br.save_current_best(self.task, 2)
        self.assertFalse(br.should_rollback(self.task, 2),
                         "单轮下降 (att2) 不应回滚，否则丢掉后续 att3 的修复")

    def test_two_rounds_no_improvement_rollback(self):
        """连续 2 轮都劣于 best → 回滚。"""
        _write_val(self.task, 0, 50.0, 57.62, False); br.save_current_best(self.task, 0)
        _write_val(self.task, 1, 50.0, 64.60, False); br.save_current_best(self.task, 1)
        _write_val(self.task, 2, 45.0, 50.0, False); br.save_current_best(self.task, 2)
        _write_val(self.task, 3, 42.0, 48.0, False); br.save_current_best(self.task, 3)
        self.assertTrue(br.should_rollback(self.task, 3),
                        "att2/att3 连续劣于 best(att1) → 回滚")

    def test_no_rollback_without_best(self):
        _write_val(self.task, 0, 50.0, 57.62, False)
        self.assertFalse(br.should_rollback(self.task, 0), "首轮无 best 不回滚")

    def test_improvement_resets(self):
        """att2 提升成新 best 后，不该回滚。"""
        _write_val(self.task, 0, 50.0, 57.62, False); br.save_current_best(self.task, 0)
        _write_val(self.task, 1, 40.0, 54.30, False); br.save_current_best(self.task, 1)
        _write_val(self.task, 2, 60.0, 70.0, False); br.save_current_best(self.task, 2)
        self.assertFalse(br.should_rollback(self.task, 2), "att2 成为新 best，不回滚")
        self.assertEqual(br.read_best_metric(self.task)["attempt"], 2)


class TestDoRollback(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.task = _mk_task(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_rollback_overwrites_kernel(self):
        # att0 是 best(kernel=v0)；之后 kernel 被改坏成 v_bad
        _write_val(self.task, 0, 80.0, 90.0, False)
        br.save_current_best(self.task, 0)
        (self.task / "kernel" / "k.cpp").write_text("v_bad", encoding="utf-8")
        (self.task / "kernel" / "junk.cpp").write_text("junk", encoding="utf-8")
        best = br.do_rollback(self.task)
        self.assertIsNotNone(best)
        self.assertEqual((self.task / "kernel" / "k.cpp").read_text(), "v0",
                         "回滚应把 kernel 恢复到 best 版本")
        self.assertFalse((self.task / "kernel" / "junk.cpp").exists(),
                         "回滚应覆盖源码层，删除 best 中不存在的新增源码")

    def test_build_not_saved_nor_touched(self):
        """决策 2 (修订): 只存真源码，build/ 不进 best、回滚不删 build/。"""
        build = self.task / "kernel" / "build"
        build.mkdir(parents=True)
        (build / "obj.o").write_text("compiled", encoding="utf-8")
        _write_val(self.task, 0, 80.0, 90.0, False)
        br.save_current_best(self.task, 0)
        # build 不进 current_best
        self.assertFalse(
            (self.task / "precision_tuning/history/current_best/src/build").exists(),
            "build/ 不应存入 current_best")
        # 改坏源码后回滚，build/ 的编译产物应原样保留 (下轮 --clean 重建)
        (self.task / "kernel" / "k.cpp").write_text("bad", encoding="utf-8")
        br.do_rollback(self.task)
        self.assertEqual((self.task / "kernel" / "k.cpp").read_text(), "v0")
        self.assertTrue((build / "obj.o").exists(), "回滚不应删除 build/ 产物")


class TestRollbackPromptInjection(unittest.TestCase):
    """决策 4: 回滚后下一轮 prompt 注入失败方向提示。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.task = _mk_task(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_warning_injected_after_rollback(self):
        from engine import transition
        from engine.agent_backend import _rollback_warning
        # 上一轮 (attempt 2) 回滚到 best(attempt 1)
        transition.record_rollback(
            self.task, from_attempt=2,
            best_metric={"attempt": 1, "case_pass_rate": 50.0, "match_rate": "64.60"})
        # 本轮 attempt=3 的 prompt 应含回滚提示
        w = _rollback_warning(self.task, 3)
        self.assertIn("回滚", w)
        self.assertIn("attempt 1", w)  # best_attempt
        self.assertIn("tuning_directions", w)  # 指引读失败方向

    def test_no_warning_when_no_rollback(self):
        from engine.agent_backend import _rollback_warning
        self.assertEqual(_rollback_warning(self.task, 1), "")

    def test_warning_only_for_immediately_prior_attempt(self):
        from engine import transition
        from engine.agent_backend import _rollback_warning
        transition.record_rollback(
            self.task, from_attempt=2,
            best_metric={"attempt": 1, "case_pass_rate": 50.0, "match_rate": "64.60"})
        # attempt=5 时，上一轮是 4 而非 2，不该注入旧回滚
        self.assertEqual(_rollback_warning(self.task, 5), "")


class TestRunnerIntegration(unittest.TestCase):
    """集成: 在 run_debug_session 真实流程里验证 best/回滚被正确触发 (attempt 传参正确)。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.task = Path(self._tmp.name)
        (self.task / "kernel").mkdir(parents=True)
        (self.task / "kernel" / "k.cpp").write_text("orig", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def _dispatcher_factory(self, case_by_attempt):
        """validate 步按 attempt 写真实 validation_result 文件，返回 CONTINUE/PASS。"""
        def disp(action, task_dir, op_name, agent_callback):
            if action.step == "validate":
                a = int((action.skill_args or {}).get("attempt", 0))
                case = case_by_attempt[min(a, len(case_by_attempt) - 1)]
                ok = case >= 100.0
                _write_val(Path(task_dir), a, case, case, ok)
                return {"gate": "GATE-V", "passed": ok,
                        "loop_signal": "PASS" if ok else "CONTINUE",
                        "stop_reason_code": None}
            return {"success": True}
        return disp

    def test_rollback_event_emitted_on_sustained_regression(self):
        from engine.events import read_events
        # case%: 50→50→40→30→... 持续无改善(best=att1的50)，att3 应触发回滚
        disp = self._dispatcher_factory([50.0, 50.0, 40.0, 30.0, 30.0])
        run_debug_session(self.task, op_name="add", entry_failure_type="precision_failed",
                          dispatcher=disp, deadline_sec=None)
        events = read_events(self.task)
        rollbacks = [e for e in events if e.get("type") == "rollback"]
        self.assertTrue(rollbacks, "持续无改善应产生至少一次 rollback 事件")
        # best 应停在 case%=50 的轮次
        best = br.read_best_metric(self.task)
        self.assertEqual(float(best["case_pass_rate"]), 50.0)

    def test_no_rollback_when_converging(self):
        from engine.events import read_events
        # case%: 50→70→100 单调收敛，不应回滚
        disp = self._dispatcher_factory([50.0, 70.0, 100.0])
        run_debug_session(self.task, op_name="add", entry_failure_type="precision_failed",
                          dispatcher=disp, deadline_sec=None)
        rollbacks = [e for e in read_events(self.task) if e.get("type") == "rollback"]
        self.assertEqual(rollbacks, [], "单调收敛不应回滚")


if __name__ == "__main__":
    unittest.main()

