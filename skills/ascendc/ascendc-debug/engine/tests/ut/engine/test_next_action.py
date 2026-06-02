"""test_next_action.py — 引擎核心决策器: 五分支 × 三信号矩阵 + 漂移 + 预算闸 + 路由。

验收点 (REWRITE_PLAN §6.4): 五分支×三信号 + failure_type 漂移路由 + 双层预算闸 +
白名单/黑名单 + 各 stop_reason_code。喂 events/state fixture → 断言 decision 类型与字段。

注: next_action 是纯函数，这里直接构造 DebugState (不落盘)，独立于 events I/O。
"""
from __future__ import annotations

import unittest

from engine.gate_adapter import GateResult
from engine.next_action import debug_next_action
from engine.state import Counters, DebugState
from engine.types import Abort, Action, Continue, Done


def _state(*, current_ft, total_attempts=0, per_branch=None, events=None) -> DebugState:
    cf = Counters(
        total_attempts=total_attempts,
        current_failure_type=current_ft,
        entry_failure_type=current_ft,
        per_branch_attempt=dict(per_branch or {}),
    )
    return DebugState(task_dir=None, cf=cf, events=list(events or []))


def _gate(loop_signal, stop_reason_code=None) -> GateResult:
    return GateResult(gate="GATE-V", passed=(loop_signal == "PASS"),
                      loop_signal=loop_signal, stop_reason_code=stop_reason_code)


def _attempt_ev(ft):
    return {"type": "attempt_started", "failure_type": ft}


def _completed_ev(step, result=None):
    return {"type": "action_completed", "action": {"step": step}, "result": result or {}}


class TestRouting(unittest.TestCase):
    """SKILL.md Step 0.3 白名单/黑名单路由。"""

    def test_missing_failure_type_aborts_crashed(self) -> None:
        d = debug_next_action(_state(current_ft=None))
        self.assertIsInstance(d, Abort)
        self.assertEqual(d.details["session_outcome"], "crashed")

    def test_success_entry(self) -> None:
        d = debug_next_action(_state(current_ft="success"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "success")

    def test_import_env_side_skipped(self) -> None:
        # 最近 action_completed 带 import_subtype=import_env_side。
        st = _state(current_ft="import_failed", total_attempts=1,
                    per_branch={"import_failed": 1},
                    events=[_completed_ev("forensics",
                                          {"import_subtype": "import_env_side"})])
        d = debug_next_action(st)
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "skipped_env_issue")

    def test_import_kernel_side_continues(self) -> None:
        # kernel_side 不退出，进入轮内推进 (返回 Action，非 Done)。
        st = _state(current_ft="import_failed", total_attempts=1,
                    per_branch={"import_failed": 1},
                    events=[_attempt_ev("import_failed"),
                            _completed_ev("x", {"import_subtype": "import_kernel_side"})])
        d = debug_next_action(st)
        self.assertNotIsInstance(d, Done)

    def test_import_first_round_no_subtype_conservatively_enters_1I(self) -> None:
        # 首轮无 action_completed → import_subtype=None → 保守进 1-I (非 skipped_env_issue)。
        # 锁定缺口行为: runner 须喂 import_subtype，否则首轮 env_side 漏判。
        st = _state(current_ft="import_failed", total_attempts=1,
                    per_branch={"import_failed": 1},
                    events=[_attempt_ev("import_failed")])
        d = debug_next_action(st)
        self.assertNotIsInstance(d, Done)  # 不退出，进入轮内推进
        self.assertIsInstance(d, Action)
        self.assertEqual(d.skill_args["failure_type"], "import_failed")

    def test_nonwhitelist_skipped_unsupported(self) -> None:
        for ft in ("degraded", "no_kernel", "tilelang_only_failed", "execution_aborted"):
            d = debug_next_action(_state(current_ft=ft))
            self.assertIsInstance(d, Done, ft)
            self.assertEqual(d.session_outcome, "skipped_unsupported_type", ft)

    def test_unknown_failure_type_skipped(self) -> None:
        d = debug_next_action(_state(current_ft="weird_new_type"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "skipped_unsupported_type")


class TestBudgetGates(unittest.TestCase):
    """三道闸的前两道 (全局 MAX_ATTEMPTS + 分支硬上限)。"""

    def test_global_max_attempts(self) -> None:
        st = _state(current_ft="precision_failed", total_attempts=5,
                    per_branch={"precision_failed": 5})
        d = debug_next_action(st)
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "stopped_by_loop_limit")

    def test_branch_hard_cap_build(self) -> None:
        # build cap=3，撞上即停 (此时全局 total=3 < 5，不触发闸1)。
        st = _state(current_ft="build_failed", total_attempts=3,
                    per_branch={"build_failed": 3})
        d = debug_next_action(st)
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "stopped_by_loop_limit")

    def test_precision_cap_higher_than_build(self) -> None:
        # precision 在 build cap(3) 处不停 (precision cap=5)，应继续推进。
        st = _state(current_ft="precision_failed", total_attempts=3,
                    per_branch={"precision_failed": 3},
                    events=[_attempt_ev("precision_failed")])
        d = debug_next_action(st)
        self.assertNotIsInstance(d, Done)


class TestLoopSignalDispatch(unittest.TestCase):
    """五分支 × 三信号: validate 完成后读 gate_result 派发。"""

    def _state_after_validate(self, ft, total=1):
        return _state(current_ft=ft, total_attempts=total,
                      per_branch={ft: total},
                      events=[_attempt_ev(ft),
                              _completed_ev("forensics"),
                              _completed_ev("diagnose_and_fix"),
                              _completed_ev("validate")])

    def test_pass_to_success_all_branches(self) -> None:
        for ft in ("precision_failed", "build_failed", "import_failed",
                   "runtime_error", "timeout"):
            d = debug_next_action(self._state_after_validate(ft), _gate("PASS"))
            self.assertIsInstance(d, Done, ft)
            self.assertEqual(d.session_outcome, "success", ft)

    def test_continue_all_branches(self) -> None:
        for ft in ("precision_failed", "build_failed", "import_failed",
                   "runtime_error", "timeout"):
            d = debug_next_action(self._state_after_validate(ft), _gate("CONTINUE"))
            self.assertIsInstance(d, Continue, ft)

    def test_stop_prerequisite_to_gate(self) -> None:
        d = debug_next_action(self._state_after_validate("precision_failed"),
                              _gate("STOP", "prerequisite_failure"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "stopped_by_gate")

    def test_stop_max_attempts_to_loop_limit(self) -> None:
        d = debug_next_action(self._state_after_validate("precision_failed"),
                              _gate("STOP", "max_attempts_reached"))
        self.assertEqual(d.session_outcome, "stopped_by_loop_limit")

    def test_stop_nearly_success_to_failed(self) -> None:
        # 用户确认: nearly_success / fp16_ceiling 归 failed (stop_code 区分)。
        for code in ("nearly_success", "fp16_precision_ceiling",
                     "harmful_regression", "stagnant_same_direction",
                     "validation_failed"):
            d = debug_next_action(self._state_after_validate("precision_failed"),
                                  _gate("STOP", code))
            self.assertEqual(d.session_outcome, "failed", code)

    def test_stop_unknown_code_to_failed(self) -> None:
        d = debug_next_action(self._state_after_validate("precision_failed"),
                              _gate("STOP", None))
        self.assertEqual(d.session_outcome, "failed")


class TestRoundSequence(unittest.TestCase):
    """轮内 step 序列推进 forensics → diagnose_and_fix → validate。"""

    def test_first_attempt_starts_continue(self) -> None:
        # 全新 session (total=0) → Continue(0)。
        d = debug_next_action(_state(current_ft="precision_failed", total_attempts=0))
        self.assertIsInstance(d, Continue)
        self.assertEqual(d.next_attempt, 0)

    def test_step_progression(self) -> None:
        ft = "precision_failed"
        # attempt 已起 (total=1)，本轮未做任何 step → forensics。
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft)])
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "forensics")
        self.assertEqual(d.kind, "py_action")

    def test_step_after_forensics_is_spawn(self) -> None:
        ft = "precision_failed"
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft), _completed_ev("forensics")])
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "diagnose_and_fix")
        self.assertEqual(d.kind, "spawn_agent")

    def test_step_after_fix_is_validate(self) -> None:
        ft = "precision_failed"
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft), _completed_ev("forensics"),
                            _completed_ev("diagnose_and_fix")])
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "validate")
        self.assertEqual(d.kind, "py_action")


class TestDriftRouting(unittest.TestCase):
    """failure_type 漂移: next_action 按最新 ft 派发对应分支。"""

    def test_drift_routes_by_latest_ft(self) -> None:
        # 上一轮 precision，本轮漂到 build。current_ft=build → 走 build 分支步骤。
        st = _state(current_ft="build_failed", total_attempts=2,
                    per_branch={"precision_failed": 1, "build_failed": 1},
                    events=[_attempt_ev("precision_failed"),
                            _attempt_ev("build_failed")])
        d = debug_next_action(st)
        # 本轮 (build) 未做 step → forensics，skill_args.failure_type=build_failed。
        self.assertIsInstance(d, Action)
        self.assertEqual(d.skill_args["failure_type"], "build_failed")

    def test_continue_carries_current_ft(self) -> None:
        st = _state(current_ft="build_failed", total_attempts=1,
                    per_branch={"build_failed": 1},
                    events=[_attempt_ev("build_failed"),
                            _completed_ev("forensics"),
                            _completed_ev("diagnose_and_fix"),
                            _completed_ev("validate")])
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Continue)
        self.assertEqual(d.next_failure_type, "build_failed")


class TestGatePrecedenceOverBudget(unittest.TestCase):
    """H1: 本轮 gate 终判 (PASS/STOP) 优先于预算闸。

    第 5 轮 validate 已 PASS/STOP 时，不能被全局闸 (total>=5) 覆盖成
    stopped_by_loop_limit——那一轮本就是终态轮，gate 的终判才是真实结局。
    """

    def _state_fifth_validate(self, ft="precision_failed"):
        evs = [_attempt_ev(ft) for _ in range(5)]
        evs += [_completed_ev("forensics"), _completed_ev("diagnose_and_fix"),
                _completed_ev("validate")]
        return _state(current_ft=ft, total_attempts=5, per_branch={ft: 5}, events=evs)

    def test_fifth_round_pass_is_success_not_loop_limit(self) -> None:
        d = debug_next_action(self._state_fifth_validate(), _gate("PASS"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "success")

    def test_fifth_round_stop_maps_by_code(self) -> None:
        # 第 5 轮 STOP nearly_success → failed (按 code)，不被全局闸覆盖为 loop_limit。
        d = debug_next_action(self._state_fifth_validate(),
                              _gate("STOP", "nearly_success"))
        self.assertEqual(d.session_outcome, "failed")

    def test_continue_at_budget_limit_is_loop_limit(self) -> None:
        # 反向: 第 5 轮 CONTINUE 想续跑但已达预算 → stopped_by_loop_limit (预算守住)。
        d = debug_next_action(self._state_fifth_validate(), _gate("CONTINUE"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "stopped_by_loop_limit")


class TestResumeReconstructsGate(unittest.TestCase):
    """H2: gate_result 未传入 (crash-resume) 时，从事件流最后一个 validate result 重建。

    validate 的 action_completed 已持久化 loop_signal/stop_reason_code，但 runner 内存态
    gate_result 在崩溃后丢失。next_action 须从事件重建，不能无视已算出的终判径直续跑。
    """

    def _state_validate_persisted(self, loop_signal, stop_code=None, ft="precision_failed"):
        result = {"gate": "GATE-V", "passed": loop_signal == "PASS",
                  "loop_signal": loop_signal, "stop_reason_code": stop_code}
        return _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                      events=[_attempt_ev(ft), _completed_ev("forensics"),
                              _completed_ev("diagnose_and_fix"),
                              _completed_ev("validate", result)])

    def test_resume_pass_reconstructed_to_success(self) -> None:
        # gate_result=None (resume)，事件里 validate loop_signal=PASS → 重建派发 success。
        d = debug_next_action(self._state_validate_persisted("PASS"), None)
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "success")

    def test_resume_stop_reconstructed_to_outcome(self) -> None:
        d = debug_next_action(
            self._state_validate_persisted("STOP", "prerequisite_failure"), None)
        self.assertEqual(d.session_outcome, "stopped_by_gate")

    def test_resume_continue_reconstructed(self) -> None:
        d = debug_next_action(self._state_validate_persisted("CONTINUE"), None)
        self.assertIsInstance(d, Continue)
        self.assertEqual(d.next_failure_type, "precision_failed")


if __name__ == "__main__":
    unittest.main()
