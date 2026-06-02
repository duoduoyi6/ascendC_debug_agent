"""next_action.py — 引擎核心纯函数决策器。

把原本散落在 SKILL.md 散文 + LLM 自驱的编排决策，全部代码化为一个无副作用纯函数:
    debug_next_action(state, gate_result=None) -> Action | Continue | Done | Abort | Escalate

决策来源 (均已精读源码/文档核实，非臆造):
  - 路由表          ← SKILL.md Step 0.3 分支路由表 (L123-132)
  - loop_signal 派发 ← SKILL.md L1027-1046 (PASS→Step5 / CONTINUE→Step0.3 / STOP→Step6)
  - 白名单/黑名单    ← SKILL.md L29-30 + agent.md L103 (import_env_side→skipped_env_issue)
  - 三道闸预算       ← REWRITE_PLAN §2.3b (全局 MAX_ATTEMPTS + 分支硬上限 + wall-clock)
  - stop_reason_code → session_outcome 映射 ← 用户确认 (按语义映射 + nearly_success 归 failed)

纯函数契约: 输入 (DebugState 投影 + 可选 GateResult)，输出一个 decision，无副作用、不写文件、
不跑子进程。所有 I/O (跑 gate、spawn agent、落事件) 由 runner (Step 5) 承担。
"""
from __future__ import annotations

import os
from typing import Optional

from engine.gate_adapter import GateResult, parse_gate_output
from engine.state import DebugState
from engine.types import (
    DEBUGGABLE_FAILURE_TYPES,
    PIPELINE_ORDER,
    Abort,
    Action,
    Continue,
    Done,
)

# ---------------------------------------------------------------------------
# 预算配置 (REWRITE_PLAN §2.3b 三道闸的前两道；第三道 wall-clock 在 runner)。
# 全局 MAX_ATTEMPTS 与现有 gates/common.py 同源 (env 可覆盖)，保持引擎与 gate 一致。
# ---------------------------------------------------------------------------
def _max_attempts() -> int:
    try:
        v = int(os.environ.get("ASCENDC_DEBUG_MAX_ATTEMPTS", "5"))
        return v if v >= 1 else 5
    except (TypeError, ValueError):
        return 5


# 分支硬上限: 撞上即停 session (stopped_by_loop_limit)。precision 比其他分支宽，因为
# 精度调优本就需要更多轮次试探；build/import/runtime/timeout 是确定性错误，3 轮够。
# env ASCENDC_DEBUG_BRANCH_CAP_<FT>=<N> 可逐分支覆盖。
_DEFAULT_BRANCH_CAP = {
    "precision_failed": 5,
    "build_failed": 3,
    "import_failed": 3,
    "runtime_error": 3,
    "timeout": 3,
}


def _branch_cap(failure_type: str) -> int:
    env_key = f"ASCENDC_DEBUG_BRANCH_CAP_{failure_type.upper()}"
    try:
        v = int(os.environ[env_key])
        if v >= 1:
            return v
    except (KeyError, TypeError, ValueError):
        pass
    return _DEFAULT_BRANCH_CAP.get(failure_type, 3)


# ---------------------------------------------------------------------------
# 路由白名单/黑名单 (SKILL.md Step 0.3)。
# ---------------------------------------------------------------------------
# 5 个可调试分支 → 分支标签 (debug_status.json 的 session_branch 字段)。
_BRANCH_LABEL = {
    "precision_failed": "1-P",
    "build_failed": "1-B",
    "import_failed": "1-I",
    "runtime_error": "1-R",
    "timeout": "1-T",
}

# 不在白名单 → 直接退出的 failure_type → session_outcome。
# success 单独处理 (走 PASS 收尾，不在此表)。
_NONWHITELIST_OUTCOME = {
    "degraded": "skipped_unsupported_type",
    "no_kernel": "skipped_unsupported_type",
    "tilelang_only_failed": "skipped_unsupported_type",
    "execution_aborted": "skipped_unsupported_type",
}

# ---------------------------------------------------------------------------
# stop_reason_code → session_outcome 映射 (按语义)。
# 未列出的 stop_reason_code (含 None) 归 failed 兜底。
# ---------------------------------------------------------------------------
_STOP_OUTCOME = {
    "prerequisite_failure": "stopped_by_gate",
    "max_attempts_reached": "stopped_by_loop_limit",
    # 以下均归 failed；nearly_success / fp16_precision_ceiling 由 stop_reason_code
    # 字段在论文统计时单独拆「准通过」子类，session_outcome 仍是 failed。
    "harmful_regression": "failed",
    "stagnant_same_direction": "failed",
    "stagnant_new_direction": "failed",
    "nearly_success": "failed",
    "fp16_precision_ceiling": "failed",
    "validation_failed": "failed",
}


# ---------------------------------------------------------------------------
# 轮内步骤序列 (一个 attempt 内引擎逐步驱动的 Action 链)。
# 对齐 agent.md 分工边界: 取证=py_action / 诊断+修复=spawn_agent / Gate-V=py_action。
#   forensics → diagnose_and_fix → validate
# 引擎据「本轮 attempt_started 之后已完成哪些 step」决定下一步。
# ---------------------------------------------------------------------------
_ROUND_SEQUENCE = ("forensics", "diagnose_and_fix", "validate")


def _completed_steps_this_attempt(state: DebugState) -> set:
    """本轮 (最后一个 attempt_started 之后) 已完成的 step 名集合。"""
    done: set = set()
    # 反向扫到最近的 attempt_started，收集其后的 action_completed.step。
    for e in reversed(state.events):
        if e.get("type") == "attempt_started":
            break
        if e.get("type") == "action_completed":
            step = (e.get("action") or {}).get("step")
            if step:
                done.add(step)
    return done


def _route_label(failure_type: Optional[str]) -> Optional[str]:
    return _BRANCH_LABEL.get(failure_type or "")


def _make_action_for_step(step: str, failure_type: str, attempt: int) -> Action:
    """把轮内 step 名构造成具体 Action。

    forensics / validate → py_action (跑确定性脚本: precision_gate.py --step)。
    diagnose_and_fix → spawn_agent (拉起 constructive/discovery worker 诊断+改 kernel)。
    """
    if step == "diagnose_and_fix":
        return Action(
            kind="spawn_agent",
            name="debug_worker",
            step=step,
            skill_args={"failure_type": failure_type, "attempt": attempt},
        )
    return Action(
        kind="py_action",
        name="precision_gate",
        step=step,
        skill_args={"failure_type": failure_type, "attempt": attempt,
                    "gate_step": "forensics" if step == "forensics" else "validate"},
    )


def debug_next_action(state: DebugState, gate_result: Optional[GateResult] = None):
    """引擎核心决策。返回 Action | Continue | Done | Abort。

    gate_result: 若上一步是 validate (Gate-V)，runner 把解析好的 GateResult 传入，
    据其 loop_signal/stop_reason_code 决定 PASS/CONTINUE/STOP。其余情况传 None；
    None 且本轮 validate 已完成时，从事件流重建 (crash-resume，见 _resolve_gate)。
    """
    ft = state.current_failure_type

    # ---- 0. 不可恢复前置: 无 failure_type → crashed ----
    if ft is None:
        return Abort(category="missing_failure_type",
                     reason="无 failure_type (verify_status 缺失或未初始化)",
                     details={"session_outcome": "crashed"})

    # ---- 1. 路由白名单/黑名单 (SKILL.md Step 0.3) ----
    if ft == "success":
        # 不经 attempt 直接 success 入口 (极少见): 视为已通过。
        return Done(session_outcome="success", reason="入口即 success")
    if ft == "import_failed":
        subtype = _latest_import_subtype(state)
        if subtype == "import_env_side":
            return Done(session_outcome="skipped_env_issue",
                        reason="import_env_side: 环境库问题，本 skill 不处理")
        # import_kernel_side (或未知子类，保守按 kernel_side 进入 1-I)。
    if ft in _NONWHITELIST_OUTCOME:
        return Done(session_outcome=_NONWHITELIST_OUTCOME[ft],
                    reason=f"failure_type={ft} 不在白名单")
    if ft not in DEBUGGABLE_FAILURE_TYPES:
        return Done(session_outcome="skipped_unsupported_type",
                    reason=f"未知 failure_type={ft}")

    # ---- 2. 本轮 validate 已完成 → gate 终判优先于预算闸 (H1) ----
    # 本轮是终态轮时，gate 的 PASS/STOP 才是真实结局，不能被预算闸覆盖成
    # stopped_by_loop_limit。gate_result 缺失 (crash-resume) 时从事件流重建 (H2)。
    completed = _completed_steps_this_attempt(state)
    if "validate" in completed:
        gate = _resolve_gate(state, gate_result)
        if gate is not None:
            sig = gate.loop_signal
            if sig in ("PASS", "STOP"):
                # 终判: 直接收尾，预算闸不介入。
                return _dispatch_loop_signal(state, gate)
            # CONTINUE: 落到下方预算闸——想续跑但已达上限则 stopped_by_loop_limit。

    # ---- 3. 闸 1 全局 MAX_ATTEMPTS ----
    if state.total_attempts >= _max_attempts():
        return Done(session_outcome="stopped_by_loop_limit",
                    reason=f"达全局 MAX_ATTEMPTS={_max_attempts()}")

    # ---- 4. 闸 2 分支硬上限 ----
    if state.branch_attempt(ft) >= _branch_cap(ft):
        return Done(session_outcome="stopped_by_loop_limit",
                    reason=f"分支 {ft} 撞硬上限 {_branch_cap(ft)}")

    # ---- 5. 本轮 validate 已完成且 gate=CONTINUE 且预算未尽 → 进下一轮 ----
    if "validate" in completed:
        gate = _resolve_gate(state, gate_result)
        if gate is not None and gate.loop_signal == "CONTINUE":
            return _dispatch_loop_signal(state, gate)

    # ---- 6. 尚未开始任何 attempt → 起第一轮 ----
    if state.total_attempts == 0:
        return Continue(next_attempt=0, next_failure_type=ft,
                        reason="session 首轮")

    # ---- 7. 轮内推进: 找本轮下一个未完成的 step ----
    for step in _ROUND_SEQUENCE:
        if step not in completed:
            return _make_action_for_step(step, ft, state.total_attempts - 1)

    # ---- 8. 本轮全部 step 完成但无 gate 信号 (异常) → 兜底当作需继续 ----
    return Continue(next_attempt=state.total_attempts, next_failure_type=ft,
                    reason="本轮步骤完成，进入下一轮")


def _resolve_gate(state: DebugState, gate_result: Optional[GateResult]):
    """取本轮 validate 的 GateResult: 优先用 runner 传入的；缺失 (crash-resume) 时
    从事件流最后一个 validate 的 action_completed.result 重建 (H2)。

    事件已持久化 loop_signal/stop_reason_code (_gate_result_to_dict)，故重放可还原，
    符合 event-sourcing「内存态可由事实源重建」契约。重建不出则返回 None。
    """
    if gate_result is not None:
        return gate_result
    result = _validate_result_this_attempt(state)
    if result is None:
        return None
    return parse_gate_output(result)


def _validate_result_this_attempt(state: DebugState) -> Optional[dict]:
    """本轮 (最后一个 attempt_started 之后) 最近一个 validate 步的 result dict。"""
    for e in reversed(state.events):
        if e.get("type") == "attempt_started":
            return None
        if e.get("type") == "action_completed" and \
                (e.get("action") or {}).get("step") == "validate":
            return e.get("result") or {}
    return None


def _dispatch_loop_signal(state: DebugState, gate: GateResult):
    """读 Gate-V 的 loop_signal 派发终态/续跑 (SKILL.md L1027-1046)。"""
    sig = gate.loop_signal
    if sig == "PASS":
        return Done(session_outcome="success", reason="Gate-V loop_signal=PASS")
    if sig == "STOP":
        outcome = _STOP_OUTCOME.get(gate.stop_reason_code or "", "failed")
        return Done(session_outcome=outcome,
                    reason=f"Gate-V STOP: {gate.stop_reason_code or 'unspecified'}")
    # CONTINUE: 进下一轮，failure_type 取「下一轮入口的最新 ft」。这里用当前 ft；
    # runner 在落 attempt_started 前会刷新 verify_status，故真实 ft 由下一拍 reload 决定。
    return Continue(next_attempt=state.total_attempts,
                    next_failure_type=state.current_failure_type,
                    reason=f"Gate-V CONTINUE: {gate.loop_reason or ''}")


def _latest_import_subtype(state: DebugState) -> Optional[str]:
    """从最近的 action_completed.result 里取 import_subtype (runner 跑 verify 后写入)。

    首轮无 action_completed 时返回 None → 调用方保守按 import_kernel_side 进入 1-I。
    runner (Step 5) 须在 session_started / 首次 verify 把 import_subtype 写进事件 result，
    否则首轮 import_env_side 会漏判进 1-I (纯函数不读文件，靠 runner 喂数据)。
    """
    last = state.last_action_completed()
    if last is None:
        return None
    return (last.get("result") or {}).get("import_subtype")
