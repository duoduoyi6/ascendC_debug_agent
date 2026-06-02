"""runner.py — debug session 通用主循环 (方案 B: runner 是 owner)。

镜像 lingxi runtime/engine/runner.py 的 run_autonomous 形态，裁剪到 debug 所需 +
加 lingxi 没有的两件 debug 特有逻辑:
  1. Continue 决策 = attempt 边界 (落 attempt_started 事件，lingxi 无此概念)。
  2. wall-clock timeout 闸 (REWRITE_PLAN §2.3b 闸 3): 超时主动 emit 终态事件，不裸死。

主循环每拍:
    decision = debug_next_action(state, gate_result)
      ├ Done/Abort        → record_decision + write_exit_artifacts + return
      ├ Continue          → record_attempt_started + reload
      └ Action            → record_started → dispatch → record_completed → reload
                            (validate 步的 GateResult 缓存，喂给下一拍 next_action)

dispatcher 解耦 (可注入 mock，UT 不需 NPU): 默认实现 py_action→run_gate / spawn_agent→
agent_callback。resume = DebugState.load 重放，无额外逻辑。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from engine import transition
from engine.exit_artifacts import write_exit_artifacts
from engine.gate_adapter import GateResult, parse_gate_output, run_gate
from engine.next_action import debug_next_action
from engine.state import DebugState
from engine.types import Abort, Action, Continue, Done

# agent_callback 签名: (action, task_dir, op_name, attempt) -> result dict。
AgentCallback = Callable[[Action, Path, str, int], dict]
# dispatcher 签名: (action, task_dir, op_name, agent_callback) -> result dict。
Dispatcher = Callable[[Action, Path, str, Optional[AgentCallback]], dict]


class RunnerError(RuntimeError):
    pass


# 引擎可派发的 Action 白名单 (对应 lingxi validate_action)。
_AVAILABLE_PY_ACTIONS = ("precision_gate",)
_AVAILABLE_AGENTS = ("debug_worker",)


def validate_action(action: Action) -> None:
    if action.kind == "py_action":
        if action.name not in _AVAILABLE_PY_ACTIONS:
            raise RunnerError(
                f"py_action '{action.name}' 不在白名单 {_AVAILABLE_PY_ACTIONS}")
    elif action.kind == "spawn_agent":
        if action.name not in _AVAILABLE_AGENTS:
            raise RunnerError(
                f"spawn_agent '{action.name}' 不在白名单 {_AVAILABLE_AGENTS}")
    else:
        raise RunnerError(f"未知 action.kind={action.kind!r}")


def _default_dispatcher(
    action: Action,
    task_dir: Path,
    op_name: str,
    agent_callback: Optional[AgentCallback],
) -> dict:
    """默认派发: py_action 跑 run_gate / spawn_agent 调 agent_callback。

    py_action(precision_gate): 跑 gate CLI，把 GateResult 摊平进 result，并履行
    import_subtype 喂数据契约 (Step 4 标注: next_action 靠 result.import_subtype 判
    import_env_side)。
    spawn_agent(debug_worker): 委托 agent_callback (NPU 上拉起 agent 进程)。
    """
    args = action.skill_args or {}
    if action.kind == "py_action":
        gate_step = args.get("gate_step", "validate")
        attempt = int(args.get("attempt", 0))
        gr = run_gate(task_dir, step=gate_step, op_name=op_name, attempt=attempt)
        return _gate_result_to_dict(gr)
    if action.kind == "spawn_agent":
        if agent_callback is None:
            raise RunnerError("spawn_agent 需要 agent_callback，但未提供")
        attempt = int(args.get("attempt", 0))
        return agent_callback(action, task_dir, op_name, attempt)
    raise RunnerError(f"无法派发 action.kind={action.kind!r}")


def _gate_result_to_dict(gr: GateResult) -> dict:
    """GateResult → 事件 result dict。摊平 loop_signal/stop_reason_code +
    透传 import_subtype (从 checks 提取，履行喂数据契约)。"""
    return {
        "gate": gr.gate,
        "passed": gr.passed,
        "loop_signal": gr.loop_signal,
        "loop_reason": gr.loop_reason,
        "stop_reason_code": gr.stop_reason_code,
        "prerequisite_error": gr.prerequisite_error,
        "import_subtype": (gr.checks or {}).get("import_subtype"),
        "checks": gr.checks,
    }


def _result_to_gate(result: dict) -> Optional[GateResult]:
    """把 validate 步的 result dict 还原成 GateResult，喂给下一拍 next_action。
    非 gate result (无 loop_signal 字段) 返回 None。"""
    if "loop_signal" not in result and "gate" not in result:
        return None
    return parse_gate_output(result)


# 终态事件保护: 防止 Action 数远超合理上限时无限循环 (next_action 应已收敛，这是
# 防御性硬上限，按 MAX_ATTEMPTS * 每轮步数 * 安全系数估算)。
_MAX_TICKS = 200


def run_debug_session(
    task_dir: Path,
    *,
    op_name: str,
    agent: str = "constructive",
    entry_failure_type: Optional[str] = None,
    agent_callback: Optional[AgentCallback] = None,
    dispatcher: Optional[Dispatcher] = None,
    deadline_sec: Optional[float] = None,
    _now: Callable[[], float] = time.monotonic,
) -> dict:
    """驱动一个 debug session 到终态。返回 debug_status dict。

    task_dir: session 工作目录 (events.jsonl 落在 .debug_events/ 下)。
    op_name: 算子名 (传给 gate / agent)。
    agent: constructive | discovery (写进 session_started，供追溯)。
    entry_failure_type: 入口 failure_type；None 时由首拍 reload 决定 (resume 场景)。
    agent_callback: spawn_agent 的执行体 (NPU 上拉起 agent)；mock 时注入。
    dispatcher: 自定义派发器 (UT 注入 mock)；默认 _default_dispatcher。
    deadline_sec: wall-clock 总时长上限 (秒)；None 不限。超时主动 emit timeout 终态。
    _now: 单调时钟注入点 (UT 控制时间)。

    幂等/resume: 若 events.jsonl 已有 session_started，不重复写 (据已有事件续跑)。
    """
    task_dir = Path(task_dir)
    dispatcher = dispatcher or _default_dispatcher
    start = _now()

    # 初始化 / resume。
    existing = DebugState.load(task_dir)
    if not existing.events:
        # 全新 session: 写 session_started。
        transition.record_session_started(
            task_dir, op_name=op_name, agent=agent,
            entry_failure_type=entry_failure_type,
        )
    else:
        # H4: 已终态 session 不重启——直接返回原 status，绝不在终态事件后追加。
        if existing.is_terminal():
            from engine.exit_artifacts import build_debug_status
            return build_debug_status(task_dir)
        # H3: resume 时 reconcile 崩溃残留的 dangling action_started。补一条
        # action_completed{success:false}，使下一轮是「干净重派」而非续跑叠加
        # (尤其 spawn_agent 重派 = 二次改 kernel)。
        _reconcile_dangling(task_dir, existing)

    gate_result: Optional[GateResult] = None
    tick = 0
    while True:
        tick += 1
        if tick > _MAX_TICKS:
            return _terminate(task_dir, Abort(
                category="state_machine_stuck",
                reason=f"超过最大 tick 数 {_MAX_TICKS}，疑似决策不收敛",
                details={"session_outcome": "crashed"}))

        # 闸 3: wall-clock timeout。超时主动 emit 终态，不裸死。
        if deadline_sec is not None and (_now() - start) >= deadline_sec:
            return _terminate(task_dir, Done(
                session_outcome="timeout",
                reason=f"wall-clock 超时 (>{deadline_sec}s)"))

        state = DebugState.load(task_dir)
        decision = debug_next_action(state, gate_result)

        if isinstance(decision, (Done, Abort)):
            return _terminate(task_dir, decision)

        if isinstance(decision, Continue):
            transition.record_attempt_started(task_dir, decision)
            gate_result = None  # 新一轮，清空上一轮 gate 缓存
            continue

        if isinstance(decision, Action):
            validate_action(decision)
            action_id = transition.new_action_id()
            transition.record_action_started(task_dir, decision, action_id)
            try:
                result = dispatcher(decision, task_dir, op_name, agent_callback)
            except Exception as exc:  # noqa: BLE001 — dispatch 失败转 Abort，不裸死
                transition.record_action_completed(
                    task_dir, decision, action_id,
                    {"success": False, "error": str(exc), "fatal": True})
                return _terminate(task_dir, Abort(
                    category="dispatch_error",
                    reason=f"派发 {decision.kind}:{decision.name} 失败: {exc}",
                    details={"session_outcome": "crashed"}))
            transition.record_action_completed(task_dir, decision, action_id, result)
            # validate 步: 缓存 GateResult 喂下一拍；其余步清空。
            if decision.step == "validate":
                gate_result = _result_to_gate(result)
            else:
                gate_result = None
            continue

        raise RunnerError(f"未知 decision 类型: {type(decision).__name__}")


def _reconcile_dangling(task_dir: Path, state: DebugState) -> None:
    """resume 时收尾崩溃残留: 对每个有 started 无 completed 的 action_id，补一条
    action_completed{success:false, reason:crash-recovery}。

    意义: 让 next_action 的「本轮已完成 step」集合把残留步算作「已尝试且失败」，
    下一拍重新派发是干净重试，而不是把半完成的 spawn_agent 当作未开始再拉一次
    (二次改 kernel)。补的 completed 不带 loop_signal，故不会被误当作 Gate-V 结果。
    """
    for ev in state.dangling_started():
        payload = ev.get("action") or {}
        action = Action(
            kind=payload.get("kind", "py_action"),
            name=payload.get("name", ""),
            step=payload.get("step"),
            skill_args=payload.get("skill_args"),
        )
        transition.record_action_completed(
            task_dir, action, ev.get("action_id"),
            {"success": False, "reason": "crash-recovery: 上次进程在此步崩溃，已收尾"})


def _terminate(task_dir: Path, decision) -> dict:
    """落终态事件 + 生成退出产物 + 返回 debug_status dict。"""
    transition.record_decision(task_dir, decision)
    write_exit_artifacts(task_dir)
    from engine.exit_artifacts import build_debug_status
    return build_debug_status(task_dir)
