"""exit_artifacts.py — 退出产物从 events.jsonl 重建 (不靠 LLM 手写)。

REWRITE_PLAN §2.7: runner 抵达终态时调用，从事件流重放生成两份强制退出产物:
  debug_status.json — 机器可读 verdict (10 键，schema 见 exit-protocols.md §7.2)
  debug_trace.md    — 4 节叙事 (入口快照 / 迭代历史 / Verdict / 产物清单，§7.1)

determinism 对论文 reproducibility 的直接贡献: 退出产物不再靠 LLM 手写 (消除漏写/编造
风险)，而是从权威事实源 events.jsonl 确定性重建。timeout/crashed 终态同样从 events 重建，
不依赖进程正常退出。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from engine.events import read_events
from engine.state import DebugState

SCHEMA_VERSION = 1

# failure_type → session_branch 标签 (exit-protocols.md §7.2: 填入口分支，后续漂移不更新)。
_BRANCH_LABEL = {
    "precision_failed": "1-P",
    "build_failed": "1-B",
    "import_failed": "1-I",
    "runtime_error": "1-R",
    "timeout": "1-T",
}

_TERMINAL_TYPES = {"session_done", "session_aborted", "session_escalated"}


def _iso(ts_ns: Optional[int]) -> Optional[str]:
    """事件 seq (time.time_ns()) → ISO8601 UTC。None 透传。"""
    if ts_ns is None:
        return None
    return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _final_failure_type(events: list[dict]) -> Optional[str]:
    """最后一个 attempt_started 的 failure_type；无 attempt 则 None (skipped_* 场景)。"""
    for e in reversed(events):
        if e.get("type") == "attempt_started":
            return e.get("failure_type")
    return None


def _terminal_event(events: list[dict]) -> Optional[dict]:
    for e in reversed(events):
        if e.get("type") in _TERMINAL_TYPES:
            return e
    return None


def _session_outcome(events: list[dict]) -> str:
    """从终态事件取 session_outcome；无终态事件 (异常中断) 归 crashed。"""
    term = _terminal_event(events)
    if term is None:
        return "crashed"
    if term.get("type") == "session_done":
        return term.get("session_outcome", "failed")
    if term.get("type") == "session_aborted":
        details = term.get("details") or {}
        return details.get("session_outcome", "crashed")
    return "crashed"  # escalated (debug 不用) 保守归 crashed


def build_debug_status(task_dir: Path) -> dict:
    """从 events 派生 debug_status.json 的 10 键 dict (exit-protocols.md §7.2)。"""
    events = read_events(task_dir)
    state = DebugState.load(task_dir)
    entry_ft = state.entry_failure_type
    final_ft = _final_failure_type(events) or entry_ft

    started_at = _iso(events[0].get("seq")) if events else None
    term = _terminal_event(events)
    ended_at = _iso(term.get("seq")) if term else None

    final_status_path: Optional[str] = None
    if final_ft is not None and state.total_attempts > 0:
        # 指向最后一轮的 phase8 verify 产物 (runner 跑 verify 落盘的约定路径)。
        last_attempt = state.total_attempts - 1
        final_status_path = str(
            Path(task_dir) / ".verify_status" / f"phase8_attempt{last_attempt}.json"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "session_outcome": _session_outcome(events),
        "session_branch": _BRANCH_LABEL.get(entry_ft or "", None),
        "started_at": started_at,
        "ended_at": ended_at,
        "attempts_used": state.total_attempts,
        "entry_failure_type": entry_ft,
        "final_failure_type": final_ft,
        "final_verify_status_path": final_status_path,
        "notes": _terminal_reason(term),
    }


def _terminal_reason(term: Optional[dict]) -> str:
    if term is None:
        return "无终态事件 (异常中断)"
    return term.get("reason") or ""


def write_debug_status(task_dir: Path) -> Path:
    status = build_debug_status(task_dir)
    path = Path(task_dir) / "debug_status.json"
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _iter_attempts(events: list[dict]) -> list[dict]:
    """把事件按 attempt 分组: [{attempt, failure_type, completed:[step,...]}, ...]。"""
    attempts: list[dict] = []
    cur: Optional[dict] = None
    for e in events:
        t = e.get("type")
        if t == "attempt_started":
            cur = {
                "attempt": e.get("attempt"),
                "failure_type": e.get("failure_type"),
                "completed": [],
            }
            attempts.append(cur)
        elif t == "action_completed" and cur is not None:
            step = (e.get("action") or {}).get("step")
            result = e.get("result") or {}
            cur["completed"].append({
                "step": step,
                "loop_signal": result.get("loop_signal"),
                "stop_reason_code": result.get("stop_reason_code"),
            })
    return attempts


def build_debug_trace(task_dir: Path) -> str:
    """从 events 重建 debug_trace.md (4 节强制叙事)。"""
    events = read_events(task_dir)
    status = build_debug_status(task_dir)
    lines: list[str] = ["# AscendC Debug Trace", ""]

    # 1. 调用入口快照
    lines += ["## 1. 调用入口快照", ""]
    lines.append(f"- 调用时间: {status['started_at'] or '(未记录)'}")
    lines.append(f"- task_dir: {task_dir}")
    lines.append(f"- session_branch: {status['session_branch'] or '(未路由)'}")
    lines.append(f"- entry_failure_type: {status['entry_failure_type'] or '(无)'}")
    lines.append("")

    # 2. 迭代历史 (每轮一节)
    lines += ["## 2. 迭代历史", ""]
    attempts = _iter_attempts(events)
    if not attempts:
        lines.append("- (无 attempt: 入口即退出，见 Verdict)")
        lines.append("")
    for a in attempts:
        lines.append(f"### Attempt {a['attempt']}")
        lines.append(f"- failure_type: {a['failure_type']}")
        steps = ", ".join(c["step"] or "?" for c in a["completed"]) or "(无完成步骤)"
        lines.append(f"- 已完成步骤: {steps}")
        # 取本轮 validate 的 loop_signal / stop_reason_code (若有)。
        for c in a["completed"]:
            if c["step"] == "validate" and c["loop_signal"]:
                extra = f"- Gate-V: loop_signal={c['loop_signal']}"
                if c["stop_reason_code"]:
                    extra += f", stop_reason_code={c['stop_reason_code']}"
                lines.append(extra)
        lines.append("")

    # 3. 最终 Verdict
    lines += ["## 3. 最终 Verdict", ""]
    lines.append(f"- session_outcome: {status['session_outcome']}")
    lines.append(f"- attempts_used: {status['attempts_used']}")
    lines.append(f"- final_failure_type: {status['final_failure_type'] or '(无)'}")
    if status["notes"]:
        lines.append(f"- 原因: {status['notes']}")
    lines.append("")

    # 4. 产物清单
    lines += ["## 4. 产物清单", ""]
    lines.append("- debug_status.json")
    lines.append(f"- .debug_events/events.jsonl ({len(events)} 条事件)")
    if status["final_verify_status_path"]:
        rel = Path(status["final_verify_status_path"]).relative_to(task_dir)
        lines.append(f"- {rel}")
    lines.append("")

    return "\n".join(lines)


def write_debug_trace(task_dir: Path) -> Path:
    path = Path(task_dir) / "debug_trace.md"
    path.write_text(build_debug_trace(task_dir), encoding="utf-8")
    return path


def write_exit_artifacts(task_dir: Path) -> tuple[Path, Path]:
    """runner 终态调用: 同时落地 debug_status.json + debug_trace.md。

    顺序: 先 status (trace 引用其字段)，再 trace。
    """
    status_path = write_debug_status(task_dir)
    trace_path = write_debug_trace(task_dir)
    return status_path, trace_path
