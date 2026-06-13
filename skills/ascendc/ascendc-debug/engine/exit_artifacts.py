"""exit_artifacts.py — 退出产物从 events.jsonl 重建 (不靠 LLM 手写)。

REWRITE_PLAN §2.7: runner 抵达终态时调用，从事件流重放生成两份强制退出产物:
  debug_status.json — 机器可读 verdict (10 键，schema 见 exit-protocols.md §7.2)
  debug_trace.md    — 4 节叙事 (入口快照 / 迭代历史 / Verdict / 产物清单，§7.1)

determinism 对论文 reproducibility 的直接贡献: 退出产物不再靠 LLM 手写 (消除漏写/编造
风险)，而是从权威事实源 events.jsonl 确定性重建。timeout/crashed 终态同样从 events 重建，
不依赖进程正常退出。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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

# 项 12b: kernel 源文件快照 (per-file hash)，runner 在 diagnose 派发前后各取一次，
# diff 出本轮真实改动文件名 → diagnosis_summary.changed_files。globs 与 forensics
# 输入对齐 (validate_runner._FORENSICS_SRC_GLOBS)，覆盖 kernel/ 下 .cpp/.h/.hpp/.py
# + task_dir 根 model_new_ascendc.py。code_snapshot/ 不可用 (仅 Gate-A 写，引擎不调)。
_KERNEL_SRC_GLOBS = ("*.cpp", "*.h", "*.hpp", "*.py")


def _kernel_file_hashes(task_dir: Path) -> dict[str, str]:
    """{相对 task_dir 的源文件路径: sha256}。读异常的文件跳过 (best-effort)。"""
    task_dir = Path(task_dir)
    out: dict[str, str] = {}
    files: list[Path] = []
    kernel_dir = task_dir / "kernel"
    if kernel_dir.is_dir():
        for pat in _KERNEL_SRC_GLOBS:
            files.extend(kernel_dir.rglob(pat))
    model_new = task_dir / "model_new_ascendc.py"
    if model_new.exists():
        files.append(model_new)
    for f in sorted(set(files), key=str):
        # 排除点文件: macOS AppleDouble (._foo.cpp，归档经 mac 中转的真实残留，见
        # hyena 快照) 及其他隐藏文件，不是真实源码，会污染 changed_files 文件名列表。
        if f.name.startswith("."):
            continue
        try:
            rel = f.relative_to(task_dir).as_posix()
            out[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
        except OSError:
            continue
    return out


def diff_changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """两次 _kernel_file_hashes 快照 diff → 新增/修改/删除的文件名 (排序)。"""
    changed = set()
    for path, h in after.items():
        if before.get(path) != h:
            changed.add(path)
    for path in before:
        if path not in after:
            changed.add(path)
    return sorted(changed)


def _iso(ts_ns: Optional[int]) -> Optional[str]:
    """事件 seq (time.time_ns()) → ISO8601 UTC。None 透传。"""
    if ts_ns is None:
        return None
    return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _duration(start_ns: Optional[int], end_ns: Optional[int]) -> str:
    if start_ns is None or end_ns is None:
        return "(未记录)"
    elapsed = max(0.0, (end_ns - start_ns) / 1e9)
    if elapsed < 10:
        return f"{elapsed:.3f}s"
    return f"{elapsed:.1f}s"


def _md_cell(value: Any) -> str:
    text = str(value) if value is not None else ""
    return text.replace("\n", " ").replace("|", "\\|")


def _action_label(action: dict) -> str:
    step = action.get("step") or action.get("name") or action.get("kind")
    return str(step or "?")


def _result_summary(result: dict) -> str:
    fields = []
    for key in (
        "loop_signal",
        "stop_reason_code",
        "failure_type",
        "claude_state",
        "api_error_status",
        "status",
        "returncode",
        "error",
    ):
        value = result.get(key)
        if value not in (None, ""):
            fields.append(f"{key}={value}")
    return ", ".join(fields) or "ok"


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


def _latest_existing_phase8(task_dir: Path, attempts: int) -> Optional[str]:
    status_dir = Path(task_dir) / ".verify_status"
    for attempt in range(attempts - 1, -1, -1):
        path = status_dir / f"phase8_attempt{attempt}.json"
        if path.exists():
            return str(path)
    latest = status_dir / "latest.json"
    if latest.exists():
        return str(latest)
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
        final_status_path = _latest_existing_phase8(task_dir, state.total_attempts)

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


# ---------------------------------------------------------------------------
# 诊断摘要 (6.11 文档 Tier 2 项 12b): engine 每轮 validate 后产出独立的
# diagnosis_summary_attempt_{N}.json，数据源 = agent final_response + changed_files
# (均来自 diagnose_and_fix 事件 result) + validation summary (validation_result_attempt_N
# .json)，**不依赖 precision_audit_N.md** (该文件实测缺失率 100%: Gate-A 不被引擎调用、
# 约束层不强制、无 CLI 引擎生成——正是 12b 要解决的问题)。precision_audit_N.md 若存在
# 则降级为可选 enrichment (折叠 root_cause/fix_plan)。
#
# 范式 (方案 B, runner = owner): 诊断"内容"是 agent 推理产物 (final_response)，"汇编"
# 是 engine 确定性搬运。每轮 diagnose 后即写，贴 attempt_N 语义、CONTINUE 时也留档。
# best-effort: 任何文件缺失/读异常 → None/降级，绝不影响终态产物落地。
# ---------------------------------------------------------------------------

DIAGNOSIS_SCHEMA_VERSION = 1

# 诊断摘要单 section 截断上限 (防 ROOT_CAUSE 长篇撑爆 trace；trace 是概览非全文)。
_DIAGNOSIS_SECTION_MAXLEN = 600


def _extract_audit_section(content: str, tag: str) -> Optional[str]:
    """从 audit 文本抽取 `[TAG]` section 正文 (与 gate 侧 _extract_section 同源)。

    audit 文件历史上两种 marker 格式并存: 裸 `[ROOT_CAUSE]` (SKILL.md 模板规定，多数
    产物) 与 `## [ROOT_CAUSE]` (部分产物带 markdown 标题)。裸 `[{tag}]` 是两者公共子串，
    故用裸前缀定位即可同时命中。边界: 到下一 section 头 (`\n[` 或 `\n## [`) 或
    `=== END AUDIT ===` 或文末。返回去首尾空白的正文；未找到 marker 或正文空 → None。
    """
    marker = f"[{tag}]"
    idx = content.find(marker)
    if idx == -1:
        return None
    body_start = idx + len(marker)
    rest = content[body_start:]
    # 找下一个 section 头或结束标记 (兼容裸 [ 与 ## [ 两种格式)。
    end = len(rest)
    for sentinel in ("\n[", "\n## [", "=== END AUDIT ==="):
        pos = rest.find(sentinel)
        if pos != -1:
            end = min(end, pos)
    body = rest[:end].strip()
    return body or None


def _diagnose_result_for_attempt(events: list[dict], attempt: int) -> dict:
    """取指定 attempt 的 diagnose_and_fix 事件 result (final_response/changed_files 源)。

    按 attempt_started 分组定位；同 attempt 多次 diagnose (重派) 取最后一次。无 → {}。
    """
    cur: Optional[int] = None
    found: dict = {}
    for e in events:
        t = e.get("type")
        if t == "attempt_started":
            cur = e.get("attempt")
        elif t == "action_completed" and cur == attempt:
            if (e.get("action") or {}).get("step") == "diagnose_and_fix":
                found = e.get("result") or {}
    return found


def _validation_summary_for_attempt(task_dir: Path, attempt: int) -> Optional[dict]:
    """读 validation_result_attempt_{N}.json 的核心字段 (engine 每轮 validate 产出)。"""
    path = Path(task_dir) / "precision_tuning" / f"validation_result_attempt_{attempt}.json"
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    out: dict[str, Any] = {}
    for key in ("correctness_passed", "match_rate", "max_diff",
                "passed_cases", "total_cases", "first_error_lines"):
        if data.get(key) is not None:
            out[key] = data[key]
    return out or None


def _audit_enrichment(task_dir: Path, attempt: int) -> dict:
    """可选 enrichment: precision_audit_{N}.md 若存在则折叠 root_cause/fix_plan。

    audit 实测缺失率 100% (Gate-A 不被引擎调用)，故仅作锦上添花，缺失静默跳过。
    """
    path = Path(task_dir) / "precision_tuning" / f"precision_audit_{attempt}.md"
    out: dict[str, str] = {}
    try:
        if not path.exists():
            return out
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return out
    for tag, key in (("ROOT_CAUSE", "root_cause"), ("FIX_PLAN", "fix_plan")):
        sec = _extract_audit_section(content, tag)
        if sec:
            if len(sec) > _DIAGNOSIS_SECTION_MAXLEN:
                sec = sec[:_DIAGNOSIS_SECTION_MAXLEN].rstrip() + " …(截断)"
            out[key] = sec
    return out


def build_diagnosis_summary(
    task_dir: Path, attempt: int, events: Optional[list[dict]] = None,
) -> Optional[dict]:
    """汇编单轮诊断摘要 (项 12b)。三源全缺 → None。

    final_response/changed_files ← diagnose_and_fix 事件 result;
    validation ← validation_result_attempt_{N}.json;
    root_cause/fix_plan ← precision_audit_{N}.md (可选 enrichment)。
    """
    task_dir = Path(task_dir)
    if events is None:
        events = read_events(task_dir)
    diag = _diagnose_result_for_attempt(events, attempt)

    out: dict[str, Any] = {
        "schema_version": DIAGNOSIS_SCHEMA_VERSION,
        "attempt": attempt,
    }
    final_response = diag.get("final_response")
    if final_response:
        out["final_response"] = final_response
    changed = diag.get("changed_files")
    if changed is not None:
        out["changed_files"] = changed
    validation = _validation_summary_for_attempt(task_dir, attempt)
    if validation:
        out["validation"] = validation
    out.update(_audit_enrichment(task_dir, attempt))

    # 仅有 schema_version/attempt 两个骨架键 → 本轮无任何实质内容 → None。
    if set(out) <= {"schema_version", "attempt"}:
        return None
    return out


def write_diagnosis_summary(task_dir: Path, attempt: int) -> Optional[Path]:
    """落 diagnosis_summary_attempt_{N}.json。best-effort: 失败返回 None，绝不抛异常。"""
    try:
        summary = build_diagnosis_summary(task_dir, attempt)
        if summary is None:
            return None
        tuning = Path(task_dir) / "precision_tuning"
        tuning.mkdir(parents=True, exist_ok=True)
        path = tuning / f"diagnosis_summary_attempt_{attempt}.json"
        path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    except Exception:  # noqa: BLE001
        return None


def _load_diagnosis_summary(task_dir: Path, attempt: int) -> Optional[dict]:
    """读已归档的 diagnosis_summary_attempt_{N}.json (debug_trace §2 消费)。缺/坏 → None。"""
    path = Path(task_dir) / "precision_tuning" / f"diagnosis_summary_attempt_{attempt}.json"
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _iter_attempts(events: list[dict]) -> list[dict]:
    """把事件按 attempt 分组，并配对 action started/completed 时间戳。"""
    attempts: list[dict[str, Any]] = []
    cur: Optional[dict[str, Any]] = None
    last_seq: Optional[int] = None

    def close_current(end_ns: Optional[int]) -> None:
        if cur is not None and cur.get("ended_ns") is None:
            cur["ended_ns"] = end_ns

    for e in events:
        t = e.get("type")
        if t == "attempt_started":
            close_current(last_seq)
            cur = {
                "attempt": e.get("attempt"),
                "failure_type": e.get("failure_type"),
                "started_ns": e.get("seq"),
                "ended_ns": None,
                "last_event_ns": e.get("seq"),
                "completed": [],
                "actions": [],
                "_actions_by_id": {},
            }
            attempts.append(cur)
        elif t == "action_started" and cur is not None:
            action = e.get("action") or {}
            action_id = e.get("action_id") or f"anon_{len(cur['actions'])}"
            item = {
                "action_id": action_id,
                "step": _action_label(action),
                "started_ns": e.get("seq"),
                "ended_ns": None,
                "result": {},
            }
            cur["actions"].append(item)
            cur["_actions_by_id"][action_id] = item
            cur["last_event_ns"] = e.get("seq")
        elif t == "action_completed" and cur is not None:
            action = e.get("action") or {}
            action_id = e.get("action_id") or f"anon_{len(cur['actions'])}"
            step = (e.get("action") or {}).get("step")
            result = e.get("result") or {}
            cur["completed"].append({
                "step": step,
                "loop_signal": result.get("loop_signal"),
                "stop_reason_code": result.get("stop_reason_code"),
            })
            item = cur["_actions_by_id"].get(action_id)
            if item is None:
                item = {
                    "action_id": action_id,
                    "step": _action_label(action),
                    "started_ns": None,
                    "ended_ns": None,
                    "result": {},
                }
                cur["actions"].append(item)
                cur["_actions_by_id"][action_id] = item
            item["step"] = item.get("step") or _action_label(action)
            item["ended_ns"] = e.get("seq")
            item["result"] = result
            cur["last_event_ns"] = e.get("seq")
        elif t in _TERMINAL_TYPES:
            close_current(e.get("seq"))
            if cur is not None:
                cur["last_event_ns"] = e.get("seq")
        elif cur is not None:
            cur["last_event_ns"] = e.get("seq")
        last_seq = e.get("seq") or last_seq

    # 异常中断无终态时，保留 last_event_at，但不伪造 ended_at。
    for attempt in attempts:
        attempt.pop("_actions_by_id", None)
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
        lines.append(f"- started_at: {_iso(a.get('started_ns')) or '(未记录)'}")
        if a.get("ended_ns") is not None:
            lines.append(f"- ended_at: {_iso(a.get('ended_ns')) or '(未记录)'}")
            lines.append(f"- duration: {_duration(a.get('started_ns'), a.get('ended_ns'))}")
        elif a.get("last_event_ns") is not None:
            lines.append(f"- last_event_at: {_iso(a.get('last_event_ns')) or '(未记录)'}")
        steps = ", ".join(c["step"] or "?" for c in a["completed"]) or "(无完成步骤)"
        lines.append(f"- 已完成步骤: {steps}")
        if a["actions"]:
            lines.append("")
            lines.append("| step | started_at | ended_at | duration | result |")
            lines.append("|------|------------|----------|----------|--------|")
            for action in a["actions"]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _md_cell(action.get("step")),
                            _md_cell(_iso(action.get("started_ns")) or "(未记录)"),
                            _md_cell(_iso(action.get("ended_ns")) or "(未记录)"),
                            _md_cell(_duration(action.get("started_ns"), action.get("ended_ns"))),
                            _md_cell(_result_summary(action.get("result") or {})),
                        ]
                    )
                    + " |"
                )
        # 取本轮 validate 的 loop_signal / stop_reason_code (若有)。
        for c in a["completed"]:
            if c["step"] == "validate" and c["loop_signal"]:
                extra = f"- Gate-V: loop_signal={c['loop_signal']}"
                if c["stop_reason_code"]:
                    extra += f", stop_reason_code={c['stop_reason_code']}"
                lines.append(extra)
        # 诊断摘要 (engine 每轮 validate 后归档的独立 JSON，项 12b；精度分支有，余分支降级)。
        diag = _load_diagnosis_summary(task_dir, a["attempt"])
        if diag is None:
            lines.append("- 诊断摘要: (无)")
        else:
            lines.append("- 诊断摘要:")
            if diag.get("changed_files") is not None:
                files = ", ".join(diag["changed_files"]) or "(无改动)"
                lines.append(f"  - changed_files: {files}")
            val = diag.get("validation") or {}
            if val.get("correctness_passed") is not None:
                lines.append(f"  - correctness_passed: {val['correctness_passed']}")
            if val.get("match_rate") is not None:
                lines.append(f"  - match_rate: {val['match_rate']}")
            if diag.get("root_cause"):
                lines.append(f"  - root_cause: {_md_cell(diag['root_cause'])}")
            if diag.get("fix_plan"):
                lines.append(f"  - fix_plan: {_md_cell(diag['fix_plan'])}")
            if diag.get("final_response"):
                lines.append(f"  - final_response: {_md_cell(diag['final_response'])}")
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


# ===========================================================================
# run_summary.json — 批跑后处理直接消费的机器摘要 (6.11 文档 Tier 2 项 11)。
#
# 设计: 独立 JSON，不混入 debug_trace.md (trace 是人读叙事)。聚合 turns / gate /
# anti_cheat / kb_finalize / forensics，免 batch report 再扫各处日志。
#
# 成本口径只用 turns (agent_backend 透传的 num_turns)，不含 total_cost_usd——
# 美元随 --model 浮动、跨模型不可比，turns 模型无关 (沿用 Tier 1 同源决策)。
# best-effort: 任何子项读取失败降级为空/None，绝不影响终态。
# ===========================================================================

RUN_SUMMARY_SCHEMA_VERSION = 1


def _turns_summary(events: list[dict]) -> dict:
    """累加 diagnose_and_fix 步的 agent_turns，并按 attempt 拆分。

    事件流是唯一事实源 (与 next_action._total_agent_turns 同源)。缺失/非正整数按
    0 计 (timeout/spawn_failed 早返回路径无此字段)。
    """
    by_attempt: dict[int, int] = {}
    total = 0
    cur_attempt: Optional[int] = None
    for e in events:
        t = e.get("type")
        if t == "attempt_started":
            cur_attempt = e.get("attempt")
            by_attempt.setdefault(cur_attempt, 0)
        elif t == "action_completed":
            if (e.get("action") or {}).get("step") != "diagnose_and_fix":
                continue
            turns = (e.get("result") or {}).get("agent_turns")
            if isinstance(turns, int) and turns > 0:
                total += turns
                if cur_attempt is not None:
                    by_attempt[cur_attempt] = by_attempt.get(cur_attempt, 0) + turns
    return {
        "total": total,
        "by_attempt": {str(k): v for k, v in sorted(by_attempt.items())},
    }


def _gate_summary(events: list[dict]) -> dict:
    """各轮 validate 步的 loop_signal/stop_reason_code，及最终一轮的终判信号。"""
    by_attempt: list[dict] = []
    cur_attempt: Optional[int] = None
    for e in events:
        t = e.get("type")
        if t == "attempt_started":
            cur_attempt = e.get("attempt")
        elif t == "action_completed":
            if (e.get("action") or {}).get("step") != "validate":
                continue
            result = e.get("result") or {}
            sig = result.get("loop_signal")
            if sig is None:
                continue
            by_attempt.append({
                "attempt": cur_attempt,
                "loop_signal": sig,
                "stop_reason_code": result.get("stop_reason_code"),
            })
    final = by_attempt[-1] if by_attempt else {}
    return {
        "final_loop_signal": final.get("loop_signal"),
        "final_stop_reason_code": final.get("stop_reason_code"),
        "by_attempt": by_attempt,
    }


def _anti_cheat_summary(task_dir: Path) -> dict:
    """cheat_history.json 的 cheating_attempts 计数 (violation/warning 分列)。

    文件不存在 = 从未触发任何检查 = 全 0 (clean)。解析失败同样降级全 0。
    """
    path = Path(task_dir) / "precision_tuning" / "cheat_history.json"
    empty = {"total": 0, "violations": 0, "warnings": 0}
    if not path.exists():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return empty
    attempts = data.get("cheating_attempts") or []
    violations = sum(1 for a in attempts if a.get("severity") == "violation")
    warnings = sum(1 for a in attempts if a.get("severity") == "warning")
    return {"total": len(attempts), "violations": violations, "warnings": warnings}


def _forensics_summary(task_dir: Path, attempts: int) -> dict:
    """哪些 attempt 落了 forensics_report (cache 复用也会复制成当轮名，一并计入)。"""
    tuning = Path(task_dir) / "precision_tuning"
    reports = [
        a for a in range(max(attempts, 0))
        if (tuning / f"forensics_report_{a}.json").exists()
    ]
    return {"reports": reports}


def build_run_summary(task_dir: Path, status: Optional[dict] = None) -> dict:
    """从 events + 各产物文件聚合 run_summary (batch report 程序化消费)。

    status 可由调用方 (runner._terminate) 传入已构建好的 debug_status (含 kb_finalize)，
    避免重复 build；未传则现场 build。
    """
    task_dir = Path(task_dir)
    events = read_events(task_dir)
    if status is None:
        status = build_debug_status(task_dir)
    attempts_used = status.get("attempts_used") or 0
    return {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "session_outcome": status.get("session_outcome"),
        "session_branch": status.get("session_branch"),
        "attempts_used": attempts_used,
        "turns": _turns_summary(events),
        "gate": _gate_summary(events),
        "anti_cheat": _anti_cheat_summary(task_dir),
        "kb_finalize": status.get("kb_finalize"),
        "forensics": _forensics_summary(task_dir, attempts_used),
    }


def write_run_summary(task_dir: Path, status: Optional[dict] = None) -> Optional[Path]:
    """落 run_summary.json。best-effort: 失败返回 None，绝不抛异常影响终态。"""
    try:
        summary = build_run_summary(task_dir, status)
        path = Path(task_dir) / "run_summary.json"
        path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    except Exception:
        return None
