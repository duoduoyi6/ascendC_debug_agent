"""agent_backend.py — diagnose_and_fix 的真实执行体 (方案 C 的唯一 spawn 点)。

引擎 next_action 里 forensics/validate 是确定性 py_action (runner 直接跑 precision_gate.py)，
**唯一需要拉起 agent 的是 diagnose_and_fix**。本模块把该 Action 翻译成一次 `claude --bare -p`
调用 (命令形态对齐批处理脚本的 Claude CLI 调用约定)，做「单个 attempt 的
诊断 + 改 kernel」，改完进程退出，控制权交回 runner (runner 随后跑 Gate-V 客观判定这一轮)。

方案 C 的核心边界 (防 reward-hack):
  - backend 只判断「claude 这一轮进程是否正常跑完」(读 --output-format json 的
    is_error/stop_reason/api_error_status)，返回的 success **不是「修好了」**。
  - 「修没修好」由 runner 之后的 validate gate 客观判定，agent 自报不算数。
  - 收窄 prompt 显式禁止 agent 自己跑 MAX_ATTEMPTS 循环 / 读 loop_signal 自决 / 写
    debug_status.json (这些编排职责已收归引擎)。

可注入: claude 命令通过 runner 参数构造，subprocess 调用点 (_run) 可被 UT monkeypatch，
UT 不真跑 claude。
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Callable, Optional

from engine.types import Action

_PROVIDER_LIMIT_STATUSES = {401, 402, 403, 429}

# 项 12b: agent final response 截断上限 (诊断摘要的主数据源——claude --output-format
# json 的顶层 `result` 字段是 agent 本轮的最终诊断陈述，实测含完整根因/证据链/修复说明，
# 通常 1~2KB)。透传给 next 层做 diagnosis_summary，超限截断防撑爆事件流/产物。
_FINAL_RESPONSE_MAXLEN = 4000

# cc 脚本默认值 (utils/run_ascendc_debug_batch_cc.sh L44-47)，保持一致。
_DEFAULT_CLAUDE_BIN = "claude"
_DEFAULT_ALLOWED_TOOLS = "Bash,Read,Write,Edit,Glob,Grep,Skill"
_DEFAULT_AGENT = "ascendc-debug-agent-constructive"  # AAAI 主力 (cc 默认是 discovery，这里默认 constructive)
_EFFORT_ENV = "ASCENDC_DEBUG_AGENT_EFFORT"

# 单轮约束覆盖层: 方案 C 下 agent 只做一个 attempt，编排归引擎。
# 不改 agent spec 本体 (留 Step 7)，在 prompt 末尾追加此约束覆盖自驱散文。
_SINGLE_ROUND_CONSTRAINT = """

═══════════════════════════════════════════════════════════════
【本次调用 = 单个 attempt，由确定性引擎驱动，务必遵守以下覆盖约束】
═══════════════════════════════════════════════════════════════
本次调用只负责**一个 attempt 的诊断 + 修复**，循环与终态由外层引擎掌控:
- 禁止自己跑 MAX_ATTEMPTS 循环；禁止读 Gate loop_signal 自行决定继续/停止/跳 Step5/Step6。
- 禁止写 debug_status.json / debug_trace.md (退出产物由引擎从事件流确定性重建)。
- 【受保护文件，严禁以任何方式修改/删除/重命名/覆写】: model.py、参考实现 wrapper
  (model_new_ascendc.py / model_new_tilelang.py)、算子配置 <op>.json 及其 .json.bak 备份、
  .verify_status/ 目录下任何文件。这些是客观评测的基准与产物，改动它们等同作弊 (会被反作弊
  hash 校验与 C++/AST 扫描捕获并判定 cheat_detected，本轮成果作废)。只允许修改
  {task_dir}/kernel/ 下的 .cpp/.h/.hpp 算子源码。
- 不要把自测/自评结果写成最终结论；最终 build/eval/classify/precision_gate 由引擎
  在本轮结束后统一执行。诊断中如确有必要，可以运行局部 build/eval/forensics 命令，
  但其结果只作为本轮根因分析证据，不作为流程终态。
- 读取上下文采用按需原则: 不要全文读取 SKILL.md、branch_precision.py、CANN API 文档、
  build/eval 日志或 .json 大文件，优先用 Grep/Read offset+limit/sed/head/tail 定位
  当前根因相关片段；只有诊断必须时才扩大读取范围。
- 知识库检索由 engine 在本轮 diagnose 前确定性执行，并已在 prompt 中注入摘要；
  不要重复运行 precision_knowledge.py search。若没有摘要，按当前 forensics/code 证据继续。
- Bash 命令应有明确诊断目的；长输出请重定向到文件并查看摘要/关键片段，避免完整递归
  grep、完整编译/验证输出反复进入上下文。
- 修复保持聚焦、最小且可解释；可以按根因需要修改 {task_dir}/kernel/ 下相关文件。
  完成本轮预期修改后停止，把验证交回引擎。
本次只做: 按当前 failure_type 走对应分支方法论 (精度走 Phase A→B→C 等) → 诊断根因 →
最小化修改 {task_dir}/kernel/ 下文件 → 完成即停 (不要追加任何收尾动作)。
如需参考 SKILL.md，只读取/遵循当前 failure_type 对应的诊断/修复方法论片段。
不要读取或执行 SKILL.md 中的全局调度、MAX_ATTEMPTS/loop_signal、Gate 验证、
Step5/Step6/Step7、退出协议、归档、批处理、report 生成等流程章节；这些由 engine 负责。
═══════════════════════════════════════════════════════════════
"""

# PLACEHOLDER_REST


def _cheat_warning(task_dir: Path) -> str:
    """读 cheat_history.json，若上一轮被判作弊则生成警告注入下一轮 prompt (修复4 场景A)。

    仅取最近一条 violation (非 warning——validator 异常不该警告 agent 它作弊)。无则返回空串。
    """
    path = task_dir / "precision_tuning" / "cheat_history.json"
    if not path.exists():
        return ""
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return ""
    violations = [e for e in history.get("cheating_attempts", [])
                  if e.get("severity") != "warning"]
    if not violations:
        return ""
    last = violations[-1]
    return (
        f"\n⚠️ 上一轮 (attempt {last.get('attempt')}) 的修复被检测为作弊:\n"
        f"  - 类型: {last.get('cheat_type')}\n"
        f"  - 要求: {last.get('instruction')}\n"
        f"  - 本轮必须用真实 AscendC kernel 实现，禁止用 torch 原生算子绕过。\n"
    )


def _knowledge_search_context(task_dir: Path, attempt: int) -> str:
    """Return a compact KB retrieval summary injected by engine, if present."""
    path = task_dir / "precision_tuning" / "knowledge_search_log.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return ""
    if not isinstance(data, list):
        return ""
    entries = [
        e for e in data
        if isinstance(e, dict) and e.get("attempt") == attempt
    ][-2:]
    if not entries:
        return ""

    lines = [
        "\n【引擎知识库检索摘要】",
        f"- log_path: {path}",
        "- 你应参考这些命中项，但根因仍必须由 forensics/code 证据验证。",
    ]
    for entry in entries:
        query = entry.get("query") or {}
        lines.append(
            f"- call_index={entry.get('call_index')} "
            f"op_type={query.get('op_type')} pattern={query.get('pattern')} "
            f"position={query.get('position')} "
            f"matched={entry.get('matched_count')} checklist={entry.get('checklist_count')} "
            f"fallback={entry.get('fallback_to_full_load')}"
        )
        for item in (entry.get("match_reasons") or [])[:3]:
            title = item.get("title")
            score = item.get("score")
            reason = ",".join(str(x) for x in (item.get("reason") or [])[:4])
            if title:
                lines.append(f"  - {title} (score={score}, reason={reason})")
        if not (entry.get("match_reasons") or []) and entry.get("top_titles"):
            for title in (entry.get("top_titles") or [])[:3]:
                lines.append(f"  - {title}")
    return "\n".join(lines) + "\n"


def _build_prompt(task_dir: Path, op_name: str, failure_type: str,
                  attempt: int, npu: Optional[str]) -> str:
    """构造收窄到「单个 attempt」的 diagnose prompt。

    agent spec 的 System Prompt 由 --agent 加载 (含分支方法论)，这里只传本轮参数 +
    单轮约束覆盖层。对齐 cc 脚本 PROMPT_TEMPLATE 的 "debug {task_dir} npu=" 入口形态。
    """
    npu_line = f"npu={npu}" if npu is not None else ""
    head = (
        f"debug {task_dir} {npu_line}\n\n"
        f"本轮上下文 (由引擎注入，勿自行改动):\n"
        f"  task_dir: {task_dir}\n"
        f"  op_name: {op_name}\n"
        f"  failure_type: {failure_type}\n"
        f"  attempt: {attempt}\n"
    )
    return (head + _cheat_warning(task_dir)
            + _knowledge_search_context(task_dir, attempt)
            + _SINGLE_ROUND_CONSTRAINT.replace("{task_dir}", str(task_dir)))


def _classify_claude_result(result_file: Path) -> dict:
    """读 claude --output-format json 结果，判定本轮进程是否正常完成。

    success=进程正常跑完本轮 (非「修好了」——后者由 runner 的 Gate-V 判定)。
    对齐 cc 脚本 read_claude_state / read_fatal_claude_error 的字段语义。
    """
    if not result_file.exists():
        return {"success": False, "claude_state": "missing_claude_result",
                "error": f"claude 未产出结果文件 {result_file.name}"}
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        return {"success": False, "claude_state": "invalid_claude_result",
                "error": f"claude 结果文件无法解析: {e}"}

    api_status = data.get("api_error_status")
    stop_reason = data.get("stop_reason")
    subtype = data.get("subtype")
    is_error = bool(data.get("is_error"))
    # api_error_status 更具体且是 fatal 判定依据 (对齐 cc 脚本 read_fatal_claude_error)，
    # 优先于泛化的 is_error。真实 claude API 错误两者常同时出现。
    if api_status is not None:
        state = f"api_error_{api_status}"
    elif is_error and subtype == "error_max_turns":
        state = "max_turns_exceeded"
    elif is_error:
        state = "claude_error"
    elif stop_reason == "pause_turn":
        state = "claude_pause_turn"  # 本轮未自然结束 (引擎可据此判断重试/收尾)
    else:
        state = "ok"
    result = {
        "success": state == "ok",
        "claude_state": state,
        "stop_reason": stop_reason,
        "subtype": subtype,
        "api_error_status": api_status,
        "error": None if state == "ok" else f"claude_state={state}",
    }
    # 透传本轮 agentic turn 数 (claude --output-format json 顶层 num_turns)。
    # next_action 跨 attempt 累加做任务级 turns 硬闸；非整数/缺失置 None (累加层按 0 计)。
    turns = data.get("num_turns")
    result["agent_turns"] = turns if isinstance(turns, int) else None
    # 项 12b: 透传 agent 本轮最终诊断陈述 (顶层 `result` 字段)，作为 diagnosis_summary
    # 的主数据源 (取代 100% 缺失的 precision_audit_N.md)。
    # 关键: 仅 state=="ok" 才提取——真实产物核实 (artifacts 20 样例) 表明 is_error/
    # stop_sequence 态的 `result` 是错误串 ("API Error: 400 ... token limit")，非诊断
    # 内容；error 态恒置 None，避免把 API 错误噪声塞进诊断摘要。非 str/空同样 None；超限截断。
    final = data.get("result")
    if state == "ok" and isinstance(final, str) and final.strip():
        final = final.strip()
        if len(final) > _FINAL_RESPONSE_MAXLEN:
            final = final[:_FINAL_RESPONSE_MAXLEN].rstrip() + " …(截断)"
        result["final_response"] = final
    else:
        result["final_response"] = None
    if api_status in _PROVIDER_LIMIT_STATUSES:
        result["fatal"] = True
        result["provider_error"] = True
    return result


def _build_claude_cmd(prompt: str, *, claude_bin: str, model: Optional[str],
                      agent_name: str, session_id: str, workdir: Path,
                      allowed_tools: str, result_file: Path,
                      max_turns: Optional[str] = None,
                      effort: Optional[str] = None) -> list:
    """构造 `claude --bare -p` 命令。

    输出写 result_file (--output-format json)；调用方负责把 stdout 重定向到该文件。
    """
    cmd = [claude_bin, "--bare", "-p"]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    # --max-turns: 单 attempt agentic turn 数硬闸，模型无关。
    if max_turns:
        cmd += ["--max-turns", str(max_turns)]
    cmd += [
        "--agent", agent_name,
        "--session-id", session_id,
        "--add-dir", str(workdir),
        "--allowedTools", allowed_tools,
        "--output-format", "json",
        prompt,
    ]
    return cmd


def spawn_diagnose_agent(
    action: Action,
    task_dir: Path,
    op_name: str,
    attempt: int,
    *,
    agent_name: str = _DEFAULT_AGENT,
    claude_bin: str = _DEFAULT_CLAUDE_BIN,
    model: Optional[str] = None,
    allowed_tools: str = _DEFAULT_ALLOWED_TOOLS,
    workdir: Optional[Path] = None,
    npu: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    max_turns: Optional[str] = None,
    effort: Optional[str] = None,
    _run: Callable = subprocess.run,
) -> dict:
    """把 diagnose_and_fix Action 翻译成一次 claude 调用，返回本轮 result dict。

    _run 注入点: 默认 subprocess.run，UT 用 fake 替换 (不真跑 claude)。
    result dict 进 events.jsonl 的 action_completed.result (success/claude_state/...)。
    """
    args = action.skill_args or {}
    failure_type = args.get("failure_type", "")
    wd = Path(workdir) if workdir is not None else Path(task_dir)
    session_id = str(uuid.uuid4())
    result_file = Path(task_dir) / f"_claude_result_attempt{attempt}.json"

    prompt = _build_prompt(Path(task_dir), op_name, failure_type, attempt, npu)
    effort = effort if effort is not None else os.environ.get(_EFFORT_ENV)
    if effort == "":
        effort = None
    cmd = _build_claude_cmd(
        prompt, claude_bin=claude_bin, model=model, agent_name=agent_name,
        session_id=session_id, workdir=wd, allowed_tools=allowed_tools,
        result_file=result_file,
        max_turns=max_turns, effort=effort)

    try:
        with result_file.open("w", encoding="utf-8") as out:
            _run(cmd, stdout=out, stderr=subprocess.PIPE, timeout=timeout_sec,
                 check=False, text=True)
    except subprocess.TimeoutExpired:
        return {"success": False, "claude_state": "timeout", "fatal": False,
                "error": f"claude 调用超时 (>{timeout_sec}s)", "session_id": session_id}
    except Exception as e:  # noqa: BLE001 — 拉起失败转结构化结果，不让 runner 裸死
        return {"success": False, "claude_state": "spawn_failed", "fatal": True,
                "error": f"拉起 claude 失败: {e}", "session_id": session_id}

    result = _classify_claude_result(result_file)
    result["session_id"] = session_id
    return result


def make_agent_callback(
    *,
    agent_name: str = _DEFAULT_AGENT,
    claude_bin: str = _DEFAULT_CLAUDE_BIN,
    model: Optional[str] = None,
    allowed_tools: str = _DEFAULT_ALLOWED_TOOLS,
    workdir: Optional[Path] = None,
    npu: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    max_turns: Optional[str] = None,
    effort: Optional[str] = None,
    _run: Callable = subprocess.run,
) -> Callable[[Action, Path, str, int], dict]:
    """造一个符合 runner AgentCallback 签名 (action, task_dir, op_name, attempt)->dict
    的闭包，把上述配置固化进去，供 run_debug_session(agent_callback=...) 注入。"""
    def _cb(action: Action, task_dir: Path, op_name: str, attempt: int) -> dict:
        return spawn_diagnose_agent(
            action, task_dir, op_name, attempt,
            agent_name=agent_name, claude_bin=claude_bin, model=model,
            allowed_tools=allowed_tools, workdir=workdir, npu=npu,
            timeout_sec=timeout_sec,
            max_turns=max_turns,
            effort=effort, _run=_run)
    return _cb
