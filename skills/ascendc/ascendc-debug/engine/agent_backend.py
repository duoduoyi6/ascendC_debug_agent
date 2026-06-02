"""agent_backend.py — diagnose_and_fix 的真实执行体 (方案 C 的唯一 spawn 点)。

引擎 next_action 里 forensics/validate 是确定性 py_action (runner 直接跑 precision_gate.py)，
**唯一需要拉起 agent 的是 diagnose_and_fix**。本模块把该 Action 翻译成一次 `claude --bare -p`
调用 (命令形态对齐 utils/run_ascendc_debug_batch_cc.sh::run_claude_turn)，做「单个 attempt 的
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
import subprocess
import uuid
from pathlib import Path
from typing import Callable, Optional

from engine.types import Action

# cc 脚本默认值 (utils/run_ascendc_debug_batch_cc.sh L44-47)，保持一致。
_DEFAULT_CLAUDE_BIN = "claude"
_DEFAULT_ALLOWED_TOOLS = "Bash,Read,Write,Edit,Glob,Grep,Skill"
_DEFAULT_AGENT = "ascendc-debug-agent-constructive"  # AAAI 主力 (cc 默认是 discovery，这里默认 constructive)

# 单轮约束覆盖层: 方案 C 下 agent 只做一个 attempt，编排归引擎。
# 不改 agent spec 本体 (留 Step 7)，在 prompt 末尾追加此约束覆盖自驱散文。
_SINGLE_ROUND_CONSTRAINT = """

═══════════════════════════════════════════════════════════════
【本次调用 = 单个 attempt，由确定性引擎驱动，务必遵守以下覆盖约束】
═══════════════════════════════════════════════════════════════
本次调用只负责**一个 attempt 的诊断 + 修复**，循环与终态由外层引擎掌控:
- 禁止自己跑 MAX_ATTEMPTS 循环；禁止读 Gate loop_signal 自行决定继续/停止/跳 Step5/Step6。
- 禁止写 debug_status.json / debug_trace.md (退出产物由引擎从事件流确定性重建)。
- 禁止调用 Gate (forensics/validate)；引擎会在你改完后自己跑 Gate 客观判定本轮效果。
本次只做: 按当前 failure_type 走对应分支方法论 (精度走 Phase A→B→C 等) → 诊断根因 →
最小化修改 {task_dir}/kernel/ 下文件 → 完成即停 (不要追加任何收尾动作)。
SKILL.md 的分支方法论散文仍须遵循 (Read 取方法论)，仅「编排/循环/退出」段落被本约束覆盖。
═══════════════════════════════════════════════════════════════
"""

# PLACEHOLDER_REST


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
    return head + _SINGLE_ROUND_CONSTRAINT.replace("{task_dir}", str(task_dir))


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
    is_error = bool(data.get("is_error"))
    # api_error_status 更具体且是 fatal 判定依据 (对齐 cc 脚本 read_fatal_claude_error)，
    # 优先于泛化的 is_error。真实 claude API 错误两者常同时出现。
    if api_status is not None:
        state = f"api_error_{api_status}"
    elif is_error:
        state = "claude_error"
    elif stop_reason == "pause_turn":
        state = "claude_pause_turn"  # 本轮未自然结束 (引擎可据此判断重试/收尾)
    else:
        state = "ok"
    return {
        "success": state == "ok",
        "claude_state": state,
        "stop_reason": stop_reason,
        "api_error_status": api_status,
        "error": None if state == "ok" else f"claude_state={state}",
    }


def _build_claude_cmd(prompt: str, *, claude_bin: str, model: Optional[str],
                      agent_name: str, session_id: str, workdir: Path,
                      allowed_tools: str, result_file: Path,
                      max_budget_usd: Optional[str]) -> list:
    """构造 `claude --bare -p` 命令 (命令形态对齐 cc 脚本 run_claude_turn 的 turn=0 分支)。

    输出写 result_file (--output-format json)；调用方负责把 stdout 重定向到该文件。
    """
    cmd = [claude_bin, "--bare", "-p"]
    if model:
        cmd += ["--model", model]
    if max_budget_usd:
        cmd += ["--max-budget-usd", str(max_budget_usd)]
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
    max_budget_usd: Optional[str] = None,
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
    cmd = _build_claude_cmd(
        prompt, claude_bin=claude_bin, model=model, agent_name=agent_name,
        session_id=session_id, workdir=wd, allowed_tools=allowed_tools,
        result_file=result_file, max_budget_usd=max_budget_usd)

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
    max_budget_usd: Optional[str] = None,
    _run: Callable = subprocess.run,
) -> Callable[[Action, Path, str, int], dict]:
    """造一个符合 runner AgentCallback 签名 (action, task_dir, op_name, attempt)->dict
    的闭包，把上述配置固化进去，供 run_debug_session(agent_callback=...) 注入。"""
    def _cb(action: Action, task_dir: Path, op_name: str, attempt: int) -> dict:
        return spawn_diagnose_agent(
            action, task_dir, op_name, attempt,
            agent_name=agent_name, claude_bin=claude_bin, model=model,
            allowed_tools=allowed_tools, workdir=workdir, npu=npu,
            timeout_sec=timeout_sec, max_budget_usd=max_budget_usd, _run=_run)
    return _cb


