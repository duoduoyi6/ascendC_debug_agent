#!/bin/bash
# 批量调度 Claude Code 跨多个 docker 容器 + 多 NPU 执行 AscendC debug。
#
# 输入：一组"主 agent 已产出的算子目录"（由 ascend-kernel-developer-anti-cheat
#       生成；每个目录应含 model.py / model_new_ascendc.py / kernel/ / trace.md /
#       <op_name>.json(.bak)）。
# 行为：为每个 task_dir 独立调用 agents/ascendc-debug-agent-discovery.md（通过
#       claude --bare -p --agent），按 ASCENDC_DEBUG_MAX_ATTEMPTS 上限迭代修复。
#
# 动态工作队列：容器间并行（每容器绑定一张 NPU），容器内串行；
# 谁先完成当前任务，就从共享队列拉下一个，不做预分配。
#
# 典型用法:
#   # 方式 A：显式传入逗号分隔的 task_dir 列表
#   bash utils/run_ascendc_debug_batch_cc.sh \
#        --task-dirs /home/c00959374/AscendOpGenAgent/outputs/run_20260422_1900/31_ELU,/home/c00959374/AscendOpGenAgent/outputs/run_20260422_1900/32_GELU \
#        --containers cjm_cann1,cjm_cann2 --npus 1,6 \
#        --output /home/c00959374/AscendOpGenAgent/outputs/debug_$(date +%Y%m%d_%H%M)
#
#   # 方式 B：从文件读，每行一个 task_dir
#   bash utils/run_ascendc_debug_batch_cc.sh \
#        --task-dirs-file debug_targets.txt \
#        --containers cjm_cann1,cjm_cann2,cjm_cann3 --npus 1,6,7 \
#        --output /home/c00959374/AscendOpGenAgent/outputs/debug_run_01 \
#        --max-attempts 7

set -euo pipefail

# ── 默认值 ──
TASK_DIRS=""
TASK_DIRS_FILE=""
CONTAINERS=""
NPUS=""
OUTPUT_DIR=""
MODEL=""
TIMEOUT_SEC="5400"          # 单任务超时（秒），默认 1.5 小时
MAX_ATTEMPTS="5"            # ASCENDC_DEBUG_MAX_ATTEMPTS 默认值
MAX_RESUMES="3"             # pause_turn 最大恢复次数
MAX_BUDGET_USD=""           # 单任务预算上限（空=不限）
STALE_AFTER_FAILURE_SEC="3600"   # 失败后停滞多久判定为 stale
STALE_CHECK_INTERVAL_SEC="60"    # 停滞检测间隔
WORKDIR_IN_CONTAINER="/home/c00959374/AscendOpGenAgent"
TILELANG_ENV_SH="/home/c00959374/tilelang/tilelang-ascend/set_env.sh"
CLAUDE_ENV_SH=""            # API 凭证脚本路径（可选）
CLAUDE_BIN="claude"
AGENT="ascendc-debug-agent-discovery"
ALLOWED_TOOLS="Bash,Read,Write,Edit,Glob,Grep,Skill"
ANTICHEAT_SCRIPT="skills/ascendc/ascendc-debug/scripts/anticheat.py"

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --task-dirs)              TASK_DIRS="$2"; shift 2 ;;
        --task-dirs-file)         TASK_DIRS_FILE="$2"; shift 2 ;;
        --containers)             CONTAINERS="$2"; shift 2 ;;
        --npus)                   NPUS="$2"; shift 2 ;;
        --output)                 OUTPUT_DIR="$2"; shift 2 ;;
        --model)                  MODEL="$2"; shift 2 ;;
        --timeout)                TIMEOUT_SEC="$2"; shift 2 ;;
        --max-attempts)           MAX_ATTEMPTS="$2"; shift 2 ;;
        --max-resumes)            MAX_RESUMES="$2"; shift 2 ;;
        --max-budget-usd)         MAX_BUDGET_USD="$2"; shift 2 ;;
        --stale-after-failure)    STALE_AFTER_FAILURE_SEC="$2"; shift 2 ;;
        --stale-check-interval)   STALE_CHECK_INTERVAL_SEC="$2"; shift 2 ;;
        --workdir)                WORKDIR_IN_CONTAINER="$2"; shift 2 ;;
        --tilelang-env)           TILELANG_ENV_SH="$2"; shift 2 ;;
        --claude-env)             CLAUDE_ENV_SH="$2"; shift 2 ;;
        --claude-bin)             CLAUDE_BIN="$2"; shift 2 ;;
        --agent)                  AGENT="$2"; shift 2 ;;
        --allowed-tools)          ALLOWED_TOOLS="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,30p' "$0"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 校验 ──
[[ -z "$TASK_DIRS" && -z "$TASK_DIRS_FILE" ]] && {
    echo "错误: 必须指定 --task-dirs 或 --task-dirs-file"; exit 1;
}
[[ -n "$TASK_DIRS" && -n "$TASK_DIRS_FILE" ]] && {
    echo "错误: --task-dirs 与 --task-dirs-file 互斥"; exit 1;
}
[[ -z "$CONTAINERS" ]] && { echo "错误: 必须 --containers (逗号分隔)"; exit 1; }
[[ -z "$NPUS" ]]       && { echo "错误: 必须 --npus (逗号分隔，与 containers 一一对应)"; exit 1; }
[[ -z "$OUTPUT_DIR" ]] && { echo "错误: 必须 --output"; exit 1; }
[[ "$MAX_ATTEMPTS" =~ ^[0-9]+$ ]] || { echo "错误: --max-attempts 必须是正整数"; exit 1; }
[[ "$MAX_RESUMES" =~ ^[0-9]+$ ]]  || { echo "错误: --max-resumes 必须是正整数"; exit 1; }

IFS=',' read -ra CONTAINER_ARR <<< "$CONTAINERS"
IFS=',' read -ra NPU_ARR <<< "$NPUS"
(( ${#CONTAINER_ARR[@]} == ${#NPU_ARR[@]} )) \
    || { echo "错误: containers 与 npus 数量不一致"; exit 1; }

# ── 构造 task_dir 列表 ──
TASK_LIST=()
if [[ -n "$TASK_DIRS_FILE" ]]; then
    [[ -f "$TASK_DIRS_FILE" ]] || { echo "错误: 不存在文件 $TASK_DIRS_FILE"; exit 1; }
    while IFS= read -r line; do
        line="${line%%#*}"                    # 去注释
        line="$(echo "$line" | xargs)"        # 去首尾空白
        [[ -z "$line" ]] && continue
        TASK_LIST+=("$line")
    done < "$TASK_DIRS_FILE"
else
    IFS=',' read -ra TASK_LIST <<< "$TASK_DIRS"
fi
(( ${#TASK_LIST[@]} > 0 )) || { echo "错误: 无可用 task_dir"; exit 1; }

# ── Prompt 模板 ──
# agent 的 system prompt 由 --agent 加载，这里只传参数和执行指令
PROMPT_TEMPLATE="debug __TASK_DIR__ npu=__NPU__

独立调用预授权（非交互，一次跑完）：
本脚本批量调度即视为用户对本 agent 的显式调用授权；遇到分支/决策按 spec 默认路径处理，不得停机等待确认。必填产物缺失或不可恢复错误时直接终止并写 {task_dir}/debug_trace.md + {task_dir}/debug_status.json（session_outcome=crashed + crash_reason=<原因>）。

attempt 上限 = __MAX_ATTEMPTS__。"

mkdir -p "$OUTPUT_DIR"
QUEUE="$OUTPUT_DIR/.queue"
LOCK="$OUTPUT_DIR/.lock"
REPORT="$OUTPUT_DIR/batch_report.md"
FATAL="$OUTPUT_DIR/.fatal"

# ── 初始化队列（按原始顺序写入） ──
: > "$QUEUE"
for td in "${TASK_LIST[@]}"; do
    echo "$td" >> "$QUEUE"
done
: > "$LOCK"
: > "$FATAL"

# ── 初始化报告 ──
{
    echo "# Claude Code AscendC Debug 批量执行报告"
    echo
    echo "- containers: $CONTAINERS"
    echo "- npus: $NPUS"
    echo "- task_dirs: ${#TASK_LIST[@]} 个"
    echo "- max_attempts: $MAX_ATTEMPTS"
    echo "- max_resumes: $MAX_RESUMES"
    echo "- agent: $AGENT"
    echo "- model: ${MODEL:-<env default>}"
    echo "- claude env: ${CLAUDE_ENV_SH:-<none>}"
    echo "- tilelang env: $TILELANG_ENV_SH"
    echo "- timeout: ${TIMEOUT_SEC}s/task"
    echo "- stale_after_failure: ${STALE_AFTER_FAILURE_SEC}s"
    echo "- start: $(date '+%F %T')"
    echo
    echo "| # | task_dir | session_outcome | 耗时(s) | 容器@NPU |"
    echo "|---|----------|-----------------|---------|----------|"
} > "$REPORT"

TOTAL=${#TASK_LIST[@]}
echo "================================================================"
echo "总 debug 任务数: $TOTAL    workers: ${#CONTAINER_ARR[@]}    timeout: ${TIMEOUT_SEC}s    max_attempts: $MAX_ATTEMPTS"
for i in "${!CONTAINER_ARR[@]}"; do
    echo "  worker[$i]: ${CONTAINER_ARR[$i]} → npu=${NPU_ARR[$i]}"
done
echo "================================================================"

# ══════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════

# 从 debug_status.json 读 session_outcome
read_debug_outcome() {
    local target_dir="$1"
    local status_file="$target_dir/debug_status.json"
    if [[ ! -f "$status_file" ]]; then
        echo "missing_debug_status"
        return 0
    fi
    python3 - "$status_file" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get("session_outcome", "unknown"))
except Exception:
    print("invalid_debug_status")
PY
}

# 从 debug_status.json 读 attempts_used
read_debug_attempts() {
    local target_dir="$1"
    local status_file="$target_dir/debug_status.json"
    python3 - "$status_file" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get("attempts_used", 0))
except Exception:
    print(0)
PY
}

# 读 Claude Code 输出的 JSON result 文件，判断是否有致命错误
read_fatal_claude_error() {
    local result_file="$1"
    if [[ ! -f "$result_file" ]]; then
        return 1
    fi
    python3 - "$result_file" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(1)

status = data.get("api_error_status")
parts = []
for key in ("error", "message", "result"):
    value = data.get(key)
    if value is not None:
        parts.append(str(value))
text = "\n".join(parts).lower()

fatal_status = status in (401, 402, 403)
fatal_markers = (
    "usage limit", "quota", "credit", "billing cycle",
    "permission_error", "failed to authenticate", "invalid api key",
)
if fatal_status or (data.get("is_error") and any(marker in text for marker in fatal_markers)):
    reason = f"api_error_status={status}"
    for marker in fatal_markers:
        if marker in text:
            reason += f" marker={marker}"
            break
    print(reason)
    raise SystemExit(0)

raise SystemExit(1)
PY
}

# 读 Claude result 判断是否 pause_turn
read_claude_state() {
    local result_file="$1"
    if [[ ! -f "$result_file" ]]; then
        echo "missing_claude_result"
        return 0
    fi
    python3 - "$result_file" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    print("invalid_claude_result")
    raise SystemExit(0)

if data.get("is_error"):
    print("claude_error")
elif data.get("api_error_status") is not None:
    print(f"api_error_{data.get('api_error_status')}")
elif data.get("stop_reason") == "pause_turn":
    print("claude_pause_turn")
elif data.get("terminal_reason") not in (None, "completed"):
    print(f"terminal_{data.get('terminal_reason')}")
else:
    print("ok")
PY
}

# 标记全局致命错误
mark_fatal_error() {
    local reason="$1" wlog="$2"
    exec 8>"$LOCK"
    flock -x 8
    if [[ ! -s "$FATAL" ]]; then
        echo "$reason" > "$FATAL"
    fi
    flock -u 8
    exec 8>&-
    echo "[fatal] stopping queue: $reason" >> "$wlog"
}

# 获取 task_dir 下关键文件的最新 mtime
latest_task_progress_mtime() {
    local target_dir="$1"
    python3 - "$target_dir" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
paths = []
for rel in [".verify_status", ".verify_logs", "kernel"]:
    path = root / rel
    if path.exists():
        paths.extend(p for p in path.rglob("*") if p.is_file())
for name in ["trace.md", "debug_trace.md", "debug_status.json",
             "model_new_ascendc.py", "model_new_tilelang.py"]:
    path = root / name
    if path.exists():
        paths.append(path)

latest = 0
for path in paths:
    try:
        latest = max(latest, int(path.stat().st_mtime))
    except OSError:
        pass
print(latest)
PY
}

# 检查 task_dir 下是否有活跃的子进程（编译/验证等）
has_active_task_subprocesses() {
    local target_dir="$1"
    python3 - "$target_dir" <<'PY'
import os, re, sys
target = sys.argv[1]
patterns = re.compile(
    r"(build_ascendc\.py|verification_ascendc|classify_verify_result\.py|"
    r"cmake|gmake|/make\b|c\+\+|cc1plus|ccec|ld\.lld|ascendc_pack_kernel)"
)
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read()
        cmd = raw.replace(b"\0", b" ").decode("utf-8", "replace")
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except Exception:
        continue
    if not cmd or "claude --bare" in cmd:
        continue
    if not patterns.search(cmd):
        continue
    if target in cmd or cwd == target or cwd.startswith(target + "/"):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

# 停滞检测：失败后文件长时间无变化且无活跃进程
should_stop_stale_after_failure() {
    local target_dir="$1" wlog="$2"
    [[ "$STALE_AFTER_FAILURE_SEC" -gt 0 ]] || return 1

    local failure_type
    failure_type=$(read_debug_outcome "$target_dir" 2>/dev/null) || return 1
    # 只在非成功、非进行中的状态触发
    [[ "$failure_type" != "success" && "$failure_type" != "missing_debug_status" ]] || return 1

    if has_active_task_subprocesses "$target_dir"; then
        return 1
    fi

    local latest_mtime now age
    latest_mtime=$(latest_task_progress_mtime "$target_dir")
    [[ "$latest_mtime" -gt 0 ]] || return 1
    now=$(date +%s)
    age=$((now - latest_mtime))

    if [[ "$age" -ge "$STALE_AFTER_FAILURE_SEC" ]]; then
        echo "[watchdog] stale_after_failure outcome=${failure_type} age=${age}s threshold=${STALE_AFTER_FAILURE_SEC}s" >> "$wlog"
        return 0
    fi
    return 1
}

# 清理 task_dir 相关的残留进程
cleanup_task_processes() {
    local container="$1" task_dir="$2" token="$3" wlog="$4"
    {
        echo "[cleanup] stopping leftover processes for $task_dir token=$token"
        docker exec "$container" bash -lc '
            set +e
            target="$1"
            token="$2"
            kill_by_pattern() {
                sig="$1"; pat="$2"
                [ -z "$pat" ] && return 0
                pgrep -f "$pat" 2>/dev/null | while read -r pid; do
                    [ -z "$pid" ] && continue
                    [ "$pid" = "$$" ] && continue
                    [ "$pid" = "$BASHPID" ] && continue
                    [ "$pid" = "$PPID" ] && continue
                    cmdline="$(tr "\0" " " < "/proc/$pid/cmdline" 2>/dev/null || true)"
                    case "$cmdline" in *pgrep*|*pkill*) continue ;; esac
                    kill "-$sig" "$pid" 2>/dev/null || true
                done
            }
            kill_by_cwd() {
                sig="$1"
                for proc in /proc/[0-9]*; do
                    pid="${proc##*/}"
                    [ "$pid" = "1" ] && continue
                    [ "$pid" = "$$" ] && continue
                    [ "$pid" = "$BASHPID" ] && continue
                    [ "$pid" = "$PPID" ] && continue
                    cwd="$(readlink "$proc/cwd" 2>/dev/null || true)"
                    case "$cwd" in "$target"|"$target"/*) kill "-$sig" "$pid" 2>/dev/null || true ;; esac
                done
            }
            kill_by_pattern TERM "$target"
            kill_by_pattern TERM "$token"
            kill_by_cwd TERM
            sleep 2
            kill_by_pattern KILL "$target"
            kill_by_pattern KILL "$token"
            kill_by_cwd KILL
        ' _ "$task_dir" "$token" || true
    } >> "$wlog" 2>&1
}

# ══════════════════════════════════════════════════════════════════
# 单次 Claude Code 调用（支持首次和 resume）
# ══════════════════════════════════════════════════════════════════
run_claude_turn() {
    local container="$1" npu="$2" session_id="$3" turn="$4" prompt="$5" result_file="$6" wlog="$7"
    local resume_args="--session-id $session_id"
    if [[ "$turn" -gt 0 ]]; then
        resume_args="--resume $session_id"
    fi

    {
        echo "[claude] turn=$turn result=$result_file args=$resume_args"
        echo "[claude] start=$(date '+%F %T')"
    } >> "$wlog"

    set +e
    timeout --signal=TERM --kill-after=30 "$TIMEOUT_SEC" \
        docker exec \
            -e "ASCEND_RT_VISIBLE_DEVICES=$npu" \
            -e "ASCENDC_DEBUG_MAX_ATTEMPTS=$MAX_ATTEMPTS" \
            -e "CLAUDE_PROMPT=$prompt" \
            "$container" bash -lc '
                set -e
                claude_env="$1"
                tilelang_env="$2"
                workdir="$3"
                requested_model="$4"
                turn="$5"
                session_id="$6"
                agent="$7"
                allowed_tools="$8"
                result_file="$9"
                claude_bin="${10}"
                max_budget_usd="${11}"

                [ -n "$claude_env" ] && [ -f "$claude_env" ] && source "$claude_env"
                [ -f "$tilelang_env" ] && source "$tilelang_env"
                cd "$workdir"

                model="$requested_model"
                [ -n "$model" ] || model="${ANTHROPIC_MODEL:-}"

                budget_args=()
                if [ -n "$max_budget_usd" ]; then
                    budget_args=(--max-budget-usd "$max_budget_usd")
                fi

                if [ "$turn" = "0" ]; then
                    "$claude_bin" --bare -p \
                        --model "$model" \
                        "${budget_args[@]}" \
                        --agent "$agent" \
                        --session-id "$session_id" \
                        --add-dir "$workdir" \
                        --allowedTools "$allowed_tools" \
                        --output-format json \
                        "$CLAUDE_PROMPT" \
                        > "$result_file"
                else
                    "$claude_bin" --bare -p \
                        --model "$model" \
                        "${budget_args[@]}" \
                        --resume "$session_id" \
                        --add-dir "$workdir" \
                        --allowedTools "$allowed_tools" \
                        --output-format json \
                        "$CLAUDE_PROMPT" \
                        > "$result_file"
                fi
            ' _ "$CLAUDE_ENV_SH" "$TILELANG_ENV_SH" "$WORKDIR_IN_CONTAINER" \
                "${MODEL:-}" "$turn" "$session_id" "$AGENT" "$ALLOWED_TOOLS" \
                "$result_file" "$CLAUDE_BIN" "$MAX_BUDGET_USD" >> "$wlog" 2>&1 &
    local cmd_pid=$!
    local turn_status=0
    local stale_stop=0

    # 停滞检测轮询
    while kill -0 "$cmd_pid" 2>/dev/null; do
        sleep "$STALE_CHECK_INTERVAL_SEC"
        if ! kill -0 "$cmd_pid" 2>/dev/null; then
            break
        fi
        if should_stop_stale_after_failure "$(dirname "$result_file")" "$wlog"; then
            stale_stop=1
            cleanup_task_processes "$container" "$(dirname "$result_file")" "$session_id" "$wlog"
            kill -TERM "$cmd_pid" 2>/dev/null || true
            sleep 2
            kill -KILL "$cmd_pid" 2>/dev/null || true
            break
        fi
    done

    wait "$cmd_pid"
    turn_status=$?
    if [[ "$stale_stop" -eq 1 ]]; then
        turn_status=86
    fi
    return "$turn_status"
}

# ══════════════════════════════════════════════════════════════════
# Worker：从队列拉 task_dir，跨 docker 执行 Claude Code debug
# ══════════════════════════════════════════════════════════════════
run_worker() {
    local container="$1" npu="$2"
    local wlog="$OUTPUT_DIR/worker_${container}_npu${npu}.log"
    : > "$wlog"

    while true; do
        # 全局熔断检查
        if [[ -s "$FATAL" ]]; then
            echo "[worker] stop: fatal $(cat "$FATAL")" >> "$wlog"
            break
        fi

        local task_dir=""
        # 原子出队
        exec 9>"$LOCK"
        flock -x 9
        if [[ -s "$QUEUE" ]]; then
            task_dir=$(head -n1 "$QUEUE")
            sed -i '1d' "$QUEUE"
        fi
        flock -u 9
        exec 9>&-

        [[ -z "$task_dir" ]] && break
        local op_name; op_name=$(basename "$task_dir")

        local prompt="${PROMPT_TEMPLATE//__TASK_DIR__/$task_dir}"
        prompt="${prompt//__NPU__/$npu}"
        prompt="${prompt//__MAX_ATTEMPTS__/$MAX_ATTEMPTS}"

        local session_id
        session_id=$(cat /proc/sys/kernel/random/uuid 2>/dev/null || uuidgen)

        local start end elapsed status
        start=$(date +%s)

        {
            echo "[task] op=$op_name task_dir=$task_dir session_id=$session_id"
            echo "[task] start=$(date '+%F %T')"
        } >> "$wlog"

        # ── 首次 + resume 循环 ──
        status=0
        local turn claude_state result_file
        for turn in $(seq 0 "$MAX_RESUMES"); do
            result_file="$task_dir/_claude_result_${turn}.json"

            set +e
            run_claude_turn "$container" "$npu" "$session_id" "$turn" "$prompt" "$result_file" "$wlog"
            status=$?
            set -e

            # 保存最新 result
            if [[ -f "$result_file" ]]; then
                cp "$result_file" "$task_dir/_claude_result.json"
                # 致命错误检测
                local fatal_reason
                if fatal_reason=$(read_fatal_claude_error "$result_file" 2>/dev/null); then
                    mark_fatal_error "$fatal_reason" "$wlog"
                fi
            fi

            # 超时或非零退出 → 不再 resume
            if [[ "$status" -eq 124 || "$status" -eq 137 || "$status" -eq 143 ]]; then
                break
            fi
            if [[ "$status" -ne 0 ]]; then
                break
            fi

            # 检查是否需要 resume
            claude_state=$(read_claude_state "$task_dir/_claude_result.json")
            if [[ "$claude_state" != "claude_pause_turn" ]]; then
                break
            fi
            if [[ "$turn" -lt "$MAX_RESUMES" ]]; then
                echo "[claude] turn=$turn ended with pause_turn; resuming session_id=$session_id" >> "$wlog"
                prompt="Continue the previous AscendC debug task until required outputs and verification are complete. Do not ask for confirmation."
            fi
        done

        end=$(date +%s); elapsed=$((end - start))

        # 超时后清理残留进程
        if [[ "$status" -eq 124 || "$status" -eq 137 || "$status" -eq 143 ]]; then
            cleanup_task_processes "$container" "$task_dir" "$session_id" "$wlog"
        fi

        # ── 跨分支自动重入（progressed_to_new_failure_type） ──
        local session_outcome attempts_used total_attempts_used reentry_count
        session_outcome=$(read_debug_outcome "$task_dir" 2>/dev/null || echo "unknown")
        attempts_used=$(read_debug_attempts "$task_dir" 2>/dev/null || echo "0")
        total_attempts_used=0
        reentry_count=0

        if [[ "$session_outcome" == "progressed_to_new_failure_type" ]]; then
            if [[ "$attempts_used" =~ ^[0-9]+$ ]]; then
                total_attempts_used=$attempts_used
            else
                total_attempts_used=1
            fi
        fi

        while [[ "$session_outcome" == "progressed_to_new_failure_type" ]]; do
            # 检查全局 attempt 配额
            if [[ $total_attempts_used -ge $MAX_ATTEMPTS ]]; then
                session_outcome="stopped_by_global_attempt_limit"
                break
            fi

            reentry_count=$((reentry_count + 1))
            local remaining=$((MAX_ATTEMPTS - total_attempts_used))
            echo "[${container}@npu${npu}] ↗ ${op_name} progressed_to_new_failure_type, reentry #${reentry_count} (total_attempts=${total_attempts_used}, remaining=${remaining})" >> "$wlog"

            # 新 session（failure type 变了，需要新的 agent session）
            session_id=$(cat /proc/sys/kernel/random/uuid 2>/dev/null || uuidgen)
            local reentry_prompt="Failure type changed. Continue debug with remaining attempts=${remaining}. task_dir=${task_dir} npu=${npu}

独立调用预授权（非交互，一次跑完）：
遇到分支/决策按 spec 默认路径处理，不得停机等待确认。attempt 上限 = ${remaining}。"

            result_file="$task_dir/_claude_result_reentry_${reentry_count}.json"

            set +e
            run_claude_turn "$container" "$npu" "$session_id" "0" "$reentry_prompt" "$result_file" "$wlog"
            local reentry_status=$?
            set -e

            # 保存最新 result
            if [[ -f "$result_file" ]]; then
                cp "$result_file" "$task_dir/_claude_result.json"
            fi

            # 重新读取 debug_status.json
            local new_attempts_used=0
            session_outcome=$(read_debug_outcome "$task_dir" 2>/dev/null || echo "unknown")
            new_attempts_used=$(read_debug_attempts "$task_dir" 2>/dev/null || echo "0")

            if [[ "$new_attempts_used" =~ ^[0-9]+$ ]]; then
                total_attempts_used=$((total_attempts_used + new_attempts_used))
            else
                total_attempts_used=$((total_attempts_used + 1))
            fi

            local reentry_end=$(date +%s)
            elapsed=$((elapsed + reentry_end - end))
            end=$reentry_end
            status=$reentry_status
        done

        # ── 反作弊后置检测 ──
        local cheat_json cheat_verdict cheat_reasons cheat_mark
        cheat_json=$(docker exec "$container" bash -lc "
            cd '$WORKDIR_IN_CONTAINER'
            python3 '$ANTICHEAT_SCRIPT' verify '$task_dir' --json 2>/dev/null
        " 2>/dev/null || true)
        cheat_verdict=$(echo "$cheat_json" | python3 -c "
import sys, json
try:
    print(json.loads(sys.stdin.read()).get('verdict', 'UNKNOWN'))
except Exception:
    print('UNKNOWN')
" 2>/dev/null || echo "UNKNOWN")
        cheat_reasons=$(echo "$cheat_json" | python3 -c "
import sys, json
try:
    print(';'.join(json.loads(sys.stdin.read()).get('reasons', [])))
except Exception:
    print('')
" 2>/dev/null || echo "")
        [[ -n "$cheat_json" ]] && echo "$cheat_json" > "$task_dir/_anticheat.json"

        cheat_mark=""
        if [[ "$cheat_verdict" == "CHEAT" ]]; then
            cheat_mark=" / 🚨 CHEAT"
            echo "[${container}@npu${npu}] 🚨 ${op_name} CHEAT: $cheat_reasons"
        fi

        # ── 判定结果并写报告 ──
        local icon
        # 重新读取最终 session_outcome（可能被重入循环更新过）
        session_outcome=$(read_debug_outcome "$task_dir" 2>/dev/null || echo "unknown")

        if [[ $status -eq 0 ]]; then
            case "$session_outcome" in
                success)                            icon="✅ $session_outcome${cheat_mark}" ;;
                stopped_by_global_attempt_limit)    icon="⛔ $session_outcome${cheat_mark}" ;;
                skipped_*)                          icon="⊘ $session_outcome${cheat_mark}" ;;
                failed|stopped_*|crashed|timeout)   icon="❌ $session_outcome${cheat_mark}" ;;
                *)                                  icon="⚠ $session_outcome${cheat_mark}" ;;
            esac
            echo "[${container}@npu${npu}] ✅ ${op_name} session_outcome=${session_outcome} (${elapsed}s, reentries=${reentry_count})"
        elif [[ "$status" -eq 124 || "$status" -eq 137 || "$status" -eq 143 ]]; then
            icon="⏱ 超时(claude)${cheat_mark}"
            echo "[${container}@npu${npu}] ⏱ ${op_name} CLAUDE_TIMEOUT (${elapsed}s)"
        elif [[ "$status" -eq 86 ]]; then
            icon="🧊 stale_after_failure${cheat_mark}"
            echo "[${container}@npu${npu}] 🧊 ${op_name} STALE (${elapsed}s)"
        else
            icon="❌ claude_rc=$status${cheat_mark}"
            echo "[${container}@npu${npu}] ❌ ${op_name} claude_rc=$status (${elapsed}s)"
        fi

        local idx row
        exec 9>"$LOCK"; flock -x 9
        idx=$(grep -c '^| [0-9]' "$REPORT" 2>/dev/null || echo 0)
        idx=$((idx + 1))
        row="| $idx | $op_name | $icon | $elapsed | ${container}@npu${npu} |"
        echo "$row" >> "$REPORT"
        # 增量生成汇总报告（若工具存在）
        GEN_REPORT="$(dirname "$0")/generate_report_dynamic.py"
        if [[ -f "$GEN_REPORT" ]]; then
            python3 "$GEN_REPORT" -i "$OUTPUT_DIR" -o "$OUTPUT_DIR/final_batch_report.md" >>"$OUTPUT_DIR/report_gen.log" 2>&1 || true
        fi
        flock -u 9; exec 9>&-
    done
}

# ══════════════════════════════════════════════════════════════════
# 并行启动 workers
# ══════════════════════════════════════════════════════════════════
pids=()
for i in "${!CONTAINER_ARR[@]}"; do
    run_worker "${CONTAINER_ARR[$i]}" "${NPU_ARR[$i]}" &
    pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || true; done

# ── 汇总 ──
SUCCESS=$(grep -c "✅ success" "$REPORT" || echo 0)
STOPPED_GLOBAL=$(grep -c "⛔ stopped_by_global_attempt_limit" "$REPORT" || echo 0)
SKIPPED=$(grep -c "⊘ skipped" "$REPORT" || echo 0)
TIMEOUT_CNT=$(grep -c "⏱ 超时" "$REPORT" || echo 0)
STALE_CNT=$(grep -c "🧊 stale" "$REPORT" || echo 0)
FAIL=$(grep -c "❌ " "$REPORT" || echo 0)
CHEAT=$(grep -c "🚨 CHEAT" "$REPORT" || echo 0)

{
    echo
    echo "## 汇总"
    echo
    echo "- 总数: $TOTAL"
    echo "- success: $SUCCESS"
    echo "- stopped_by_global_attempt_limit: $STOPPED_GLOBAL"
    echo "- skipped_*: $SKIPPED"
    echo "- claude timeout: $TIMEOUT_CNT"
    echo "- stale_after_failure: $STALE_CNT"
    echo "- failed / crashed / stopped_* / claude_rc!=0: $FAIL"
    echo "- 作弊 (🚨 CHEAT, 与 outcome 正交): $CHEAT"
    if [[ -s "$FATAL" ]]; then
        echo "- 全局熔断: $(cat "$FATAL")"
    fi
    echo "- 结束: $(date '+%F %T')"
} >> "$REPORT"

echo "================================================================"
echo "完成: SUCCESS=$SUCCESS TIMEOUT=$TIMEOUT_CNT STALE=$STALE_CNT FAILED=$FAIL CHEAT=$CHEAT / 共 $TOTAL"
if [[ -s "$FATAL" ]]; then
    echo "全局熔断: $(cat "$FATAL")"
fi
echo "报告: $REPORT"
echo "每 worker 日志: $OUTPUT_DIR/worker_<container>_npu<N>.log"
echo "================================================================"
