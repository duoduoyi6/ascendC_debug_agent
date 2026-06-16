#!/bin/bash
# Batch-run Claude Code across docker containers + NPUs for AscendC generation.

set -euo pipefail

BENCHMARK_DIR=""
LEVEL=""
RANGE=""
IDS=""
CONTAINERS=""
NPUS=""
OUTPUT_DIR=""
MODEL=""
TIMEOUT_SEC="7200"
MAX_RESUMES="3"
MAX_BUDGET_USD=""
STALE_AFTER_FAILURE_SEC="3600"
STALE_CHECK_INTERVAL_SEC="60"
WORKDIR_IN_CONTAINER="/home/wsx/AscendOpGenAgent"
TILELANG_ENV_SH="/home/wsx/tilelang-ascend/set_env.sh"
CLAUDE_ENV_SH="/home/wsx/minimax_claude_env.sh"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
AGENT="ascend-kernel-developer-anti-cheat"
ALLOWED_TOOLS="Bash,Read,Write,Edit,MultiEdit,Glob,Grep,Task"
PROMPT_TEMPLATE='生成ascendC算子，npu=__NPU__，op_file=__FILE__，output_dir=__TARGET__/'
ANTICHEAT_SCRIPT="skills/ascendc/ascendc-debug/scripts/anticheat.py"

while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark-dir) BENCHMARK_DIR="$2"; shift 2 ;;
        --level)         LEVEL="$2"; shift 2 ;;
        --range)         RANGE="$2"; shift 2 ;;
        --ids)           IDS="$2"; shift 2 ;;
        --containers)    CONTAINERS="$2"; shift 2 ;;
        --npus)          NPUS="$2"; shift 2 ;;
        --output)        OUTPUT_DIR="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --timeout)       TIMEOUT_SEC="$2"; shift 2 ;;
        --max-resumes)   MAX_RESUMES="$2"; shift 2 ;;
        --max-budget-usd) MAX_BUDGET_USD="$2"; shift 2 ;;
        --stale-after-failure) STALE_AFTER_FAILURE_SEC="$2"; shift 2 ;;
        --stale-check-interval) STALE_CHECK_INTERVAL_SEC="$2"; shift 2 ;;
        --workdir)       WORKDIR_IN_CONTAINER="$2"; shift 2 ;;
        --tilelang-env)  TILELANG_ENV_SH="$2"; shift 2 ;;
        --claude-env)    CLAUDE_ENV_SH="$2"; shift 2 ;;
        --agent)         AGENT="$2"; shift 2 ;;
        --allowed-tools) ALLOWED_TOOLS="$2"; shift 2 ;;
        --prompt)        PROMPT_TEMPLATE="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,40p' "$0"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

[[ -z "$BENCHMARK_DIR" ]] && { echo "错误: 必须 --benchmark-dir"; exit 1; }
[[ -z "$LEVEL" ]] && { echo "错误: 必须 --level"; exit 1; }
[[ -z "$RANGE" && -z "$IDS" ]] && { echo "错误: 必须 --range 或 --ids"; exit 1; }
[[ -z "$CONTAINERS" ]] && { echo "错误: 必须 --containers"; exit 1; }
[[ -z "$NPUS" ]] && { echo "错误: 必须 --npus"; exit 1; }
[[ -z "$OUTPUT_DIR" ]] && { echo "错误: 必须 --output"; exit 1; }

LEVEL_DIR="${BENCHMARK_DIR}/level${LEVEL}"
[[ -d "$LEVEL_DIR" ]] || { echo "错误: 不存在 $LEVEL_DIR"; exit 1; }

IFS=',' read -ra CONTAINER_ARR <<< "$CONTAINERS"
IFS=',' read -ra NPU_ARR <<< "$NPUS"
(( ${#CONTAINER_ARR[@]} == ${#NPU_ARR[@]} )) || { echo "错误: containers 与 npus 数量不一致"; exit 1; }

OP_IDS=()
if [[ -n "$RANGE" ]]; then
    S=$(echo "$RANGE" | cut -d'-' -f1)
    E=$(echo "$RANGE" | cut -d'-' -f2)
    for i in $(seq "$S" "$E"); do OP_IDS+=("$i"); done
else
    IFS=',' read -ra OP_IDS <<< "$IDS"
fi

declare -A OP_FILES
for id in "${OP_IDS[@]}"; do
    matched=$(find "$LEVEL_DIR" -maxdepth 1 -name "${id}_*.py" -type f 2>/dev/null | head -1)
    if [[ -n "$matched" ]]; then
        OP_FILES[$id]="$matched"
    else
        echo "警告: 未找到算子 ${id}，跳过"
    fi
done
(( ${#OP_FILES[@]} > 0 )) || { echo "错误: 无可用算子"; exit 1; }

mkdir -p "$OUTPUT_DIR"
QUEUE="$OUTPUT_DIR/.queue"
LOCK="$OUTPUT_DIR/.lock"
REPORT="$OUTPUT_DIR/batch_report.md"
FATAL="$OUTPUT_DIR/.fatal"

: > "$QUEUE"
for id in "${OP_IDS[@]}"; do
    [[ -n "${OP_FILES[$id]+set}" ]] && echo "$id" >> "$QUEUE"
done
: > "$LOCK"
: > "$FATAL"

{
    echo "# Claude Code 批量执行报告"
    echo
    echo "- benchmark: $BENCHMARK_DIR"
    echo "- level: $LEVEL"
    echo "- containers: $CONTAINERS"
    echo "- npus: $NPUS"
    echo "- tasks: ${OP_IDS[*]}"
    echo "- model: ${MODEL:-<env default>}"
    echo "- agent: $AGENT"
    echo "- claude env: $CLAUDE_ENV_SH"
    echo "- tilelang env: $TILELANG_ENV_SH"
    echo "- timeout: ${TIMEOUT_SEC}s/task"
    echo "- max resumes: $MAX_RESUMES"
    echo "- max budget usd: ${MAX_BUDGET_USD:-<none>}"
    echo "- stale after failure: ${STALE_AFTER_FAILURE_SEC}s"
    echo "- stale check interval: ${STALE_CHECK_INTERVAL_SEC}s"
    echo "- start: $(date '+%F %T')"
    echo
    echo "| id | file | 状态 | 耗时(s) | 容器@NPU |"
    echo "|----|------|------|---------|----------|"
} > "$REPORT"

TOTAL=${#OP_FILES[@]}
echo "================================================================"
echo "总任务数: $TOTAL    workers: ${#CONTAINER_ARR[@]}    timeout: ${TIMEOUT_SEC}s"
for i in "${!CONTAINER_ARR[@]}"; do
    echo "  worker[$i]: ${CONTAINER_ARR[$i]} -> npu=${NPU_ARR[$i]}"
done
echo "================================================================"

count_report_rows() {
    local pattern="$1"
    grep -c "$pattern" "$REPORT" 2>/dev/null || true
}

is_timeout_status() {
    local status="$1"
    [[ "$status" -eq 124 || "$status" -eq 137 || "$status" -eq 143 ]]
}

is_stale_status() {
    local status="$1"
    [[ "$status" -eq 86 ]]
}

cleanup_task_processes() {
    local container="$1" task_dir="$2" token="$3" wlog="$4"
    {
        echo "[cleanup] stopping leftover processes for $task_dir token=$token"
        docker exec "$container" bash -lc '
            set +e
            target="$1"
            token="$2"

            kill_by_pattern() {
                sig="$1"
                pat="$2"
                [ -z "$pat" ] && return 0
                pgrep -f "$pat" 2>/dev/null | while read -r pid; do
                    [ -z "$pid" ] && continue
                    [ "$pid" = "$$" ] && continue
                    [ "$pid" = "$BASHPID" ] && continue
                    [ "$pid" = "$PPID" ] && continue
                    cmdline="$(tr "\0" " " < "/proc/$pid/cmdline" 2>/dev/null || true)"
                    case "$cmdline" in
                        *pgrep*|*pkill*) continue ;;
                    esac
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
                    case "$cwd" in
                        "$target"|"$target"/*) kill "-$sig" "$pid" 2>/dev/null || true ;;
                    esac
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

read_non_success_verify_failure_type() {
    local target_dir="$1"
    local status_file="$target_dir/.verify_status/latest.json"
    if [[ ! -f "$status_file" ]]; then
        return 1
    fi
    python3 - "$status_file" <<'PY'
import json
import sys

try:
    data = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(1)

failure_type = data.get("failure_type") or data.get("status") or "unknown"
if failure_type == "success":
    raise SystemExit(1)
print(failure_type)
PY
}

latest_task_progress_mtime() {
    local target_dir="$1"
    python3 - "$target_dir" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
paths = []
for rel in [
    ".verify_status",
    ".verify_logs",
    "kernel",
]:
    path = root / rel
    if path.exists():
        paths.extend(p for p in path.rglob("*") if p.is_file())

for name in [
    "trace.md",
    "debug_trace.md",
    "debug_status.json",
    "model_new_ascendc.py",
    "model_new_tilelang.py",
]:
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

has_active_task_subprocesses() {
    local target_dir="$1"
    python3 - "$target_dir" <<'PY'
import os
import re
import sys

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

should_stop_stale_after_failure() {
    local target_dir="$1" wlog="$2"
    [[ "$STALE_AFTER_FAILURE_SEC" -gt 0 ]] || return 1

    local failure_type
    failure_type=$(read_non_success_verify_failure_type "$target_dir" 2>/dev/null) || return 1

    if has_active_task_subprocesses "$target_dir"; then
        return 1
    fi

    local latest_mtime now age
    latest_mtime=$(latest_task_progress_mtime "$target_dir")
    [[ "$latest_mtime" -gt 0 ]] || return 1
    now=$(date +%s)
    age=$((now - latest_mtime))

    if [[ "$age" -ge "$STALE_AFTER_FAILURE_SEC" ]]; then
        echo "[watchdog] stale_after_failure failure=${failure_type} age=${age}s threshold=${STALE_AFTER_FAILURE_SEC}s" >> "$wlog"
        return 0
    fi
    return 1
}

has_required_outputs() {
    local target_dir="$1"
    [[ -f "$target_dir/model_new_ascendc.py" ]] || return 1
    [[ -f "$target_dir/trace.md" ]] || return 1
    [[ -d "$target_dir/kernel" ]] || return 1
    find "$target_dir/kernel" -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.so' \) -print -quit | grep -q .
}

read_verify_failure_type() {
    local target_dir="$1"
    local status_file="$target_dir/.verify_status/latest.json"
    if [[ ! -f "$status_file" ]]; then
        echo "missing_verify_status"
        return 0
    fi
    python3 - "$status_file" <<'PY'
import json
import sys

try:
    data = json.load(open(sys.argv[1]))
    print(data.get("failure_type") or data.get("status") or "unknown")
except Exception:
    print("invalid_verify_status")
PY
}

read_claude_error_state() {
    local result_file="$1"
    if [[ ! -f "$result_file" ]]; then
        echo "missing_claude_result"
        return 0
    fi
    python3 - "$result_file" <<'PY'
import json
import sys

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

read_fatal_claude_error() {
    local result_file="$1"
    if [[ ! -f "$result_file" ]]; then
        return 1
    fi
    python3 - "$result_file" <<'PY'
import json
import sys

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
    "usage limit",
    "quota",
    "credit",
    "billing cycle",
    "permission_error",
    "failed to authenticate",
    "invalid api key",
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

                source "$claude_env"
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
            ' _ "$CLAUDE_ENV_SH" "$TILELANG_ENV_SH" "$WORKDIR_IN_CONTAINER" "${MODEL:-}" "$turn" "$session_id" "$AGENT" "$ALLOWED_TOOLS" "$result_file" "$CLAUDE_BIN" "$MAX_BUDGET_USD" >> "$wlog" 2>&1 &
    local cmd_pid=$!
    local turn_status=0
    local stale_stop=0

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

run_worker() {
    local container="$1" npu="$2"
    local wlog="$OUTPUT_DIR/worker_${container}_npu${npu}.log"
    : > "$wlog"

    while true; do
        if [[ -s "$FATAL" ]]; then
            echo "[worker] stop: fatal $(cat "$FATAL")" >> "$wlog"
            break
        fi

        local id=""
        exec 9>"$LOCK"
        flock -x 9
        if [[ -s "$QUEUE" ]]; then
            id=$(head -n1 "$QUEUE")
            sed -i '1d' "$QUEUE"
        fi
        flock -u 9
        exec 9>&-

        [[ -z "$id" ]] && break
        local file="${OP_FILES[$id]}"
        local filename; filename=$(basename "$file")
        local op_name="${filename%.*}"
        local target_dir="$OUTPUT_DIR/$op_name"
        mkdir -p "$target_dir"

        local prompt="${PROMPT_TEMPLATE//__NPU__/$npu}"
        prompt="${prompt//__FILE__/$file}"
        prompt="${prompt//__TARGET__/$target_dir}"

        local session_id
        session_id=$(cat /proc/sys/kernel/random/uuid 2>/dev/null || uuidgen)

        local start end elapsed status
        start=$(date +%s)

        {
            echo "[task] id=$id file=$filename output_dir=$target_dir session_id=$session_id"
            echo "[task] start=$(date '+%F %T')"
        } >> "$wlog"

        status=0
        local turn claude_state result_file
        for turn in $(seq 0 "$MAX_RESUMES"); do
            result_file="$target_dir/_claude_result_${turn}.json"
            if [[ "$turn" -eq 0 ]]; then
                set +e
                run_claude_turn "$container" "$npu" "$session_id" "$turn" "$prompt" "$result_file" "$wlog"
                status=$?
                set -e
            else
                set +e
                run_claude_turn "$container" "$npu" "$session_id" "$turn" "Continue the previous AscendC generation task until required outputs and verification are complete. Do not ask for confirmation." "$result_file" "$wlog"
                status=$?
                set -e
            fi

            if [[ -f "$result_file" ]]; then
                cp "$result_file" "$target_dir/_claude_result.json"
                local fatal_reason
                if fatal_reason=$(read_fatal_claude_error "$result_file" 2>/dev/null); then
                    mark_fatal_error "$fatal_reason" "$wlog"
                fi
            fi

            if is_timeout_status "$status"; then
                break
            fi
            if [[ "$status" -ne 0 ]]; then
                break
            fi

            claude_state=$(read_claude_error_state "$target_dir/_claude_result.json")
            if [[ "$claude_state" != "claude_pause_turn" ]]; then
                break
            fi
            if [[ "$turn" -lt "$MAX_RESUMES" ]]; then
                echo "[claude] turn=$turn ended with pause_turn; resuming session_id=$session_id" >> "$wlog"
            fi
        done

        end=$(date +%s); elapsed=$((end - start))
        if is_timeout_status "$status"; then
            cleanup_task_processes "$container" "$target_dir" "$session_id" "$wlog"
        fi

        local cheat_json cheat_verdict cheat_reasons cheat_mark
        cheat_json=$(docker exec "$container" bash -lc "
            cd '$WORKDIR_IN_CONTAINER'
            python3 '$ANTICHEAT_SCRIPT' verify '$target_dir' --json 2>/dev/null
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
        [[ -n "$cheat_json" ]] && echo "$cheat_json" > "$target_dir/_anticheat.json"

        cheat_mark=""
        if [[ "$cheat_verdict" == "CHEAT" ]]; then
            cheat_mark=" / CHEAT"
            echo "[${container}@npu${npu}] CHEAT id=$id ${filename}: $cheat_reasons"
        fi

        local verify_failure output_state row icon has_outputs
        verify_failure=$(read_verify_failure_type "$target_dir")
        claude_state=$(read_claude_error_state "$target_dir/_claude_result.json")
        if has_required_outputs "$target_dir"; then
            has_outputs=1
            output_state="outputs=present"
        else
            has_outputs=0
            output_state="outputs=missing"
        fi

        if [[ "$cheat_verdict" != "CHEAT" && "$has_outputs" -eq 1 && "$verify_failure" == "success" && "$claude_state" == "ok" ]]; then
            if is_timeout_status "$status"; then
                icon="SUCCESS(timeout_after_verify)"
                echo "[${container}@npu${npu}] SUCCESS id=$id ${filename} verify=success after timeout race (${elapsed}s)"
            else
                icon="SUCCESS"
                echo "[${container}@npu${npu}] SUCCESS id=$id ${filename} verify=success (${elapsed}s)"
            fi
        elif is_timeout_status "$status"; then
            icon="TIMEOUT${cheat_mark}"
            echo "[${container}@npu${npu}] TIMEOUT id=$id ${filename} (${elapsed}s)"
        elif is_stale_status "$status"; then
            icon="FAILED(stale_after_failure, verify=${verify_failure}, ${output_state})${cheat_mark}"
            echo "[${container}@npu${npu}] FAILED id=$id ${filename} stale_after_failure verify=${verify_failure} ${output_state}${cheat_mark} (${elapsed}s)"
        elif [[ "$cheat_verdict" == "CHEAT" ]]; then
            icon="FAILED(cheat, verify=${verify_failure}, ${output_state})${cheat_mark}"
            echo "[${container}@npu${npu}] FAILED id=$id ${filename} CHEAT verify=${verify_failure} ${output_state} (${elapsed}s)"
        elif [[ $status -ne 0 ]]; then
            icon="FAILED(rc=$status, claude=${claude_state}, verify=${verify_failure}, ${output_state})"
            echo "[${container}@npu${npu}] FAILED id=$id ${filename} rc=$status claude=${claude_state} verify=${verify_failure} ${output_state} (${elapsed}s)"
        elif [[ "$claude_state" != "ok" ]]; then
            icon="FAILED(claude=${claude_state}, verify=${verify_failure}, ${output_state})"
            echo "[${container}@npu${npu}] FAILED id=$id ${filename} claude=${claude_state} verify=${verify_failure} ${output_state} (${elapsed}s)"
        elif [[ "$has_outputs" -ne 1 ]]; then
            icon="FAILED(missing_outputs, verify=${verify_failure})"
            echo "[${container}@npu${npu}] FAILED id=$id ${filename} missing required outputs verify=${verify_failure} (${elapsed}s)"
        elif [[ "$verify_failure" == "success" ]]; then
            icon="SUCCESS(partial_outputs)"
            echo "[${container}@npu${npu}] SUCCESS id=$id ${filename} verify=success but ${output_state} (${elapsed}s)"
        else
            icon="FAILED(verify=${verify_failure}, ${output_state})"
            echo "[${container}@npu${npu}] FAILED id=$id ${filename} verify=${verify_failure} ${output_state} (${elapsed}s)"
        fi
        row="| $id | $filename | $icon | $elapsed | ${container}@npu${npu} |"

        exec 9>"$LOCK"; flock -x 9
        echo "$row" >> "$REPORT"
        GEN_REPORT="$(dirname "$0")/generate_report_dynamic.py"
        if [[ -f "$GEN_REPORT" ]]; then
            python3 "$GEN_REPORT" -i "$OUTPUT_DIR" -o "$OUTPUT_DIR/final_batch_report.md" >>"$OUTPUT_DIR/report_gen.log" 2>&1 || true
        fi
        flock -u 9; exec 9>&-
    done
}

pids=()
for i in "${!CONTAINER_ARR[@]}"; do
    run_worker "${CONTAINER_ARR[$i]}" "${NPU_ARR[$i]}" &
    pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || true; done

SUCCESS=$(count_report_rows "SUCCESS")
TIMEOUT_CNT=$(count_report_rows "TIMEOUT")
FAIL=$(count_report_rows "FAILED")
CHEAT=$(count_report_rows "CHEAT")

{
    echo
    echo "## 汇总"
    echo
    echo "- 总数: $TOTAL"
    echo "- 成功: $SUCCESS"
    echo "- 超时: $TIMEOUT_CNT"
    echo "- 失败: $FAIL"
    echo "- 作弊 (CHEAT, 与成功/失败正交，不重跑): $CHEAT"
    if [[ -s "$FATAL" ]]; then
        echo "- 全局熔断: $(cat "$FATAL")"
    fi
    echo "- 结束: $(date '+%F %T')"
} >> "$REPORT"

echo "================================================================"
echo "完成: SUCCESS=$SUCCESS TIMEOUT=$TIMEOUT_CNT FAILED=$FAIL CHEAT=$CHEAT / 共 $TOTAL"
if [[ -s "$FATAL" ]]; then
    echo "全局熔断: $(cat "$FATAL")"
fi
echo "报告: $REPORT"
echo "每 worker 日志: $OUTPUT_DIR/worker_<container>_npu<N>.log"
echo "================================================================"

GEN="$(dirname "$0")/generate_report_dynamic.py"
if [[ -f "$GEN" ]]; then
    echo "正在调用 generate_report_dynamic.py..."
    python3 "$GEN" -i "$OUTPUT_DIR" -o "$OUTPUT_DIR/final_batch_report.md" || true
fi
