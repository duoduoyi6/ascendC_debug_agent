#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/home/wsx/AscendOpGenAgent}"
SOURCE_MAP="$WORKDIR/outputs/npukernelbench_full_xiaomi_kimi_20260506_archive/selected_cases_source_map.csv"
OUTPUT_DIR=""
CONTAINER_LABEL="${CONTAINER_LABEL:-wsx_cann}"
NPUS="4,5,6,7"
TIMEOUT_SEC="7200"
MAX_ATTEMPTS="5"
INCLUDE_CHEAT="1"
VARIANTS="agent1,agent2"
CLAUDE_ENV_SH="/home/wsx/kimi_claude_env.sh"
TILELANG_ENV_SH="/home/wsx/tilelang-ascend/set_env.sh"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
ALLOWED_TOOLS="Bash,Read,Write,Edit,MultiEdit,Glob,Grep,Task"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-map) SOURCE_MAP="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --npus) NPUS="$2"; shift 2 ;;
        --timeout) TIMEOUT_SEC="$2"; shift 2 ;;
        --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
        --include-cheat) INCLUDE_CHEAT="$2"; shift 2 ;;
        --variants) VARIANTS="$2"; shift 2 ;;
        --claude-env) CLAUDE_ENV_SH="$2"; shift 2 ;;
        --tilelang-env) TILELANG_ENV_SH="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,80p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

[[ -n "$OUTPUT_DIR" ]] || { echo "missing --output"; exit 1; }
[[ -f "$SOURCE_MAP" ]] || { echo "missing source map: $SOURCE_MAP"; exit 1; }

mkdir -p "$OUTPUT_DIR"
QUEUE_DIR="$OUTPUT_DIR/queues"
mkdir -p "$QUEUE_DIR"
MASTER_TARGETS="$OUTPUT_DIR/precision_failed_targets.tsv"

python3 - "$SOURCE_MAP" "$MASTER_TARGETS" "$INCLUDE_CHEAT" <<'PY'
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
out = Path(sys.argv[2])
include_cheat = sys.argv[3] not in {"0", "false", "False", "no", "NO"}

rows = []
for row in csv.DictReader(source.open()):
    if row.get("verify_failure_type") != "precision_failed":
        continue
    if not include_cheat and row.get("anticheat_verdict") != "CLEAN":
        continue
    rows.append(row)

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    f.write("level\tcase_id\top_name\tcategory\tanticheat_verdict\tsource_task_dir\n")
    for row in rows:
        f.write(
            "\t".join([
                row["level"],
                row["case_id"],
                row["op_name"],
                row["category"],
                row["anticheat_verdict"],
                row["selected_case_path"],
            ]) + "\n"
        )
print(f"selected precision_failed targets: {len(rows)}")
PY

IFS=',' read -ra NPU_ARR <<< "$NPUS"
IFS=',' read -ra VARIANT_ARR <<< "$VARIANTS"

make_prompt() {
    local variant="$1" task_dir="$2" npu="$3" attempts="$4"
    if [[ "$variant" == "agent1" ]]; then
        cat <<EOF
You are running a non-interactive AscendC precision debug comparison task.

Use these instructions as the method source:
- Agent spec: $WORKDIR/external_sources/xtt_main/agents/ascend-kernel-developer.md
- Precision skill: $WORKDIR/.claude/skills/ascendc-operator-precision-debug/SKILL.md

This is NOT a fresh operator generation task. Repair the existing task directory only:
- task_dir: $task_dir
- npu: $npu
- max repair attempts: $attempts

Rules:
- Only read/write files under task_dir, plus repo reference docs and archive_tasks.
- Do not read any other outputs/ directory.
- Keep fixes minimal. Prefer kernel/*.cpp and kernel/*.h changes; only change model_new_ascendc.py if the wrapper is the real root cause.
- For each attempt, run verification and classification:
  mkdir -p "$task_dir/.verify_logs"
  python3 utils/verification_ascendc.py "$task_dir" > "$task_dir/.verify_logs/agent1_attemptN.stdout" 2> "$task_dir/.verify_logs/agent1_attemptN.stderr"
  python3 utils/classify_verify_result.py --exit-code <rc> --stdout-path "$task_dir/.verify_logs/agent1_attemptN.stdout" --stderr-path "$task_dir/.verify_logs/agent1_attemptN.stderr" --task-dir "$task_dir" --phase 9 --attempt N --write-status
- Stop early if failure_type becomes success.
- Write "$task_dir/debug_compare_agent1_trace.md" with attempts, diagnosis, files changed, and final verdict.
EOF
    elif [[ "$variant" == "agent2" ]]; then
        cat <<EOF
Use the Claude agent "ascendc-debug-agent-constructive" for this non-interactive debug task.

Input:
debug $task_dir npu=$npu

Hard constraints:
- Only read/write files under task_dir, plus repo reference docs and archive_tasks.
- Do not read any other outputs/ directory.
- max attempts is controlled by ASCENDC_DEBUG_MAX_ATTEMPTS=$attempts.
- Complete the required debug_trace.md and debug_status.json before exiting.
EOF
    else
        echo "unsupported variant: $variant" >&2
        return 1
    fi
}

prepare_variant() {
    local variant="$1"
    local variant_dir="$OUTPUT_DIR/$variant"
    local task_list="$variant_dir/task_dirs.tsv"
    mkdir -p "$variant_dir/tasks"
    : > "$task_list"

    tail -n +2 "$MASTER_TARGETS" | while IFS=$'\t' read -r level case_id op_name category verdict source_dir; do
        local padded
        padded="$(printf "%03d" "$case_id")"
        local dest="$variant_dir/tasks/$level/${padded}_${op_name}"
        mkdir -p "$(dirname "$dest")"
        rm -rf "$dest"
        mkdir -p "$dest"
        cp -a "$source_dir/." "$dest/"
        printf "%s\t%s\t%s\t%s\t%s\n" "$level" "$case_id" "$op_name" "$category" "$dest" >> "$task_list"
    done

    echo "$task_list"
}

final_verify() {
    local variant="$1" task_dir="$2" npu="${3:-}"
    mkdir -p "$task_dir/.verify_logs"
    local stdout="$task_dir/.verify_logs/${variant}_final.stdout"
    local stderr="$task_dir/.verify_logs/${variant}_final.stderr"
    set +e
    if [[ -n "$npu" ]]; then
        ASCEND_RT_VISIBLE_DEVICES="$npu" timeout --signal=TERM --kill-after=30 600 \
            python3 utils/verification_ascendc.py "$task_dir" > "$stdout" 2> "$stderr"
    else
        timeout --signal=TERM --kill-after=30 600 \
            python3 utils/verification_ascendc.py "$task_dir" > "$stdout" 2> "$stderr"
    fi
    local rc=$?
    python3 utils/classify_verify_result.py \
        --exit-code "$rc" \
        --stdout-path "$stdout" \
        --stderr-path "$stderr" \
        --task-dir "$task_dir" \
        --phase 9 \
        --attempt 999 \
        --write-status >/dev/null 2>&1 || true
    python3 skills/ascendc/ascendc-debug/scripts/anticheat.py verify --json "$task_dir" \
        > "$task_dir/_anticheat_${variant}.json" 2> "$task_dir/_anticheat_${variant}.stderr" || true
    set -e
}

run_variant() {
    local variant="$1"
    local task_list="$2"
    local variant_dir="$OUTPUT_DIR/$variant"
    local queue="$QUEUE_DIR/${variant}.queue"
    local lock="$QUEUE_DIR/${variant}.lock"
    local report="$variant_dir/batch_report.md"
    cp "$task_list" "$queue"
    : > "$lock"
    {
        echo "# precision debug batch: $variant"
        echo
        echo "- start: $(date '+%F %T')"
        echo "- npus: $NPUS"
        echo "- timeout: $TIMEOUT_SEC"
        echo "- max_attempts: $MAX_ATTEMPTS"
        echo
        echo "| task | status | elapsed_sec | worker |"
        echo "|------|--------|-------------|--------|"
    } > "$report"

    for npu in "${NPU_ARR[@]}"; do
        (
            local worker="npu${npu}"
            local wlog="$variant_dir/worker_${worker}.log"
            : > "$wlog"
            while true; do
                local line=""
                {
                    flock -x 200
                    line="$(head -n 1 "$queue" || true)"
                    if [[ -n "$line" ]]; then
                        tail -n +2 "$queue" > "${queue}.tmp"
                        mv "${queue}.tmp" "$queue"
                    fi
                } 200>"$lock"
                [[ -n "$line" ]] || break

                IFS=$'\t' read -r level case_id op_name category task_dir <<< "$line"
                local started ended elapsed status result_file prompt agent_args=()
                started="$(date +%s)"
                result_file="$task_dir/_claude_${variant}.json"
                prompt="$(make_prompt "$variant" "$task_dir" "$npu" "$MAX_ATTEMPTS")"
                if [[ "$variant" == "agent2" ]]; then
                    agent_args=(--agent ascendc-debug-agent-constructive)
                fi

                {
                    echo "================================================================"
                    echo "[task] $variant $level/$case_id $op_name task_dir=$task_dir npu=$npu"
                    echo "[start] $(date '+%F %T')"
                } >> "$wlog"

                set +e
                (
                    source "$CLAUDE_ENV_SH"
                    [[ -f "$TILELANG_ENV_SH" ]] && source "$TILELANG_ENV_SH"
                    cd "$WORKDIR"
                    model="${ANTHROPIC_MODEL:-}"
                    ASCEND_RT_VISIBLE_DEVICES="$npu" \
                    ASCENDC_DEBUG_MAX_ATTEMPTS="$MAX_ATTEMPTS" \
                    timeout --signal=TERM --kill-after=30 "$TIMEOUT_SEC" \
                        "$CLAUDE_BIN" --bare -p \
                            --model "$model" \
                            --add-dir "$WORKDIR" \
                            --allowedTools "$ALLOWED_TOOLS" \
                            --output-format json \
                            "${agent_args[@]}" \
                            "$prompt" \
                        > "$result_file"
                ) >> "$wlog" 2>&1
                local rc=$?
                set -e

                final_verify "$variant" "$task_dir" "$npu" >> "$wlog" 2>&1 || true
                ended="$(date +%s)"
                elapsed=$((ended - started))
                if [[ "$rc" -eq 0 ]]; then
                    status="claude_ok"
                elif [[ "$rc" -eq 124 || "$rc" -eq 137 || "$rc" -eq 143 ]]; then
                    status="timeout"
                else
                    status="claude_rc_${rc}"
                fi
                {
                    flock -x 201
                    echo "| $level/$case_id $op_name | $status | $elapsed | $worker |" >> "$report"
                } 201>"$lock.report"
                echo "[done] status=$status elapsed=$elapsed" >> "$wlog"
            done
        ) &
    done
    wait
    echo "- end: $(date '+%F %T')" >> "$report"
}

for variant in "${VARIANT_ARR[@]}"; do
    echo "===== preparing $variant ====="
    task_list="$(prepare_variant "$variant")"
    echo "===== running $variant ====="
    run_variant "$variant" "$task_list"
done

python3 "$WORKDIR/utils/summarize_precision_debug_compare.py" --run-dir "$OUTPUT_DIR" || true
echo "done: $OUTPUT_DIR"
