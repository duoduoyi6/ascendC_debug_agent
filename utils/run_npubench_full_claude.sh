#!/bin/bash
# Run all NPUKernelBench levels. Each level is processed sequentially, while
# cases inside a level are distributed across the configured container/NPU workers.

set -euo pipefail

BENCHMARK_DIR="benchmarks/NPUKernelBench"
OUTPUT_ROOT=""
LEVELS=""
FIRST_LEVEL_IDS=""
CONTAINERS="wsx_cann1,wsx_cann1,wsx_cann1"
NPUS="5,6,7"
MODEL="MiniMax-M2.7-highspeed"
TIMEOUT_SEC="7200"
MAX_RESUMES="3"
STALE_AFTER_FAILURE_SEC="300"
STALE_CHECK_INTERVAL_SEC="30"
CLAUDE_ENV_SH="/home/wsx/minimax_claude_env.sh"
TILELANG_ENV_SH="/home/wsx/tilelang-ascend/set_env.sh"
WORKDIR_IN_CONTAINER="/home/wsx/AscendOpGenAgent"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark-dir) BENCHMARK_DIR="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --levels) LEVELS="$2"; shift 2 ;;
        --first-level-ids) FIRST_LEVEL_IDS="$2"; shift 2 ;;
        --containers) CONTAINERS="$2"; shift 2 ;;
        --npus) NPUS="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --timeout) TIMEOUT_SEC="$2"; shift 2 ;;
        --max-resumes) MAX_RESUMES="$2"; shift 2 ;;
        --stale-after-failure) STALE_AFTER_FAILURE_SEC="$2"; shift 2 ;;
        --stale-check-interval) STALE_CHECK_INTERVAL_SEC="$2"; shift 2 ;;
        --claude-env) CLAUDE_ENV_SH="$2"; shift 2 ;;
        --tilelang-env) TILELANG_ENV_SH="$2"; shift 2 ;;
        --workdir) WORKDIR_IN_CONTAINER="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,40p' "$0"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

[[ -d "$BENCHMARK_DIR" ]] || { echo "错误: 不存在 $BENCHMARK_DIR"; exit 1; }

if [[ -z "$OUTPUT_ROOT" ]]; then
    OUTPUT_ROOT="outputs/claude_minimax_npubench_full_npu567_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_ROOT"

RUNNER="$(dirname "$0")/run_benchmark_ascendc_claude.sh"
SUMMARIZER="$(dirname "$0")/summarize_debug_bench.py"
MANIFEST="$BENCHMARK_DIR/manifest.json"
REPORT="$OUTPUT_ROOT/run_all_report.md"

level_list() {
    if [[ -n "$LEVELS" ]]; then
        echo "$LEVELS" | tr ',' ' '
        return
    fi
    python3 - "$BENCHMARK_DIR" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
levels = []
for path in root.glob("level*"):
    match = re.fullmatch(r"level(\d+)", path.name)
    if match and path.is_dir():
        levels.append(int(match.group(1)))
print(" ".join(str(x) for x in sorted(levels)))
PY
}

level_ids() {
    local level="$1"
    python3 - "$BENCHMARK_DIR/level${level}" <<'PY'
from pathlib import Path
import re
import sys

level_dir = Path(sys.argv[1])
ids = []
for path in level_dir.glob("*.py"):
    match = re.match(r"(\d+)_", path.name)
    if match:
        ids.append(int(match.group(1)))
print(",".join(str(x) for x in sorted(ids)))
PY
}

{
    echo "# NPUKernelBench Full Run"
    echo
    echo "- benchmark: $BENCHMARK_DIR"
    echo "- output_root: $OUTPUT_ROOT"
    echo "- levels: $(level_list)"
    echo "- first level ids override: ${FIRST_LEVEL_IDS:-<none>}"
    echo "- containers: $CONTAINERS"
    echo "- npus: $NPUS"
    echo "- model: $MODEL"
    echo "- claude env: $CLAUDE_ENV_SH"
    echo "- start: $(date '+%F %T')"
    echo
    echo "| level | ids | status | output |"
    echo "| --- | --- | --- | --- |"
} > "$REPORT"

first_level=""
for level in $(level_list); do
    if [[ -z "$first_level" ]]; then
        first_level="$level"
    fi
    if [[ -n "$FIRST_LEVEL_IDS" && "$level" == "$first_level" ]]; then
        ids="$FIRST_LEVEL_IDS"
    else
        ids=$(level_ids "$level")
    fi
    if [[ -z "$ids" ]]; then
        echo "| level${level} |  | SKIPPED(no_cases) |  |" >> "$REPORT"
        continue
    fi

    level_output="$OUTPUT_ROOT/level${level}"
    mkdir -p "$level_output"

    echo "================================================================"
    echo "[level${level}] ids=$ids"
    echo "[level${level}] output=$level_output"
    echo "================================================================"

    bash "$RUNNER" \
        --benchmark-dir "$BENCHMARK_DIR" \
        --level "$level" \
        --ids "$ids" \
        --containers "$CONTAINERS" \
        --npus "$NPUS" \
        --output "$level_output" \
        --model "$MODEL" \
        --timeout "$TIMEOUT_SEC" \
        --max-resumes "$MAX_RESUMES" \
        --stale-after-failure "$STALE_AFTER_FAILURE_SEC" \
        --stale-check-interval "$STALE_CHECK_INTERVAL_SEC" \
        --claude-env "$CLAUDE_ENV_SH" \
        --tilelang-env "$TILELANG_ENV_SH" \
        --workdir "$WORKDIR_IN_CONTAINER"

    if [[ -f "$SUMMARIZER" && -f "$MANIFEST" ]]; then
        python3 "$SUMMARIZER" --run-dir "$level_output" --level "$level" --manifest "$MANIFEST" || true
    fi

    if [[ -s "$level_output/.fatal" ]]; then
        reason=$(cat "$level_output/.fatal")
        echo "| level${level} | $ids | FATAL($reason) | $level_output |" >> "$REPORT"
        echo "全局熔断: level${level}: $reason"
        exit 88
    fi

    echo "| level${level} | $ids | DONE | $level_output |" >> "$REPORT"
done

{
    echo
    echo "- end: $(date '+%F %T')"
} >> "$REPORT"

echo "全量运行完成: $OUTPUT_ROOT"
