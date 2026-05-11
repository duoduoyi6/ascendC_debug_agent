#!/bin/bash
# 跨多 docker 容器按项目过滤并归档 Codex CLI / Claude Code session。
#
# 用法:
#   bash pack_codex_sessions.sh \
#       --project /home/c00959374/AscendOpGenAgent \
#       --containers cjm_cann1,cjm_cann2 \
#       --ssh-host npu_server \
#       --staging /home/c00959374/codex_pack_tmp \
#       --output archive_$(date +%Y%m%d).tar.gz \
#       --tool codex|claude \
#       [--claude-home /home/c00959374/.claude]
#
# 输出：远端打好 tar.gz，再 scp 回本地同名文件。

set -euo pipefail

SSH_HOST="npu_server"
PROJECT=""
CONTAINERS=""
OUTPUT=""
STAGING="/home/c00959374/codex_pack_tmp"
TOOL="codex"
CLAUDE_HOME="/home/c00959374/.claude"

usage() {
    cat <<'EOF'
用法:
  bash pack_codex_sessions.sh \
      --project <path> \
      --containers <c1,c2,...> \
      --output <file.tar.gz> \
      [--ssh-host <host>] \
      [--staging <dir>] \
      [--tool codex|claude] \
      [--claude-home <dir>]

参数:
  --project     项目绝对路径（过滤用）
  --containers  容器名列表，逗号分隔
  --output      本地输出的 tar.gz 路径
  --ssh-host    SSH 跳板主机名（默认 npu_server）
  --staging     远端临时目录（默认 /home/c00959374/codex_pack_tmp）
  --tool        工具类型：codex（默认）或 claude
  --claude-home Claude Code 数据根目录（默认 /root/.claude）
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --project)     PROJECT="$2"; shift 2 ;;
        --containers)  CONTAINERS="$2"; shift 2 ;;
        --ssh-host)    SSH_HOST="$2"; shift 2 ;;
        --staging)     STAGING="$2"; shift 2 ;;
        --output)      OUTPUT="$2"; shift 2 ;;
        --tool)        TOOL="$2"; shift 2 ;;
        --claude-home) CLAUDE_HOME="$2"; shift 2 ;;
        -h|--help)     usage ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

[[ -z "$PROJECT" ]]    && { echo "必须 --project";    exit 1; }
[[ -z "$CONTAINERS" ]] && { echo "必须 --containers"; exit 1; }
[[ -z "$OUTPUT" ]]     && { echo "必须 --output";     exit 1; }
[[ "$TOOL" != "codex" && "$TOOL" != "claude" ]] && { echo "--tool 必须是 codex 或 claude"; exit 1; }

IFS=',' read -ra CARR <<< "$CONTAINERS"
ARCHIVE_BASE=$(basename "$OUTPUT")
REMOTE_ARCHIVE="$STAGING/$ARCHIVE_BASE"

echo "=========================================================="
echo " tool       : $TOOL"
echo " project    : $PROJECT"
echo " containers : $CONTAINERS"
echo " ssh        : $SSH_HOST"
echo " staging    : $STAGING"
echo " output     : $OUTPUT  <-  $REMOTE_ARCHIVE"
if [[ "$TOOL" == "claude" ]]; then
    echo " claude-home: $CLAUDE_HOME"
fi
echo "=========================================================="

# ------------------------------------------------------------------
# 编码项目路径（Claude 用）：将 / \ : 替换为 -
# ------------------------------------------------------------------
encode_project_path() {
    echo "$1" | sed 's/[\/\\:]/-/g'
}

# ------------------------------------------------------------------
# 1) 清理 + 建 staging/merged/
# ------------------------------------------------------------------
ssh "$SSH_HOST" "rm -rf '$STAGING' && mkdir -p '$STAGING/merged'"

# ------------------------------------------------------------------
# Codex 模式过滤
# ------------------------------------------------------------------
run_codex_filter() {
    local C="$1"
    echo
    echo "=== [$C] scanning ==="
    ssh "$SSH_HOST" "docker exec $C bash -lc \"python3 - <<PYEOF
import json, shutil
from pathlib import Path
SRC = Path('/root/.codex/sessions')
DST = Path('$STAGING/merged/$C')
DST.mkdir(parents=True, exist_ok=True)
n = 0
buckets = {}
for f in SRC.rglob('rollout-*.jsonl'):
    try:
        meta = json.loads(open(f).readline())
        if meta.get('payload', {}).get('cwd') == '$PROJECT':
            rel = f.relative_to(SRC)
            dst = DST / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n += 1
            ymd = '-'.join(rel.parts[:3])
            buckets[ymd] = buckets.get(ymd, 0) + 1
    except Exception:
        pass
print(f'matched {n} rollouts')
for ymd in sorted(buckets):
    print(f'  {ymd}: {buckets[ymd]}')
PYEOF
\""
}

# ------------------------------------------------------------------
# Claude 模式过滤
# ------------------------------------------------------------------
run_claude_filter() {
    local C="$1"
    local ENCODED
    ENCODED=$(encode_project_path "$PROJECT")
    echo
    echo "=== [$C] Claude scanning (encoded: $ENCODED) ==="
    ssh "$SSH_HOST" "docker exec $C bash -lc \"python3 - <<PYEOF
import json, shutil, base64, os
from pathlib import Path

SRC = Path('$CLAUDE_HOME')
DST = Path('$STAGING/merged/$C')
PROJECT = '$PROJECT'
ENCODED = '$ENCODED'
DST.mkdir(parents=True, exist_ok=True)

session_ids = set()

# 1) projects/<encoded>/
proj_dir = SRC / 'projects' / ENCODED
if proj_dir.exists():
    dst_proj = DST / 'projects' / ENCODED
    dst_proj.mkdir(parents=True, exist_ok=True)
    for f in proj_dir.iterdir():
        if f.is_file() and f.suffix == '.jsonl':
            session_ids.add(f.stem)
            shutil.copy2(f, dst_proj / f.name)
        elif f.is_dir():
            session_ids.add(f.name)
            shutil.copytree(f, dst_proj / f.name, dirs_exist_ok=True)
    print(f'projects: {len(session_ids)} sessions')

# 2) history.jsonl 补充 sessionId
history_file = SRC / 'history.jsonl'
n_new = 0
if history_file.exists():
    with open(history_file) as hf:
        for line in hf:
            try:
                data = json.loads(line)
                if data.get('project') == PROJECT:
                    sid = data.get('sessionId')
                    if sid and sid not in session_ids:
                        session_ids.add(sid)
                        n_new += 1
            except Exception:
                pass
if n_new:
    print(f'history.jsonl added {n_new} sessions')

# 3) transcripts/ 兼容（存在才扫，无法按项目过滤）
trans_dir = SRC / 'transcripts'
if trans_dir.exists():
    dst_trans = DST / 'transcripts'
    dst_trans.mkdir(parents=True, exist_ok=True)
    n_trans = 0
    for f in trans_dir.glob('*.jsonl'):
        shutil.copy2(f, dst_trans / f.name)
        n_trans += 1
    print(f'transcripts: copied {n_trans} files (unfiltered)')

# 4) 按 sessionId 关联收集
for sid in session_ids:
    src_fh = SRC / 'file-history' / sid
    if src_fh.exists():
        shutil.copytree(src_fh, DST / 'file-history' / sid, dirs_exist_ok=True)
    src_tasks = SRC / 'tasks' / sid
    if src_tasks.exists():
        shutil.copytree(src_tasks, DST / 'tasks' / sid, dirs_exist_ok=True)

# 5) sessions/ 活跃 session
sessions_dir = SRC / 'sessions'
if sessions_dir.exists():
    dst_sessions = DST / 'sessions'
    dst_sessions.mkdir(parents=True, exist_ok=True)
    for f in sessions_dir.glob('*.json'):
        try:
            data = json.load(open(f))
            if data.get('cwd') == PROJECT:
                shutil.copy2(f, dst_sessions / f.name)
        except Exception:
            pass

# 6) history.jsonl 按 sessionId 过滤写出
n_hist = 0
if history_file.exists():
    dst_history = DST / 'history.jsonl'
    with open(history_file) as hf, open(dst_history, 'w') as out:
        for line in hf:
            try:
                data = json.loads(line)
                if data.get('sessionId') in session_ids:
                    out.write(line)
                    n_hist += 1
            except Exception:
                pass
    print(f'history.jsonl: {n_hist} lines')

# 7) telemetry 按 sessionId 过滤 + Base64 解码汇总
tele_dir = SRC / 'telemetry'
total = {'messageTokens': 0, 'inputTokens': 0, 'outputTokens': 0, 'costUSD': 0.0}
n_tele = 0
if tele_dir.exists():
    dst_tele = DST / 'telemetry'
    dst_tele.mkdir(parents=True, exist_ok=True)
    for f in tele_dir.glob('*.json'):
        lines_out = []
        with open(f) as inf:
            for line in inf:
                try:
                    data = json.loads(line)
                    ev = data.get('event_data', {})
                    if ev.get('session_id') in session_ids:
                        lines_out.append(line)
                        meta_b64 = ev.get('additional_metadata', '')
                        if meta_b64:
                            meta = json.loads(base64.b64decode(meta_b64))
                            total['messageTokens'] += meta.get('messageTokens', 0)
                            total['inputTokens'] += meta.get('inputTokens', 0)
                            total['outputTokens'] += meta.get('outputTokens', 0)
                            total['costUSD'] += meta.get('costUSD', 0.0)
                except Exception:
                    pass
        if lines_out:
            out_path = dst_tele / f.name
            with open(out_path, 'w') as outf:
                outf.writelines(lines_out)
            n_tele += len(lines_out)

print(f'matched {len(session_ids)} sessions')
print(f'telemetry lines: {n_tele}')
print('token_summary: ' + json.dumps(total))
PYEOF
\""
}

# ------------------------------------------------------------------
# 2) 在每个容器内过滤 + 复制
# ------------------------------------------------------------------
for C in "${CARR[@]}"; do
    if [[ "$TOOL" == "codex" ]]; then
        run_codex_filter "$C"
    else
        run_claude_filter "$C"
    fi
done

# ------------------------------------------------------------------
# 3) 在 host 端打 tar.gz
# ------------------------------------------------------------------
echo
echo "=== tar on host ==="
ssh "$SSH_HOST" "cd '$STAGING' && tar -czf '$REMOTE_ARCHIVE' -C '$STAGING' merged && ls -la '$REMOTE_ARCHIVE' && du -h '$REMOTE_ARCHIVE'"

# ------------------------------------------------------------------
# 4) scp 回本地
# ------------------------------------------------------------------
echo
echo "=== scp ${SSH_HOST}:$REMOTE_ARCHIVE -> $OUTPUT ==="
scp "${SSH_HOST}:${REMOTE_ARCHIVE}" "$OUTPUT"

echo
echo "=== done ==="
ls -la "$OUTPUT"
if [[ "$TOOL" == "codex" ]]; then
    tar -tzf "$OUTPUT" | awk -F/ '/rollout-.*\.jsonl$/ {print $2}' | sort | uniq -c
else
    tar -tzf "$OUTPUT" | head -n 20
fi
