#!/usr/bin/env python3
"""
precision_knowledge.py — 精度问题知识库管理

职责:
  1. load   — 全量加载知识库 (fallback 用)
  2. search — 结构化 RAG 检索: 按 op_types + patterns + position 评分排序, 返回 top-K + CHECKLIST
  3. check  — dump 前检查候选条目与知识库的相似度，辅助 Agent 决策 new/merge/abandon
  4. dump   — 精度通过后，将 Agent 生成的候选条目写入知识库（支持 new / merge / abandon 三种操作）

知识库格式: 七字段 JSON，RAG-ready。
每条记录:
  {
    "title":    "标准化中文标题 (含英文关键词)",
    "patterns": ["tail_spike", "boundary_concentration"],   # 误差模式数组，可为空 []
    "op_types": ["reduction", "pooling"],                   # 算子类型数组，可为空 []
    "feature":  "自然语言描述错误特征 (泛化, 中文)，不嵌入 pattern=/op_type= 标签",
    "reason":   "深层原因 (中文)",
    "fix":      "通用修复指南 (代码级别, 中文)",
    "type":     "FIX_PRECISION_xxx"
  }

patterns 合法值（VALID_PATTERNS）:
  tail_spike, uniform_offset, scattered, magnitude_correlated,
  nan_inf_contamination, dimension_concentration, boundary_concentration, all_wrong

op_types: 自由字符串数组，如 "reduction"、"pooling"、"matmul"、"convolution" 等，无枚举限制。

type 枚举 (精度专项):
  FIX_PRECISION_PADDING     — Padding 值导致精度问题
  FIX_PRECISION_TAIL        — 尾块处理精度问题
  FIX_PRECISION_REDUCTION   — 归约操作精度损失
  FIX_PRECISION_TYPECAST    — 类型转换精度问题
  FIX_PRECISION_LAYOUT      — 数据布局导致精度错误
  FIX_PRECISION_SYNC        — 同步问题导致精度随机错误
  FIX_PRECISION_OVERFLOW    — 数值溢出 (NaN/Inf)
  FIX_PRECISION_LOGIC       — 算法逻辑导致精度偏差
  FIX_PRECISION_OTHER       — 其他精度问题

用法:
    # 全量加载知识库 (fallback, stdout 输出 JSON)
    python3 precision_knowledge.py load --kb-path <path>

    # 结构化 RAG 检索 (推荐, stdout 输出 JSON)
    python3 precision_knowledge.py search --kb-path <path> --op-type <type> --pattern <hint> [--position <pos>] [--top-k 3]

    # dump 前检查候选条目与知识库的相似度 (stdout 输出 JSON)
    python3 precision_knowledge.py check --kb-path <path> --candidate-path <path> [--top-k 3] [--threshold 0.10]

    # 写入知识库 (精度通过后)
    python3 precision_knowledge.py dump --kb-path <path> --task-name <name> --op-name <name> [--action new|merge|abandon] [--merge-target-title "<title>"]
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent.parent.parent.parent


VALID_TYPES = [
    "FIX_PRECISION_PADDING",
    "FIX_PRECISION_TAIL",
    "FIX_PRECISION_REDUCTION",
    "FIX_PRECISION_TYPECAST",
    "FIX_PRECISION_LAYOUT",
    "FIX_PRECISION_SYNC",
    "FIX_PRECISION_OVERFLOW",
    "FIX_PRECISION_LOGIC",
    "FIX_PRECISION_OTHER",
]

VALID_PATTERNS = [
    "tail_spike",
    "uniform_offset",
    "scattered",
    "magnitude_correlated",
    "nan_inf_contamination",
    "dimension_concentration",
    "boundary_concentration",
    "all_wrong",
]

# 文本必填字段（非空字符串）
REQUIRED_FIELDS = ["title", "feature", "reason", "fix", "type"]


def _is_valid_entry(entry: dict) -> bool:
    """校验知识库条目格式：文本字段非空 + patterns/op_types 为 list。"""
    for k in REQUIRED_FIELDS:
        if not entry.get(k):
            return False
    if not isinstance(entry.get("patterns"), list):
        return False
    if not isinstance(entry.get("op_types"), list):
        return False
    return True


# ============================================================
# Tokenize / Similarity helpers (用于 check 子命令)
# ============================================================

def _tokenize(text: str) -> set:
    """
    提取混合中英文文本中的 token 集合。
    - 英文: 按非字母边界切分，snake_case / camelCase 进一步拆分，min 长度 2
    - 中文: 每个汉字单独作为 token
    """
    tokens = set()
    for ch in re.findall(r'[一-鿿]', text):
        tokens.add(ch)
    for word in re.findall(r'[a-zA-Z][a-zA-Z0-9_]*', text):
        w = word.lower()
        if len(w) >= 2:
            tokens.add(w)
        for part in w.split('_'):
            if len(part) >= 2:
                tokens.add(part)
        for part in re.sub(r'([A-Z][a-z]+)', r' \1', word).split():
            if len(part) >= 2:
                tokens.add(part.lower())
    return tokens


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard 相似度: |A∩B| / |A∪B|"""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ============================================================
# Load
# ============================================================

def load_knowledge_base(kb_path: str) -> list:
    """加载知识库, 返回条目列表"""
    if not os.path.exists(kb_path):
        print(f"[KB] ⚠️ 知识库文件不存在: {kb_path}, 使用空知识库", file=sys.stderr)
        return []

    with open(kb_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[KB] ⚠️ 知识库格式错误 (期望 list), 使用空知识库", file=sys.stderr)
        return []

    valid = [e for e in data if _is_valid_entry(e)]

    print(f"[KB] ✅ 已加载 {len(valid)} 条精度知识")
    print(json.dumps(valid, indent=2, ensure_ascii=False))
    return valid


# ============================================================
# Search (结构化 RAG 检索)
# ============================================================

# pattern → 最可能关联的 type 映射 (用于 type 字段交叉加权)
PATTERN_TYPE_AFFINITY = {
    "tail_spike": ["FIX_PRECISION_TAIL", "FIX_PRECISION_PADDING", "FIX_PRECISION_REDUCTION"],
    "uniform_offset": ["FIX_PRECISION_PADDING", "FIX_PRECISION_LOGIC", "FIX_PRECISION_REDUCTION"],
    "scattered": ["FIX_PRECISION_SYNC", "FIX_PRECISION_REDUCTION"],
    "magnitude_correlated": ["FIX_PRECISION_TYPECAST", "FIX_PRECISION_REDUCTION"],
    "nan_inf_contamination": ["FIX_PRECISION_OVERFLOW"],
    "dimension_concentration": ["FIX_PRECISION_LAYOUT", "FIX_PRECISION_LOGIC"],
    "boundary_concentration": ["FIX_PRECISION_PADDING", "FIX_PRECISION_TAIL"],
    "all_wrong": ["FIX_PRECISION_LOGIC", "FIX_PRECISION_LAYOUT", "FIX_PRECISION_REDUCTION", "FIX_PRECISION_SYNC"],
}

# position 特征 → 关联的 pattern (用于第二次检索的辅助加分)
POSITION_PATTERN_AFFINITY = {
    "tail": ["tail_spike", "boundary_concentration"],
    "boundary": ["boundary_concentration", "tail_spike"],
    "head": ["uniform_offset", "all_wrong"],
    "scattered": ["scattered", "magnitude_correlated"],
}

# 评分权重
W_PATTERN = 3
W_OP_TYPE = 2
W_TYPE = 1
W_POSITION = 1
W_OP_TYPE_ALL_WRONG_BOOST = 2  # all_wrong 是泛化 hint, op_type 精确匹配时额外加权


def _is_checklist(entry: dict) -> bool:
    """判断条目是否为 CHECKLIST 类型"""
    return entry.get("title", "").startswith("[CHECKLIST]")


def _score_entry(entry: dict, query_pattern: str | None, query_op_type: str | None,
                 query_position: str | None) -> float:
    """对单条知识库条目评分。直接从 patterns / op_types 数组读取，无需解析 feature 文本。"""
    score = 0.0
    entry_patterns = entry.get("patterns", [])
    entry_op_types = entry.get("op_types", [])
    entry_type = entry.get("type", "")

    # 1. pattern 精确匹配 (权重 3)
    if query_pattern and query_pattern in entry_patterns:
        score += W_PATTERN

    # 2. op_type 精确匹配 (权重 2) — 仅对非 CHECKLIST 条目
    if query_op_type and not _is_checklist(entry):
        if query_op_type in entry_op_types:
            score += W_OP_TYPE

    # 3. type 字段交叉匹配 (权重 1)
    if query_pattern and query_pattern in PATTERN_TYPE_AFFINITY:
        if entry_type in PATTERN_TYPE_AFFINITY[query_pattern]:
            score += W_TYPE

    # 4. position 辅助加分 (仅第二次检索时使用, 权重 1)
    if query_position and query_position in POSITION_PATTERN_AFFINITY:
        affine_patterns = POSITION_PATTERN_AFFINITY[query_position]
        for p in affine_patterns:
            if p in entry_patterns:
                score += W_POSITION
                break

    # 5. all_wrong 特例: op_type 额外加权
    if query_pattern == "all_wrong" and query_op_type and not _is_checklist(entry):
        if query_op_type in entry_op_types:
            score += W_OP_TYPE_ALL_WRONG_BOOST

    return score


def search_knowledge_base(kb_path: str, op_type: str | None = None,
                          pattern: str | None = None, position: str | None = None,
                          top_k: int = 3) -> dict:
    """
    结构化 RAG 检索: 根据 op_type + pattern + position 筛选并评分排序。

    返回:
      {
        "query": {"op_type": ..., "pattern": ..., "position": ..., "top_k": ...},
        "matched_entries": [...],       # top-K 普通条目 (按 score 降序)
        "checklists": [...],            # op_type 匹配的 CHECKLIST (不占 K 配额)
        "total_kb_size": N,
        "fallback_to_full_load": bool
      }
    """
    if not os.path.exists(kb_path):
        print(f"[KB-SEARCH] ⚠️ 知识库文件不存在: {kb_path}", file=sys.stderr)
        return _empty_search_result(op_type, pattern, position, top_k)

    try:
        with open(kb_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[KB-SEARCH] ⚠️ 知识库读取失败: {e}", file=sys.stderr)
        return _empty_search_result(op_type, pattern, position, top_k)

    if not isinstance(data, list):
        print(f"[KB-SEARCH] ⚠️ 知识库格式错误 (期望 list)", file=sys.stderr)
        return _empty_search_result(op_type, pattern, position, top_k)

    valid = [e for e in data if _is_valid_entry(e)]

    # 分离 CHECKLIST 和普通条目
    checklists = [e for e in valid if _is_checklist(e)]
    normal_entries = [e for e in valid if not _is_checklist(e)]

    # CHECKLIST 按 op_types 数组精确匹配
    matched_checklists = []
    if op_type:
        for cl in checklists:
            if op_type in cl.get("op_types", []):
                matched_checklists.append(cl)

    # 普通条目评分排序
    scored = []
    for entry in normal_entries:
        score = _score_entry(entry, pattern, op_type, position)
        if score > 0:
            scored.append({"score": score, "entry": entry})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_entries = scored[:top_k]

    # Fallback: 无任何命中 → 返回前 top_k 条
    fallback = len(top_entries) == 0 and len(matched_checklists) == 0
    if fallback:
        print(f"[KB-SEARCH] ⚠️ 无匹配条目, fallback 到前 {top_k} 条", file=sys.stderr)
        top_entries = [{"score": 0, "entry": e} for e in normal_entries[:top_k]]
        matched_checklists = checklists[:top_k]

    result = {
        "query": {
            "op_type": op_type,
            "pattern": pattern,
            "position": position,
            "top_k": top_k,
        },
        "matched_entries": [
            {
                "index": valid.index(s["entry"]) if s["entry"] in valid else -1,
                "score": s["score"],
                "title": s["entry"]["title"],
                "patterns": s["entry"].get("patterns", []),
                "op_types": s["entry"].get("op_types", []),
                "feature": s["entry"]["feature"],
                "reason": s["entry"]["reason"],
                "fix": s["entry"]["fix"],
                "type": s["entry"]["type"],
            }
            for s in top_entries
        ],
        "checklists": [
            {
                "title": cl["title"],
                "patterns": cl.get("patterns", []),
                "op_types": cl.get("op_types", []),
                "feature": cl["feature"],
                "reason": cl["reason"],
                "fix": cl["fix"],
                "type": cl["type"],
            }
            for cl in matched_checklists
        ],
        "total_kb_size": len(valid),
        "fallback_to_full_load": fallback,
    }

    n_matched = len(result["matched_entries"])
    n_checklists = len(result["checklists"])
    print(f"[KB-SEARCH] ✅ 检索完成 (op_type={op_type}, pattern={pattern}, position={position})")
    print(f"  知识库总条目: {len(valid)}")
    print(f"  命中普通条目: {n_matched} / top-K={top_k}")
    print(f"  命中 CHECKLIST: {n_checklists}")
    if fallback:
        print(f"  ⚠️ FALLBACK: 无匹配, 已返回全量 {len(valid)} 条")
    for s in top_entries[:top_k]:
        print(f"    [{s['score']:.1f}] {s['entry']['title']}")
    for cl in matched_checklists:
        print(f"    [CL] {cl['title']}")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _empty_search_result(op_type, pattern, position, top_k) -> dict:
    return {
        "query": {"op_type": op_type, "pattern": pattern, "position": position, "top_k": top_k},
        "matched_entries": [],
        "checklists": [],
        "total_kb_size": 0,
        "fallback_to_full_load": True,
    }


# ============================================================
# Check (相似度检查，辅助 dump 前决策)
# ============================================================

def check_similarity(kb_path: str, candidate_path: str,
                     top_k: int = 3, threshold: float = 0.10) -> dict:
    """
    检查候选条目与知识库现有条目的相似度，辅助 Agent 判断 new/merge/abandon。

    计算方式: 对 (title + feature + patterns + op_types) 拼接文本提取 token 集合，
    与候选条目同字段 token 集合计算 Jaccard 相似度，返回 score >= threshold 的 top-K 条。

    返回 (stdout JSON):
      {
        "candidate_title": ...,
        "candidate_type": ...,
        "similar_entries": [{"index": N, "score": 0.xx, "title": ..., ...}],
        "suggestion": "new | review_needed"
      }
    """
    if not os.path.exists(candidate_path):
        print(f"[KB-CHECK] ⚠️ 候选条目文件不存在: {candidate_path}", file=sys.stderr)
        return {}

    try:
        with open(candidate_path) as f:
            candidate = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[KB-CHECK] ⚠️ 候选条目读取失败: {e}", file=sys.stderr)
        return {}

    missing = [field for field in REQUIRED_FIELDS if not candidate.get(field)]
    if missing:
        print(f"[KB-CHECK] ⚠️ 候选条目缺少必填字段: {missing}", file=sys.stderr)
        return {}

    # patterns / op_types 缺失时警告但不中止（check 仅预览）
    if not isinstance(candidate.get("patterns"), list):
        print(f"[KB-CHECK] ⚠️ 候选条目 patterns 不是数组，相似度计算可能偏低", file=sys.stderr)
    if not isinstance(candidate.get("op_types"), list):
        print(f"[KB-CHECK] ⚠️ 候选条目 op_types 不是数组，相似度计算可能偏低", file=sys.stderr)

    if not os.path.exists(kb_path):
        print(f"[KB-CHECK] ⚠️ 知识库不存在: {kb_path}，建议 new", file=sys.stderr)
        result = {
            "candidate_title": candidate["title"],
            "candidate_type": candidate["type"],
            "similar_entries": [],
            "suggestion": "new",
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    try:
        with open(kb_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[KB-CHECK] ⚠️ 知识库读取失败: {e}", file=sys.stderr)
        return {}

    valid = [e for e in data if isinstance(e, dict) and _is_valid_entry(e)]

    # 候选条目: title + feature + patterns + op_types 全部参与 tokenize
    cand_patterns = candidate.get("patterns", []) if isinstance(candidate.get("patterns"), list) else []
    cand_op_types = candidate.get("op_types", []) if isinstance(candidate.get("op_types"), list) else []
    cand_text = (candidate["title"] + " " + candidate["feature"]
                 + " " + " ".join(cand_patterns) + " " + " ".join(cand_op_types))
    cand_tokens = _tokenize(cand_text)

    scored = []
    for i, entry in enumerate(valid):
        entry_patterns = entry.get("patterns", [])
        entry_op_types = entry.get("op_types", [])
        entry_text = (entry["title"] + " " + entry["feature"]
                      + " " + " ".join(entry_patterns) + " " + " ".join(entry_op_types))
        entry_tokens = _tokenize(entry_text)
        score = _jaccard(cand_tokens, entry_tokens)
        if score >= threshold:
            scored.append({
                "index": i,
                "score": round(score, 4),
                "title": entry["title"],
                "type": entry["type"],
                "patterns": entry_patterns,
                "op_types": entry_op_types,
                "feature": entry["feature"],
                "reason": entry["reason"],
                "fix": entry["fix"],
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_entries = scored[:top_k]

    suggestion = "review_needed" if top_entries else "new"

    result = {
        "candidate_title": candidate["title"],
        "candidate_type": candidate["type"],
        "similar_entries": top_entries,
        "suggestion": suggestion,
    }

    print(f"[KB-CHECK] 检查完成: {candidate['title']}", file=sys.stderr)
    print(f"  知识库条目总数: {len(valid)}", file=sys.stderr)
    print(f"  相似条目 (score>={threshold}): {len(top_entries)}", file=sys.stderr)
    for e in top_entries:
        print(f"    [{e['score']:.4f}] {e['title']}", file=sys.stderr)
    print(f"  建议: {suggestion}", file=sys.stderr)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


# ============================================================
# Dump (精度通过后写入知识库)
# ============================================================

def dump_success_knowledge(kb_path: str, task_dir: str, op_name: str,
                           action: str = "new",
                           merge_target_title: str | None = None) -> dict | None:
    """
    将 Agent 生成的候选知识库条目写入知识库。

    action:
      new    — 追加为新条目（默认）
      merge  — 用候选内容替换 merge_target_title 对应的已有条目
      abandon — 候选已被现有知识完全覆盖，跳过写入

    读取: {task_dir}/precision_tuning/candidate_kb_entry.json
    """
    tuning_dir = os.path.join(task_dir, "precision_tuning")

    # 1. 读取候选条目
    candidate_path = os.path.join(tuning_dir, "candidate_kb_entry.json")
    if not os.path.exists(candidate_path):
        print(f"[KB] ⚠️ 候选条目文件不存在: {candidate_path}", file=sys.stderr)
        print(f"    请先执行 Step 5.2: 生成候选知识库条目", file=sys.stderr)
        return None

    try:
        with open(candidate_path) as f:
            candidate = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[KB] ⚠️ 候选条目 JSON 解析失败: {e}", file=sys.stderr)
        return None

    # 2. 验证文本字段完整性
    missing = [f for f in REQUIRED_FIELDS if not candidate.get(f)]
    if missing:
        print(f"[KB] ⚠️ 候选条目缺少必填字段: {missing}", file=sys.stderr)
        return None

    # 3. 验证 type 枚举值
    if candidate["type"] not in VALID_TYPES:
        print(f"[KB] ⚠️ type 值非法: {candidate['type']}，应为以下之一: {VALID_TYPES}", file=sys.stderr)
        return None

    # 4. 严格校验 patterns 字段（写入端强制）
    patterns_val = candidate.get("patterns")
    if not isinstance(patterns_val, list):
        print(f"[KB] ❌ patterns 字段必须为数组，当前类型: {type(patterns_val).__name__}", file=sys.stderr)
        print(f"    正确示例: \"patterns\": [\"all_wrong\", \"scattered\"]", file=sys.stderr)
        return None
    invalid_patterns = [p for p in patterns_val if p not in VALID_PATTERNS]
    if invalid_patterns:
        print(f"[KB] ❌ patterns 包含非法值: {invalid_patterns}", file=sys.stderr)
        print(f"    合法值: {VALID_PATTERNS}", file=sys.stderr)
        return None

    # 5. 严格校验 op_types 字段（写入端强制）
    op_types_val = candidate.get("op_types")
    if not isinstance(op_types_val, list):
        print(f"[KB] ❌ op_types 字段必须为数组，当前类型: {type(op_types_val).__name__}", file=sys.stderr)
        print(f"    正确示例: \"op_types\": [\"reduction\"] 或空数组 []", file=sys.stderr)
        return None
    invalid_op_types = [t for t in op_types_val if not isinstance(t, str) or not t.strip()]
    if invalid_op_types:
        print(f"[KB] ❌ op_types 包含非法值（必须为非空字符串）: {invalid_op_types}", file=sys.stderr)
        return None

    # 6. abandon: 记录日志后直接返回
    if action == "abandon":
        print(f"[KB] ℹ️ action=abandon — 候选已被现有知识覆盖，跳过写入")
        print(f"  title: {candidate['title']}")
        return None

    # 7. 补充 _meta
    import glob as _glob
    forensics_files = _glob.glob(os.path.join(tuning_dir, "forensics_report_*.json"))
    num_attempts = 1
    if forensics_files:
        try:
            attempt_nums = []
            for fp in forensics_files:
                stem = os.path.basename(fp)[len("forensics_report_"):-len(".json")]
                if stem.isdigit():
                    attempt_nums.append(int(stem))
            if attempt_nums:
                num_attempts = max(attempt_nums)
        except (ValueError, OSError):
            pass

    entry = dict(candidate)
    entry["_meta"] = {
        "op_name": op_name,
        "created_at": datetime.now().isoformat(),
        "attempts_needed": num_attempts,
        "action": action,
    }

    # 8. 加载知识库
    kb = []
    if os.path.exists(kb_path):
        with open(kb_path) as f:
            kb = json.load(f)

    # 9. 执行写入操作
    if action == "merge":
        if not merge_target_title:
            print(f"[KB] ⚠️ action=merge 需要 --merge-target-title 参数", file=sys.stderr)
            return None
        target_idx = next(
            (i for i, e in enumerate(kb) if e.get("title") == merge_target_title), None
        )
        if target_idx is None:
            print(f"[KB] ⚠️ 未找到目标条目 (title): {merge_target_title}", file=sys.stderr)
            return None
        kb[target_idx] = entry
        print(f"[KB] ✅ 已合并更新条目:")
        print(f"  原 title: {merge_target_title}")
        print(f"  新 title: {entry['title']}")
        print(f"  type: {entry['type']}")
        print(f"  kb_size: {len(kb)} 条 (条目数不变)")
    else:  # action == "new"
        existing_titles = {e.get("title") for e in kb}
        if entry["title"] in existing_titles:
            print(f"[KB] ⚠️ 知识条目已存在 (title 重复), 跳过: {entry['title']}")
            return None
        kb.append(entry)
        print(f"[KB] ✅ 已写入新知识条目: {entry['title']}")
        print(f"  type: {entry['type']}")
        print(f"  patterns: {entry.get('patterns', [])}")
        print(f"  op_types: {entry.get('op_types', [])}")
        print(f"  attempts_needed: {num_attempts}")
        print(f"  kb_size: {len(kb)} 条")

    with open(kb_path, "w") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

    return entry


# ============================================================
# Search log
# ============================================================

def _append_search_log(log_dir: str, call_index: int, op_type, pattern,
                       position, top_k: int, result: dict,
                       attempt: int | None = None) -> None:
    if log_dir.endswith(".json"):
        log_path = log_dir
    else:
        log_path = os.path.join(log_dir, "knowledge_search_log.json")

    existing_entries = []
    if os.path.exists(log_path):
        try:
            with open(log_path, encoding="utf-8") as f:
                existing_entries = json.load(f)
            if not isinstance(existing_entries, list):
                existing_entries = []
        except (json.JSONDecodeError, OSError):
            existing_entries = []

    entry = {
        "attempt": attempt,
        "call_index": call_index,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": {
            "op_type": op_type,
            "pattern": pattern,
            "position": position,
            "top_k": top_k,
        },
        "matched_count": len(result.get("matched_entries", [])),
        "checklist_count": len(result.get("checklists", [])),
        "fallback_to_full_load": result.get("fallback_to_full_load", False),
        "top_titles": [e["title"] for e in result.get("matched_entries", [])[:top_k]],
    }
    existing_entries.append(entry)

    try:
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(existing_entries, f, indent=2, ensure_ascii=False)
    except OSError:
        pass


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="精度问题知识库管理")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # load
    p_load = subparsers.add_parser("load", help="全量加载知识库 (fallback)")
    p_load.add_argument("--kb-path", required=True, help="知识库 JSON 路径")

    # search
    p_search = subparsers.add_parser("search", help="结构化 RAG 检索")
    p_search.add_argument("--kb-path", required=True, help="知识库 JSON 路径")
    p_search.add_argument("--op-type", default=None,
                          help="算子类型 (来自 L8, 如 reduction/pooling/loss/matmul/normalization)")
    p_search.add_argument("--pattern", default=None,
                          help=f"误差模式 (来自取证 primary_hint, 合法值: {VALID_PATTERNS})")
    p_search.add_argument("--position", default=None,
                          help="误差位置特征 (第二次检索用, 如 tail/boundary/head/scattered)")
    p_search.add_argument("--top-k", type=int, default=3)
    p_search.add_argument("--log-path", default=None)
    p_search.add_argument("--call-index", type=int, default=0)
    p_search.add_argument("--attempt", type=int, default=None)

    # check
    p_check = subparsers.add_parser("check", help="检查候选条目与知识库的相似度")
    p_check.add_argument("--kb-path", required=True)
    p_check.add_argument("--candidate-path", required=True)
    p_check.add_argument("--top-k", type=int, default=3)
    p_check.add_argument("--threshold", type=float, default=0.10)

    # dump
    p_dump = subparsers.add_parser("dump", help="精度通过后写入知识库 (支持 new/merge/abandon)")
    p_dump.add_argument("--kb-path", required=True)
    p_dump.add_argument("--task-name", required=True)
    p_dump.add_argument("--task-dir", default=None)
    p_dump.add_argument("--op-name", required=True)
    p_dump.add_argument("--action", choices=["new", "merge", "abandon"], default="new")
    p_dump.add_argument("--merge-target-title", default=None)

    args = parser.parse_args()

    if args.command == "load":
        load_knowledge_base(args.kb_path)
    elif args.command == "search":
        result = search_knowledge_base(
            args.kb_path,
            op_type=args.op_type,
            pattern=args.pattern,
            position=args.position,
            top_k=args.top_k,
        )
        if getattr(args, "log_path", None):
            _append_search_log(
                log_dir=args.log_path,
                call_index=getattr(args, "call_index", 0),
                op_type=args.op_type,
                pattern=args.pattern,
                position=args.position,
                top_k=args.top_k,
                result=result,
                attempt=getattr(args, "attempt", None),
            )
    elif args.command == "check":
        check_similarity(
            args.kb_path,
            args.candidate_path,
            top_k=args.top_k,
            threshold=args.threshold,
        )
    elif args.command == "dump":
        task_dir = args.task_dir or str(REPO_ROOT / args.task_name)
        result = dump_success_knowledge(
            args.kb_path,
            task_dir,
            args.op_name,
            action=args.action,
            merge_target_title=args.merge_target_title,
        )
        if result is None:
            sys.exit(1)


if __name__ == "__main__":
    main()
