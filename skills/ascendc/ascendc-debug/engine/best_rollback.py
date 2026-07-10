"""best_rollback.py — 串行覆盖的 current_best 保存 + 匹配率回滚 (问题 7)。

debug 每轮 agent 原地覆盖 kernel/，无 best 保护时某轮改劣会让下一轮在错误代码上继续改。
本模块把老版本 SKILL.md 里丢失的 current_best/回滚机制迁移到 engine 代码级 (决策 5)。

设计决策 (均经 success16 产物举证，见 消融实验问题反馈.md 问题 7):
  - 指标 (决策 1): 主 case_pass_rate，tie-break match_rate。排序键 (case%, match%) 元组。
    match_rate 未通过时是 avg_mismatch_ratio 逐元素派生，可与 case 通过数背离 (NLLLoss m=0/c=70)，
    故不单用 match_rate。
  - 粒度 (决策 2, 2026-07-11 修订): 只存 kernel/ 真源码 (排除 kernel/build/)。依据:
    (a) 每个算子真源码固定为 kernel/ 下非 build 的 .cpp/.h/.hpp/.json (6-9 个)，规律通用；
    (b) validate 每轮强制 --clean 重编译 (validate_runner 契约 + ASCENDC_DEBUG_CLEAN_BUILD
    默认 1)，build/ 产物完全可重建，存旧 build 反而可能与回滚后源码 clean 重编结果不一致；
    (c) 只存 6-9 个小文件，回滚仅覆盖这几个，rmtree/copytree 失败窗口几乎消失 (原比整目录
    33-53MB/400+ 文件更快更原子)，并规避 build/ 内 cmake 绝对路径缓存的跨机隐患。
  - 时机 (决策 3): 单轮下降绝不回滚；连续 2 轮无改善才回滚 (产物: 8 下降事件 0 连续下降，
    5 例单轮自愈，如 NMS 50→40→100)。
  - 上下文 (决策 4): 回滚后代码级注入失败方向到下一轮 prompt (见 agent_backend)。

best-effort: 所有函数吞异常，绝不阻断主循环。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

_BEST_DIR = "precision_tuning/history/current_best"
_BEST_SRC = "src"          # current_best 下存真源码镜像的子目录
_BEST_METRIC = "metric.json"
# 真源码后缀 (与 exit_artifacts._KERNEL_SRC_GLOBS / validate_runner._FORENSICS_SRC_GLOBS
# 完全对齐: 只 .cpp/.h/.hpp/.py)。不含 .json —— kernel/ 非 build 的唯一 .json 是
# fusion_result.json，属编译融合报告 (build 产物，--clean 重建)，非 agent 源码。
_SRC_SUFFIXES = (".cpp", ".h", ".hpp", ".py")
_BUILD_DIRNAME = "build"


def _source_files(kernel_dir: Path) -> list[Path]:
    """kernel/ 下的真源码文件 (排除 kernel/build/ 生成物)。返回绝对路径列表。

    真源码 = 非 build/ 的 .cpp/.h/.hpp/.py/.json (agent 手改对象)。build/ 每轮 validate
    --clean 重建，不纳入 best。跳过点文件 (macOS AppleDouble 等，见 exit_artifacts)。
    """
    out: list[Path] = []
    if not kernel_dir.is_dir():
        return out
    for f in kernel_dir.rglob("*"):
        if not f.is_file():
            continue
        if _BUILD_DIRNAME in f.relative_to(kernel_dir).parts:
            continue  # 排除 kernel/build/**
        if f.name.startswith("."):
            continue
        if f.suffix in _SRC_SUFFIXES:
            out.append(f)
    return out


def _metric_key(metric: dict) -> tuple[float, float]:
    """排序键 (case_pass_rate, match_rate)。缺失按 -1 (最差)，保证任何有效轮都优于无数据。"""
    try:
        case = float(metric.get("case_pass_rate"))
    except (TypeError, ValueError):
        case = -1.0
    try:
        match = float(metric.get("match_rate"))
    except (TypeError, ValueError):
        match = -1.0
    return (case, match)


def read_attempt_metric(task_dir: Path, attempt: int) -> Optional[dict]:
    """读 validation_result_attempt_N.json 的指标子集。缺失/损坏返回 None。"""
    path = Path(task_dir) / "precision_tuning" / f"validation_result_attempt_{attempt}.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return {
        "attempt": attempt,
        "case_pass_rate": d.get("case_pass_rate"),
        "match_rate": d.get("match_rate"),
        "correctness_passed": d.get("correctness_passed"),
    }


def read_best_metric(task_dir: Path) -> Optional[dict]:
    """读 current_best/metric.json。无则 None (还没存过 best)。"""
    path = Path(task_dir) / _BEST_DIR / _BEST_METRIC
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def save_current_best(task_dir: Path, attempt: int, metric: Optional[dict] = None) -> bool:
    """若本轮指标 > 已存 best，把整个 kernel/ 覆盖存入 current_best/ 并更新 metric.json。

    返回 True=更新了 best。best-effort，异常吞掉返回 False。
    """
    task_dir = Path(task_dir)
    if metric is None:
        metric = read_attempt_metric(task_dir, attempt)
    if metric is None:
        return False
    best = read_best_metric(task_dir)
    if best is not None and _metric_key(metric) <= _metric_key(best):
        return False  # 未提升 (含相等)，不覆盖
    kernel = task_dir / "kernel"
    srcs = _source_files(kernel)
    if not srcs:
        return False
    best_src = task_dir / _BEST_DIR / _BEST_SRC
    try:
        if best_src.exists():
            shutil.rmtree(best_src)
        best_src.mkdir(parents=True, exist_ok=True)
        for f in srcs:
            rel = f.relative_to(kernel)
            dst = best_src / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
        (task_dir / _BEST_DIR / _BEST_METRIC).write_text(
            json.dumps(metric, ensure_ascii=False), encoding="utf-8")
    except OSError:
        return False
    return True


def should_rollback(task_dir: Path, current_attempt: int) -> bool:
    """决策 3: 连续 2 轮无改善才回滚，单轮下降不回滚。

    判据: 本轮 (current_attempt) 与上一轮 (current_attempt-1) 的指标**都未超过 best**。
    单轮下降时上一轮可能仍是 best，key 相等不算"未超过"→ 不触发 (容忍单轮下降)。
    需要至少 2 个已完成轮次 + 已有 best，否则不回滚。
    """
    if current_attempt < 1:
        return False
    best = read_best_metric(task_dir)
    if best is None:
        return False
    best_key = _metric_key(best)
    cur = read_attempt_metric(task_dir, current_attempt)
    prev = read_attempt_metric(task_dir, current_attempt - 1)
    if cur is None or prev is None:
        return False
    # 两轮都严格劣于 best (未达到 best) → 连续无改善 → 回滚。
    return _metric_key(cur) < best_key and _metric_key(prev) < best_key


def do_rollback(task_dir: Path) -> Optional[dict]:
    """用 current_best/src/ 覆盖工作区 kernel/ 的真源码 (build/ 不动，下轮 validate --clean 重建)。

    语义: 删除当前 kernel/ 全部真源码 (含 agent 新增的、best 里没有的) 后写回 best 源码，
    实现源码层的整体替换。返回 best metric (供注入 prompt)；失败 None。
    """
    task_dir = Path(task_dir)
    best_src = task_dir / _BEST_DIR / _BEST_SRC
    if not best_src.is_dir():
        return None
    kernel = task_dir / "kernel"
    try:
        # 1. 删除当前真源码 (只删源码，保留 build/)。
        for f in _source_files(kernel):
            f.unlink()
        # 2. 从 best/src 写回。
        for f in best_src.rglob("*"):
            if not f.is_file():
                continue
            dst = kernel / f.relative_to(best_src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
    except OSError:
        return None
    return read_best_metric(task_dir)

