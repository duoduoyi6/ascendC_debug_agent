"""Objective validation runner for engine-owned Gate-V inputs.

The debug engine owns the validate step.  A validate action must first rebuild
the AscendC extension, run the canonical verification script, classify that
result into .verify_status/phase8_attemptN.json, and then let precision_gate.py
consume those artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_ENGINE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ENGINE_DIR.parents[3]

_MISMATCH_RATIO_RE = re.compile(r"mismatch_ratio=([0-9.]+)%")
_MAX_DIFF_RE = re.compile(r"max_abs_diff=([0-9.eE+\-]+|inf|-inf|nan)", re.IGNORECASE)
_CASE_RE = re.compile(r"^case\[(\d+)\]:\s*(.+)$", re.MULTILINE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    prepend = [str(repo_root)]
    archive = repo_root / "archive_tasks"
    if archive.is_dir():
        prepend.insert(0, str(archive))
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = ":".join(prepend + ([old] if old else []))
    return env


def _append_header(path: Path, title: str, cmd: list[str]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {title} start={_now_iso()} =====\n")
        f.write("$ " + " ".join(cmd) + "\n")


def _append_footer(path: Path, title: str, rc: int) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(f"===== {title} end={_now_iso()} rc={rc} =====\n")


def _run_logged(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    title: str,
    timeout: Optional[float],
) -> int:
    _append_header(stdout_path, title, cmd)
    _append_header(stderr_path, title, cmd)
    with stdout_path.open("a", encoding="utf-8") as out, stderr_path.open(
        "a", encoding="utf-8"
    ) as err:
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                stdout=out,
                stderr=err,
                text=True,
                timeout=timeout,
                check=False,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired as exc:
            rc = 124
            err.write(f"\n[engine_validate] timeout after {timeout}s: {exc}\n")
    _append_footer(stdout_path, title, rc)
    _append_footer(stderr_path, title, rc)
    return rc


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_first_json(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        return {}
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _case_stats(stdout_text: str) -> tuple[int, int, float]:
    total = 0
    passed = 0
    for match in _CASE_RE.finditer(stdout_text):
        total += 1
        detail = match.group(2)
        if "matched" in detail:
            passed += 1
    rate = (passed * 100.0 / total) if total else 0.0
    return passed, total, rate


def _validation_metrics(stdout_text: str, exit_code: int) -> dict:
    ratios = [float(x) for x in _MISMATCH_RATIO_RE.findall(stdout_text)]
    max_diffs = _MAX_DIFF_RE.findall(stdout_text)
    passed_cases, total_cases, case_pass_rate = _case_stats(stdout_text)
    if ratios:
        match_rate = max(0.0, 100.0 - (sum(ratios) / len(ratios)))
        source = "avg_mismatch_ratio"
    elif exit_code == 0:
        match_rate = 100.0
        source = "verification_exit_code"
    elif total_cases:
        match_rate = case_pass_rate
        source = "case_pass_rate_fallback"
    else:
        match_rate = 0.0
        source = "no_case_data"
    return {
        "match_rate": f"{match_rate:.2f}",
        "match_rate_source": source,
        "max_diff": max_diffs[-1] if max_diffs else "0.0",
        "passed_cases": passed_cases,
        "total_cases": total_cases,
        "case_pass_rate": round(case_pass_rate, 4),
    }


_ERROR_LINE_RE = re.compile(
    r"(error|fail|mismatch|traceback|exception|assert|❌|not match)", re.IGNORECASE
)


def _extract_first_error(stdout_text: str, *, max_lines: int = 30) -> list[str]:
    """从 stdout 摘出首个错误相关行起的最多 max_lines 行，作为 first_error_lines 指针。

    修复 4a: validation_result 不再内嵌完整 stdout/stderr (数千行)，改存这段紧凑摘要 +
    stdout_path/stderr_path 指针，需要全文时 agent 按指针读盘。无错误标记时回退到
    stdout 末尾 max_lines 行 (失败信息通常在尾部)。
    """
    lines = stdout_text.splitlines()
    if not lines:
        return []
    for i, line in enumerate(lines):
        if _ERROR_LINE_RE.search(line):
            return lines[i:i + max_lines]
    return lines[-max_lines:]


def _write_validation_result(
    task_dir: Path,
    *,
    attempt: int,
    exit_code: int,
    stdout_text: str,
    stderr_text: str,
    stdout_path: Path,
    stderr_path: Path,
) -> Path:
    tuning_dir = task_dir / "precision_tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    metrics = _validation_metrics(stdout_text, exit_code)
    payload = {
        "attempt": attempt,
        "correctness_passed": exit_code == 0,
        "match_rate": metrics["match_rate"],
        "max_diff": metrics["max_diff"],
        "match_rate_source": metrics["match_rate_source"],
        "passed_cases": metrics["passed_cases"],
        "total_cases": metrics["total_cases"],
        "case_pass_rate": metrics["case_pass_rate"],
        "source": "engine_objective_validate",
        # 修复 4a: 删 evaluate_stdout/evaluate_stderr 全文字段，改紧凑摘要 + 路径指针。
        "first_error_lines": _extract_first_error(stdout_text),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    path = tuning_dir / f"validation_result_attempt_{attempt}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_objective_validation(
    task_dir: Path,
    *,
    attempt: int,
    repo_root: Optional[Path] = None,
    timeout: Optional[float] = None,
) -> dict:
    """Run build + verification + classify for phase8 attempt artifacts."""
    task_dir = Path(task_dir).resolve()
    repo_root = Path(repo_root or _REPO_ROOT).resolve()
    logs_dir = task_dir / ".verify_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"phase8_attempt{attempt}.stdout"
    stderr_path = logs_dir / f"phase8_attempt{attempt}.stderr"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    env = _repo_env(repo_root)
    soc_version = env.get("ASCENDC_SOC_VERSION", "Ascend910B3")
    clean_build = env.get("ASCENDC_DEBUG_CLEAN_BUILD", "1") != "0"

    build_cmd = [
        sys.executable,
        str(repo_root / "utils" / "build_ascendc.py"),
        str(task_dir),
        "-v",
        soc_version,
    ]
    if clean_build:
        build_cmd.append("--clean")
    build_rc = _run_logged(
        build_cmd,
        cwd=repo_root,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        title="build_ascendc",
        timeout=timeout,
    )

    verification_ran = False
    verify_rc = build_rc
    if build_rc == 0:
        verification_ran = True
        verify_cmd = [
            sys.executable,
            str(repo_root / "utils" / "verification_ascendc.py"),
            str(task_dir),
        ]
        verify_rc = _run_logged(
            verify_cmd,
            cwd=repo_root,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            title="verification_ascendc",
            timeout=timeout,
        )

    stdout_text = _read(stdout_path)
    stderr_text = _read(stderr_path)
    validation_result_path = _write_validation_result(
        task_dir,
        attempt=attempt,
        exit_code=verify_rc,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    classify_cmd = [
        sys.executable,
        str(repo_root / "utils" / "classify_verify_result.py"),
        "--exit-code",
        str(verify_rc),
        "--stdout-path",
        str(stdout_path),
        "--stderr-path",
        str(stderr_path),
        "--task-dir",
        str(task_dir),
        "--phase",
        "8",
        "--attempt",
        str(attempt),
        "--write-status",
    ]
    classify_proc = subprocess.run(
        classify_cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    status = _extract_first_json(classify_proc.stdout)
    status_path = task_dir / ".verify_status" / f"phase8_attempt{attempt}.json"

    return {
        "success": classify_proc.returncode == 0,
        "build_exit_code": build_rc,
        "verification_exit_code": verify_rc,
        "verification_ran": verification_ran,
        "classify_exit_code": classify_proc.returncode,
        "failure_type": status.get("failure_type"),
        "failed_step": status.get("failed_step"),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "verify_status_path": str(status_path),
        "validation_result_path": str(validation_result_path),
    }


_FORENSICS_SRC_GLOBS = ("*.cpp", "*.h", "*.hpp", "*.py")


def _forensics_input_hash(task_dir: Path) -> str:
    """对取证输入 (kernel/ 下源文件 + model_new_ascendc.py) 算合并内容 hash。

    建议A (6.11 文档 Tier 1): forensics 子进程 (含 OperatorExecutor) 昂贵 (上限 1800s)，
    若本轮 kernel 源码与上轮完全一致 (agent 修复未落盘 / 仅改了无关文件)，重跑取证产出
    必然相同，可直接复用上轮 report。hash 覆盖会影响取证结果的全部源文件: kernel/ 下
    .cpp/.h/.hpp/.py 与 task_dir 根的 model_new_ascendc.py。文件名也并入 hash，使增删
    文件 (而非仅改内容) 同样判为变化。
    """
    h = hashlib.sha256()
    files: list[Path] = []
    kernel_dir = task_dir / "kernel"
    if kernel_dir.is_dir():
        for pat in _FORENSICS_SRC_GLOBS:
            files.extend(kernel_dir.rglob(pat))
    model_new = task_dir / "model_new_ascendc.py"
    if model_new.exists():
        files.append(model_new)
    for f in sorted(set(files), key=lambda p: str(p)):
        try:
            h.update(str(f.relative_to(task_dir)).encode("utf-8"))
            h.update(b"\0")
            h.update(f.read_bytes())
            h.update(b"\0")
        except OSError:
            continue
    return h.hexdigest()


def _forensics_cache_load(task_dir: Path) -> dict:
    path = task_dir / "precision_tuning" / ".forensics_cache.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError):
        return {}


def _forensics_cache_save(task_dir: Path, *, src_hash: str, attempt: int) -> None:
    path = task_dir / "precision_tuning" / ".forensics_cache.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"src_hash": src_hash, "attempt": attempt}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def run_forensics(
    task_dir: Path,
    *,
    attempt: int,
    repo_root: Optional[Path] = None,
    timeout: Optional[float] = None,
) -> dict:
    """Engine-owned: 主动调用 precision_forensics.py 产出 forensics_report。

    修复 1 (6.11 文档): forensics step 此前从不由 engine 编排，磁盘上的 report 是
    agent 在 diagnose 内自发产出、时机不受 Gate-F 控制。此函数让 engine 在 forensics
    step 主动产出 report，使 Gate-F 当轮可读到。

    与 run_objective_validation 对齐: 同样 cwd=repo_root + _repo_env(PYTHONPATH) 以便
    precision_forensics.py 的 _forensics_child 子进程能 import 任务模块。timeout 默认
    None——脚本内部 OperatorExecutor 子进程自带 1800s 上限，外层不再叠加更短的硬杀，
    避免误杀合法长取证后触发无谓重试。
    """
    task_dir = Path(task_dir).resolve()
    repo_root = Path(repo_root or _REPO_ROOT).resolve()
    report = task_dir / "precision_tuning" / f"forensics_report_{attempt}.json"

    # 建议A: staleness 缓存。当前取证输入 (kernel 源码 + model_new) hash 命中上轮缓存，
    # 且上轮 report 仍在 → 复用，省一次昂贵子进程 (上限 1800s)。复用时把上轮 report 复制
    # 成当前 attempt 名 (Gate-F 按 forensics_report_{attempt}.json 精确读取)。
    # env ASCENDC_DEBUG_FORENSICS_NO_CACHE=1 禁用。源码或缓存任一异常 → 回退到重跑 (向
    # 正确性倾斜: 宁可多跑一次，不复用可能过期的取证)。
    use_cache = os.environ.get("ASCENDC_DEBUG_FORENSICS_NO_CACHE", "0") != "1"
    src_hash = _forensics_input_hash(task_dir) if use_cache else ""
    if use_cache and src_hash:
        cache = _forensics_cache_load(task_dir)
        prev_attempt = cache.get("attempt")
        if cache.get("src_hash") == src_hash and prev_attempt is not None:
            prev_report = task_dir / "precision_tuning" / f"forensics_report_{prev_attempt}.json"
            if prev_report.exists():
                if prev_report != report:
                    try:
                        report.write_bytes(prev_report.read_bytes())
                    except OSError:
                        pass
                if report.exists():
                    _forensics_cache_save(task_dir, src_hash=src_hash, attempt=attempt)
                    return {
                        "success": True,
                        "exit_code": 0,
                        "report_path": str(report),
                        "error": None,
                        "cached": True,
                        "cached_from_attempt": prev_attempt,
                    }

    script = repo_root / "skills" / "ascendc" / "ascendc-debug" / "scripts" / "precision_forensics.py"
    # 第一位置参 = task_name (目录名)，不是 op_name；--task-dir 给绝对路径。
    cmd = [sys.executable, str(script), task_dir.name,
           "--attempt", str(attempt), "--task-dir", str(task_dir)]
    env = _repo_env(repo_root)
    try:
        proc = subprocess.run(
            cmd, cwd=repo_root, env=env,
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        exit_code = proc.returncode
        stderr_tail = proc.stderr[-500:] if proc.returncode != 0 else None
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stderr_tail = f"forensics timeout after {timeout}s: {exc}"
    success = exit_code == 0 and report.exists()
    # 重跑成功 → 记录本轮 hash，使下轮 (源码未变时) 命中缓存。
    if success and use_cache and src_hash:
        _forensics_cache_save(task_dir, src_hash=src_hash, attempt=attempt)
    return {
        "success": success,
        "exit_code": exit_code,
        "report_path": str(report) if report.exists() else None,
        "error": stderr_tail,
        "cached": False,
    }
