"""common.py — Gate 通用层。

契约（findings.md §3.3 ② / §7.6）:
  - 反作弊 hash 未破坏
  - AST 退化未引入
  - task_dir 目录结构完整
  - verify_status.json 产出存在且 schema_version 正确
  - {op}.json.bak 未被破坏
  - audit_{attempt}.md 文件存在（audit/fix 步 gating；validate 步只作为诊断）

不检查 audit section 格式；不在 fix 步骤单独加 Gate——由下一轮 Gate-V 通过 verify_status 差分间接验证。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 跨分支共享的 attempt 上限（single source of truth）。
#
# 默认值 5。优先级（高 → 低）:
#   1. Subagent 在 Step 0 根据 failure_type / 错误复杂度动态设置 env var
#      （`export ASCENDC_DEBUG_MAX_ATTEMPTS=<N>` 或 inline `ASCENDC_DEBUG_MAX_ATTEMPTS=<N> python3 ...`）
#   2. Launcher / CI 层通过 `docker exec -e` 注入
#      （见 `utils/run_benchmark_ascendc_codex_with_debug.sh`）
#   3. 本文件默认值 5
#
# 每次 Gate 脚本以新 Python 进程启动时按 env 当场读值，不缓存；因此 subagent
# 在 bash session 里 `export` 后的所有后续 gate 调用都会生效。
# ---------------------------------------------------------------------------
try:
    MAX_ATTEMPTS = int(os.environ.get("ASCENDC_DEBUG_MAX_ATTEMPTS", "5"))
    if MAX_ATTEMPTS < 1:
        MAX_ATTEMPTS = 5
except (TypeError, ValueError):
    MAX_ATTEMPTS = 5

_PROTECTED_PY_FILES = ("model.py", "model_new_ascendc.py", "model_new_tilelang.py")


@dataclass
class GateOutcome:
    gate: str
    ok: bool
    checks: dict
    loop_signal: Optional[str] = None
    reason: Optional[str] = None

    def to_gate_output(self) -> dict:
        out = {"gate": self.gate, "passed": self.ok, "checks": self.checks}
        if self.loop_signal is not None:
            out["loop_signal"] = self.loop_signal
        if self.reason is not None:
            out["loop_reason"] = self.reason
        return out


def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_anticheat(task_dir: Path) -> dict:
    """对 reference/wrapper 文件对比 .bench_baseline/ 的 hash。

    若 .bench_baseline/ 不存在（非 bench 环境），跳过并视为通过。
    """
    baseline_dir = task_dir / ".bench_baseline"
    if not baseline_dir.is_dir():
        return {"anticheat_baseline_present": False, "anticheat_pass": True}
    result = {"anticheat_baseline_present": True}
    hash_keys = []
    for name in _PROTECTED_PY_FILES:
        bp = baseline_dir / name
        cp = task_dir / name
        if not bp.exists():
            continue
        key = f"hash_{name}"
        result[key] = _sha256(bp) == _sha256(cp)
        hash_keys.append(key)
    # 如果没有可比 hash，也视为通过（baseline dir 存在但无文件）
    result["anticheat_pass"] = all(result[k] for k in hash_keys) if hash_keys else True
    return result


_cpp_checker_fn = None
_cpp_checker_loaded = False


def _load_cpp_checker():
    """Lazy-load _check_cpp_regression from scripts/anticheat.py (gates/ 的同级父目录)。

    用 importlib 按文件路径加载，独立于 sys.path。加载失败 (文件缺失/语法错误/无该
    符号) 返回 None → check_cpp_regression fail-open，不把 gate 卡死。
    """
    global _cpp_checker_fn, _cpp_checker_loaded
    if _cpp_checker_loaded:
        return _cpp_checker_fn
    _cpp_checker_loaded = True
    try:
        import importlib.util
        anticheat_path = Path(__file__).resolve().parent.parent / "anticheat.py"
        if not anticheat_path.exists():
            return None
        spec = importlib.util.spec_from_file_location("_anticheat_for_gate", str(anticheat_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _cpp_checker_fn = getattr(module, "_check_cpp_regression", None)
    except Exception:
        _cpp_checker_fn = None
    return _cpp_checker_fn


def check_cpp_regression(task_dir: Path) -> dict:
    """C++ kernel 源码扫描 (复用 anticheat._check_cpp_regression，DRY)。

    检测 kernel/*.{cpp,h} 里 at::/torch:: 算子调用、ATen 头文件、禁用 tensor 计算方法、
    缺 kernel launch (NO_KERNEL_LAUNCH)。这是静态源码扫描 (非 fail-open validator)，
    命中即确证作弊 (绕过 AscendC kernel 在 C++ 层直接调 torch/aten)。

    checker 加载失败 / 扫描异常 → fail-open (cpp_regression_pass=True，errored 标记)，
    不卡死 gate。no_kernel_dir → 中性通过 (无可扫描内容，不确证作弊)。
    """
    fn = _load_cpp_checker()
    if fn is None:
        return {"cpp_checker_present": False, "cpp_regression_pass": True}
    try:
        res = fn(task_dir)
    except Exception:
        return {"cpp_checker_present": True, "cpp_regression_pass": True,
                "cpp_checker_errored": True}
    status = res.get("status")
    return {
        "cpp_checker_present": True,
        "cpp_regression_pass": status != "fail",
        "cpp_violations": len(res.get("violations", [])),
    }


def _find_ast_validator(task_dir: Path) -> Optional[Path]:
    """向上查找 skills/ascendc/ascendc-translator/scripts/validate_ascendc_impl.py。"""
    for cand in [task_dir] + list(task_dir.parents):
        p = cand / "skills" / "ascendc" / "ascendc-translator" / "scripts" / "validate_ascendc_impl.py"
        if p.exists():
            return p
    return None


def check_ast_degrade(task_dir: Path) -> dict:
    """调用 validate_ascendc_impl.py 检测退化（退化子类型 1-4）。缺失或 exit=0 视为通过。"""
    script = _find_ast_validator(task_dir)
    target = task_dir / "model_new_ascendc.py"
    if script is None or not target.exists():
        return {
            "ast_validator_present": script is not None,
            "ast_degrade_pass": True,
        }
    try:
        r = subprocess.run(
            ["python3", str(script), str(target)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {"ast_validator_present": True, "ast_degrade_pass": r.returncode == 0}
    except (subprocess.TimeoutExpired, OSError):
        # 验证器本身出错不能把 gate 卡死 → 视为通过但标记
        return {"ast_validator_present": True, "ast_degrade_pass": True, "ast_validator_errored": True}


def check_structure(task_dir: Path, op_name: str) -> dict:
    bak = task_dir / f"{op_name}.json.bak"
    bak_ok = True if not bak.exists() else bak.stat().st_size > 0
    return {
        "has_kernel_dir": (task_dir / "kernel").is_dir(),
        "has_model_new_ascendc": (task_dir / "model_new_ascendc.py").exists(),
        "json_bak_preserved_if_exists": bak_ok,
    }


def check_verify_status_present(task_dir: Path) -> dict:
    latest = task_dir / ".verify_status" / "latest.json"
    ok = latest.exists()
    schema_ok = False
    if ok:
        try:
            schema_ok = json.loads(latest.read_text()).get("schema_version") == 1
        except Exception:
            schema_ok = False
    return {
        "verify_status_latest_present": ok,
        "verify_status_schema_ok": schema_ok,
    }


def check_audit_file_present(task_dir: Path, attempt: int) -> dict:
    """仅检查文件存在；section schema 由分支层各自负责。"""
    path = task_dir / "precision_tuning" / f"precision_audit_{attempt}.md"
    return {
        "audit_file_present": path.exists(),
        "audit_file_nonempty": path.exists() and path.stat().st_size > 0,
    }


# 这些是"必须为 True 才视为通过"的 gating key；其余为纯诊断信息不参与 ok 判定。
# 设计契约（findings.md §3.3 ② / §7.6）：只有明确反映"前置 / 不变量"失败的 key 才 gating。
#
# 修复4 (6.11 文档 §5.1): anticheat_pass / ast_degrade_pass 已**移出** gating——作弊不再
# 通过 ok=False 硬阻断 (会触发问题3的兜底 + 让 batch1 的 forensics 重试在下一轮把仍带
# 作弊的 kernel 反复判失败而耗尽重试)。改由 run_common 的 validate step 结合 objective
# 结果分场景给 loop_signal (success+作弊→STOP cheat_detected / fail+作弊→CONTINUE)，
# 见下方 run_common。两者保留在 checks 里作纯诊断 + 触发 cheat_history 记录。
_BASE_GATING_KEYS = {
    # structure
    "has_kernel_dir",
    "has_model_new_ascendc",
    "json_bak_preserved_if_exists",
}

_VALIDATE_GATING_KEYS = {
    "verify_status_latest_present",
    "verify_status_schema_ok",
}

_AUDIT_GATING_KEYS = {
    "audit_file_present",
    "audit_file_nonempty",
}


def _gating_keys_for_step(step: str) -> set[str]:
    keys = set(_BASE_GATING_KEYS)
    if step == "validate":
        keys.update(_VALIDATE_GATING_KEYS)
    if step in ("audit", "fix"):
        keys.update(_AUDIT_GATING_KEYS)
    return keys


def _read_correctness_passed(task_dir: Path, attempt: int) -> bool:
    """读 validation_result_attempt_{N}.json 的 correctness_passed；缺失/不可解析→False。

    修复4 A/B 判定靠它区分 objective success/fail。dispatcher 在 validate step 先跑
    run_objective_validation 写好该文件再跑 gate，故此处可读到当轮结果。保守缺省 False
    (读不到时按 objective 未过处理 → 作弊走 CONTINUE 而非误判 success terminal)。
    """
    path = task_dir / "precision_tuning" / f"validation_result_attempt_{attempt}.json"
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("correctness_passed", False))
    except (ValueError, OSError):
        return False


def _record_cheat_attempt(task_dir: Path, attempt: int, checks: dict, *, cheat_type: str) -> None:
    """把作弊轮记入 precision_tuning/cheat_history.json，按 (attempt, cheat_type) 去重。

    修复4 + N5: 同一 attempt 内 forensics/validate 两次 run_common 都会检测到作弊，去重防
    重复记录。AST_VALIDATOR_ERROR (validator 超时/缺失 fail-open) 单独记为 warning，不静默
    吞 (N5)，供 exit_artifacts 区分"真作弊"与"未检测"。
    """
    path = task_dir / "precision_tuning" / "cheat_history.json"
    data = {"cheating_attempts": []}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("cheating_attempts"), list):
                data = loaded
        except (ValueError, OSError):
            pass

    for e in data["cheating_attempts"]:
        if e.get("attempt") == attempt and e.get("cheat_type") == cheat_type:
            return  # 已记录 (去重)

    is_warning = cheat_type == "AST_VALIDATOR_ERROR"
    entry = {
        "attempt": attempt,
        "cheat_type": cheat_type,
        "severity": "warning" if is_warning else "violation",
        "detected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evidence": {
            "ast_degrade_pass": checks.get("ast_degrade_pass"),
            "ast_validator_errored": checks.get("ast_validator_errored", False),
            "anticheat_pass": checks.get("anticheat_pass"),
            "cpp_regression_pass": checks.get("cpp_regression_pass"),
            "cpp_violations": checks.get("cpp_violations"),
        },
        "instruction": (
            "AST validator 异常 (超时/缺失)，本轮反作弊未能确证，按未检测处理"
            if is_warning else
            "禁止用 torch 原生算子绕过 AscendC kernel；只能修改 kernel/ 下的 .cpp/.h"
        ),
    }
    data["cheating_attempts"].append(entry)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def run_common(step: str, task_dir: Path, op_name: str, attempt: int) -> GateOutcome:
    """对给定 step 组合适用的通用检查。返回 GateOutcome。

    - forensics: 结构 / 反作弊 / AST
    - audit:     +audit 文件存在
    - fix:       +audit 文件存在（fix 不单独 Gate，此处复用 audit）
    - validate:  +verify_status；audit 文件只作为诊断信息

    只有当前 step 的 gating keys 参与 ok 判定；其它 (baseline_present / validator_present 等)
    为纯诊断信息。
    """
    checks: dict = {}
    _ablate_ac = os.environ.get("ABLATE_ANTICHEAT") == "1"
    if not _ablate_ac:
        checks.update(check_anticheat(task_dir))
        checks.update(check_ast_degrade(task_dir))
    checks.update(check_structure(task_dir, op_name))
    if step == "validate":
        checks.update(check_verify_status_present(task_dir))
        # C++ 源码扫描只在 validate step 跑: forensics/audit 阶段 kernel 可能仍在构造，
        # NO_KERNEL_LAUNCH 会误报污染 cheat_history。validate 时 kernel 已成型，扫描可信。
        if not _ablate_ac:
            checks.update(check_cpp_regression(task_dir))
    if step in ("audit", "fix", "validate"):
        checks.update(check_audit_file_present(task_dir, attempt))

    ok = all(
        checks.get(k, True) is True
        for k in _gating_keys_for_step(step)
        if k in checks
    )

    # ---- 修复 4 (6.11 文档): anti-cheat 分场景 ----
    # ast_degrade_pass / anticheat_pass 已移出 gating keys (纯诊断)，作弊不再硬阻断
    # step (否则 fail+cheat→CONTINUE 后下一轮 forensics 会被批次1的重试逻辑误判耗尽)。
    # 作弊判定与 A/B 派信号在此集中处理:
    #   - 检测到 ast_degrade fail → 记录 cheat_history (按 (attempt,cheat_type) 去重)。
    #   - validate step 能拿到 objective 结果 (dispatcher 已先跑 run_objective_validation
    #     写好 validation_result)，据此分 A/B:
    #       success + cheat → STOP + stop_reason_code=cheat_detected (靠作弊绕过 kernel 的
    #         「假成功」，终止并标记，不计 clean success；继续修会以作弊态为基础污染状态)。
    #       fail + cheat    → CONTINUE (数值本就没过，作弊只是一次失败尝试；保留诊断成本，
    #         下一轮 prompt 注入警告告知 agent)。
    #   - 非 validate step (forensics/audit) 无 objective 结论 → 仅记录，不在此派终判信号
    #     (保守不阻断，等 validate step 统一裁决)。
    # cheat 路径显式给出 loop_signal (非 None)，规避批次1「validate 无信号 → Abort」误触。
    ast_failed = checks.get("ast_degrade_pass", True) is False
    anticheat_failed = checks.get("anticheat_pass", True) is False
    cpp_failed = checks.get("cpp_regression_pass", True) is False
    if ast_failed or anticheat_failed or cpp_failed:
        if ast_failed:
            # N5: validator 异常 (超时/缺失 fail-open) 与真作弊区分，单独记 warning，
            # 不静默当 pass，也不等价真作弊 (避免误杀)。
            cheat_type = ("AST_VALIDATOR_ERROR"
                          if checks.get("ast_validator_errored") else "AST_DEGRADE")
            _record_cheat_attempt(task_dir, attempt, checks, cheat_type=cheat_type)
        if anticheat_failed:
            _record_cheat_attempt(task_dir, attempt, checks, cheat_type="WRAPPER_HASH")
        if cpp_failed:
            # C++ 扫描是静态确证 (非 fail-open validator)，命中即真作弊，记 violation。
            _record_cheat_attempt(task_dir, attempt, checks, cheat_type="CPP_REGRESSION")

        # validator 异常 (errored) 不等同确证作弊，不据此终止/续跑；按原 ok 判定走。
        # cpp 扫描命中是确证作弊 (静态源码层面无 fail-open 歧义)。
        confirmed_cheat = anticheat_failed or cpp_failed or (
            ast_failed and not checks.get("ast_validator_errored")
        )
        if confirmed_cheat and step == "validate":
            if _read_correctness_passed(task_dir, attempt):
                # stop_reason_code 经 checks 透传 (与 branch 层 _legacy_to_outcome 同路径:
                # GateOutcome 无该字段，to_gate_output 输出 checks，parse_gate_output 再提升)。
                checks["stop_reason_code"] = "cheat_detected"
                return GateOutcome(
                    gate=f"GATE-COMMON-{step}", ok=False, checks=checks,
                    loop_signal="STOP",
                    reason=("objective success 但检测到作弊 (绕过 AscendC kernel)，"
                            "终止并标记 cheat_detected，不计 clean success"),
                )
            return GateOutcome(
                gate=f"GATE-COMMON-{step}", ok=False, checks=checks,
                loop_signal="CONTINUE",
                reason=("检测到作弊但 objective 未通过，已记录 cheat_history，"
                        "下一轮告知 agent"),
            )

    return GateOutcome(gate=f"GATE-COMMON-{step}", ok=ok, checks=checks)
