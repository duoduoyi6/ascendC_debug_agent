#!/usr/bin/env python3
"""Build the evidence-backed V5 ablation closure package."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ARMS = (
    "full",
    "no_kb",
    "no_diagnostic_evidence",
    "no_loopguard",
    "no_anticheat",
    "no_fulleval",
    "baseline",
)
EVIDENCE_CAPABLE_ARMS = {
    "full", "no_kb", "no_diagnostic_evidence", "no_loopguard",
}
PROVIDER_ERROR_RE = re.compile(
    r"\b(?:400|401|403|429|5\d\d)\b|refusal|context[_ -]?limit|"
    r"output[_ -]?limit|max[_ -]?turn",
    re.I,
)


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def claude_result_paths(task: Path) -> list[Path]:
    archive = sorted(
        (task / "precision_tuning" / "claude_results").glob("*.json")
    )
    if archive:
        return archive
    return sorted(task.glob("_claude_result_attempt*.json"))


def summarize_claude_cost(paths: Iterable[Path]) -> dict[str, Any]:
    totals: dict[str, float] = {
        "turns": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "cost_usd": 0.0,
    }
    models: set[str] = set()
    provider_errors: list[str] = []
    result_count = 0
    for path in paths:
        payload = _load_json(path, {})
        if not isinstance(payload, dict):
            continue
        result_count += 1
        totals["turns"] += _number(payload.get("num_turns"))
        usage = payload.get("modelUsage")
        usage_cost = 0.0
        if isinstance(usage, dict):
            for model, record in usage.items():
                if isinstance(record, dict):
                    models.add(str(model))
                    totals["input_tokens"] += _number(record.get("inputTokens"))
                    totals["output_tokens"] += _number(record.get("outputTokens"))
                    totals["cache_creation_tokens"] += _number(
                        record.get("cacheCreationInputTokens"))
                    totals["cache_read_tokens"] += _number(
                        record.get("cacheReadInputTokens"))
                    usage_cost += _number(record.get("costUSD"))
        top_cost = payload.get("total_cost_usd")
        totals["cost_usd"] += (
            _number(top_cost) if top_cost is not None else usage_cost
        )
        error_text = json.dumps({
            key: payload.get(key)
            for key in (
                "api_error_status", "is_error", "terminal_reason",
                "claude_state", "error", "subtype",
            )
            if key in payload
        }, ensure_ascii=False)
        provider_errors.extend(
            match.group(0).lower()
            for match in PROVIDER_ERROR_RE.finditer(error_text)
        )
    totals["turns"] = int(totals["turns"])
    for key in (
        "input_tokens", "output_tokens",
        "cache_creation_tokens", "cache_read_tokens",
    ):
        totals[key] = int(totals[key])
    totals["total_tokens"] = sum(
        totals[key]
        for key in (
            "input_tokens", "output_tokens",
            "cache_creation_tokens", "cache_read_tokens",
        )
    )
    totals["cost_usd"] = round(float(totals["cost_usd"]), 8)
    totals["result_count"] = result_count
    totals["models"] = sorted(models)
    totals["provider_errors"] = sorted(provider_errors)
    return totals


def _coverage_map(root: Path) -> dict[str, dict[str, Any]]:
    audit = _load_json(
        root / "experiment_control" / "dataset_audit.json", {})
    return {
        str(row.get("task")): row.get("full_eval_coverage") or {}
        for row in audit.get("tasks", [])
        if isinstance(row, dict) and row.get("task")
    }


def _posthoc_map(root: Path, arm: str) -> dict[str, dict[str, Any]]:
    payload = _load_json(
        root / "posthoc" / f"arm_{arm}" / "posthoc_results.json", {})
    return {
        str(row.get("task")): row
        for row in payload.get("rows", [])
        if isinstance(row, dict) and row.get("task")
    }


def _full_eval_evidence(task: Path, applicable: bool) -> tuple[bool, str]:
    if not applicable:
        return True, "not_applicable_or_coverage_equivalent"
    paths = sorted(
        (task / "precision_tuning").glob(
            "validation_result_attempt_*_full.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not paths:
        return False, "missing"
    payload = _load_json(paths[-1], {})
    passed = bool(
        payload.get("correctness_passed")
        or payload.get("coverage_equivalent")
    )
    return passed, paths[-1].name


def _observability(task: Path, status: dict[str, Any]) -> dict[str, bool]:
    tuning = task / "precision_tuning"
    attempts = int(_number(status.get("attempts_used")))
    return {
        "status": (task / "debug_status.json").is_file(),
        "events": (task / ".debug_events" / "events.jsonl").is_file(),
        "run_summary": (task / "run_summary.json").is_file(),
        "claude_results": bool(claude_result_paths(task)),
        "probe": bool(list(tuning.glob("probe_*attempt*.json"))) or attempts == 0,
        "direction": bool(list(tuning.glob("direction_*attempt*.json"))) or attempts == 0,
        "kb": bool(list(tuning.glob("*knowledge*.json"))) or attempts == 0,
        "forensics": bool(list(tuning.glob("forensics_report_*.json"))) or attempts == 0,
        "checkpoint": (
            (tuning / "checkpoints").exists()
            or (tuning / "best_checkpoint.json").exists()
            or attempts == 0
        ),
        "rollback": (
            bool(list(tuning.glob("*rollback*.json")))
            or status.get("objective_success") is True
            or attempts == 0
        ),
    }


def task_row(
    root: Path,
    arm: str,
    task: Path,
    *,
    coverage: dict[str, dict[str, Any]],
    posthoc: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rel = str(task.relative_to(root / f"arm_{arm}" / "tasks"))
    status = _load_json(task / "debug_status.json", {})
    run_summary = _load_json(task / "run_summary.json", {})
    cost = summarize_claude_cost(claude_result_paths(task))
    coverage_row = coverage.get(rel, {})
    full_pass, full_source = _full_eval_evidence(
        task, bool(coverage_row.get("full_eval_applicable")))
    posthoc_row = posthoc.get(rel, {})
    objective = status.get("objective_success") is True
    reportable = status.get("reportable_success") is True
    anti = status.get("anti_cheat_pass") is True
    ast = status.get("ast_degrade_pass") is True
    evidence_backed = (
        arm in EVIDENCE_CAPABLE_ARMS
        and objective and reportable and anti and ast and full_pass
    )
    row: dict[str, Any] = {
        "arm": arm,
        "task": rel,
        "session_outcome": status.get("session_outcome"),
        "entry_failure_type": status.get("entry_failure_type"),
        "final_failure_type": status.get("final_failure_type"),
        "attempts_used": int(_number(status.get("attempts_used"))),
        "objective_success": objective,
        "reportable_success": reportable,
        "anti_cheat_pass": anti,
        "ast_degrade_pass": ast,
        "full_eval_applicable": bool(
            coverage_row.get("full_eval_applicable")),
        "full_eval_pass": full_pass,
        "full_eval_evidence": full_source,
        "evidence_backed_success": evidence_backed,
        "posthoc_clean_success": (
            posthoc_row.get("posthoc_clean_success") is True),
        "posthoc_build_pass": (
            (posthoc_row.get("build") or {}).get("passed") is True),
        "notes": str(status.get("notes") or ""),
        "observability": _observability(task, status),
        **cost,
    }
    if isinstance(run_summary, dict):
        row["no_improvement_signal"] = (
            (run_summary.get("gate") or {}).get("final_loop_signal"))
    return row


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def classify_long_failures(rows: list[dict[str, Any]]) -> None:
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        thresholds = {
            key: percentile(
                [float(row[key]) for row in arm_rows if float(row[key]) > 0],
                0.75,
            )
            for key in ("turns", "total_tokens", "cost_usd")
        }
        for row in arm_rows:
            components = [
                key for key, threshold in thresholds.items()
                if threshold > 0 and float(row[key]) >= threshold
            ]
            row["long_failure_components"] = components
            row["long_failure"] = (
                not row["posthoc_clean_success"] and len(components) >= 2
            )


def mcnemar_exact(full: list[bool], treatment: list[bool]) -> dict[str, Any]:
    b = sum(left and not right for left, right in zip(full, treatment))
    c = sum(not left and right for left, right in zip(full, treatment))
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index) * (0.5 ** discordant)
            for index in range(0, min(b, c) + 1)
        )
        p_value = min(1.0, 2.0 * tail)
    return {
        "full_only": b,
        "treatment_only": c,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def paired_bootstrap_ci(
    full: list[bool],
    treatment: list[bool],
    *,
    samples: int = 10000,
    seed: int = 20260724,
) -> dict[str, float]:
    if not full:
        return {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    diffs = []
    n = len(full)
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(
            int(treatment[index]) - int(full[index]) for index in indices
        ) / n)
    return {
        "difference": (
            sum(map(int, treatment)) - sum(map(int, full))) / n,
        "ci_low": percentile(diffs, 0.025),
        "ci_high": percentile(diffs, 0.975),
    }


def _manifest_compliance(
    root: Path, arm: str, task_count: int, posthoc_count: int
) -> dict[str, Any]:
    frozen = _load_json(
        root / "experiment_control" / "arm_manifests" / f"arm_{arm}.json", {})
    runtime = _load_json(root / f"arm_{arm}" / "experiment_manifest.json", {})
    mapping = {
        "task_count": len(runtime.get("targets") or []),
        "containers": str(runtime.get("containers") or "").split(","),
        "npus": [
            int(value) for value in str(runtime.get("npus") or "").split(",")
            if value
        ],
        "max_attempts": runtime.get("max_attempts"),
        "max_turns": int(runtime.get("max_turns") or 0),
        "soft_task_turns": runtime.get("soft_task_turns"),
        "max_task_turns": runtime.get("max_task_turns"),
        "timeout": runtime.get("timeout_sec"),
        "agent": runtime.get("agent"),
        "entry_failure_type": runtime.get("entry_failure_type"),
        "arm": runtime.get("ablate_profile"),
        "kb_path": runtime.get("kb_path"),
        "model": runtime.get("model"),
        "provider_assignment_mode": runtime.get("provider_assignment_mode"),
        "provider_names": runtime.get("providers"),
        "provider_env_storage": runtime.get("provider_env_storage"),
        "usage_query_enabled": runtime.get("usage_query_enabled"),
        "mixed_provider_enabled": runtime.get("mixed_provider_enabled"),
        "kb_read_only": runtime.get("kb_read_only"),
    }
    mismatches = {
        key: {"expected": frozen.get(key), "actual": value}
        for key, value in mapping.items()
        if frozen.get(key) != value
    }
    task_root = root / f"arm_{arm}" / "tasks"
    prohibited: list[str] = []
    if arm in {"no_kb", "no_diagnostic_evidence", "baseline"}:
        for path in task_root.glob("level*/*/precision_tuning/*knowledge*.json"):
            prohibited.append(str(path.relative_to(root)))
    if arm in {"no_diagnostic_evidence", "baseline"}:
        for path in task_root.glob("level*/*/precision_tuning/forensics_report_*.json"):
            prohibited.append(str(path.relative_to(root)))
    if arm in {"no_fulleval", "baseline"}:
        for path in task_root.glob(
            "level*/*/precision_tuning/validation_result_attempt_*_full.json"
        ):
            prohibited.append(str(path.relative_to(root)))
    if arm == "no_anticheat":
        for path in task_root.glob("level*/*/_anticheat.json"):
            prohibited.append(str(path.relative_to(root)))
    complete = (
        task_count == int(frozen.get("task_count") or 0)
        and posthoc_count == int(frozen.get("task_count") or 0)
    )
    return {
        "arm": arm,
        "passed": complete and not mismatches and not prohibited,
        "terminal_task_count": task_count,
        "posthoc_task_count": posthoc_count,
        "expected_task_count": frozen.get("task_count"),
        "manifest_mismatches": mismatches,
        "prohibited_artifacts": prohibited,
    }


def _failed_cycles(root: Path) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        for path in sorted(
            (root / f"arm_{arm}" / "retry_evidence").glob(
                "**/archive_manifest.json")
        ):
            payload = _load_json(path, {})
            rows.append({
                "arm": arm,
                "archive": str(path.relative_to(root)),
                "task": payload.get("target") or path.parent.parent.name,
                "reason": payload.get("reason"),
                "excluded_from_primary_cost": (
                    (payload.get("cost_accounting") or {}).get(
                        "excluded_from_primary_cost") is True
                ),
            })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _arm_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        ebs = sum(row["evidence_backed_success"] for row in selected)
        output.append({
            "arm": arm,
            "tasks": len(selected),
            "objective_success": sum(row["objective_success"] for row in selected),
            "reportable_success": sum(row["reportable_success"] for row in selected),
            "evidence_backed_success": ebs,
            "posthoc_clean_success": sum(
                row["posthoc_clean_success"] for row in selected),
            "long_failures": sum(row["long_failure"] for row in selected),
            "final_cycle_turns": sum(row["turns"] for row in selected),
            "final_cycle_tokens": sum(row["total_tokens"] for row in selected),
            "final_cycle_cost_usd": round(
                sum(row["cost_usd"] for row in selected), 8),
            "tokens_per_ebs": (
                round(sum(row["total_tokens"] for row in selected) / ebs, 2)
                if ebs else None
            ),
            "cost_per_ebs_usd": (
                round(sum(row["cost_usd"] for row in selected) / ebs, 8)
                if ebs else None
            ),
        })
    return output


def _paired_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_arm = {
        arm: {row["task"]: row for row in rows if row["arm"] == arm}
        for arm in ARMS
    }
    common_posthoc = set.intersection(*[
        {task for task, row in by_arm[arm].items()
         if row["posthoc_clean_success"]}
        for arm in ARMS
    ])
    common_ebs = set.intersection(*[
        {task for task, row in by_arm[arm].items()
         if row["evidence_backed_success"]}
        for arm in sorted(EVIDENCE_CAPABLE_ARMS)
    ])
    paired = []
    for scope, tasks, arms in (
        ("common_posthoc_all_arms", common_posthoc, ARMS),
        ("common_ebs_evidence_arms", common_ebs, sorted(EVIDENCE_CAPABLE_ARMS)),
    ):
        for arm in arms:
            selected = [by_arm[arm][task] for task in sorted(tasks)]
            paired.append({
                "scope": scope,
                "arm": arm,
                "task_count": len(selected),
                "turns": sum(row["turns"] for row in selected),
                "tokens": sum(row["total_tokens"] for row in selected),
                "cost_usd": round(sum(row["cost_usd"] for row in selected), 8),
                "tokens_per_task": (
                    round(sum(row["total_tokens"] for row in selected) / len(selected), 2)
                    if selected else None
                ),
                "cost_per_task_usd": (
                    round(sum(row["cost_usd"] for row in selected) / len(selected), 8)
                    if selected else None
                ),
            })
    return paired, {
        "common_posthoc_tasks": sorted(common_posthoc),
        "common_evidence_backed_tasks": sorted(common_ebs),
    }


def _paired_success_tests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arm = {
        arm: {row["task"]: row for row in rows if row["arm"] == arm}
        for arm in ARMS
    }
    tasks = sorted(set.intersection(*[
        set(by_arm[arm]) for arm in ARMS
    ]))
    output = []
    for metric in (
        "objective_success", "reportable_success", "posthoc_clean_success",
    ):
        full = [bool(by_arm["full"][task][metric]) for task in tasks]
        for arm in ARMS[1:]:
            treatment = [bool(by_arm[arm][task][metric]) for task in tasks]
            output.append({
                "metric": metric,
                "arm": arm,
                **mcnemar_exact(full, treatment),
                **paired_bootstrap_ci(full, treatment),
            })
    return output


def _write_report(
    output: Path,
    summaries: list[dict[str, Any]],
    compliance: list[dict[str, Any]],
    intersections: dict[str, Any],
    failed_cycles: list[dict[str, Any]],
) -> None:
    lines = [
        "# V5 Ablation Analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Primary cost includes only each operator's final valid supervisor cycle. "
        "Archived retry cycles are reported separately and are not summed.",
        "",
        "## Arm Summary",
        "",
        "| Arm | OSR | WRSR | EBSR | Post-hoc clean | Long failures | Tokens | Cost USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        n = row["tasks"]
        lines.append(
            f"| {row['arm']} | {row['objective_success']}/{n} | "
            f"{row['reportable_success']}/{n} | "
            f"{row['evidence_backed_success']}/{n} | "
            f"{row['posthoc_clean_success']}/{n} | "
            f"{row['long_failures']} | {row['final_cycle_tokens']} | "
            f"{row['final_cycle_cost_usd']:.6f} |"
        )
    lines.extend([
        "",
        "## Paired Cost Sets",
        "",
        f"- Common post-hoc clean tasks across all arms: "
        f"{len(intersections['common_posthoc_tasks'])}.",
        f"- Common evidence-backed tasks across evidence-capable arms: "
        f"{len(intersections['common_evidence_backed_tasks'])}.",
        "",
        "## Contract Closure",
        "",
    ])
    for row in compliance:
        lines.append(
            f"- `{row['arm']}`: {'PASS' if row['passed'] else 'FAIL'}; "
            f"terminal={row['terminal_task_count']}, "
            f"posthoc={row['posthoc_task_count']}."
        )
    lines.extend([
        "",
        "## Failed Provider Cycles",
        "",
        f"{len(failed_cycles)} archived cycles are listed in `failed_cycles.csv`; "
        "their cost is intentionally excluded from primary aggregates.",
        "",
        "See `per_task.csv`, `paired_success_tests.json`, "
        "`paired_success_cost.csv`, `long_failures.csv`, and "
        "`arm_compliance.json` for auditable evidence.",
    ])
    (output / "V5_ANALYSIS_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=27)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    coverage = _coverage_map(args.experiment_root)
    rows: list[dict[str, Any]] = []
    compliance = []
    for arm in ARMS:
        task_root = args.experiment_root / f"arm_{arm}" / "tasks"
        tasks = sorted(path for path in task_root.glob("level*/*") if path.is_dir())
        posthoc = _posthoc_map(args.experiment_root, arm)
        rows.extend(task_row(
            args.experiment_root,
            arm,
            task,
            coverage=coverage,
            posthoc=posthoc,
        ) for task in tasks)
        compliance.append(_manifest_compliance(
            args.experiment_root, arm, len(tasks), len(posthoc)))

    classify_long_failures(rows)
    summaries = _arm_summary(rows)
    paired, intersections = _paired_rows(rows)
    success_tests = _paired_success_tests(rows) if rows else []
    failed_cycles = _failed_cycles(args.experiment_root)
    long_failures = [row for row in rows if row["long_failure"]]

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_cost_scope": "final_valid_cycle_only",
        "failed_cycles_excluded_from_primary_cost": True,
        "arm_summary": summaries,
        "paired_intersections": intersections,
        "paired_success_tests": success_tests,
        "compliance": compliance,
        "rows": rows,
    }
    (args.output / "v5_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (args.output / "paired_success_tests.json").write_text(
        json.dumps(success_tests, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (args.output / "arm_compliance.json").write_text(
        json.dumps(compliance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    _write_csv(
        args.output / "arm_summary.csv",
        summaries,
        list(summaries[0]) if summaries else ["arm"],
    )
    per_task_fields = [
        "arm", "task", "session_outcome", "entry_failure_type",
        "final_failure_type", "attempts_used", "objective_success",
        "reportable_success", "evidence_backed_success",
        "posthoc_clean_success", "full_eval_pass", "turns",
        "input_tokens", "output_tokens", "cache_creation_tokens",
        "cache_read_tokens", "total_tokens", "cost_usd",
        "long_failure", "long_failure_components", "models",
        "provider_errors", "notes",
    ]
    _write_csv(args.output / "per_task.csv", rows, per_task_fields)
    _write_csv(
        args.output / "paired_success_cost.csv",
        paired,
        list(paired[0]) if paired else ["scope", "arm"],
    )
    _write_csv(
        args.output / "long_failures.csv",
        long_failures,
        per_task_fields,
    )
    _write_csv(
        args.output / "failed_cycles.csv",
        failed_cycles,
        ["arm", "task", "reason", "excluded_from_primary_cost", "archive"],
    )
    _write_report(
        args.output, summaries, compliance, intersections, failed_cycles)

    complete = (
        all(row["passed"] for row in compliance)
        and all(row["tasks"] == args.expected_tasks for row in summaries)
    )
    return 0 if complete or args.allow_incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
