#!/usr/bin/env python3
"""Summarize generation/debug bench outputs by level, op and category."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return None


def read_text(path: Path, limit: int = 200_000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    if len(text) > limit:
        return text[-limit:]
    return text


def read_claude_result(task_dir: Path) -> dict[str, Any] | None:
    candidates = [task_dir / "_claude_result.json"]
    candidates.extend(sorted(task_dir.glob("_claude_result_*.json")))
    latest: dict[str, Any] | None = None
    for path in candidates:
        payload = read_json(path)
        if payload:
            latest = payload
    return latest


def parse_case_name(path: Path) -> tuple[int | None, str]:
    match = re.match(r"^(\d+)_(.+)$", path.name)
    if not match:
        return None, path.name
    return int(match.group(1)), match.group(2)


def load_manifest(path: Path | None) -> dict[tuple[int, int], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    if not payload:
        return {}
    index: dict[tuple[int, int], dict[str, Any]] = {}
    for entry in payload.get("cases", []):
        level_num = entry.get("level_num")
        case_id = entry.get("case_id")
        if isinstance(level_num, int) and isinstance(case_id, int):
            index[(level_num, case_id)] = entry
    return index


def parse_batch_report(path: Path) -> dict[int, dict[str, str]]:
    text = read_text(path)
    rows: dict[int, dict[str, str]] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 5 or not cols[0].isdigit():
            continue
        case_id = int(cols[0])
        status_text = cols[2]
        status_lower = status_text.lower()
        if "成功" in status_text or "success" in status_lower:
            status = "runner_success"
        elif "超时" in status_text or "timeout" in status_lower:
            status = "runner_timeout"
        elif "失败" in status_text or "failed" in status_lower or "rc=" in status_text:
            status = "runner_failed"
        else:
            status = "runner_unknown"
        rows[case_id] = {
            "runner_status": status,
            "runner_status_text": status_text,
            "elapsed_sec": cols[3],
            "worker": cols[4],
        }
    return rows


def parse_worker_logs(run_dir: Path) -> dict[str, str]:
    """Map task directory basename to the runner log segment for that task."""
    segments_by_task: dict[str, str] = {}
    for log_path in sorted(run_dir.glob("worker_*.log")):
        text = read_text(log_path, limit=2_000_000)
        if "[task] id=" in text:
            parts = re.split(r"(?=^\[task\] id=)", text, flags=re.MULTILINE)
        else:
            parts = re.split(r"(?=^OpenAI Codex v)", text, flags=re.MULTILINE)
        for part in parts:
            if not part.strip():
                continue
            match = re.search(r"output_dir=([^,\s]+/(\d+_[^/\s]+)/?)", part)
            if match:
                segments_by_task[match.group(2)] = part
    return segments_by_task


def infer_generation_failure(text: str, runner_status: str) -> str:
    lower = text.lower()
    if runner_status == "runner_success":
        return "success"
    if "invalid authentication" in lower or "unauthorized" in lower or "unexpected status 401" in lower:
        return "provider_auth_failed"
    if "model_not_found" in lower or "no available channel for model" in lower:
        return "provider_model_not_found"
    if "access_terminated_error" in lower:
        return "provider_access_terminated"
    if "max budget" in lower or "budget exceeded" in lower or "maximum dollar amount" in lower:
        return "provider_budget_exceeded"
    if "stale_after_failure" in lower:
        return "stale_after_failure"
    if "api_error_status" in lower and "null" not in lower:
        return "provider_api_error"
    if '"is_error": true' in lower or '"is_error":true' in lower:
        return "claude_error"
    if "invalid_claude_result" in lower:
        return "invalid_claude_result"
    if '"stop_reason": "pause_turn"' in lower or '"stop_reason":"pause_turn"' in lower or "claude_pause_turn" in lower:
        return "claude_pause_turn"
    if "invalid function arguments json string" in lower:
        return "provider_tool_call_bad_request"
    if "unexpected status 400 bad request" in lower:
        return "provider_bad_request"
    if "unexpected status 404" in lower and "/responses" in lower:
        return "provider_responses_api_unsupported"
    if "unexpected status 404" in lower:
        return "provider_endpoint_not_found"
    if "unexpected status 429" in lower or "rate limit" in lower:
        return "provider_rate_limited"
    if "reconnecting... 10/10" in lower and "error:" in lower:
        return "provider_reconnect_failed"
    if "no last agent message" in lower:
        return "codex_no_last_message"
    if runner_status == "runner_timeout":
        return "timeout"
    if runner_status == "runner_failed":
        if "claude" in lower or "_claude_result" in lower:
            return "claude_exec_failed"
        return "codex_exec_failed"
    if runner_status == "runner_no_kernel_output":
        return "missing_kernel_output"
    return "unknown"


def infer_failure_from_text(text: str) -> str:
    lower = text.lower()
    if not lower:
        return "unknown"
    if "timeout" in lower or "timed out" in lower:
        return "timeout"
    if "cheat" in lower or "反作弊" in text:
        return "cheat"
    if "mismatch_ratio=" in lower or "max_abs_diff=" in lower or "precision" in lower:
        return "precision_failed"
    if "aicore exception" in lower or "acl stream synchronize failed" in lower or "kernel task happen error" in lower:
        return "runtime_error"
    if "undefined reference" in lower or "fatal error:" in lower or "error:" in lower and ("compile" in lower or "build" in lower):
        return "build_failed"
    if "importerror" in lower or "modulenotfounderror" in lower or "cannot open shared object file" in lower:
        return "import_failed"
    if "pass" in lower or "all cases passed" in lower:
        return "success"
    return "unknown"


def discover_verify_failure(
    task_dir: Path,
    runner_status: str,
) -> tuple[str, str | None, str, str, dict[str, Any] | None]:
    """Return raw, inferred, final failure type, source, and structured status.

    raw reflects whether the canonical .verify_status/latest.json exists.
    final preserves structured status when available, otherwise uses a best-effort
    inference so legacy outputs do not collapse into missing_verify_status.
    """
    latest_path = task_dir / ".verify_status" / "latest.json"
    latest = read_json(latest_path)
    if latest:
        failure_type = latest.get("failure_type") or latest.get("status") or "unknown"
        return failure_type, None, failure_type, "verify_status", latest

    raw_failure = "invalid_verify_status" if latest_path.exists() else "missing_verify_status"

    verify_logs = task_dir / ".verify_logs"
    if verify_logs.is_dir():
        chunks: list[str] = []
        for path in sorted(verify_logs.glob("*"))[-6:]:
            if path.is_file():
                chunks.append(read_text(path, limit=30_000))
        if chunks:
            inferred = infer_failure_from_text("\n".join(chunks))
            if inferred != "unknown":
                return raw_failure, inferred, inferred, "verify_logs", None

    combined_parts = [
        read_text(task_dir / name, limit=60_000)
        for name in ("_codex_last.txt", "_claude_result.json", "trace.md", "debug_trace.md")
    ]
    combined_parts.extend(read_text(path, limit=60_000) for path in sorted(task_dir.glob("_claude_result_*.json")))
    combined = "\n".join(combined_parts)
    inferred = infer_failure_from_text(combined)
    if inferred != "unknown":
        return raw_failure, inferred, inferred, "task_artifacts", None

    if runner_status == "runner_success":
        inferred = "legacy_success_without_verify_status"
        return raw_failure, inferred, inferred, "runner_success_without_verify_status", None

    return raw_failure, None, raw_failure, "missing", None


def discover_task_dirs(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []
    dirs = [p for p in run_dir.iterdir() if p.is_dir() and re.match(r"^\d+_", p.name)]
    return sorted(dirs, key=lambda p: (parse_case_name(p)[0] or 10**9, p.name))


def has_kernel_outputs(task_dir: Path) -> bool:
    if (task_dir / "model_new_ascendc.py").is_file():
        return True
    kernel_dir = task_dir / "kernel"
    return kernel_dir.is_dir() and any(p.is_file() for p in kernel_dir.rglob("*"))


def summarize_task(
    task_dir: Path,
    level: int,
    manifest: dict[tuple[int, int], dict[str, Any]],
    report_rows: dict[int, dict[str, str]],
    worker_logs: dict[str, str],
) -> dict[str, Any]:
    case_id, parsed_op_name = parse_case_name(task_dir)
    manifest_entry = manifest.get((level, case_id or -1), {})
    op_name = manifest_entry.get("op_name") or parsed_op_name
    category = manifest_entry.get("category") or "unknown"

    anticheat = read_json(task_dir / "_anticheat.json") or {}
    anticheat_verdict = anticheat.get("verdict") or "UNKNOWN"
    anticheat_reasons = anticheat.get("reasons") or []
    output_present = has_kernel_outputs(task_dir)

    debug_status = read_json(task_dir / "debug_status.json") or {}
    debug_attempts = debug_status.get("attempts_used") or debug_status.get("attempt") or None

    report_row = report_rows.get(case_id or -1, {})
    runner_status = report_row.get("runner_status")
    if not runner_status:
        if output_present:
            runner_status = "runner_output_present"
        elif (task_dir / "_codex_last.txt").exists() or (task_dir / "_claude_result.json").exists():
            runner_status = "runner_no_kernel_output"
        else:
            runner_status = "runner_unknown"

    raw_verify_failure, inferred_verify_failure, verify_failure, verify_source, verify_status = discover_verify_failure(
        task_dir,
        runner_status,
    )
    debug_outcome = (
        debug_status.get("session_outcome")
        or debug_status.get("phase8_outcome")
        or ("success" if verify_failure == "success" else "not_debugged")
    )
    if verify_failure == "success":
        debug_outcome = "success"

    if not output_present and anticheat_verdict == "CLEAN":
        anticheat_verdict = "NO_OUTPUT"
        anticheat_reasons = ["no generated model_new_ascendc.py or kernel outputs"]

    generation_log_parts = [
        report_row.get("runner_status_text", ""),
        worker_logs.get(task_dir.name, ""),
        read_text(task_dir / "_codex_last.txt", limit=100_000),
        read_text(task_dir / "_claude_result.json", limit=100_000),
    ]
    generation_log_parts.extend(read_text(path, limit=100_000) for path in sorted(task_dir.glob("_claude_result_*.json")))
    generation_log = "\n".join(generation_log_parts)
    if output_present and verify_failure == "success":
        generation_failure = "success"
    else:
        generation_failure = infer_generation_failure(generation_log, runner_status)
    debug_passed = verify_failure == "success" or debug_outcome == "success"
    include_in_pass_rate = anticheat_verdict != "CHEAT" and verify_failure != "legacy_success_without_verify_status"
    if anticheat_verdict == "CHEAT":
        evaluation_status = "excluded_cheat"
    elif verify_failure == "legacy_success_without_verify_status":
        evaluation_status = "legacy_unevaluable"
    else:
        evaluation_status = "evaluated"
    claude_result = read_claude_result(task_dir) or {}
    if anticheat_verdict == "CHEAT":
        case_status = "cheat"
    elif runner_status == "runner_timeout":
        case_status = "timeout"
    elif verify_failure == "legacy_success_without_verify_status":
        case_status = "legacy_success_without_verify_status"
    elif debug_passed:
        case_status = "passed"
    elif generation_failure != "unknown":
        case_status = generation_failure
    elif verify_failure != "unknown":
        case_status = verify_failure
    else:
        case_status = runner_status

    return {
        "schema_version": SCHEMA_VERSION,
        "level": f"level{level}",
        "level_num": level,
        "case_id": case_id,
        "op_name": op_name,
        "category": category,
        "task_dir": str(task_dir),
        "runner_status": runner_status,
        "generation_failure_type": generation_failure,
        "runner_status_text": report_row.get("runner_status_text"),
        "elapsed_sec": report_row.get("elapsed_sec"),
        "worker": report_row.get("worker"),
        "has_kernel_outputs": output_present,
        "anticheat_verdict": anticheat_verdict,
        "anticheat_reasons": anticheat_reasons,
        "raw_verify_failure_type": raw_verify_failure,
        "inferred_verify_failure_type": inferred_verify_failure,
        "verify_failure_type": verify_failure,
        "verify_status_source": verify_source,
        "verify_status_path": str(task_dir / ".verify_status" / "latest.json") if verify_status else None,
        "evaluation_status": evaluation_status,
        "include_in_pass_rate": include_in_pass_rate,
        "debug_outcome": debug_outcome,
        "debug_attempts": debug_attempts,
        "debug_passed": debug_passed,
        "claude_stop_reason": claude_result.get("stop_reason"),
        "claude_terminal_reason": claude_result.get("terminal_reason"),
        "claude_turns": claude_result.get("num_turns"),
        "claude_cost_usd": claude_result.get("total_cost_usd"),
        "case_status": case_status,
    }


def counter_dict(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "unknown") for row in rows).items()))


def category_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("category") or "unknown")].append(row)

    summary: list[dict[str, Any]] = []
    for category, items in sorted(grouped.items()):
        total = len(items)
        cheats = sum(1 for item in items if item.get("anticheat_verdict") == "CHEAT")
        unevaluable = sum(1 for item in items if item.get("evaluation_status") == "legacy_unevaluable")
        valid_total = sum(1 for item in items if item.get("include_in_pass_rate"))
        passed = sum(1 for item in items if item.get("debug_passed") and item.get("include_in_pass_rate"))
        summary.append(
            {
                "category": category,
                "total": total,
                "valid_total": valid_total,
                "debug_passed": passed,
                "debug_pass_rate": round(passed / valid_total, 4) if valid_total else None,
                "cheat": cheats,
                "legacy_unevaluable": unevaluable,
                "failure_types": counter_dict(items, "verify_failure_type"),
                "case_status": counter_dict(items, "case_status"),
            }
        )
    return summary


def build_summary(rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    total = len(rows)
    cheats = sum(1 for row in rows if row.get("anticheat_verdict") == "CHEAT")
    legacy_unevaluable = sum(1 for row in rows if row.get("evaluation_status") == "legacy_unevaluable")
    valid_total = sum(1 for row in rows if row.get("include_in_pass_rate"))
    passed = sum(1 for row in rows if row.get("debug_passed") and row.get("include_in_pass_rate"))
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "total": total,
        "valid_total": valid_total,
        "debug_passed": passed,
        "debug_pass_rate": round(passed / valid_total, 4) if valid_total else None,
        "excluded": {
            "cheat": cheats,
            "legacy_unevaluable": legacy_unevaluable,
        },
        "counts": {
            "runner_status": counter_dict(rows, "runner_status"),
            "generation_failure_type": counter_dict(rows, "generation_failure_type"),
            "anticheat_verdict": counter_dict(rows, "anticheat_verdict"),
            "raw_verify_failure_type": counter_dict(rows, "raw_verify_failure_type"),
            "inferred_verify_failure_type": counter_dict(rows, "inferred_verify_failure_type"),
            "verify_failure_type": counter_dict(rows, "verify_failure_type"),
            "verify_status_source": counter_dict(rows, "verify_status_source"),
            "evaluation_status": counter_dict(rows, "evaluation_status"),
            "debug_outcome": counter_dict(rows, "debug_outcome"),
            "case_status": counter_dict(rows, "case_status"),
        },
        "by_category": category_summary(rows),
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join("" if value is None else str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(out)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    rows = payload["cases"]
    lines = [
        "# Debug Bench Summary",
        "",
        f"- run_dir: {summary['run_dir']}",
        f"- total: {summary['total']}",
        f"- valid_total: {summary['valid_total']}",
        f"- debug_passed: {summary['debug_passed']}",
        f"- debug_pass_rate: {summary['debug_pass_rate']}",
        "",
        "## By Category",
        "",
        md_table(
            ["category", "total", "valid_total", "debug_passed", "pass_rate", "cheat", "legacy_unevaluable"],
            [
                [
                    item["category"],
                    item["total"],
                    item["valid_total"],
                    item["debug_passed"],
                    item["debug_pass_rate"],
                    item["cheat"],
                    item["legacy_unevaluable"],
                ]
                for item in summary["by_category"]
            ],
        ),
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(summary["counts"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Cases",
        "",
        md_table(
            [
                "level",
                "id",
                "op",
                "category",
                "runner",
                "generation_failure",
                "verify_raw",
                "verify_inferred",
                "verify_final",
                "verify_source",
                "debug_outcome",
                "claude_turns",
                "passed",
                "anticheat",
            ],
            [
                [
                    row["level"],
                    row["case_id"],
                    row["op_name"],
                    row["category"],
                    row["runner_status"],
                    row["generation_failure_type"],
                    row["raw_verify_failure_type"],
                    row["inferred_verify_failure_type"],
                    row["verify_failure_type"],
                    row["verify_status_source"],
                    row["debug_outcome"],
                    row["claude_turns"],
                    row["debug_passed"],
                    row["anticheat_verdict"],
                ]
                for row in rows
            ],
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Generation output directory containing case subdirs.")
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/NPUKernelBench/manifest.json"))
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"run dir does not exist: {run_dir}")

    manifest = load_manifest(args.manifest)
    report_rows = parse_batch_report(run_dir / "batch_report.md")
    worker_logs = parse_worker_logs(run_dir)
    task_dirs = discover_task_dirs(run_dir)
    rows = [summarize_task(task_dir, args.level, manifest, report_rows, worker_logs) for task_dir in task_dirs]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "summary": build_summary(rows, run_dir),
        "cases": rows,
    }

    output_json = args.output_json or (run_dir / "debug_bench_summary.json")
    output_md = args.output_md or (run_dir / "debug_bench_summary.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    write_markdown(output_md, payload)

    summary = payload["summary"]
    print(f"cases: {summary['total']}")
    print(f"debug_passed: {summary['debug_passed']}/{summary['valid_total']} rate={summary['debug_pass_rate']}")
    print(f"json: {output_json}")
    print(f"md: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
