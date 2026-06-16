#!/usr/bin/env python3
"""Merge NPUKernelBench results split across provider runs."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from summarize_debug_bench import (
    build_summary,
    load_manifest,
    parse_batch_report,
    parse_worker_logs,
    summarize_task,
    write_markdown as write_debug_markdown,
)
from supervise_npubench_full_claude import is_provider_retryable


@dataclass(frozen=True)
class Candidate:
    provider: str
    attempt: str
    level: int
    case_id: int
    level_dir: Path
    task_dir: Path
    provider_retryable: bool


def parse_ids(value: str) -> set[int]:
    ids: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ids.update(range(int(start), int(end) + 1))
        else:
            ids.add(int(part))
    return ids


def level_num(path: Path) -> int | None:
    match = re.fullmatch(r"level(\d+)", path.name)
    return int(match.group(1)) if match else None


def case_id_from_task(path: Path) -> int | None:
    match = re.match(r"(\d+)_", path.name)
    return int(match.group(1)) if match else None


def sorted_attempt_dirs(supervisor_root: Path) -> list[Path]:
    return sorted(
        [p for p in supervisor_root.iterdir() if p.is_dir() and p.name.startswith("attempt_")],
        key=lambda p: p.name,
    )


def task_dir_for_case(level_dir: Path, case_id: int) -> Path | None:
    matches = sorted(level_dir.glob(f"{case_id}_*"))
    return matches[0] if matches else None


def benchmark_plan(benchmark_dir: Path, levels: list[int]) -> dict[int, set[int]]:
    plan: dict[int, set[int]] = {}
    for level in levels:
        ids: set[int] = set()
        for path in (benchmark_dir / f"level{level}").glob("*.py"):
            case_id = case_id_from_task(path)
            if case_id is not None:
                ids.add(case_id)
        plan[level] = ids
    return plan


def collect_xiaomi_level1(level_dir: Path, ids: set[int]) -> dict[tuple[int, int], Candidate]:
    selected: dict[tuple[int, int], Candidate] = {}
    report_rows = parse_batch_report(level_dir / "batch_report.md")
    for case_id in sorted(ids):
        task_dir = task_dir_for_case(level_dir, case_id)
        if task_dir is None or case_id not in report_rows:
            continue
        selected[(1, case_id)] = Candidate(
            provider="xiaomi",
            attempt=level_dir.parent.name,
            level=1,
            case_id=case_id,
            level_dir=level_dir,
            task_dir=task_dir,
            provider_retryable=False,
        )
    return selected


def collect_kimi(
    supervisor_root: Path,
    plan: dict[int, set[int]],
    level1_ids: set[int],
) -> tuple[dict[tuple[int, int], Candidate], dict[tuple[int, int], Candidate]]:
    selected: dict[tuple[int, int], Candidate] = {}
    provider_fallback: dict[tuple[int, int], Candidate] = {}
    for attempt_dir in sorted_attempt_dirs(supervisor_root):
        for level_dir in sorted(attempt_dir.glob("level*")):
            level = level_num(level_dir)
            if level is None or level not in plan:
                continue
            wanted = level1_ids if level == 1 else plan[level]
            report_rows = parse_batch_report(level_dir / "batch_report.md")
            raw_rows = {
                case_id: row.get("runner_status_text", "")
                for case_id, row in report_rows.items()
            }
            for case_id, status_text in raw_rows.items():
                if case_id not in wanted:
                    continue
                task_dir = task_dir_for_case(level_dir, case_id)
                if task_dir is None:
                    continue
                retryable = is_provider_retryable(level_dir, case_id, status_text)
                candidate = Candidate(
                    provider="kimi",
                    attempt=attempt_dir.name,
                    level=level,
                    case_id=case_id,
                    level_dir=level_dir,
                    task_dir=task_dir,
                    provider_retryable=retryable,
                )
                key = (level, case_id)
                if retryable:
                    provider_fallback[key] = candidate
                    continue
                selected[key] = candidate
    return selected, provider_fallback


def summarize_candidates(
    candidates: dict[tuple[int, int], Candidate],
    manifest: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    cache: dict[Path, tuple[dict[int, dict[str, str]], dict[str, str]]] = {}
    rows: list[dict[str, Any]] = []
    for key in sorted(candidates):
        candidate = candidates[key]
        if candidate.level_dir not in cache:
            cache[candidate.level_dir] = (
                parse_batch_report(candidate.level_dir / "batch_report.md"),
                parse_worker_logs(candidate.level_dir),
            )
        report_rows, worker_logs = cache[candidate.level_dir]
        row = summarize_task(
            candidate.task_dir,
            candidate.level,
            manifest,
            report_rows,
            worker_logs,
        )
        row["provider"] = candidate.provider
        row["attempt"] = candidate.attempt
        row["source_level_dir"] = str(candidate.level_dir)
        row["provider_retryable"] = candidate.provider_retryable
        rows.append(row)
    return rows


def counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "unknown") for row in rows).items()))


def level_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["level_num"])].append(row)
    out: list[dict[str, Any]] = []
    for level, items in sorted(grouped.items()):
        valid_total = sum(1 for item in items if item.get("include_in_pass_rate"))
        passed = sum(1 for item in items if item.get("include_in_pass_rate") and item.get("debug_passed"))
        out.append(
            {
                "level": f"level{level}",
                "total": len(items),
                "valid_total": valid_total,
                "debug_passed": passed,
                "debug_pass_rate": round(passed / valid_total, 4) if valid_total else None,
                "cheat": sum(1 for item in items if item.get("anticheat_verdict") == "CHEAT"),
                "precision_failed": sum(1 for item in items if item.get("verify_failure_type") == "precision_failed"),
                "case_status": counter(items, "case_status"),
                "verify_failure_type": counter(items, "verify_failure_type"),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "level",
        "case_id",
        "op_name",
        "category",
        "provider",
        "attempt",
        "runner_status_text",
        "case_status",
        "verify_failure_type",
        "generation_failure_type",
        "debug_passed",
        "include_in_pass_rate",
        "anticheat_verdict",
        "elapsed_sec",
        "worker",
        "task_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def write_merged_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    level_rows = payload["by_level"]
    lines = [
        "# Merged NPUKernelBench Summary",
        "",
        f"- total: {summary['total']}",
        f"- valid_total: {summary['valid_total']}",
        f"- debug_passed: {summary['debug_passed']}",
        f"- debug_pass_rate: {summary['debug_pass_rate']}",
        f"- excluded_cheat: {summary['excluded']['cheat']}",
        "",
        "## By Level",
        "",
        md_table(
            ["level", "total", "valid_total", "debug_passed", "pass_rate", "cheat", "precision_failed"],
            [
                [
                    item["level"],
                    item["total"],
                    item["valid_total"],
                    item["debug_passed"],
                    item["debug_pass_rate"],
                    item["cheat"],
                    item["precision_failed"],
                ]
                for item in level_rows
            ],
        ),
        "",
        "## By Category",
        "",
        md_table(
            ["category", "total", "valid_total", "debug_passed", "pass_rate", "cheat", "precision_failed"],
            [
                [
                    item["category"],
                    item["total"],
                    item["valid_total"],
                    item["debug_passed"],
                    item["debug_pass_rate"],
                    item["cheat"],
                    item["failure_types"].get("precision_failed", 0),
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
        "## Missing Cases",
        "",
        "```json",
        json.dumps(payload["missing_cases"], indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks/NPUKernelBench"))
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/NPUKernelBench/manifest.json"))
    parser.add_argument("--xiaomi-level1-dir", type=Path, required=True)
    parser.add_argument("--xiaomi-level1-ids", default="1-16")
    parser.add_argument("--kimi-supervisor-root", type=Path, required=True)
    parser.add_argument("--kimi-level1-ids", default="17-31")
    parser.add_argument("--levels", default="1-7")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    levels = sorted(parse_ids(args.levels))
    plan = benchmark_plan(args.benchmark_dir, levels)
    manifest = load_manifest(args.manifest)

    selected = collect_xiaomi_level1(args.xiaomi_level1_dir, parse_ids(args.xiaomi_level1_ids))
    kimi_selected, kimi_fallback = collect_kimi(
        args.kimi_supervisor_root,
        plan,
        parse_ids(args.kimi_level1_ids),
    )
    selected.update(kimi_selected)
    for key, candidate in kimi_fallback.items():
        selected.setdefault(key, candidate)

    rows = summarize_candidates(selected, manifest)
    expected = {(level, case_id) for level, ids in plan.items() for case_id in ids}
    missing = sorted(expected - set(selected))
    fallback_keys = sorted(key for key, item in selected.items() if item.provider_retryable)

    debug_payload = {
        "schema_version": 2,
        "summary": build_summary(rows, args.output_dir),
        "cases": rows,
    }
    payload = {
        **debug_payload,
        "sources": {
            "xiaomi_level1_dir": str(args.xiaomi_level1_dir),
            "xiaomi_level1_ids": sorted(parse_ids(args.xiaomi_level1_ids)),
            "kimi_supervisor_root": str(args.kimi_supervisor_root),
            "kimi_level1_ids": sorted(parse_ids(args.kimi_level1_ids)),
        },
        "by_level": level_summary(rows),
        "missing_cases": [{"level": f"level{level}", "case_id": case_id} for level, case_id in missing],
        "provider_retryable_fallback_cases": [
            {"level": f"level{level}", "case_id": case_id} for level, case_id in fallback_keys
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "merged_debug_bench_summary.json"
    md_path = args.output_dir / "merged_debug_bench_summary.md"
    cases_csv = args.output_dir / "merged_cases.csv"
    detailed_md = args.output_dir / "merged_debug_bench_details.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_merged_markdown(md_path, payload)
    write_csv(cases_csv, rows)
    write_debug_markdown(detailed_md, debug_payload)

    summary = payload["summary"]
    print(f"cases: {summary['total']}")
    print(f"debug_passed: {summary['debug_passed']}/{summary['valid_total']} rate={summary['debug_pass_rate']}")
    print(f"missing_cases: {len(missing)}")
    print(f"provider_retryable_fallback_cases: {len(fallback_keys)}")
    print(f"json: {json_path}")
    print(f"md: {md_path}")
    print(f"csv: {cases_csv}")
    print(f"details_md: {detailed_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
