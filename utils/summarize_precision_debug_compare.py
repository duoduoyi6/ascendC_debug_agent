#!/usr/bin/env python3
"""Summarize precision debug comparison runs."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return None


def parse_task(path: Path) -> tuple[str, int | None, str]:
    level = path.parent.name
    match = re.match(r"^(\d+)_(.+)$", path.name)
    if not match:
        return level, None, path.name
    return level, int(match.group(1)), match.group(2)


def summarize_task(variant: str, task_dir: Path) -> dict[str, Any]:
    status = read_json(task_dir / ".verify_status" / "latest.json") or {}
    anticheat = read_json(task_dir / f"_anticheat_{variant}.json") or {}
    claude_result = read_json(task_dir / f"_claude_{variant}.json") or {}
    level, case_id, op_name = parse_task(task_dir)
    failure_type = status.get("failure_type") or status.get("status") or "missing_status"
    row = {
        "variant": variant,
        "level": level,
        "case_id": case_id if case_id is not None else "",
        "op_name": op_name,
        "failure_type": failure_type,
        "success": failure_type == "success",
        "anticheat_verdict": anticheat.get("verdict") or anticheat.get("status") or "missing",
        "claude_error": bool(claude_result.get("is_error")),
        "task_dir": str(task_dir),
    }
    return row


def build_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("agent")):
        tasks_dir = variant_dir / "tasks"
        if not tasks_dir.exists():
            continue
        for task_dir in sorted(tasks_dir.glob("level*/*")):
            if task_dir.is_dir():
                rows.append(summarize_task(variant_dir.name, task_dir))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "variant",
        "level",
        "case_id",
        "op_name",
        "failure_type",
        "success",
        "anticheat_verdict",
        "claude_error",
        "task_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, Any]], path: Path) -> None:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    lines = ["# Precision Debug Compare Summary", ""]
    for variant, items in sorted(by_variant.items()):
        total = len(items)
        success = sum(1 for r in items if r["success"])
        clean_success = sum(
            1
            for r in items
            if r["success"] and str(r["anticheat_verdict"]).upper() in {"CLEAN", "PASS", "OK"}
        )
        lines.extend([
            f"## {variant}",
            "",
            f"- total: {total}",
            f"- success: {success}",
            f"- success_clean_or_ok: {clean_success}",
            "",
            "| level | case | op | final failure_type | anticheat |",
            "|---|---:|---|---|---|",
        ])
        for row in sorted(items, key=lambda r: (str(r["level"]), int(r["case_id"] or 0))):
            lines.append(
                f"| {row['level']} | {row['case_id']} | {row['op_name']} | "
                f"{row['failure_type']} | {row['anticheat_verdict']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows(args.run_dir)
    write_csv(rows, args.run_dir / "comparison_summary.csv")
    write_md(rows, args.run_dir / "comparison_summary.md")
    print(f"wrote {len(rows)} rows to {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
