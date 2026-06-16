#!/usr/bin/env python3
"""Build README and per-case links for an archived NPUKernelBench run."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_table(rows: list[dict[str, Any]], fields: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for _, key in fields) + " |")
    return "\n".join(lines)


def write_source_maps(archive_dir: Path, rows: list[dict[str, Any]]) -> None:
    json_path = archive_dir / "selected_cases_source_map.json"
    csv_path = archive_dir / "selected_cases_source_map.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_selected_links(repo_root: Path, archive_dir: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_roots = {
        repo_root / "outputs" / "npubench_supervisor_xiaomi_full_20260503_192601": archive_dir
        / "raw_runs"
        / "xiaomi_full_20260503_192601",
        repo_root / "outputs" / "npubench_supervisor_kimi_remaining_newkey_20260504_032149": archive_dir
        / "raw_runs"
        / "kimi_remaining_newkey_20260504_032149",
    }

    source_rows: list[dict[str, Any]] = []
    selected_root = archive_dir / "selected_cases"
    selected_root.mkdir(parents=True, exist_ok=True)

    for row in sorted(cases, key=lambda item: (item["level_num"], item["case_id"])):
        task_dir = Path(row["task_dir"])
        archive_task: Path | None = None
        for source_root, archived_root in raw_roots.items():
            try:
                rel = task_dir.relative_to(source_root)
            except ValueError:
                continue
            archive_task = archived_root / rel
            break
        if archive_task is None:
            raise RuntimeError(f"cannot map task_dir into archive raw_runs: {task_dir}")
        if not archive_task.exists():
            raise RuntimeError(f"archived task dir does not exist: {archive_task}")

        op_safe = str(row["op_name"]).replace("/", "_")
        case_link = selected_root / row["level"] / f"{int(row['case_id']):03d}_{op_safe}"
        case_link.parent.mkdir(parents=True, exist_ok=True)
        if case_link.exists() or case_link.is_symlink():
            if case_link.is_dir() and not case_link.is_symlink():
                raise RuntimeError(f"refusing to replace real directory: {case_link}")
            case_link.unlink()
        case_link.symlink_to(os.path.relpath(archive_task, start=case_link.parent))

        source_rows.append(
            {
                "level": row["level"],
                "case_id": row["case_id"],
                "op_name": row["op_name"],
                "category": row.get("category"),
                "provider": row.get("provider"),
                "attempt": row.get("attempt"),
                "case_status": row.get("case_status"),
                "verify_failure_type": row.get("verify_failure_type"),
                "debug_passed": row.get("debug_passed"),
                "anticheat_verdict": row.get("anticheat_verdict"),
                "selected_case_path": str(case_link),
                "archive_raw_task_dir": str(archive_task),
                "original_task_dir": str(task_dir),
            }
        )

    return source_rows


def write_readme(archive_dir: Path, data: dict[str, Any]) -> None:
    summary = data["summary"]
    by_level = data.get("by_level", [])
    by_category = data["summary"].get("by_category", [])

    level_rows = []
    for item in by_level:
        level_rows.append(
            {
                **item,
                "precision_failed": item.get(
                    "precision_failed",
                    item.get("verify_failure_type", {}).get("precision_failed", 0),
                ),
            }
        )

    category_rows = []
    for item in by_category:
        category_rows.append(
            {
                **item,
                "precision_failed": item.get("failure_types", {}).get("precision_failed", 0),
            }
        )

    readme = f"""# NPUKernelBench Full Batch Archive

本目录归档了本次 NPUKernelBench 全量批跑结果，后续实验可直接引用这里的稳定路径。

## Scope

- Xiaomi: `level1/1-16`
- Kimi: `level1/17-31` + `level2-7`
- total cases: `{summary["total"]}`
- valid_total: `{summary["valid_total"]}`
- debug_passed: `{summary["debug_passed"]}`
- debug_pass_rate: `{summary["debug_pass_rate"]}`
- excluded cheat: `{summary["excluded"]["cheat"]}`
- missing_cases: `{len(data.get("missing_cases", []))}`

注意：这里的 `debug_passed` 沿用汇总脚本字段名，实际含义是最终 verify 通过；本批结果没有经过 Codex debug subagent 修复成功的 case。

## Directory Layout

- `raw_runs/xiaomi_full_20260503_192601/`: Xiaomi 原始 supervisor 输出，包含 batch report、worker log、每个 case 的生成产物和验证日志。
- `raw_runs/kimi_remaining_newkey_20260504_032149/`: Kimi 原始 supervisor 输出，包含所有 attempt 和原始输出。
- `selected_cases/`: 按最终合并结果整理的 case 入口，每个条目是指向 `raw_runs/` 内对应原始 task dir 的相对软链接。
- `summary/`: 合并后的汇总结果，包括 markdown/json/csv。
- `manifest/`: 本次 bench 使用的 NPUKernelBench manifest。
- `scripts/`: 生成汇总用到的脚本快照。
- `selected_cases_source_map.csv`: 每个最终 case 到 provider、attempt、原始 task dir、归档 task dir 的映射。
- `selected_cases_source_map.json`: 同上，JSON 格式。

## By Level

{row_table(level_rows, [("level", "level"), ("total", "total"), ("valid_total", "valid_total"), ("debug_passed", "debug_passed"), ("pass_rate", "debug_pass_rate"), ("cheat", "cheat"), ("precision_failed", "precision_failed")])}

## By Category

{row_table(category_rows, [("category", "category"), ("total", "total"), ("valid_total", "valid_total"), ("debug_passed", "debug_passed"), ("pass_rate", "debug_pass_rate"), ("cheat", "cheat"), ("precision_failed", "precision_failed")])}

## Main Files

- `summary/merged_debug_bench_summary.md`: 总览汇总。
- `summary/merged_debug_bench_summary.json`: 机器可读完整汇总。
- `summary/merged_debug_bench_details.md`: case 级详细表。
- `summary/merged_cases.csv`: case 级 CSV。

## Example

查看 GELU 的最终输出：

```bash
ls -la selected_cases/level1/001_GELU
```

查看 GELU 的原始结果、验证状态和日志：

```bash
find selected_cases/level1/001_GELU -maxdepth 2 -type f | sort
```
"""
    (archive_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/home/wsx/AscendOpGenAgent"))
    parser.add_argument("--archive-dir", type=Path, required=True)
    args = parser.parse_args()

    archive_dir = args.archive_dir.resolve()
    data = read_json(archive_dir / "summary" / "merged_debug_bench_summary.json")
    source_rows = build_selected_links(args.repo_root.resolve(), archive_dir, data["cases"])
    write_source_maps(archive_dir, source_rows)
    write_readme(archive_dir, data)

    print(f"archive: {archive_dir}")
    print(f"selected_cases: {len(source_rows)}")
    print(f"readme: {archive_dir / 'README.md'}")
    print(f"source_map_csv: {archive_dir / 'selected_cases_source_map.csv'}")
    print(f"source_map_json: {archive_dir / 'selected_cases_source_map.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
