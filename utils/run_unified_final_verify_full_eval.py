#!/usr/bin/env python3
"""Run fair final verification for agent variants on one frozen eval set.

The normal debug workflow lets agents edit files inside each task directory.
That includes JSONL input files, so a final verification may run on 10 cases for
one variant and 50 cases for another. This tool builds temporary verification
directories and forces every variant of the same operator to use the same
source/full JSONL cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CASE_LINE_RE = re.compile(r"^case\[\d+\]: output(?:\[[^\]]+\])+:\s*(.*)$", re.M)
STATUS_RE = re.compile(r"^Status\s*:\s*(\w+)", re.M)
TEST_JSON_EXCLUDE = {
    "debug_status.json",
    "round_summary.json",
}


@dataclass(frozen=True)
class Target:
    task: str
    level: str
    case_id: int
    op_name: str
    source_dir: Path


def empty_initial_metrics() -> dict[str, object]:
    return {
        "initial_failure_type": "",
        "initial_matched_count": "",
        "initial_total_outputs": "",
        "initial_match_rate_pct": "",
        "initial_case_status": "",
        "initial_provider": "",
        "initial_attempt": "",
        "initial_metrics_source": "",
        "initial_log": "",
    }


def parse_match_metrics_from_text(text: str, default_failure: str = "") -> tuple[str, object, object, object]:
    comps = CASE_LINE_RE.findall(text)
    total = len(comps)
    matched = sum(1 for comp in comps if comp.strip() == "matched" or comp.strip().startswith("matched"))
    status_match = STATUS_RE.search(text)
    status = status_match.group(1).upper() if status_match else ""
    if status == "PASS":
        failure = "success"
    elif total:
        failure = "precision_failed"
    else:
        failure = default_failure
    rate: object = "" if not total else round(matched * 100.0 / total, 2)
    return failure, matched if total else "", total if total else "", rate


def resolve_verify_log(raw_task_dir: Path, status_data: dict[str, object]) -> Path | None:
    log_path = status_data.get("log_path")
    if isinstance(log_path, str) and log_path:
        direct = Path(log_path)
        if direct.is_file():
            return direct
        local = raw_task_dir / ".verify_logs" / direct.name
        if local.is_file():
            return local

    phase = status_data.get("phase")
    attempt = status_data.get("attempt")
    if phase is not None and attempt is not None:
        local = raw_task_dir / ".verify_logs" / f"phase{phase}_attempt{attempt}.stdout"
        if local.is_file():
            return local

    logs = sorted((raw_task_dir / ".verify_logs").glob("*.stdout"))
    if not logs:
        return None
    return max(logs, key=lambda path: path.stat().st_mtime)


def parse_initial_metrics(raw_task_dir: Path, source_row: dict[str, str]) -> dict[str, object]:
    metrics = empty_initial_metrics()
    metrics.update(
        {
            "initial_failure_type": source_row.get("verify_failure_type", ""),
            "initial_case_status": source_row.get("case_status", ""),
            "initial_provider": source_row.get("provider", ""),
            "initial_attempt": source_row.get("attempt", ""),
            "initial_metrics_source": "source_map",
        }
    )

    status_path = raw_task_dir / ".verify_status" / "latest.json"
    status_data: dict[str, object] = {}
    if status_path.is_file():
        try:
            status_data = json.loads(status_path.read_text())
            if status_data.get("failure_type"):
                metrics["initial_failure_type"] = status_data["failure_type"]
            metrics["initial_metrics_source"] = str(status_path)
        except Exception:
            status_data = {}

    log_path = resolve_verify_log(raw_task_dir, status_data)
    text = ""
    if log_path is not None and log_path.is_file():
        text = log_path.read_text(errors="replace")
        metrics["initial_log"] = str(log_path)
    elif isinstance(status_data.get("stdout_tail"), str):
        text = str(status_data["stdout_tail"])
        metrics["initial_log"] = str(status_path)

    if text:
        failure, matched, total, rate = parse_match_metrics_from_text(
            text, default_failure=str(metrics.get("initial_failure_type", ""))
        )
        metrics.update(
            {
                "initial_failure_type": failure or metrics["initial_failure_type"],
                "initial_matched_count": matched,
                "initial_total_outputs": total,
                "initial_match_rate_pct": rate,
            }
        )
    return metrics


def load_initial_metrics(source_map: Path | None) -> dict[str, dict[str, object]]:
    if source_map is None or not source_map.is_file():
        return {}
    out: dict[str, dict[str, object]] = {}
    with source_map.open(newline="") as f:
        for row in csv.DictReader(f):
            level = row.get("level", "")
            try:
                case_id = int(row.get("case_id", ""))
            except ValueError:
                continue
            op_name = row.get("op_name", "")
            task = f"{level}/{case_id:03d}_{op_name}"
            raw_dir_text = row.get("archive_raw_task_dir") or row.get("original_task_dir") or ""
            raw_dir = Path(raw_dir_text)
            if not raw_dir.is_dir():
                out[task] = empty_initial_metrics()
                out[task].update(
                    {
                        "initial_failure_type": row.get("verify_failure_type", ""),
                        "initial_case_status": row.get("case_status", ""),
                        "initial_provider": row.get("provider", ""),
                        "initial_attempt": row.get("attempt", ""),
                        "initial_metrics_source": "source_map_missing_raw_task_dir",
                    }
                )
                continue
            out[task] = parse_initial_metrics(raw_dir, row)
    return out


def add_initial_metrics(row: dict[str, object], metrics: dict[str, object]) -> dict[str, object]:
    enriched = dict(row)
    for key, value in empty_initial_metrics().items():
        enriched[key] = metrics.get(key, value)
    return enriched


def count_nonempty_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(errors="replace").splitlines() if line.strip())


def looks_like_case_jsonl(path: Path) -> bool:
    if path.name.startswith("_") or path.name.startswith("."):
        return False
    if path.name in TEST_JSON_EXCLUDE:
        return False
    if not (path.name.endswith(".json") or path.name.endswith(".json.bak") or path.name.endswith(".json.full")):
        return False
    try:
        first = next(line for line in path.read_text(errors="replace").splitlines() if line.strip())
        data = json.loads(first)
    except Exception:
        return False
    return isinstance(data, dict) and "inputs" in data


def active_json_name(path: Path) -> str:
    name = path.name
    if name.endswith(".json.bak"):
        return name[:-4]
    if name.endswith(".json.full"):
        return name[:-5]
    return name


def find_targets(source_cases: Path) -> list[Target]:
    targets: list[Target] = []
    for level_dir in sorted(source_cases.glob("level*")):
        if not level_dir.is_dir():
            continue
        for task_dir in sorted(level_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            m = re.match(r"^(\d+)_(.+)$", task_dir.name)
            if not m:
                continue
            targets.append(
                Target(
                    task=f"{level_dir.name}/{task_dir.name}",
                    level=level_dir.name,
                    case_id=int(m.group(1)),
                    op_name=m.group(2),
                    source_dir=task_dir,
                )
            )
    return targets


def choose_full_json(source_dir: Path) -> tuple[Path | None, int | str]:
    candidates = [p for p in source_dir.iterdir() if p.is_file() and looks_like_case_jsonl(p)]
    if not candidates:
        return None, ""

    def rank(path: Path) -> tuple[int, int, str]:
        lines = count_nonempty_lines(path)
        suffix_score = 2 if path.name.endswith(".json.bak") or path.name.endswith(".json.full") else 1
        return (lines, suffix_score, path.name)

    best = max(candidates, key=rank)
    return best, count_nonempty_lines(best)


def json_names_to_populate(source_dir: Path, agent_task_dir: Path, full_json: Path | None) -> list[str]:
    names: set[str] = set()
    if full_json is not None:
        names.add(active_json_name(full_json))
    for root in (source_dir, agent_task_dir):
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if path.is_file() and looks_like_case_jsonl(path):
                name = active_json_name(path)
                if name.endswith(".json"):
                    names.add(name)
    return sorted(names)


def safe_remove(path: Path, root: Path) -> None:
    path = path.resolve() if path.exists() and not path.is_symlink() else path
    root_resolved = root.resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError:
        raise RuntimeError(f"refusing to remove outside output root: {path}") from None
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def link_item(src: Path, dst: Path, output_root: Path) -> None:
    if dst.exists() or dst.is_symlink():
        safe_remove(dst, output_root)
    os.symlink(src, dst)


def should_skip_item(path: Path) -> bool:
    if path.name in {".verify_logs", "__pycache__"}:
        return True
    if path.name.startswith("."):
        return True
    return False


def prepare_eval_dir(
    *,
    target: Target,
    variant: str,
    agent_task_dir: Path,
    eval_task_dir: Path,
    full_json: Path | None,
    full_json_lines: int | str,
    output_root: Path,
) -> None:
    if eval_task_dir.exists() or eval_task_dir.is_symlink():
        safe_remove(eval_task_dir, output_root)
    eval_task_dir.mkdir(parents=True, exist_ok=True)

    # Start with source-side ancillary files, then overlay the variant candidate.
    for src_root in (target.source_dir, agent_task_dir):
        for item in src_root.iterdir():
            if should_skip_item(item):
                continue
            if item.name == "model.py":
                continue
            if item.is_file() and looks_like_case_jsonl(item):
                continue
            link_item(item.resolve(), eval_task_dir / item.name, output_root)

    source_model = target.source_dir / "model.py"
    if not source_model.is_file():
        source_model = agent_task_dir / "model.py"
    shutil.copy2(source_model, eval_task_dir / "model.py")

    full_text = full_json.read_text(errors="replace") if full_json is not None else ""
    names = json_names_to_populate(target.source_dir, agent_task_dir, full_json)
    for name in names:
        if not name.endswith(".json"):
            continue
        (eval_task_dir / name).write_text(full_text)

    manifest = {
        "task": target.task,
        "variant": variant,
        "source_model": str(source_model),
        "agent_task_dir": str(agent_task_dir),
        "full_json_source": "" if full_json is None else str(full_json),
        "full_json_cases": full_json_lines,
        "json_names_populated": names,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (eval_task_dir / "_unified_eval_manifest.json").write_text(json.dumps(manifest, indent=2))


def classify(stdout_text: str, stderr_text: str, returncode: int) -> tuple[str, int | None, int | None, float | str]:
    comps = CASE_LINE_RE.findall(stdout_text)
    total = len(comps)
    matched = sum(1 for comp in comps if comp.strip() == "matched" or comp.strip().startswith("matched"))
    status_match = STATUS_RE.search(stdout_text)
    status = status_match.group(1).upper() if status_match else ""
    combined = f"{stdout_text}\n{stderr_text}"
    if returncode == 124:
        return "timeout", matched if total else None, total if total else None, "" if total == 0 else round(matched * 100.0 / total, 2)
    if status == "PASS":
        return "success", matched if total else None, total if total else None, "" if total == 0 else round(matched * 100.0 / total, 2)
    if total:
        return "precision_failed", matched, total, round(matched * 100.0 / total, 2)
    if "ModuleNotFoundError" in combined or "ImportError" in combined or "No module named" in combined:
        return "import_failed", None, None, ""
    if "RuntimeError" in combined or "Traceback" in combined or "Exception" in combined or returncode != 0:
        return "runtime_error", None, None, ""
    return "failed", None, None, ""


def load_agent2_completed(state_path: Path | None) -> set[str]:
    if not state_path or not state_path.is_file():
        return set()
    state = json.loads(state_path.read_text())
    return {item["task"] for item in state.get("completed", []) if "task" in item}


def run_verify(
    *,
    repo_root: Path,
    eval_task_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    npu: str,
    timeout_sec: int,
) -> int:
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(npu)
    with stdout_path.open("w", encoding="utf-8") as so, stderr_path.open("w", encoding="utf-8") as se:
        proc = subprocess.run(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=30",
                str(timeout_sec),
                "python3",
                "utils/verification_ascendc.py",
                str(eval_task_dir),
            ],
            cwd=repo_root,
            env=env,
            stdout=so,
            stderr=se,
        )
    return proc.returncode


def write_reports(output_dir: Path, rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "unified_final_verify_latest.csv"
    md_path = output_dir / "unified_final_verify_latest.md"
    if rows:
        fields = list(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    by_variant = Counter()
    by_run_state = Counter()
    for row in rows:
        by_run_state[(str(row["variant"]), str(row["run_state"]))] += 1
        if str(row["run_state"]) == "completed":
            by_variant[(str(row["variant"]), str(row["final_failure_type"]))] += 1

    tasks = sorted({str(row["task"]) for row in rows})
    by_task_variant = {(str(row["task"]), str(row["variant"])): row for row in rows}
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Unified Final Verify On Full Eval Set\n\n")
        f.write(f"- generated_at: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"- rows: {len(rows)}\n\n")
        f.write("## Summary\n\n")
        f.write("| variant | run_state | count |\n|---|---|---:|\n")
        for (variant, run_state), count in sorted(by_run_state.items()):
            f.write(f"| {variant} | {run_state} | {count} |\n")
        f.write("\n")
        f.write("| variant | final_failure_type | count |\n|---|---|---:|\n")
        for (variant, failure), count in sorted(by_variant.items()):
            f.write(f"| {variant} | {failure} | {count} |\n")
        f.write("\n## Comparison\n\n")
        f.write("| task | initial status | initial matched | eval_cases | agent1 final | agent1 matched | agent2 final | agent2 matched | delta agent2-agent1 |\n")
        f.write("|---|---|---:|---:|---|---:|---|---:|---:|\n")
        for task in tasks:
            a1 = by_task_variant.get((task, "agent1"))
            a2 = by_task_variant.get((task, "agent2"))
            eval_cases = a1.get("full_json_cases") if a1 else (a2.get("full_json_cases") if a2 else "")
            initial_src = a1 or a2 or {}
            initial_status = initial_src.get("initial_failure_type", "")
            initial_match = (
                ""
                if initial_src.get("initial_matched_count", "") == ""
                else f"{initial_src.get('initial_matched_count')}/{initial_src.get('initial_total_outputs')}"
            )
            a1_match = "" if not a1 or a1.get("matched_count") == "" else f"{a1.get('matched_count')}/{a1.get('total_outputs')}"
            a2_match = "" if not a2 or a2.get("matched_count") == "" else f"{a2.get('matched_count')}/{a2.get('total_outputs')}"
            delta: object = ""
            if a1 and a2 and a1.get("matched_count") != "" and a2.get("matched_count") != "":
                try:
                    delta = int(a2["matched_count"]) - int(a1["matched_count"])
                except Exception:
                    delta = ""
            f.write(
                f"| {task} | {initial_status} | {initial_match} | {eval_cases} | "
                f"{a1.get('final_failure_type') if a1 else ''} | {a1_match} | "
                f"{a2.get('final_failure_type') if a2 else ''} | {a2_match} | {delta} |\n"
            )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--compare-dir", type=Path, required=True)
    parser.add_argument("--source-cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", default="agent1,agent2")
    parser.add_argument("--agent2-state", type=Path)
    parser.add_argument("--initial-source-map", type=Path)
    parser.add_argument("--agent2-completed-only", action="store_true")
    parser.add_argument("--npu", default="0")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    output_dir = args.output_dir
    work_root = output_dir / "work"
    log_root = output_dir / "logs"
    result_root = output_dir / "results"
    for path in (work_root, log_root, result_root):
        path.mkdir(parents=True, exist_ok=True)

    completed_agent2 = load_agent2_completed(args.agent2_state)
    initial_by_task = load_initial_metrics(args.initial_source_map)
    targets = find_targets(args.source_cases)
    rows: list[dict[str, object]] = []
    runs_done = 0

    for target in targets:
        full_json, full_json_lines = choose_full_json(target.source_dir)
        initial_metrics = initial_by_task.get(target.task, empty_initial_metrics())
        for variant in variants:
            agent_task_dir = args.compare_dir / variant / "tasks" / target.task
            result_path = result_root / variant / target.level / f"{target.case_id:03d}_{target.op_name}.json"
            stdout_path = log_root / variant / target.level / f"{target.case_id:03d}_{target.op_name}.stdout"
            stderr_path = log_root / variant / target.level / f"{target.case_id:03d}_{target.op_name}.stderr"
            eval_task_dir = work_root / variant / target.task
            result_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.parent.mkdir(parents=True, exist_ok=True)

            if variant == "agent2" and args.agent2_completed_only and target.task not in completed_agent2:
                row = {
                    "task": target.task,
                    "variant": variant,
                    "level": target.level,
                    "case_id": target.case_id,
                    "op_name": target.op_name,
                    "run_state": "not_completed",
                    "final_failure_type": "",
                    "matched_count": "",
                    "total_outputs": "",
                    "match_rate_pct": "",
                    "returncode": "",
                    "full_json_source": "" if full_json is None else str(full_json),
                    "full_json_cases": full_json_lines,
                    "eval_task_dir": "",
                    "stdout": "",
                    "stderr": "",
                }
                rows.append(add_initial_metrics(row, initial_metrics))
                continue

            if not agent_task_dir.is_dir() or not (agent_task_dir / "model_new_ascendc.py").is_file():
                row = {
                    "task": target.task,
                    "variant": variant,
                    "level": target.level,
                    "case_id": target.case_id,
                    "op_name": target.op_name,
                    "run_state": "missing_candidate",
                    "final_failure_type": "",
                    "matched_count": "",
                    "total_outputs": "",
                    "match_rate_pct": "",
                    "returncode": "",
                    "full_json_source": "" if full_json is None else str(full_json),
                    "full_json_cases": full_json_lines,
                    "eval_task_dir": "",
                    "stdout": "",
                    "stderr": "",
                }
                rows.append(add_initial_metrics(row, initial_metrics))
                continue

            if result_path.is_file() and not args.force:
                data = json.loads(result_path.read_text())
                rows.append(add_initial_metrics(data, initial_metrics))
                continue

            if args.max_runs and runs_done >= args.max_runs:
                row = {
                    "task": target.task,
                    "variant": variant,
                    "level": target.level,
                    "case_id": target.case_id,
                    "op_name": target.op_name,
                    "run_state": "deferred",
                    "final_failure_type": "",
                    "matched_count": "",
                    "total_outputs": "",
                    "match_rate_pct": "",
                    "returncode": "",
                    "full_json_source": "" if full_json is None else str(full_json),
                    "full_json_cases": full_json_lines,
                    "eval_task_dir": "",
                    "stdout": "",
                    "stderr": "",
                }
                rows.append(add_initial_metrics(row, initial_metrics))
                continue

            prepare_eval_dir(
                target=target,
                variant=variant,
                agent_task_dir=agent_task_dir,
                eval_task_dir=eval_task_dir,
                full_json=full_json,
                full_json_lines=full_json_lines,
                output_root=output_dir,
            )
            print(f"[run] {variant} {target.task} full_cases={full_json_lines}", flush=True)
            rc = run_verify(
                repo_root=args.repo_root,
                eval_task_dir=eval_task_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                npu=args.npu,
                timeout_sec=args.timeout,
            )
            stdout_text = stdout_path.read_text(errors="replace") if stdout_path.is_file() else ""
            stderr_text = stderr_path.read_text(errors="replace") if stderr_path.is_file() else ""
            failure, matched, total, rate = classify(stdout_text, stderr_text, rc)
            row = {
                "task": target.task,
                "variant": variant,
                "level": target.level,
                "case_id": target.case_id,
                "op_name": target.op_name,
                "run_state": "completed",
                "final_failure_type": failure,
                "matched_count": "" if matched is None else matched,
                "total_outputs": "" if total is None else total,
                "match_rate_pct": rate,
                "returncode": rc,
                "full_json_source": "" if full_json is None else str(full_json),
                "full_json_cases": full_json_lines,
                "eval_task_dir": str(eval_task_dir),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
            result_path.write_text(json.dumps(row, indent=2))
            rows.append(add_initial_metrics(row, initial_metrics))
            runs_done += 1
            write_reports(output_dir, rows)

    write_reports(output_dir, rows)
    print(f"[done] wrote {output_dir / 'unified_final_verify_latest.csv'}")
    print(f"[done] wrote {output_dir / 'unified_final_verify_latest.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
