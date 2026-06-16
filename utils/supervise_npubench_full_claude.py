#!/usr/bin/env python3
"""Supervise NPUKernelBench runs across multiple Claude-compatible providers.

The supervisor runs the existing run_npubench_full_claude.sh in attempt
directories. If a provider hits an API/budget limit, it switches to the next
provider and resumes from unfinished or provider-failed cases. If every provider
is currently limited, it sleeps and retries.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


API_LIMIT_MARKERS = (
    "api_error_status=401",
    "api_error_status=402",
    "api_error_status=403",
    "api_error_status=429",
    "usage limit",
    "rate limit",
    "quota",
    "credit",
    "billing",
    "budget",
    "too many requests",
)


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    env: str


@dataclass
class Attempt:
    index: int
    provider: str
    output_root: str
    adopted: bool
    pid: int | None = None
    returncode: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    fatal_reason: str | None = None


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def slug_time() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_text(path: Path, limit: int = 200_000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text[-limit:] if len(text) > limit else text


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return None


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{now()}] {message}\n")


def parse_levels(value: str | None, benchmark_dir: Path) -> list[int]:
    if value:
        return [int(x) for x in value.replace(" ", "").split(",") if x]
    levels: list[int] = []
    for path in benchmark_dir.glob("level*"):
        m = re.fullmatch(r"level(\d+)", path.name)
        if m and path.is_dir():
            levels.append(int(m.group(1)))
    return sorted(levels)


def parse_ids(value: str | None) -> list[int] | None:
    if not value:
        return None
    ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return sorted(dict.fromkeys(ids))


def level_ids(benchmark_dir: Path, level: int) -> list[int]:
    ids: list[int] = []
    for path in (benchmark_dir / f"level{level}").glob("*.py"):
        m = re.match(r"(\d+)_", path.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def planned_ids_by_level(benchmark_dir: Path, levels: list[int], first_level_ids: list[int] | None) -> dict[int, list[int]]:
    plan: dict[int, list[int]] = {}
    first = levels[0] if levels else None
    for level in levels:
        if first_level_ids is not None and level == first:
            plan[level] = first_level_ids
        else:
            plan[level] = level_ids(benchmark_dir, level)
    return plan


def parse_batch_rows(batch_report: Path) -> dict[int, str]:
    rows: dict[int, str] = {}
    for line in read_text(batch_report, limit=2_000_000).splitlines():
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 3 or not cols[0].isdigit():
            continue
        rows[int(cols[0])] = cols[2]
    return rows


def case_dir_for_id(level_dir: Path, case_id: int) -> Path | None:
    matches = sorted(level_dir.glob(f"{case_id}_*"))
    return matches[0] if matches else None


def claude_result_provider_issue(task_dir: Path | None) -> bool:
    if task_dir is None:
        return False
    result_paths = sorted(task_dir.glob("_claude_result*.json"))
    for path in result_paths:
        data = read_json(path)
        if not data:
            continue
        status = data.get("api_error_status")
        if status in (401, 402, 403, 429):
            return True
        parts = []
        for key in ("error", "message", "result"):
            value = data.get(key)
            if value is not None:
                parts.append(str(value))
        text = "\n".join(parts).lower()
        if data.get("is_error") and any(marker in text for marker in API_LIMIT_MARKERS):
            return True
    return False


def is_provider_retryable(level_dir: Path, case_id: int, status_text: str) -> bool:
    lower = status_text.lower()
    if any(marker in lower for marker in API_LIMIT_MARKERS):
        return True
    if "api_error" in lower or "provider_api_error" in lower:
        return True
    return claude_result_provider_issue(case_dir_for_id(level_dir, case_id))


def output_root_fatal(output_root: Path) -> str | None:
    for fatal in sorted(output_root.glob("level*/.fatal")):
        text = read_text(fatal, limit=20_000).strip()
        if text:
            return text
    return None


def is_api_limit_reason(reason: str | None) -> bool:
    if not reason:
        return False
    lower = reason.lower()
    return any(marker in lower for marker in API_LIMIT_MARKERS)


def collect_completed(
    output_roots: list[Path],
    retry_provider_failures: bool,
) -> tuple[dict[int, set[int]], bool]:
    completed: dict[int, set[int]] = {}
    provider_issue_seen = False
    for output_root in output_roots:
        for level_dir in sorted(output_root.glob("level*")):
            m = re.fullmatch(r"level(\d+)", level_dir.name)
            if not m or not level_dir.is_dir():
                continue
            level = int(m.group(1))
            for case_id, status_text in parse_batch_rows(level_dir / "batch_report.md").items():
                retryable = is_provider_retryable(level_dir, case_id, status_text)
                provider_issue_seen = provider_issue_seen or retryable
                if retry_provider_failures and retryable:
                    continue
                completed.setdefault(level, set()).add(case_id)
    return completed, provider_issue_seen


def remaining_plan(
    benchmark_dir: Path,
    levels: list[int],
    first_level_ids: list[int] | None,
    output_roots: list[Path],
    retry_provider_failures: bool,
) -> tuple[list[int], dict[int, list[int]], bool]:
    plan = planned_ids_by_level(benchmark_dir, levels, first_level_ids)
    completed, provider_issue_seen = collect_completed(output_roots, retry_provider_failures)
    remaining_by_level: dict[int, list[int]] = {}
    remaining_levels: list[int] = []
    for level in levels:
        ids = [case_id for case_id in plan.get(level, []) if case_id not in completed.get(level, set())]
        if ids:
            remaining_by_level[level] = ids
            remaining_levels.append(level)
    if not remaining_levels:
        return [], {}, provider_issue_seen
    first = remaining_levels[0]
    # Existing run_npubench_full_claude.sh only accepts a first-level id override,
    # so pass all later levels and override the first remaining level only.
    return remaining_levels, {first: remaining_by_level[first]}, provider_issue_seen


def write_state(path: Path, attempts: list[Attempt], limited: set[str], status: str) -> None:
    payload = {
        "schema_version": 1,
        "updated_at": now(),
        "status": status,
        "limited_providers_this_cycle": sorted(limited),
        "attempts": [asdict(a) for a in attempts],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_report(path: Path, attempts: list[Attempt], status: str) -> None:
    lines = [
        "# NPUKernelBench Provider Supervisor",
        "",
        f"- status: {status}",
        f"- updated_at: {now()}",
        "",
        "| index | provider | adopted | returncode | fatal | output |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for attempt in attempts:
        lines.append(
            "| {index} | {provider} | {adopted} | {returncode} | {fatal} | {output} |".format(
                index=attempt.index,
                provider=attempt.provider,
                adopted=attempt.adopted,
                returncode="" if attempt.returncode is None else attempt.returncode,
                fatal=(attempt.fatal_reason or "").replace("|", "/"),
                output=attempt.output_root,
            )
        )
    path.write_text("\n".join(lines) + "\n")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def wait_pid(pid: int, poll_interval: int, log_path: Path) -> None:
    while pid_alive(pid):
        append_log(log_path, f"adopted pid still running: {pid}")
        time.sleep(poll_interval)


def provider_order(providers: list[Provider], start_after: str | None = None) -> list[Provider]:
    if not start_after:
        return providers
    names = [p.name for p in providers]
    if start_after not in names:
        return providers
    idx = names.index(start_after)
    return providers[idx + 1 :] + providers[: idx + 1]


def build_runner_cmd(
    args: argparse.Namespace,
    provider: Provider,
    attempt_output: Path,
    remaining_levels: list[int],
    first_remaining_ids: list[int],
) -> list[str]:
    cmd = [
        "bash",
        "utils/run_npubench_full_claude.sh",
        "--benchmark-dir",
        str(args.benchmark_dir),
        "--output-root",
        str(attempt_output),
        "--levels",
        ",".join(str(x) for x in remaining_levels),
        "--first-level-ids",
        ",".join(str(x) for x in first_remaining_ids),
        "--containers",
        args.containers,
        "--npus",
        args.npus,
        "--model",
        provider.model,
        "--timeout",
        str(args.timeout),
        "--max-resumes",
        str(args.max_resumes),
        "--stale-after-failure",
        str(args.stale_after_failure),
        "--stale-check-interval",
        str(args.stale_check_interval),
        "--claude-env",
        provider.env,
        "--tilelang-env",
        args.tilelang_env,
        "--workdir",
        args.workdir,
    ]
    return cmd


def run_attempt(
    args: argparse.Namespace,
    provider: Provider,
    attempt_index: int,
    remaining_levels: list[int],
    first_remaining_ids: list[int],
    log_path: Path,
) -> Attempt:
    attempt_output = args.output_root / f"attempt_{attempt_index:03d}_{provider.name}_{slug_time()}"
    attempt_output.mkdir(parents=True, exist_ok=True)
    attempt = Attempt(
        index=attempt_index,
        provider=provider.name,
        output_root=str(attempt_output),
        adopted=False,
        started_at=now(),
    )
    cmd = build_runner_cmd(args, provider, attempt_output, remaining_levels, first_remaining_ids)
    append_log(log_path, "launch " + " ".join(cmd))
    with (attempt_output / "supervisor_attempt.log").open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=args.repo_root, stdout=f, stderr=subprocess.STDOUT)
    attempt.returncode = proc.returncode
    attempt.ended_at = now()
    attempt.fatal_reason = output_root_fatal(attempt_output)
    append_log(log_path, f"attempt done provider={provider.name} rc={proc.returncode} fatal={attempt.fatal_reason}")
    return attempt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks/NPUKernelBench"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--levels", default="1,2,3,4,5,6,7")
    parser.add_argument("--first-level-ids", help="Comma/range list for the first requested level, e.g. 7-31.")
    parser.add_argument("--containers", default="wsx_cann1,wsx_cann1,wsx_cann1")
    parser.add_argument("--npus", default="5,6,7")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--max-resumes", type=int, default=3)
    parser.add_argument("--stale-after-failure", type=int, default=300)
    parser.add_argument("--stale-check-interval", type=int, default=30)
    parser.add_argument("--tilelang-env", default="/home/wsx/tilelang-ascend/set_env.sh")
    parser.add_argument("--workdir", default="/home/wsx/AscendOpGenAgent")
    parser.add_argument("--providers", default="minimax,kimi")
    parser.add_argument("--minimax-model", default="MiniMax-M2.7-highspeed")
    parser.add_argument("--minimax-env", default="/home/wsx/minimax_claude_env.sh")
    parser.add_argument("--kimi-model", default="kimi-for-coding/k2p6")
    parser.add_argument("--kimi-env", default="/home/wsx/kimi_claude_env.sh")
    parser.add_argument("--xiaomi-model", default="mimo-v2.5-pro")
    parser.add_argument("--xiaomi-env", default="/home/wsx/xiaomi_claude_env.sh")
    parser.add_argument("--sleep-when-all-limited", type=int, default=3600)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--retry-provider-failures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--history-output-root", action="append", type=Path, default=[])
    parser.add_argument("--adopt-output-root", type=Path)
    parser.add_argument("--adopt-pid", type=int)
    parser.add_argument("--adopt-provider", default="minimax")
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    args.benchmark_dir = (args.repo_root / args.benchmark_dir).resolve() if not args.benchmark_dir.is_absolute() else args.benchmark_dir
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    providers_by_name = {
        "minimax": Provider("minimax", args.minimax_model, args.minimax_env),
        "kimi": Provider("kimi", args.kimi_model, args.kimi_env),
        "xiaomi": Provider("xiaomi", args.xiaomi_model, args.xiaomi_env),
    }
    providers = [providers_by_name[name] for name in args.providers.split(",") if name in providers_by_name]
    if not providers:
        raise SystemExit("no valid providers configured")

    levels = parse_levels(args.levels, args.benchmark_dir)
    first_level_ids = parse_ids(args.first_level_ids)
    history_roots = [p.resolve() for p in args.history_output_root]
    attempts: list[Attempt] = []
    limited: set[str] = set()
    log_path = args.output_root / "supervisor.log"
    state_path = args.output_root / "supervisor_state.json"
    report_path = args.output_root / "supervisor_report.md"

    append_log(log_path, "supervisor start")

    if args.adopt_output_root:
        adopted = Attempt(
            index=0,
            provider=args.adopt_provider,
            output_root=str(args.adopt_output_root.resolve()),
            adopted=True,
            pid=args.adopt_pid,
            started_at=None,
        )
        attempts.append(adopted)
        write_state(state_path, attempts, limited, "waiting_adopted")
        write_report(report_path, attempts, "waiting_adopted")
        if args.adopt_pid:
            append_log(log_path, f"waiting for adopted pid={args.adopt_pid} output={adopted.output_root}")
            wait_pid(args.adopt_pid, args.poll_interval, log_path)
        adopted.ended_at = now()
        adopted.returncode = None
        adopted.fatal_reason = output_root_fatal(Path(adopted.output_root))
        _, adopted_provider_issue = collect_completed([Path(adopted.output_root)], args.retry_provider_failures)
        if is_api_limit_reason(adopted.fatal_reason) or adopted_provider_issue:
            limited.add(adopted.provider)
        append_log(log_path, f"adopted done provider={adopted.provider} fatal={adopted.fatal_reason}")

    attempt_index = len(attempts) + 1
    last_provider = attempts[-1].provider if attempts else None
    while True:
        roots = history_roots + [Path(a.output_root) for a in attempts]
        remaining_levels, remaining_by_first_level, _provider_issue_seen = remaining_plan(
            args.benchmark_dir,
            levels,
            first_level_ids,
            roots,
            args.retry_provider_failures,
        )
        if not remaining_levels:
            write_state(state_path, attempts, limited, "completed")
            write_report(report_path, attempts, "completed")
            append_log(log_path, "completed: no remaining cases")
            return 0

        first_level = remaining_levels[0]
        first_remaining_ids = remaining_by_first_level[first_level]
        selected: Provider | None = None
        for provider in provider_order(providers, last_provider):
            if provider.name not in limited:
                selected = provider
                break

        if selected is None:
            status = f"sleeping_all_providers_limited_{args.sleep_when_all_limited}s"
            write_state(state_path, attempts, limited, status)
            write_report(report_path, attempts, status)
            append_log(log_path, f"all providers limited: {sorted(limited)}; sleeping {args.sleep_when_all_limited}s")
            time.sleep(args.sleep_when_all_limited)
            limited.clear()
            last_provider = None
            continue

        if not Path(selected.env).exists():
            append_log(log_path, f"provider env missing: {selected.name} {selected.env}")
            limited.add(selected.name)
            continue

        write_state(state_path, attempts, limited, f"running_{selected.name}")
        write_report(report_path, attempts, f"running_{selected.name}")
        attempt = run_attempt(args, selected, attempt_index, remaining_levels, first_remaining_ids, log_path)
        attempts.append(attempt)
        attempt_index += 1
        last_provider = selected.name

        if is_api_limit_reason(attempt.fatal_reason):
            limited.add(selected.name)
            continue

        roots_after = history_roots + [Path(a.output_root) for a in attempts]
        remaining_after, _, _provider_issue_after = remaining_plan(
            args.benchmark_dir,
            levels,
            first_level_ids,
            roots_after,
            args.retry_provider_failures,
        )
        _, attempt_provider_issue = collect_completed([Path(attempt.output_root)], args.retry_provider_failures)
        if attempt_provider_issue and remaining_after:
            limited.add(selected.name)
            continue
        if attempt.returncode not in (0, 88) and remaining_after:
            status = f"stopped_non_retriable_rc_{attempt.returncode}"
            write_state(state_path, attempts, limited, status)
            write_report(report_path, attempts, status)
            append_log(log_path, status)
            return attempt.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
