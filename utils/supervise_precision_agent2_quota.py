#!/usr/bin/env python3
"""Quota-aware supervisor for precision debug agent2 runs.

This runner is intentionally separate from run_precision_debug_compare_claude.sh
because provider quota failures should not be finalized as task results.  When a
Claude/Kimi call returns a usage-limit error, the supervisor:

1. marks that key as cooling down,
2. resets the current task from the original selected_case_path,
3. requeues the same task,
4. pauses workers until another key is available, or until the cooldown expires.

Only non-quota Claude exits are followed by final verification and anticheat.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


API_LIMIT_STATUSES = {401, 402, 403, 429}
API_LIMIT_MARKERS = (
    "usage limit",
    "rate limit",
    "quota",
    "billing cycle",
    "credit",
    "too many requests",
    "membership",
    "permission",
    "subscribe",
    "unable to verify",
    "not have permission",
)
ALLOWED_TOOLS = "Bash,Read,Write,Edit,MultiEdit,Glob,Grep,Task"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return None


def read_text(path: Path, limit: int = 200_000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text[-limit:] if len(text) > limit else text


def parse_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def is_kimi_coding_provider(base_url: str) -> bool:
    return "api.kimi.com/coding" in (base_url or "").lower()


def quota_window_state(detail: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(detail, dict):
        return None
    limit = parse_float(detail.get("limit"))
    remaining = parse_float(detail.get("remaining"))
    used = parse_float(detail.get("used"))
    if remaining is None and limit is not None and used is not None:
        remaining = max(0.0, limit - used)
    if used is None and limit is not None and remaining is not None:
        used = max(0.0, limit - remaining)
    utilization = None
    if limit and used is not None:
        utilization = max(0.0, min(100.0, used / limit * 100.0))
    return {
        "limit": limit,
        "used": used,
        "remaining": remaining,
        "utilization_percent": utilization,
        "reset_time": detail.get("resetTime"),
    }


def query_kimi_usage(provider: "Provider", timeout_sec: int) -> dict[str, Any]:
    if not is_kimi_coding_provider(provider.base_url):
        return {
            "supported": False,
            "success": None,
            "queried_at": now(),
        }

    req = urllib.request.Request(
        "https://api.kimi.com/coding/v1/usages",
        headers={
            "Authorization": f"Bearer {provider.api_key}",
            "Accept": "application/json",
            "User-Agent": "claude-code",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            status = resp.status
            body = resp.read(20_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read(4_000).decode("utf-8", "replace")
    except Exception as exc:
        return {
            "supported": True,
            "success": False,
            "queried_at": now(),
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }

    state: dict[str, Any] = {
        "supported": True,
        "success": 200 <= int(status) < 300,
        "queried_at": now(),
        "http_status": int(status),
    }
    try:
        data = json.loads(body)
    except Exception:
        state["error"] = body[:500]
        return state

    if not state["success"]:
        error = data.get("error") if isinstance(data, dict) else None
        state["error"] = error if error is not None else data
        return state

    usage = data.get("usage") if isinstance(data, dict) else None
    weekly = quota_window_state(usage)
    if weekly:
        state["weekly_limit"] = weekly

    five_hour = None
    limits = data.get("limits") if isinstance(data, dict) else None
    if isinstance(limits, list) and limits:
        first = limits[0] if isinstance(limits[0], dict) else {}
        five_hour = quota_window_state(first.get("detail"))
        window = first.get("window")
        if five_hour and isinstance(window, dict):
            five_hour["window_duration"] = parse_float(window.get("duration"))
            five_hour["window_time_unit"] = window.get("timeUnit")
    if five_hour:
        state["five_hour"] = five_hour

    total_quota = data.get("totalQuota") if isinstance(data, dict) else None
    total = quota_window_state(total_quota)
    if total:
        state["total_quota"] = total

    parallel = data.get("parallel") if isinstance(data, dict) else None
    if isinstance(parallel, dict):
        state["parallel_limit"] = parse_float(parallel.get("limit"))
    authentication = data.get("authentication") if isinstance(data, dict) else None
    if isinstance(authentication, dict):
        state["authentication_scope"] = authentication.get("scope")
        state["authentication_method"] = authentication.get("method")
    return state


def append_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{now()}] {message}\n")


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def remove_tree_with_retry(path: Path, attempts: int = 6) -> None:
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


@dataclass
class Provider:
    name: str
    api_key: str
    model: str
    base_url: str
    cooldown_until: float = 0.0
    quota_hits: int = 0
    attempts: int = 0
    usage_checked_at: float = 0.0
    usage_state: dict[str, Any] = field(default_factory=dict)

    def public_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "cooldown_until": datetime.fromtimestamp(self.cooldown_until).strftime("%F %T")
            if self.cooldown_until
            else None,
            "cooldown_remaining_sec": max(0, int(self.cooldown_until - time.time())),
            "quota_hits": self.quota_hits,
            "attempts": self.attempts,
            "usage": self.usage_state
            or {
                "supported": is_kimi_coding_provider(self.base_url),
                "success": None,
                "queried_at": None,
            },
        }


@dataclass(frozen=True)
class Target:
    level: str
    case_id: int
    op_name: str
    category: str
    source_dir: Path
    task_dir: Path

    @property
    def task_rel(self) -> str:
        return f"{self.level}/{self.task_dir.name}"

    @property
    def report_task(self) -> str:
        return f"{self.level}/{self.case_id} {self.op_name}"


@dataclass
class SupervisorState:
    started_at: str
    output_dir: str
    variant: str
    total_targets: int
    skipped_existing_success: list[str] = field(default_factory=list)
    completed: list[dict[str, Any]] = field(default_factory=list)
    quota_events: list[dict[str, Any]] = field(default_factory=list)
    active: dict[str, str] = field(default_factory=dict)
    queue_remaining: int = 0
    providers: list[dict[str, Any]] = field(default_factory=list)
    ended_at: str | None = None


def parse_key_config(path: Path) -> list[Provider]:
    data = read_json(path)
    if not data:
        raise SystemExit(f"missing or invalid key config: {path}")
    default_model = data.get("model", "kimi-for-coding/k2p6")
    default_base_url = data.get("base_url", "https://api.kimi.com/coding/")
    providers: list[Provider] = []
    for item in data.get("keys", []):
        name = item.get("name")
        api_key = item.get("api_key")
        if not name or not api_key:
            continue
        providers.append(
            Provider(
                name=name,
                api_key=api_key,
                model=item.get("model", default_model),
                base_url=item.get("base_url", default_base_url),
            )
        )
    if not providers:
        raise SystemExit(f"no usable keys in key config: {path}")
    return providers


def task_name_from_row(row: dict[str, str]) -> str:
    selected = Path(row.get("selected_case_path") or "")
    if selected.name:
        return selected.name
    return f"{int(row['case_id']):03d}_{row['op_name']}"


def build_targets(
    source_map: Path,
    output_dir: Path,
    variant: str,
    include_cheat: bool,
) -> list[Target]:
    targets: list[Target] = []
    variant_tasks = output_dir / variant / "tasks"
    for row in csv.DictReader(source_map.open(encoding="utf-8")):
        if row.get("verify_failure_type") != "precision_failed":
            continue
        if not include_cheat and row.get("anticheat_verdict") != "CLEAN":
            continue
        case_id = int(row["case_id"])
        name = task_name_from_row(row)
        source_dir = Path(row["selected_case_path"])
        task_dir = variant_tasks / row["level"] / name
        targets.append(
            Target(
                level=row["level"],
                case_id=case_id,
                op_name=row["op_name"],
                category=row.get("category", ""),
                source_dir=source_dir,
                task_dir=task_dir,
            )
        )
    return targets


def has_existing_success(target: Target, variant: str) -> bool:
    status = read_json(target.task_dir / ".verify_status" / "latest.json") or {}
    claude = read_json(target.task_dir / f"_claude_{variant}.json") or {}
    anticheat = read_json(target.task_dir / f"_anticheat_{variant}.json") or {}
    verdict = str(anticheat.get("verdict") or anticheat.get("status") or "").upper()
    return (
        status.get("failure_type") == "success"
        and claude.get("is_error") is False
        and claude.get("api_error_status") in (None, "")
        and verdict in {"CLEAN", "PASS", "OK"}
    )


def reset_task(target: Target) -> None:
    if not target.source_dir.exists():
        raise FileNotFoundError(f"missing source task dir: {target.source_dir}")
    if target.task_dir.exists():
        remove_tree_with_retry(target.task_dir)
    target.task_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(target.source_dir, target.task_dir)


def make_prompt(task_dir: Path, npu: str, attempts: int) -> str:
    return f"""Use the Claude agent "ascendc-debug-agent-constructive" for this non-interactive debug task.

Input:
debug {task_dir} npu={npu}

Hard constraints:
- Only read/write files under task_dir, plus repo reference docs and archive_tasks.
- Do not read any other outputs/ directory.
- max attempts is controlled by ASCENDC_DEBUG_MAX_ATTEMPTS={attempts}.
- Complete the required debug_trace.md and debug_status.json before exiting.
"""


def is_quota_result(result_file: Path, log_tail: str = "") -> tuple[bool, str]:
    data = read_json(result_file) or {}
    text_parts = [log_tail]
    for key in ("result", "message", "error"):
        value = data.get(key)
        if value is not None:
            text_parts.append(str(value))
    text = "\n".join(text_parts).lower()
    status = data.get("api_error_status")
    if status in API_LIMIT_STATUSES and any(marker in text for marker in API_LIMIT_MARKERS):
        return True, f"api_error_status={status}"
    if data.get("is_error") and any(marker in text for marker in API_LIMIT_MARKERS):
        return True, "usage_limit_marker"
    return False, ""


def final_verify(workdir: Path, variant: str, target: Target, npu: str, worker_log: Path) -> None:
    verify_logs = target.task_dir / ".verify_logs"
    verify_logs.mkdir(parents=True, exist_ok=True)
    stdout = verify_logs / f"{variant}_final.stdout"
    stderr = verify_logs / f"{variant}_final.stderr"
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(npu)
    with stdout.open("w", encoding="utf-8") as so, stderr.open("w", encoding="utf-8") as se:
        proc = subprocess.run(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=30",
                "600",
                "python3",
                "utils/verification_ascendc.py",
                str(target.task_dir),
            ],
            cwd=workdir,
            env=env,
            stdout=so,
            stderr=se,
        )
    subprocess.run(
        [
            "python3",
            "utils/classify_verify_result.py",
            "--exit-code",
            str(proc.returncode),
            "--stdout-path",
            str(stdout),
            "--stderr-path",
            str(stderr),
            "--task-dir",
            str(target.task_dir),
            "--phase",
            "9",
            "--attempt",
            "999",
            "--write-status",
        ],
        cwd=workdir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with (target.task_dir / f"_anticheat_{variant}.json").open("w", encoding="utf-8") as out, (
        target.task_dir / f"_anticheat_{variant}.stderr"
    ).open("w", encoding="utf-8") as err:
        subprocess.run(
            ["python3", "skills/ascendc/ascendc-debug/scripts/anticheat.py", "verify", "--json", str(target.task_dir)],
            cwd=workdir,
            stdout=out,
            stderr=err,
        )
    append_line(worker_log, f"[final_verify] rc={proc.returncode}")


def kill_stale_task_processes(output_dir: Path, variant: str, supervisor_log: Path) -> None:
    task_root = str(output_dir / variant / "tasks")
    markers = ("claude --bare", "timeout --signal=TERM", "verification_ascendc.py")
    try:
        ps = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True, errors="replace")
    except Exception as exc:
        append_line(supervisor_log, f"stale_process_scan_failed error={exc!r}")
        return

    pids: list[int] = []
    for line in ps.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, cmd = parts
        if task_root not in cmd or not any(marker in cmd for marker in markers):
            continue
        try:
            pids.append(int(pid_text))
        except ValueError:
            continue

    if not pids:
        return
    append_line(supervisor_log, f"kill_stale_task_processes pids={pids}")
    for pid in pids:
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
    time.sleep(5)
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


class QuotaSupervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.workdir = args.workdir
        self.output_dir = args.output
        self.variant_dir = self.output_dir / args.variant
        self.variant_dir.mkdir(parents=True, exist_ok=True)
        (self.variant_dir / "tasks").mkdir(parents=True, exist_ok=True)
        self.report = self.variant_dir / "batch_report.md"
        self.supervisor_log = self.variant_dir / "quota_supervisor.log"
        self.state_path = self.variant_dir / "quota_supervisor_state.json"
        self.providers = parse_key_config(args.key_config)
        self.provider_cv = threading.Condition()
        self.report_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.usage_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.targets = build_targets(args.source_map, args.output, args.variant, args.include_cheat)
        self.todo: queue.Queue[Target] = queue.Queue()
        self.state = SupervisorState(
            started_at=now(),
            output_dir=str(args.output),
            variant=args.variant,
            total_targets=len(self.targets),
        )

    def refresh_usage_if_due(self, force: bool = False) -> None:
        if self.args.disable_usage_query:
            return
        current = time.time()
        with self.usage_lock:
            for provider in self.providers:
                if not is_kimi_coding_provider(provider.base_url):
                    if not provider.usage_state:
                        provider.usage_state = {
                            "supported": False,
                            "success": None,
                            "queried_at": now(),
                        }
                    continue
                if not force and current - provider.usage_checked_at < self.args.usage_refresh_sec:
                    continue
                provider.usage_state = query_kimi_usage(provider, self.args.usage_timeout)
                provider.usage_checked_at = current

    def save_state(self) -> None:
        self.refresh_usage_if_due()
        with self.state_lock:
            self.state.queue_remaining = self.todo.qsize()
            self.state.providers = [p.public_state() for p in self.providers]
            atomic_write_json(self.state_path, self.state.__dict__)

    def heartbeat(self) -> None:
        while not self.stop_event.wait(timeout=self.args.state_refresh_sec):
            self.save_state()

    def append_report_row(self, target: Target, status: str, elapsed: float, worker: str) -> None:
        with self.report_lock:
            with self.report.open("a", encoding="utf-8") as f:
                f.write(f"| {target.report_task} | {status} | {int(elapsed)} | {worker} |\n")

    def choose_provider(self, worker_log: Path) -> Provider:
        while True:
            with self.provider_cv:
                current = time.time()
                for provider in self.providers:
                    if provider.cooldown_until <= current:
                        provider.attempts += 1
                        self.save_state()
                        return provider
                earliest = min(provider.cooldown_until for provider in self.providers)
                wait = max(1, int(earliest - current))
                names = ", ".join(f"{p.name}:{max(0, int(p.cooldown_until - current))}s" for p in self.providers)
                append_line(worker_log, f"[quota_wait] all providers cooling down ({names}); sleep {wait}s")
                self.provider_cv.wait(timeout=wait)

    def mark_quota(self, provider: Provider, target: Target, reason: str, worker: str) -> None:
        with self.provider_cv:
            provider.quota_hits += 1
            provider.cooldown_until = time.time() + self.args.wait_sec
            event = {
                "time": now(),
                "provider": provider.name,
                "task": target.task_rel,
                "worker": worker,
                "reason": reason,
                "cooldown_sec": self.args.wait_sec,
            }
            with self.state_lock:
                self.state.quota_events.append(event)
            self.provider_cv.notify_all()
        self.save_state()

    def run_claude(self, target: Target, npu: str, provider: Provider, worker_log: Path) -> tuple[int, bool, str, float]:
        result_file = target.task_dir / f"_claude_{self.args.variant}.json"
        prompt = make_prompt(target.task_dir, npu, self.args.max_attempts)
        env = os.environ.copy()
        env.update(
            {
                "ANTHROPIC_MODEL": provider.model,
                "ANTHROPIC_BASE_URL": provider.base_url,
                "ANTHROPIC_API_KEY": provider.api_key,
                "ASCEND_RT_VISIBLE_DEVICES": str(npu),
                "ASCENDC_DEBUG_MAX_ATTEMPTS": str(self.args.max_attempts),
            }
        )
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        shell = (
            f"set -e; "
            f"[ -f {self.args.tilelang_env!s} ] && source {self.args.tilelang_env!s}; "
            f"cd {self.workdir!s}; "
            f"timeout --signal=TERM --kill-after=30 {self.args.timeout} "
            f"{self.args.claude_bin} --bare -p "
            f"--model \"$ANTHROPIC_MODEL\" "
            f"--add-dir {self.workdir!s} "
            f"--allowedTools {ALLOWED_TOOLS} "
            f"--output-format json "
            f"--agent ascendc-debug-agent-constructive "
            f"{json.dumps(prompt)}"
        )
        started = time.time()
        with result_file.open("w", encoding="utf-8") as out, worker_log.open("a", encoding="utf-8") as log:
            proc = subprocess.run(["bash", "-lc", shell], cwd=self.workdir, env=env, stdout=out, stderr=log)
        elapsed = time.time() - started
        log_tail = read_text(worker_log, limit=80_000)
        quota, reason = is_quota_result(result_file, log_tail)
        return proc.returncode, quota, reason, elapsed

    def worker(self, npu: str) -> None:
        worker_name = f"npu{npu}"
        worker_log = self.variant_dir / f"worker_{worker_name}.log"
        worker_log.write_text("", encoding="utf-8")
        while True:
            try:
                target = self.todo.get_nowait()
            except queue.Empty:
                return
            with self.state_lock:
                self.state.active[worker_name] = target.task_rel
            self.save_state()

            try:
                reset_task(target)
                provider = self.choose_provider(worker_log)
                append_line(
                    worker_log,
                    f"[task] {target.task_rel} npu={npu} provider={provider.name} task_dir={target.task_dir}",
                )
                rc, quota, reason, elapsed = self.run_claude(target, npu, provider, worker_log)
                if quota:
                    append_line(
                        worker_log,
                        f"[quota] provider={provider.name} task={target.task_rel} reason={reason}; requeue",
                    )
                    self.mark_quota(provider, target, reason, worker_name)
                    reset_task(target)
                    self.todo.put(target)
                    continue

                final_verify(self.workdir, self.args.variant, target, npu, worker_log)
                if rc == 0:
                    status = "claude_ok"
                elif rc in (124, 137, 143):
                    status = "timeout"
                else:
                    status = f"claude_rc_{rc}"
                self.append_report_row(target, status, elapsed, worker_name)
                with self.state_lock:
                    self.state.completed.append(
                        {
                            "time": now(),
                            "task": target.task_rel,
                            "status": status,
                            "elapsed_sec": int(elapsed),
                            "worker": worker_name,
                            "provider": provider.name,
                        }
                    )
                append_line(worker_log, f"[done] status={status} elapsed={int(elapsed)} provider={provider.name}")
            except Exception as exc:  # Keep the supervisor alive for other tasks.
                elapsed = 0
                status = "supervisor_error"
                self.append_report_row(target, status, elapsed, worker_name)
                append_line(worker_log, f"[supervisor_error] task={target.task_rel} error={exc!r}")
                with self.state_lock:
                    self.state.completed.append(
                        {
                            "time": now(),
                            "task": target.task_rel,
                            "status": status,
                            "elapsed_sec": elapsed,
                            "worker": worker_name,
                            "error": repr(exc),
                        }
                    )
            finally:
                with self.state_lock:
                    self.state.active.pop(worker_name, None)
                self.todo.task_done()
                self.save_state()

    def prepare(self) -> None:
        append_line(self.supervisor_log, "prepare")
        kill_stale_task_processes(self.output_dir, self.args.variant, self.supervisor_log)
        skipped: list[Target] = []
        pending: list[Target] = []
        for target in self.targets:
            if self.args.skip_existing_success and has_existing_success(target, self.args.variant):
                skipped.append(target)
            else:
                pending.append(target)
        with self.report.open("w", encoding="utf-8") as f:
            f.write(f"# precision debug batch: {self.args.variant}\n\n")
            f.write(f"- start: {now()}\n")
            f.write(f"- supervisor: quota-aware\n")
            f.write(f"- npus: {','.join(self.args.npus)}\n")
            f.write(f"- timeout: {self.args.timeout}\n")
            f.write(f"- max_attempts: {self.args.max_attempts}\n")
            f.write(f"- wait_sec_on_quota: {self.args.wait_sec}\n")
            f.write("\n| task | status | elapsed_sec | worker |\n")
            f.write("|------|--------|-------------|--------|\n")
            for target in skipped:
                f.write(f"| {target.report_task} | existing_success | 0 | preserved |\n")
        for target in skipped:
            self.state.skipped_existing_success.append(target.task_rel)
        for target in pending:
            reset_task(target)
            self.todo.put(target)
        append_line(self.supervisor_log, f"targets={len(self.targets)} skipped={len(skipped)} pending={len(pending)}")
        self.save_state()

    def run(self) -> None:
        self.prepare()
        heartbeat = threading.Thread(target=self.heartbeat, name="state-heartbeat", daemon=True)
        heartbeat.start()
        threads = [threading.Thread(target=self.worker, name=f"worker-npu{npu}", args=(npu,)) for npu in self.args.npus]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.stop_event.set()
        heartbeat.join(timeout=5)
        with self.report.open("a", encoding="utf-8") as f:
            f.write(f"- end: {now()}\n")
        self.state.ended_at = now()
        self.save_state()
        subprocess.run(
            ["python3", str(self.workdir / "utils/summarize_precision_debug_compare.py"), "--run-dir", str(self.output_dir)],
            cwd=self.workdir,
        )
        append_line(self.supervisor_log, "done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-config", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path("/home/wsx/AscendOpGenAgent"))
    parser.add_argument("--variant", default="agent2")
    parser.add_argument("--npus", default="4,5")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--include-cheat", action="store_true")
    parser.add_argument("--wait-sec", type=int, default=3600)
    parser.add_argument("--state-refresh-sec", type=int, default=60)
    parser.add_argument("--usage-refresh-sec", type=int, default=300)
    parser.add_argument("--usage-timeout", type=int, default=10)
    parser.add_argument("--disable-usage-query", action="store_true")
    parser.add_argument("--tilelang-env", type=Path, default=Path("/home/wsx/tilelang-ascend/set_env.sh"))
    parser.add_argument("--claude-bin", default=os.environ.get("CLAUDE_BIN", "claude"))
    parser.add_argument("--no-skip-existing-success", dest="skip_existing_success", action="store_false")
    parser.set_defaults(skip_existing_success=True)
    args = parser.parse_args()
    args.npus = [item.strip() for item in args.npus.split(",") if item.strip()]
    return args


def main() -> int:
    supervisor = QuotaSupervisor(parse_args())
    supervisor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
