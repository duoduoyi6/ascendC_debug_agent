#!/usr/bin/env python3
"""Build an auditable CANNBench operator-effort snapshot from archived artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
from collections import defaultdict
from typing import Any


SNAPSHOT_PRECISION = {
    "AdaptiveAvgPool3D": ((20, 20), None, "platform_blocked"),
    "AddRmsNormDynamicQuant": ((20, 20), (80, 80), "strict_pass"),
    "ApplyAdamW": ((20, 20), (80, 80), "strict_pass"),
    "ApplyRotaryPosEmb": ((20, 20), (80, 80), "strict_pass"),
    "ArgMax": ((20, 20), (80, 80), "strict_pass"),
    "Conv2D": ((20, 20), (80, 80), "strict_pass"),
    "Conv3DBackpropFilter": ((20, 20), (80, 80), "strict_pass"),
    "CrossEntropyLoss": ((20, 20), (77, 80), "incomplete"),
    "Cummin": ((20, 20), (80, 80), "strict_pass"),
    "DepthwiseConv2D": ((20, 20), (80, 80), "strict_pass"),
    "DequantSwigluQuant": ((20, 20), (80, 80), "strict_pass"),
    "Dilation2D": ((20, 20), (80, 80), "strict_pass"),
    "DynamicQuant": ((20, 20), (80, 80), "strict_pass"),
    "EngramGateFusion": ((20, 20), (9, 80), "deferred"),
    "Exp": ((20, 20), (80, 80), "strict_pass"),
    "ForeachAddcdivScalar": ((20, 20), (80, 80), "strict_pass"),
    "ForeachNorm": ((20, 20), (80, 80), "strict_pass"),
    "Gather": ((20, 20), (80, 80), "strict_pass"),
    "Gcd": ((20, 20), (80, 80), "strict_pass"),
    "Gelu": ((20, 20), (80, 80), "strict_pass"),
    "GridSampler3D": ((20, 20), (80, 80), "strict_pass"),
    "GroupNorm": ((20, 20), (80, 80), "strict_pass"),
    "GroupedMatmul": ((20, 20), (55, 80), "deferred"),
    "LSTM": ((20, 20), (61, 80), "deferred"),
    "MaskedScale": ((20, 20), (80, 80), "strict_pass"),
    "Maximum": ((20, 20), (79, 80), "incomplete"),
    "MhcSinkhorn": ((20, 20), (80, 80), "strict_pass"),
    "Mish": ((20, 20), (80, 80), "strict_pass"),
    "MoeFinalizeRouting": ((20, 20), (74, 80), "platform_blocked"),
    "MoeGatingTopKSoftmax": ((20, 20), (80, 80), "strict_pass"),
    "MoeReRouting": ((20, 20), (80, 80), "strict_pass"),
    "NMS": ((20, 20), (80, 80), "strict_pass"),
    "QuantMatmul": ((20, 20), (80, 80), "strict_pass"),
    "RmsNorm": ((20, 20), (80, 80), "strict_pass"),
    "ROIAlign": ((20, 20), (80, 80), "strict_pass"),
    "Scatter": ((20, 20), None, "hidden_inflight"),
    "Sigmoid": ((20, 20), (80, 80), "strict_pass"),
    "Softmax": ((20, 20), (80, 80), "strict_pass"),
    "StridedSlice": ((20, 20), (80, 80), "strict_pass"),
    "SwiGlu": ((20, 20), (80, 80), "strict_pass"),
    "TopK": ((20, 20), (80, 80), "strict_pass"),
    "Transpose": ((20, 20), (80, 80), "strict_pass"),
    "Unique": ((20, 20), (80, 80), "strict_pass"),
    "UnsortedSegmentSum": ((20, 20), (80, 80), "strict_pass"),
    "WeightQuantBatchMatmul": ((15, 20), None, "deferred"),
}


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


TARGET_BY_NORMALIZED = {normalize(name): name for name in SNAPSHOT_PRECISION}
TARGET_KEYS_LONGEST_FIRST = sorted(TARGET_BY_NORMALIZED, key=len, reverse=True)
NON_TARGET_PREFIXES = {
    "groupedmatmulswigluquant",
    "gqa",
    "gru",
    "mha",
    "mla",
    "mlaprolog",
    "sparseflashattention",
}
GENERIC_PATH_PARTS = {
    "experiments",
    "experimentcontrol",
    "outputs",
    "runtimeevidence",
    "tasks",
    "formaltasks",
    "websitesubmissions",
}


def canonical_operator(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    key = normalize(re.sub(r"^\d+_", "", value))
    if key in TARGET_BY_NORMALIZED:
        return TARGET_BY_NORMALIZED[key]
    return None


def parse_time(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def seq_time(value: Any) -> dt.datetime | None:
    if not isinstance(value, int) or value < 1_000_000_000_000_000_000:
        return None
    return dt.datetime.fromtimestamp(value / 1_000_000_000, tz=dt.timezone.utc)


def read_json(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def discover(root: pathlib.Path) -> list[pathlib.Path]:
    names = (
        "events.jsonl",
        "submission_record_initial.json",
        "standard_followup_manifest.json",
        "hidden_request_state.json",
        "hidden_followup_manifest.json",
        "hidden_registry_receipt.json",
        "JOB_ID",
        "HIDDEN_JOB_ID",
    )
    command = ["find", str(root / "outputs"), "-type", "f", "("]
    for index, name in enumerate(names):
        if index:
            command.append("-o")
        command.extend(["-name", name])
    command.extend([
        "-o", "-path", "*/precision_tuning/claude_results/*.json", ")", "-print"
    ])
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return [pathlib.Path(line) for line in completed.stdout.splitlines() if line]


def in_campaign_scope(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        relative = path.relative_to(root / "outputs")
    except ValueError:
        return False
    if not relative.parts:
        return False
    top = relative.parts[0]
    if top in {
        "cannbench_four_slot_k3_20260807",
        "cannbench_public20_cann91_debug_agent_k3_20260806",
        "cannbench_lstm_official_public_standard_20260810_203639",
    }:
        return True
    return top.startswith((
        "lstm_hidden_failure_continuation_",
        "lstm_cycle2_duplicate_candidate_continuation_",
        "lstm_provider_recovery_20260811_",
    ))


EXCLUDED_EVENT_PARTS = {
    "initial_evidence",
    "predecessor_task_history",
    "predecessor_evidence",
    "experiment_control",
    "source_snapshot",
    "released_slot_evidence",
    "website_submissions",
}


def path_operator(path: pathlib.Path) -> str | None:
    for part in reversed(path.parts):
        match = re.match(r"^\d+_(.+)$", part)
        if match:
            operator = canonical_operator(match.group(1))
            if operator:
                return operator
    for part in reversed(path.parts):
        key = normalize(part)
        if not key or key in GENERIC_PATH_PARTS:
            continue
        if any(key.startswith(prefix) for prefix in NON_TARGET_PREFIXES):
            continue
        for target_key in TARGET_KEYS_LONGEST_FIRST:
            if key.startswith(target_key):
                return TARGET_BY_NORMALIZED[target_key]
    return None


def is_actual_event_path(path: pathlib.Path) -> bool:
    parts = set(path.parts)
    return path.name == "events.jsonl" and not (parts & EXCLUDED_EVENT_PARTS)


def token_usage(result: dict[str, Any]) -> dict[str, float]:
    totals = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_creation": 0,
        "cost_usd": 0.0,
    }
    model_usage = result.get("modelUsage")
    if isinstance(model_usage, dict) and model_usage:
        for usage in model_usage.values():
            if not isinstance(usage, dict):
                continue
            totals["input"] += int(usage.get("inputTokens") or 0)
            totals["output"] += int(usage.get("outputTokens") or 0)
            totals["cache_read"] += int(usage.get("cacheReadInputTokens") or 0)
            totals["cache_creation"] += int(usage.get("cacheCreationInputTokens") or 0)
            totals["cost_usd"] += float(usage.get("costUSD") or 0.0)
        return totals
    usage = result.get("usage")
    if isinstance(usage, dict):
        totals["input"] = int(usage.get("input_tokens") or 0)
        totals["output"] = int(usage.get("output_tokens") or 0)
        totals["cache_read"] = int(usage.get("cache_read_input_tokens") or 0)
        totals["cache_creation"] = int(usage.get("cache_creation_input_tokens") or 0)
    totals["cost_usd"] = float(result.get("total_cost_usd") or 0.0)
    return totals


def baseline_record(event: dict[str, Any], path: pathlib.Path) -> dict[str, Any] | None:
    action = event.get("action") or {}
    if event.get("type") != "action_completed" or action.get("name") != "baseline_checkpoint":
        return None
    result = event.get("result") or {}
    objective = result.get("objective_validation") or {}
    full_eval = objective.get("full_eval")
    record: dict[str, Any] = {
        "timestamp": (seq_time(event.get("seq")) or dt.datetime.max.replace(tzinfo=dt.timezone.utc)).isoformat(),
        "failure_type": objective.get("failure_type") or result.get("failure_type"),
        "build_exit_code": objective.get("build_exit_code"),
        "verification_exit_code": objective.get("verification_exit_code"),
        "evidence_path": str(path),
        "coverage": "unknown",
        "passed_cases": None,
        "total_cases": None,
    }
    if isinstance(full_eval, dict) and full_eval.get("ran"):
        record["coverage"] = "public_full"
        record["passed_cases"] = full_eval.get("passed_cases")
        record["total_cases"] = full_eval.get("total_cases")
    elif objective.get("build_exit_code") not in (None, 0):
        record["coverage"] = "build_only"
    elif objective.get("verification_ran"):
        record["coverage"] = "target_case_only"
        metric = (result.get("checkpoint") or {}).get("best_metric") or {}
        rate = metric.get("case_pass_rate")
        if isinstance(rate, (int, float)):
            record["passed_cases"] = 1 if rate >= 100 else 0
            record["total_cases"] = 1
    return record


def post_attempt_validation(event: dict[str, Any], path: pathlib.Path) -> dict[str, Any] | None:
    action = event.get("action") or {}
    result = event.get("result") or {}
    objective = result.get("objective_validation")
    if (
        event.get("type") != "action_completed"
        or action.get("name") != "precision_gate"
        or not isinstance(objective, dict)
    ):
        return None

    checks = result.get("checks") or {}
    full_eval = objective.get("full_eval")
    record: dict[str, Any] = {
        "timestamp": (seq_time(event.get("seq")) or dt.datetime.max.replace(tzinfo=dt.timezone.utc)).isoformat(),
        "gate": result.get("gate"),
        "gate_passed": result.get("passed"),
        "failure_type": objective.get("failure_type") or result.get("failure_type"),
        "build_exit_code": objective.get("build_exit_code"),
        "verification_exit_code": objective.get("verification_exit_code"),
        "verification_ran": objective.get("verification_ran"),
        "coverage": "unknown",
        "passed_cases": None,
        "total_cases": None,
        "blocked_reason": None,
        "evidence_path": str(path),
    }
    if isinstance(full_eval, dict) and full_eval.get("ran"):
        record["coverage"] = "public_full"
        record["passed_cases"] = full_eval.get("passed_cases")
        record["total_cases"] = full_eval.get("total_cases")
    elif objective.get("build_exit_code") not in (None, 0):
        record["coverage"] = "build_only"
    elif objective.get("verification_ran") is False:
        record["coverage"] = "not_run"
        if checks.get("cannbench_source_pass") is False:
            record["blocked_reason"] = "source_integrity_gate"
        elif checks.get("cpp_regression_pass") is False:
            record["blocked_reason"] = "cpp_regression_gate"
        elif checks.get("anticheat_pass") is False:
            record["blocked_reason"] = "anti_cheat_gate"
        else:
            record["blocked_reason"] = result.get("gate") or "validation_gate"
    elif objective.get("verification_ran"):
        record["coverage"] = "verification_without_full_count"
    return record


def is_provider_error(result: dict[str, Any]) -> bool:
    status = result.get("api_error_status")
    return (
        status not in (None, "", 0, 200, "200")
        or result.get("session_outcome") == "provider_api_error"
        or result.get("subtype") == "provider_api_error"
    )


def add_job(
    jobs: dict[str, dict[str, Any]],
    *,
    job_id: Any,
    kind: str,
    operator: str | None,
    operator_priority: int,
    timestamp: dt.datetime | None,
    path: pathlib.Path,
) -> None:
    if not isinstance(job_id, str) or not job_id.startswith("job_") or not operator:
        return
    record = jobs.setdefault(job_id, {
        "job_id": job_id,
        "kind": kind,
        "operator": operator,
        "operator_priority": operator_priority,
        "timestamp": timestamp,
        "paths": [],
    })
    record["paths"].append(str(path))
    if timestamp and (record["timestamp"] is None or timestamp < record["timestamp"]):
        record["timestamp"] = timestamp
    if record["operator"] != operator:
        if operator_priority > record["operator_priority"]:
            record["operator"] = operator
            record["operator_priority"] = operator_priority
        elif operator_priority == record["operator_priority"] and operator_priority >= 2:
            raise RuntimeError(f"job {job_id} maps to two explicit operators")


STATUS_LABELS = {
    "strict_pass": "严格通过",
    "deferred": "待定",
    "platform_blocked": "平台阻塞",
    "incomplete": "未严格通过",
    "hidden_inflight": "隐藏评测中",
}


def render_markdown_table(payload: dict[str, Any]) -> str:
    lines = [
        "| 算子 | 初始可审计验证 | 每次 Debug-Agent attempt 后的本地官方验证 | 榜单快照（公开 + 隐藏） | 快照状态 | Debug-Agent attempts（其中有非零 token usage） | Agent tokens：输入 / 输出 / cache-read / 处理总量 | CANNBench jobs：标准 + 隐藏 = 总计 |",
        "|---|---:|---|---:|---|---:|---:|---:|",
    ]
    for operator, record in payload["operators"].items():
        initial = record.get("initial") or {}
        coverage = initial.get("coverage")
        passed = initial.get("passed_cases")
        total = initial.get("total_cases")
        failure = initial.get("failure_type") or "unknown"
        if failure == "build_failed" or coverage == "build_only":
            initial_text = "build_failed（未执行精度）"
        elif coverage == "public_full" and passed is not None:
            initial_text = f"{passed}/{total} public（{failure}）"
        elif coverage == "target_case_only" and passed is not None:
            initial_text = f"{passed}/{total} case 1（{failure}）"
        else:
            initial_text = f"{failure}（计数未归档）"

        snapshot = record["snapshot"]
        public = snapshot["public"]
        hidden = snapshot["hidden"]
        if hidden is None:
            hidden_text = "运行中" if snapshot["status"] == "hidden_inflight" else "未运行"
        else:
            hidden_text = f'{hidden["passed"]}/{hidden["total"]}'
        snapshot_text = f'{public["passed"]}/{public["total"]} + {hidden_text}'

        debug = record["debug_agent"]
        tokens = debug["tokens"]
        attempts = debug["unique_sessions"]
        token_sessions = debug["token_sessions_with_nonzero_usage"]
        trajectory_parts = []
        for attempt in debug["attempt_trajectory"]:
            sequence = attempt["sequence"]
            validation = attempt.get("validation")
            if validation is None:
                if attempt.get("provider_error"):
                    outcome = "provider_error（未验证）"
                elif not attempt.get("agent_success"):
                    outcome = "未形成候选（未验证）"
                else:
                    outcome = "无归档验证"
            elif validation["coverage"] == "public_full":
                outcome = f'{validation["passed_cases"]}/{validation["total_cases"]}'
                if validation.get("failure_type") not in (None, "success"):
                    outcome += f'（{validation["failure_type"]}）'
            elif validation["coverage"] == "build_only":
                outcome = "build_failed（未执行精度）"
            elif validation["coverage"] == "not_run":
                outcome = f'{validation.get("blocked_reason") or "验证门禁"}（未执行精度）'
            else:
                outcome = f'{validation.get("failure_type") or "验证结果无逐 case 计数"}（无 full 计数）'
            trajectory_parts.append(f"A{sequence}: {outcome}")
        trajectory_text = "<br>".join(trajectory_parts) or "无 Agent attempt"
        jobs = record["cannbench_jobs"]
        lines.append(
            f"| {operator} | {initial_text} | {trajectory_text} | {snapshot_text} | "
            f'{STATUS_LABELS[snapshot["status"]]} | '
            f'{attempts}（{token_sessions}） | '
            f'{tokens["input"]:,} / {tokens["output"]:,} / {tokens["cache_read"]:,} / '
            f'**{tokens["processed_total"]:,}** | {jobs["standard"]} + {jobs["hidden"]} = '
            f'**{jobs["total"]}** |'
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--snapshot-at", required=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--format", choices=("json", "markdown-table"), default="json")
    args = parser.parse_args()
    cutoff = parse_time(args.snapshot_at)
    if cutoff is None:
        raise SystemExit("invalid --snapshot-at")
    paths = discover(args.root)

    session_times: dict[str, dt.datetime] = {}
    session_event_operators: dict[str, str] = {}
    session_traces: dict[str, dict[str, Any]] = {}
    baselines: dict[str, list[dict[str, Any]]] = defaultdict(list)
    event_entry_failures: dict[str, list[tuple[dt.datetime, str, str]]] = defaultdict(list)
    for path in paths:
        if not in_campaign_scope(path, args.root):
            continue
        if not is_actual_event_path(path):
            continue
        operator = path_operator(path)
        if not operator:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        active_session_id: str | None = None
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            when = seq_time(event.get("seq"))
            if when and when > cutoff:
                continue
            baseline = baseline_record(event, path)
            if baseline:
                baselines[operator].append(baseline)
            if event.get("type") == "session_started" and when:
                failure = event.get("entry_failure_type")
                if isinstance(failure, str):
                    event_entry_failures[operator].append((when, failure, str(path)))
            action = event.get("action") or {}
            result = event.get("result") or {}
            if event.get("type") == "action_completed" and action.get("name") == "debug_worker":
                session_id = result.get("session_id")
                if isinstance(session_id, str) and when:
                    previous_operator = session_event_operators.get(session_id)
                    if previous_operator and previous_operator != operator:
                        raise RuntimeError(f"session {session_id} has conflicting actual-task events")
                    session_event_operators[session_id] = operator
                    previous = session_times.get(session_id)
                    if previous is None or when < previous:
                        session_times[session_id] = when
                    engine_attempt = (action.get("skill_args") or {}).get("attempt")
                    if not isinstance(engine_attempt, int):
                        match = re.search(r"attempt(-?\d+)", str(result.get("result_path") or ""))
                        engine_attempt = int(match.group(1)) if match else None
                    trace = {
                        "session_id": session_id,
                        "operator": operator,
                        "timestamp": when,
                        "engine_attempt": engine_attempt,
                        "agent_success": bool(result.get("success")),
                        "agent_subtype": result.get("subtype"),
                        "provider_error": is_provider_error(result),
                        "validation": None,
                        "event_path": str(path),
                    }
                    previous_trace = session_traces.get(session_id)
                    if previous_trace and previous_trace["operator"] != operator:
                        raise RuntimeError(f"session {session_id} has conflicting trace operators")
                    if previous_trace is None or when < previous_trace["timestamp"]:
                        session_traces[session_id] = trace
                    active_session_id = session_id

            validation = post_attempt_validation(event, path)
            if validation and active_session_id:
                trace = session_traces[active_session_id]
                previous_validation = trace["validation"]
                coverage_rank = {
                    "unknown": 0,
                    "verification_without_full_count": 1,
                    "not_run": 2,
                    "build_only": 2,
                    "public_full": 3,
                }
                if (
                    previous_validation is None
                    or coverage_rank[validation["coverage"]]
                    >= coverage_rank[previous_validation["coverage"]]
                ):
                    trace["validation"] = validation

    sessions: dict[str, dict[str, Any]] = {}
    jobs: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not in_campaign_scope(path, args.root):
            continue
        operator = path_operator(path)
        if path.parent.name == "claude_results" and path.suffix == ".json":
            result = read_json(path)
            if not result:
                continue
            session_id = result.get("session_id") or result.get("uuid")
            if not isinstance(session_id, str):
                match = re.search(r"([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})", path.name)
                session_id = match.group(1) if match else None
            if not session_id:
                continue
            operator = session_event_operators.get(session_id)
            if not operator:
                continue
            when = session_times.get(session_id)
            if when is None:
                when = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
            if when > cutoff:
                continue
            usage = token_usage(result)
            candidate = {
                "session_id": session_id,
                "operator": operator,
                "timestamp": when,
                "tokens": usage,
                "turns": result.get("num_turns"),
                "provider_error": is_provider_error(result),
                "path": str(path),
                "operator_priority": 3,
            }
            previous = sessions.get(session_id)
            richness = sum(int(usage[key]) for key in ("input", "output", "cache_read", "cache_creation"))
            previous_richness = -1
            if previous:
                previous_richness = sum(int(previous["tokens"][key]) for key in ("input", "output", "cache_read", "cache_creation"))
            should_replace = previous is None
            if previous and previous["operator"] != operator:
                if candidate["operator_priority"] > previous["operator_priority"]:
                    should_replace = True
                elif candidate["operator_priority"] < previous["operator_priority"]:
                    should_replace = False
                else:
                    raise RuntimeError(f"session {session_id} maps to two equally strong operators")
            elif previous and richness > previous_richness:
                should_replace = True
            if should_replace:
                sessions[session_id] = candidate
            continue

        data = read_json(path) if path.suffix == ".json" else None
        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        if path.name == "submission_record_initial.json" and data:
            selected = data.get("selected_operators") or []
            mapped = canonical_operator(selected[0]) if selected else operator
            add_job(jobs, job_id=data.get("job_id"), kind="standard", operator=mapped,
                    operator_priority=2 if selected else 1,
                    timestamp=parse_time(data.get("submitted_at")) or mtime, path=path)
        elif path.name == "standard_followup_manifest.json" and data:
            summary = data.get("summary") or {}
            selected = summary.get("selected_operators") or []
            mapped = canonical_operator(selected[0]) if selected else operator
            add_job(jobs, job_id=summary.get("job_id") or data.get("job_id"), kind="standard",
                    operator=mapped, operator_priority=2 if selected else 1,
                    timestamp=parse_time(data.get("updated_at")) or mtime, path=path)
        elif path.name == "hidden_request_state.json" and data:
            selected = data.get("selected_operators") or []
            mapped = canonical_operator(selected[0]) if selected else operator
            add_job(jobs, job_id=data.get("hidden_job_id"), kind="hidden", operator=mapped,
                    operator_priority=2 if selected else 1,
                    timestamp=parse_time(data.get("created_at")) or parse_time(data.get("request_started_at")) or mtime,
                    path=path)
        elif path.name == "hidden_followup_manifest.json" and data:
            summary = data.get("hidden_summary") or {}
            selected = summary.get("selected_operators") or data.get("selected_operators") or []
            mapped = canonical_operator(selected[0]) if selected else operator
            add_job(jobs, job_id=summary.get("job_id") or data.get("hidden_job_id"), kind="hidden",
                    operator=mapped, operator_priority=2 if selected else 1,
                    timestamp=parse_time(data.get("updated_at")) or mtime, path=path)
        elif path.name == "hidden_registry_receipt.json" and data:
            selected = data.get("operators") or []
            mapped = canonical_operator(selected[0]) if selected else operator
            add_job(jobs, job_id=data.get("job_id"), kind="hidden", operator=mapped,
                    operator_priority=2 if selected else 1, timestamp=mtime, path=path)
        elif path.name in {"JOB_ID", "HIDDEN_JOB_ID"}:
            try:
                job_id = path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            add_job(jobs, job_id=job_id, kind="hidden" if path.name == "HIDDEN_JOB_ID" else "standard",
                    operator=operator, operator_priority=1, timestamp=mtime, path=path)

    operators: dict[str, Any] = {}
    sessions_by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions.values():
        sessions_by_operator[session["operator"]].append(session)
    jobs_by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs.values():
        when = job["timestamp"]
        if when is None or when <= cutoff:
            jobs_by_operator[job["operator"]].append(job)

    for operator, (public, hidden, status) in SNAPSHOT_PRECISION.items():
        operator_sessions = sorted(sessions_by_operator[operator], key=lambda item: item["timestamp"])
        totals = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0, "cost_usd": 0.0}
        for session in operator_sessions:
            for key in totals:
                totals[key] += session["tokens"][key]
        totals["processed_total"] = totals["input"] + totals["output"] + totals["cache_read"] + totals["cache_creation"]
        initial = None
        if baselines[operator]:
            initial = min(baselines[operator], key=lambda item: item["timestamp"])
        elif event_entry_failures[operator]:
            when, failure, evidence = min(event_entry_failures[operator], key=lambda item: item[0])
            initial = {
                "timestamp": when.isoformat(),
                "failure_type": failure,
                "coverage": "unknown",
                "passed_cases": None,
                "total_cases": None,
                "evidence_path": evidence,
            }
        operator_jobs = jobs_by_operator[operator]
        standard_jobs = sorted(job["job_id"] for job in operator_jobs if job["kind"] == "standard")
        hidden_jobs = sorted(job["job_id"] for job in operator_jobs if job["kind"] == "hidden")
        attempt_trajectory = []
        for sequence, session in enumerate(operator_sessions, start=1):
            trace = session_traces.get(session["session_id"])
            attempt_trajectory.append({
                "sequence": sequence,
                "session_id": session["session_id"],
                "timestamp": session["timestamp"].isoformat(),
                "engine_attempt": None if trace is None else trace["engine_attempt"],
                "agent_success": None if trace is None else trace["agent_success"],
                "agent_subtype": None if trace is None else trace["agent_subtype"],
                "provider_error": session["provider_error"] or (False if trace is None else trace["provider_error"]),
                "validation": None if trace is None else trace["validation"],
                "event_path": None if trace is None else trace["event_path"],
            })
        operators[operator] = {
            "initial": initial,
            "snapshot": {
                "public": {"passed": public[0], "total": public[1]},
                "hidden": None if hidden is None else {"passed": hidden[0], "total": hidden[1]},
                "status": status,
            },
            "debug_agent": {
                "unique_sessions": len(operator_sessions),
                "provider_error_sessions": sum(1 for session in operator_sessions if session["provider_error"]),
                "token_sessions_with_nonzero_usage": sum(
                    1
                    for session in operator_sessions
                    if sum(
                        int(session["tokens"][key])
                        for key in ("input", "output", "cache_read", "cache_creation")
                    )
                    > 0
                ),
                "tokens": totals,
                "attempt_trajectory": attempt_trajectory,
                "sessions": [
                    {
                        **{key: value for key, value in session.items() if key != "timestamp"},
                        "timestamp": session["timestamp"].isoformat(),
                    }
                    for session in operator_sessions
                ],
            },
            "cannbench_jobs": {
                "standard": len(standard_jobs),
                "hidden": len(hidden_jobs),
                "total": len(standard_jobs) + len(hidden_jobs),
                "standard_job_ids": standard_jobs,
                "hidden_job_ids": hidden_jobs,
            },
        }

    payload = {
        "schema_version": 2,
        "snapshot_at": cutoff.isoformat(),
        "root": str(args.root),
        "operator_count": len(operators),
        "method": {
            "campaign_scope": [
                "cannbench_public20_cann91_debug_agent_k3_20260806",
                "cannbench_four_slot_k3_20260807",
                "official-public LSTM roots after retirement of the colleague 61-case harness",
            ],
            "attempt_unit": "unique Debug-Agent model session_id",
            "attempt_validation": "first post-debug_worker precision_gate objective_validation in the same actual task event stream",
            "token_total": "input + output + cache_read + cache_creation",
            "job_unit": "unique CANNBench standard or hidden job_id",
            "initial_priority": "earliest actual-task baseline_checkpoint before snapshot",
        },
        "operators": operators,
    }
    if args.format == "markdown-table":
        text = render_markdown_table(payload)
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
