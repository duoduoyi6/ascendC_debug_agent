#!/usr/bin/env python3
"""Assign cycle tasks across currently usable Kimi providers.

The assignment is deterministic and sticky for the lifetime of one task run.
API keys are written only to mode-0600 env files; queue and manifest artifacts
contain provider names and env paths, never key material.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from supervise_ascendc_debug_batch_cc_quota import (
    Provider,
    load_providers,
    query_kimi_usage,
    write_provider_env,
)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def remaining_percent(window: Any) -> float | None:
    """Return a normalized remaining percentage for a quota window."""

    if not isinstance(window, dict):
        return None
    limit = _number(window.get("limit"))
    remaining = _number(window.get("remaining"))
    used = _number(window.get("used"))
    if remaining is None and limit is not None and used is not None:
        remaining = max(0.0, limit - used)
    if remaining is None:
        return None
    if limit is None or limit <= 0:
        return max(0.0, remaining)
    return max(0.0, min(100.0, remaining * 100.0 / limit))


def provider_capacity(usage: dict[str, Any]) -> float | None:
    """Use the tighter of the five-hour and weekly windows as capacity."""

    if usage.get("supported") and not usage.get("success"):
        return None
    windows = [
        remaining_percent(usage.get("five_hour")),
        remaining_percent(usage.get("weekly_limit")),
    ]
    known = [value for value in windows if value is not None]
    if usage.get("supported") and len(known) < 2:
        return None
    return min(known) if known else 100.0


def choose_eligible(
    providers: list[Provider],
    usage_by_name: dict[str, dict[str, Any]],
    min_remaining: float,
) -> tuple[list[tuple[Provider, float, int]], str]:
    """Select healthy providers, degrading to any positive quota if needed."""

    candidates: list[tuple[Provider, float, int]] = []
    positive: list[tuple[Provider, float, int]] = []
    for provider in providers:
        usage = usage_by_name.get(provider.name, {})
        capacity = provider_capacity(usage)
        if capacity is None or capacity <= 0:
            continue
        parallel = int(_number(usage.get("parallel_limit")) or 1)
        item = (provider, capacity, max(1, parallel))
        positive.append(item)
        if capacity >= min_remaining:
            candidates.append(item)
    if candidates:
        return candidates, "safety_margin"
    if positive:
        return positive, "degraded_positive_quota"
    return [], "provider_pool_exhausted"


def weighted_assign(
    tasks: list[str],
    eligible: list[tuple[Provider, float, int]],
) -> list[tuple[str, Provider]]:
    """Deterministic weighted least-loaded assignment with parallel caps."""

    assigned = {provider.name: 0 for provider, _, _ in eligible}
    result: list[tuple[str, Provider]] = []
    order = {provider.name: index for index, (provider, _, _) in enumerate(eligible)}
    for task in tasks:
        available = [
            item for item in eligible if assigned[item[0].name] < item[2]
        ]
        if not available:
            # The batch worker count normally keeps us below these limits. If
            # not, retain deterministic balancing rather than dropping tasks.
            available = eligible
        unused = [item for item in available if assigned[item[0].name] == 0]
        if unused:
            # Spread first: every usable key receives one task before any key
            # receives a second task. This reduces correlated five-hour burns.
            available = unused
        provider, _, _ = max(
            available,
            key=lambda item: (
                item[1] / (assigned[item[0].name] + 1),
                -assigned[item[0].name],
                -order[item[0].name],
            ),
        )
        assigned[provider.name] += 1
        result.append((task, provider))
    return result


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def write_assignment(
    *,
    output: Path,
    queue_path: Path,
    tasks: list[str],
    providers: list[Provider],
    usage: list[dict[str, Any]],
    min_remaining: float,
) -> dict[str, Any]:
    usage_by_name = {str(item.get("name")): item for item in usage}
    eligible, policy = choose_eligible(providers, usage_by_name, min_remaining)
    if not eligible:
        return {
            "status": "provider_pool_exhausted",
            "policy": policy,
            "min_remaining_percent": min_remaining,
            "usage": usage,
            "assignments": [],
        }

    env_root = output / ".provider_env"
    env_paths: dict[str, Path] = {}
    for provider, _, _ in eligible:
        env_path = env_root / f"{_safe_name(provider.name)}.env"
        write_provider_env(env_path, provider)
        env_paths[provider.name] = env_path

    assignments = weighted_assign(tasks, eligible)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("w", encoding="utf-8") as queue:
        for task, provider in assignments:
            queue.write(f"{task}\t{provider.name}\t{env_paths[provider.name]}\n")

    return {
        "status": "assigned",
        "policy": policy,
        "min_remaining_percent": min_remaining,
        "eligible_providers": [
            {
                "name": provider.name,
                "capacity_percent": capacity,
                "parallel_limit": parallel,
            }
            for provider, capacity, parallel in eligible
        ],
        "usage": usage,
        "assignments": [
            {
                "task_dir": task,
                "provider": provider.name,
                "provider_env": str(env_paths[provider.name]),
            }
            for task, provider in assignments
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-config", type=Path, required=True)
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--min-remaining", type=float, default=10.0)
    parser.add_argument("--usage-timeout", type=int, default=20)
    args = parser.parse_args()

    tasks = [
        line.strip()
        for line in args.tasks_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    providers = load_providers(args.key_config)
    usage = [query_kimi_usage(provider, args.usage_timeout) for provider in providers]
    payload = write_assignment(
        output=args.output,
        queue_path=args.queue,
        tasks=tasks,
        providers=providers,
        usage=usage,
        min_remaining=max(0.0, args.min_remaining),
    )
    payload.update(
        {
            "key_config": str(args.key_config),
            "task_count": len(tasks),
            "provider_names": [provider.name for provider in providers],
        }
    )
    manifest = args.output / "provider_assignments.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if payload["status"] == "assigned":
        print(manifest)
        return 0
    print("no provider has positive five-hour and weekly quota", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
