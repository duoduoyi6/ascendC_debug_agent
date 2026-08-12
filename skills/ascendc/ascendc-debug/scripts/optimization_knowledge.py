#!/usr/bin/env python3
"""Read-only retrieval for the general AscendC optimization knowledge base.

Unlike ``precision_knowledge.py``, this tool deliberately has no mutation
command.  Promotion is an offline, evidence-reviewed operation.  The CLI is
safe to use in shadow runs and in a future post-correctness optimization stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


STATUS_RANK = {"supported": 1, "promoted": 2}
REQUIRED_FIELDS = {
    "knowledge_id",
    "title",
    "patterns",
    "op_types",
    "feature",
    "reason",
    "fix",
    "type",
    "status",
    "validation",
    "evidence",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokens(value: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", value or "")
    result: set[str] = set()
    for word in words:
        normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word).lower()
        for part in re.split(r"[_\s]+", normalized):
            if len(part) >= 2:
                result.add(part)
    return result


def _load(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("optimization KB must be a JSON list")
    invalid = [index for index, entry in enumerate(data) if not REQUIRED_FIELDS <= set(entry)]
    if invalid:
        raise ValueError(f"invalid optimization KB entries: {invalid[:5]}")
    return data


def _entry_text(entry: dict) -> str:
    return " ".join(
        [
            entry.get("title", ""),
            entry.get("feature", ""),
            entry.get("reason", ""),
            entry.get("fix", ""),
            entry.get("applicability", ""),
            " ".join(entry.get("patterns") or []),
            " ".join(entry.get("op_types") or []),
        ]
    )


def search_entries(
    entries: list[dict],
    *,
    op_type: str,
    op_name: str,
    patterns: list[str],
    bottlenecks: list[str],
    phase: str,
    min_status: str,
    top_k: int,
) -> list[dict]:
    minimum = STATUS_RANK[min_status]
    query_tokens = _tokens(" ".join([op_name, op_type, *patterns, *bottlenecks]))
    ranked = []
    for entry in entries:
        if STATUS_RANK.get(entry.get("status"), 0) < minimum:
            continue
        score = 0.0
        reasons = []
        entry_types = {value.lower() for value in entry.get("op_types") or []}
        entry_patterns = {value.lower() for value in entry.get("patterns") or []}
        if op_type.lower() in entry_types or "all" in entry_types:
            score += 4.0 if op_type.lower() in entry_types else 1.0
            reasons.append("op_type")
        pattern_hits = entry_patterns.intersection(value.lower() for value in patterns)
        if pattern_hits:
            score += 4.0 * len(pattern_hits)
            reasons.append("patterns=" + ",".join(sorted(pattern_hits)))
        entry_tokens = _tokens(_entry_text(entry))
        token_hits = query_tokens.intersection(entry_tokens)
        if token_hits:
            score += min(4.0, 0.5 * len(token_hits))
            reasons.append("keywords=" + ",".join(sorted(token_hits)[:8]))
        bottleneck_hits = {
            value.lower()
            for value in bottlenecks
            if value.lower() in entry_patterns or _tokens(value).intersection(entry_tokens)
        }
        if bottleneck_hits:
            score += (3.0 if phase == "fine" else 1.0) * len(bottleneck_hits)
            reasons.append("bottlenecks=" + ",".join(sorted(bottleneck_hits)))
        if entry.get("status") == "promoted":
            score += 0.5
            reasons.append("promoted")
        if not reasons:
            continue
        ranked.append(
            {
                "score": round(score, 3),
                "match_reasons": reasons,
                "knowledge": entry,
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["knowledge"]["knowledge_id"]))
    return ranked[:top_k]


def _append_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rows = []
    if not isinstance(rows, list):
        rows = []
    rows.append(payload)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--kb-path", type=Path, required=True)
    search = sub.add_parser("search")
    search.add_argument("--kb-path", type=Path, required=True)
    search.add_argument("--op-type", default="unknown")
    search.add_argument("--op-name", default="unknown")
    search.add_argument("--pattern", action="append", default=[])
    search.add_argument("--bottleneck", action="append", default=[])
    search.add_argument("--phase", choices=("coarse", "fine"), default="coarse")
    search.add_argument("--min-status", choices=("supported", "promoted"), default="supported")
    search.add_argument("--top-k", type=int, default=3)
    search.add_argument("--log-path", type=Path)
    search.add_argument("--attempt", type=int)
    search.add_argument("--call-index", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if os.environ.get("ABLATE_OPTIMIZATION_KB") == "1":
        print(json.dumps({"success": True, "skipped": True, "reason": "ABLATE_OPTIMIZATION_KB=1"}))
        return 0
    entries = _load(args.kb_path)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "success": True,
                    "entries": len(entries),
                    "promoted": sum(e["status"] == "promoted" for e in entries),
                    "supported": sum(e["status"] == "supported" for e in entries),
                    "sha256": _sha256(args.kb_path),
                },
                ensure_ascii=False,
            )
        )
        return 0

    matches = search_entries(
        entries,
        op_type=args.op_type,
        op_name=args.op_name,
        patterns=args.pattern,
        bottlenecks=args.bottleneck,
        phase=args.phase,
        min_status=args.min_status,
        top_k=max(1, args.top_k),
    )
    payload = {
        "schema_version": 1,
        "success": True,
        "read_only": True,
        "phase": args.phase,
        "query": {
            "op_type": args.op_type,
            "op_name": args.op_name,
            "patterns": args.pattern,
            "bottlenecks": args.bottleneck,
        },
        "kb_sha256": _sha256(args.kb_path),
        "matched_count": len(matches),
        "matches": matches,
        "entry_gate": [
            "official full correctness passed",
            "runtime/source anti-cheat CLEAN",
            "target compile passed",
            "immutable source SHA and repeated same-case performance baseline frozen",
        ],
        "acceptance_gate": [
            "correctness and anti-cheat remain unchanged",
            "same environment and case set",
            "gain exceeds timing noise and minimum threshold",
            "no material per-case regression",
        ],
    }
    if args.log_path:
        log_file = args.log_path
        if log_file.suffix.lower() != ".json":
            log_file = log_file / "optimization_knowledge_search_log.json"
        _append_log(
            log_file,
            {
                "attempt": args.attempt,
                "call_index": args.call_index,
                "phase": args.phase,
                "query": payload["query"],
                "kb_sha256": payload["kb_sha256"],
                "matched_ids": [item["knowledge"]["knowledge_id"] for item in matches],
                "scores": [item["score"] for item in matches],
            },
        )
        payload["log_path"] = str(log_file)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
