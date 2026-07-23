#!/usr/bin/env python3
"""Verify V5 code, dataset, KB and credential invariants before launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNTIME_NAMES = {
    ".bench_baseline",
    ".debug_events",
    ".verify_logs",
    ".verify_status",
    "__pycache__",
    "build",
    "debug_status.json",
    "debug_trace.md",
    "precision_tuning",
}
CODE_EXCLUDED_TOP_LEVEL = {".git", ".secrets", "outputs"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(
    root: Path,
    *,
    skip_runtime: bool,
    code_root: bool = False,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if (
            ".git" in rel.parts
            or path.name == ".DS_Store"
            or path.name.startswith("._")
            or (code_root and rel.parts[0] in CODE_EXCLUDED_TOP_LEVEL)
        ):
            continue
        if skip_runtime and (
            any(part in RUNTIME_NAMES for part in rel.parts)
            or path.name.startswith("_claude_result")
            or path.name.startswith("_provider")
            or path.name == "_anticheat.json"
            or path.suffix == ".pyc"
        ):
            continue
        result[str(rel)] = sha256(path)
    return result


def compare_manifest(expected: dict[str, str], actual: dict[str, str]) -> dict[str, Any]:
    expected_keys = set(expected)
    actual_keys = set(actual)
    changed = sorted(
        key for key in expected_keys & actual_keys
        if expected[key] != actual[key]
    )
    return {
        "passed": not changed and expected_keys == actual_keys,
        "expected_files": len(expected),
        "actual_files": len(actual),
        "missing": sorted(expected_keys - actual_keys),
        "extra": sorted(actual_keys - expected_keys),
        "changed": changed,
    }


def _load_files(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"manifest has no files mapping: {path}")
    return {str(key): str(value) for key, value in files.items()}


def _write_bits(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode) & (
        stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--secret", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--formal-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    code_expected = _load_files(args.control / "code_snapshot.sha256.json")
    source_expected = _load_files(
        args.control / "source_snapshot_effective.sha256.json")
    code_actual = file_manifest(
        args.root, skip_runtime=True, code_root=True)
    source_actual = file_manifest(
        args.dataset, skip_runtime=True, code_root=False)
    checks: dict[str, Any] = {
        "code_snapshot": compare_manifest(code_expected, code_actual),
        "source_snapshot": compare_manifest(source_expected, source_actual),
    }

    kb_meta = json.loads(
        (args.control / "kb_snapshot.sha256.json").read_text(encoding="utf-8"))
    checks["kb_snapshot"] = {
        "passed": args.kb.is_file() and sha256(args.kb) == kb_meta.get("sha256"),
        "expected_sha256": kb_meta.get("sha256"),
        "actual_sha256": sha256(args.kb) if args.kb.is_file() else None,
    }
    source_writable = [
        str(path.relative_to(args.dataset))
        for path in [args.dataset, *sorted(args.dataset.rglob("*"))]
        if _write_bits(path)
    ]
    checks["source_read_only"] = {
        "passed": not source_writable,
        "writable_paths": source_writable[:100],
        "writable_path_count": len(source_writable),
    }
    checks["kb_read_only"] = {
        "passed": args.kb.is_file() and _write_bits(args.kb) == 0,
        "mode": oct(stat.S_IMODE(args.kb.stat().st_mode))
        if args.kb.exists() else None,
    }
    secret_mode = (
        stat.S_IMODE(args.secret.stat().st_mode)
        if args.secret.is_file() else None
    )
    checks["secret_mode"] = {
        "passed": secret_mode == 0o600,
        "mode": oct(secret_mode) if secret_mode is not None else None,
    }
    checks["formal_output_absent"] = {
        "passed": not args.formal_output.exists(),
        "path": str(args.formal_output),
    }
    passed = all(check["passed"] for check in checks.values())
    payload = {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
