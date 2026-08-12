#!/usr/bin/env python3
"""Build an auditable CANNBench performance-comparison evidence set.

The script compares one aggregate leaderboard export with one live leaderboard
snapshot.  It never infers hidden inputs and never executes submitted code.  It
only indexes source text from the user-provided aggregate archive and, when an
evidence root is supplied, the matching local submitted source tree.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp"}
FEATURE_PATTERNS = {
    "data_copy": r"\bDataCopy(?:Pad)?\s*\(",
    "vector_add": r"\bAdd\s*\(",
    "vector_sub": r"\bSub\s*\(",
    "vector_mul": r"\bMul(?:s)?\s*\(",
    "vector_div": r"\bDiv\s*\(",
    "vector_exp": r"\bExp\s*\(",
    "vector_cast": r"\bCast\s*\(",
    "vector_compare": r"\bCompare\s*\(",
    "vector_select": r"\bSelect\s*\(",
    "reduce_sum": r"\b(?:ReduceSum|WholeReduceSum)\s*\(",
    "reduce_max": r"\b(?:ReduceMax|WholeReduceMax)\s*\(",
    "atomic_add": r"\bSetAtomicAdd\s*\(",
    "pipeline_queue": r"\bTQue\b|\bTPipe\b",
    "double_buffer_hint": r"double\s*buffer|BUFFER_NUM\s*=\s*2|DB_BUFFER_NUM",
    "template_specialization": r"template\s*<|TILING_KEY_IS|tilingKey|TILING_KEY",
    "block_partition": r"GetBlockIdx\s*\(|block_idx|GetBlockNum\s*\(",
    "scalar_loop": r"\bfor\s*\(",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def _canonical(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _source_features(text: str) -> dict[str, int]:
    return {
        name: len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
        for name, pattern in FEATURE_PATTERNS.items()
    }


def _source_record(path: str, data: bytes) -> dict:
    text = data.decode("utf-8", errors="replace")
    return {
        "path": path,
        "sha256": _sha256_bytes(data),
        "bytes": len(data),
        "lines": len(text.splitlines()),
        "features": _source_features(text),
    }


def _member_matches_operator(member: str, rel_path: str, operator: str) -> bool:
    path = PurePosixPath(member)
    if path.suffix.lower() not in SOURCE_SUFFIXES:
        return False
    canonical_parts = [_canonical(part) for part in path.parts]
    aliases = {
        _canonical(operator),
        _canonical(PurePosixPath(rel_path).name),
    }
    return any(alias and alias in canonical_parts for alias in aliases)


def _index_first_place_submission(
    aggregate: zipfile.ZipFile,
    submission_id: str,
    rel_path: str,
    operator: str,
) -> dict:
    member = f"submissions/{submission_id}.zip"
    if member not in aggregate.namelist():
        return {"submission_id": submission_id, "missing": True, "source_files": []}
    payload = aggregate.read(member)
    from io import BytesIO

    with zipfile.ZipFile(BytesIO(payload)) as inner:
        unsafe = [item.filename for item in inner.infolist() if not _safe_member(item.filename)]
        if unsafe:
            raise ValueError(f"unsafe members in {submission_id}: {unsafe[:3]}")
        source_files = [
            _source_record(item.filename, inner.read(item))
            for item in inner.infolist()
            if not item.is_dir()
            and _member_matches_operator(item.filename, rel_path, operator)
        ]
    aggregate_features = Counter()
    for source in source_files:
        aggregate_features.update(source["features"])
    return {
        "submission_id": submission_id,
        "missing": False,
        "inner_zip_sha256": _sha256_bytes(payload),
        "inner_zip_bytes": len(payload),
        "source_file_count": len(source_files),
        "source_features": dict(sorted(aggregate_features.items())),
        "source_files": source_files,
    }


def _iter_submission_records(root: Path) -> Iterable[Path]:
    yield from root.glob("**/submission_record_initial.json")


def _task_root_for_record(record_path: Path) -> Path | None:
    excluded = {"initial_evidence", "predecessor_evidence", "released_slot_evidence"}
    if excluded.intersection(record_path.parts):
        return None
    for parent in record_path.parents:
        if (parent / "tasks").is_dir():
            return parent
    return None


def _aggregate_source_records(source_files: list[dict]) -> dict[str, object]:
    totals = Counter()
    for source in source_files:
        totals.update(source["features"])
    return {
        "source_file_count": len(source_files),
        "source_features": dict(sorted(totals.items())),
        "source_files": source_files,
    }


def _source_files_from_archive(
    archive_path: Path,
    selected: list[str],
) -> list[dict]:
    with zipfile.ZipFile(archive_path) as archive:
        unsafe = [item.filename for item in archive.infolist() if not _safe_member(item.filename)]
        if unsafe:
            raise ValueError(f"unsafe members in {archive_path}: {unsafe[:3]}")
        return [
            _source_record(item.filename, archive.read(item))
            for item in archive.infolist()
            if not item.is_dir()
            and any(_member_matches_operator(item.filename, operator, operator) for operator in selected)
        ]


def _candidate_archive(record_path: Path, inventory: dict) -> Path | None:
    archive_value = inventory.get("archive")
    candidates = []
    if archive_value:
        archive_path = Path(str(archive_value))
        candidates.extend(
            [
                archive_path,
                record_path.parent / archive_path,
                record_path.parent / "submission" / archive_path.name,
            ]
        )
    candidates.extend(sorted((record_path.parent / "submission").glob("*.zip")))
    expected_sha = inventory.get("archive_sha256")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if expected_sha and _sha256_file(candidate) != expected_sha:
            continue
        return candidate
    return None


def _local_sources_by_sha(root: Path) -> dict[str, list[Path]]:
    excluded = {
        ".git",
        ".cache",
        ".cannbench_build",
        "build",
        "build_out",
        "CMakeFiles",
    }
    result: dict[str, list[Path]] = defaultdict(list)
    for source_path in root.glob("**/*"):
        if (
            not source_path.is_file()
            or source_path.suffix.lower() not in SOURCE_SUFFIXES
            or excluded.intersection(source_path.parts)
        ):
            continue
        try:
            if source_path.stat().st_size > 5_000_000:
                continue
            result[_sha256_file(source_path)].append(source_path)
        except OSError:
            continue
    return result


def _inventory_source_files(
    inventory: dict,
    selected: list[str],
    sources_by_sha: dict[str, list[Path]],
) -> list[dict]:
    records = []
    aliases = {_canonical(operator) for operator in selected}
    for item in inventory.get("files") or []:
        path = str(item.get("path") or "")
        if Path(path).suffix.lower() not in SOURCE_SUFFIXES:
            continue
        canonical_parts = {_canonical(part) for part in PurePosixPath(path).parts}
        if not aliases.intersection(canonical_parts):
            continue
        matches = sources_by_sha.get(str(item.get("sha256") or ""), [])
        if not matches:
            continue
        source_path = matches[0]
        record = _source_record(str(source_path), source_path.read_bytes())
        record["inventory_path"] = path
        records.append(record)
    return records


def _inventory_expected_source_count(inventory: dict, selected: list[str]) -> int:
    aliases = {_canonical(operator) for operator in selected}
    count = 0
    for item in inventory.get("files") or []:
        path = str(item.get("path") or "")
        canonical_parts = {_canonical(part) for part in PurePosixPath(path).parts}
        if Path(path).suffix.lower() in SOURCE_SUFFIXES and aliases.intersection(canonical_parts):
            count += 1
    return count


def _index_ours_sources(evidence_root: Path) -> dict[str, list[dict]]:
    by_job: dict[str, list[dict]] = defaultdict(list)
    sources_by_sha: dict[str, list[Path]] | None = None
    for record_path in _iter_submission_records(evidence_root):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        job_id = record.get("job_id")
        task_root = _task_root_for_record(record_path)
        if not job_id:
            continue
        selected = record.get("selected_operators") or []
        inventory_path = record_path.parent / "candidate_inventory.json"
        if inventory_path.is_file() and selected:
            try:
                inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                inventory = {}
            archive_path = _candidate_archive(record_path, inventory)
            if archive_path is not None:
                files = _source_files_from_archive(archive_path, selected)
                entry = {
                    "submission_id": record.get("submission_id"),
                    "provenance": "immutable_submission_archive",
                    "submission_record": str(record_path),
                    "archive_path": str(archive_path),
                    "archive_sha256": _sha256_file(archive_path),
                    **_aggregate_source_records(files),
                }
                by_job[job_id].append(entry)
            else:
                if sources_by_sha is None:
                    sources_by_sha = _local_sources_by_sha(evidence_root)
                files = _inventory_source_files(inventory, selected, sources_by_sha)
                if files:
                    expected_count = _inventory_expected_source_count(inventory, selected)
                    entry = {
                        "submission_id": record.get("submission_id"),
                        "provenance": (
                            "candidate_inventory_exact_file_sha"
                            if len(files) == expected_count
                            else "candidate_inventory_partial_exact_file_sha"
                        ),
                        "submission_record": str(record_path),
                        "candidate_inventory": str(inventory_path),
                        "archive_sha256": inventory.get("archive_sha256"),
                        "inventory_expected_source_file_count": expected_count,
                        "inventory_matched_source_file_count": len(files),
                        **_aggregate_source_records(files),
                    }
                    by_job[job_id].append(entry)

        if task_root is None:
            continue
        for task_dir in sorted((task_root / "tasks").glob("*")):
            kernel_dir = task_dir / "kernel"
            if not kernel_dir.is_dir():
                continue
            task_key = _canonical(task_dir.name)
            if selected and not any(_canonical(op) in task_key for op in selected):
                continue
            files = []
            for source_path in sorted(kernel_dir.rglob("*")):
                if not source_path.is_file() or source_path.suffix.lower() not in SOURCE_SUFFIXES:
                    continue
                data = source_path.read_bytes()
                item = _source_record(str(source_path), data)
                files.append(item)
            by_job[job_id].append(
                {
                    "submission_id": record.get("submission_id"),
                    "provenance": "task_root_at_submission_record",
                    "submission_record": str(record_path),
                    "task_root": str(task_root),
                    "task_dir": str(task_dir),
                    **_aggregate_source_records(files),
                }
            )
    for job_id, candidates in by_job.items():
        unique = {}
        for candidate in candidates:
            identity = (
                candidate.get("submission_id"),
                tuple(source["sha256"] for source in candidate["source_files"]),
            )
            unique.setdefault(identity, candidate)
        by_job[job_id] = list(unique.values())
    return by_job


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_analysis(
    first_place_zip: Path,
    ours_snapshot_path: Path,
    output_dir: Path,
    ours_evidence_root: Path | None = None,
) -> dict:
    ours = json.loads(ours_snapshot_path.read_text(encoding="utf-8"))
    ours_rows = {(row["operator"], row["stage"]): row for row in ours["rows"]}
    ours_standard_rows = {
        row["operator"]: row for row in ours["rows"] if row["stage"] == "standard"
    }
    ours_source_index = _index_ours_sources(ours_evidence_root) if ours_evidence_root else {}

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(first_place_zip) as aggregate:
        unsafe = [item.filename for item in aggregate.infolist() if not _safe_member(item.filename)]
        if unsafe:
            raise ValueError(f"unsafe aggregate members: {unsafe[:3]}")
        report = json.loads(aggregate.read("report.json"))
        manifest = json.loads(aggregate.read("manifest.json"))
        hidden_by_operator_submission = {
            (item["operator"], (item.get("source") or {}).get("submission_id")): item
            for item in report["hidden"]["metric"]["operators"]
        }
        artifact_cache: dict[tuple[str, str], dict] = {}
        rows = []
        for stage in ("standard", "hidden"):
            for first in report[stage]["metric"]["operators"]:
                ours_row = ours_rows.get((first["operator"], stage))
                ours_speedup = None if ours_row is None else ours_row.get("benchmark_speedup")
                first_speedup = first.get("avg_speedup")
                comparable = first_speedup is not None and ours_speedup is not None
                outperforms = bool(
                    comparable and first_speedup > 1.0 and first_speedup > ours_speedup
                )
                first_full_pass = first.get("passed_cases") == first.get("total_cases")
                classification = "not_selected"
                if outperforms and first_full_pass:
                    classification = "strict_reference"
                elif outperforms:
                    classification = "hypothesis_only_incomplete_correctness"
                elif ours_speedup is None:
                    classification = "not_comparable_missing_ours_speedup"

                source = first.get("source") or {}
                same_submission_hidden = hidden_by_operator_submission.get(
                    (first["operator"], source.get("submission_id"))
                )
                if stage == "hidden":
                    artifact_validation = (
                        "hidden_full"
                        if first_full_pass
                        else "hidden_incomplete"
                    )
                elif same_submission_hidden is None:
                    artifact_validation = "public_full_hidden_unknown"
                elif (
                    same_submission_hidden.get("passed_cases")
                    == same_submission_hidden.get("total_cases")
                ):
                    artifact_validation = "hidden_full"
                else:
                    artifact_validation = "hidden_incomplete"

                knowledge_evidence_tier = "not_selected"
                if classification == "strict_reference":
                    if artifact_validation == "hidden_full":
                        knowledge_evidence_tier = "promotion_candidate"
                    elif artifact_validation == "public_full_hidden_unknown":
                        knowledge_evidence_tier = "supported_hypothesis"
                    else:
                        knowledge_evidence_tier = "counterexample_or_rejected"
                elif classification == "hypothesis_only_incomplete_correctness":
                    knowledge_evidence_tier = "rejected_incomplete_correctness"
                artifact = None
                if classification in {
                    "strict_reference",
                    "hypothesis_only_incomplete_correctness",
                }:
                    key = (source.get("submission_id", ""), first["rel_path"])
                    if key not in artifact_cache:
                        artifact_cache[key] = _index_first_place_submission(
                            aggregate,
                            source.get("submission_id", ""),
                            first["rel_path"],
                            first["operator"],
                        )
                    artifact = artifact_cache[key]

                # Hidden jobs bind the same immutable submission as their standard
                # job, while local source evidence is indexed by the standard
                # submission record.  Use that job for source lookup in both stages.
                ours_source_row = ours_standard_rows.get(first["operator"]) or ours_row
                ours_source_job_id = (
                    None if ours_source_row is None else ours_source_row.get("job_id")
                )
                ours_sources = (
                    [] if not ours_source_job_id else ours_source_index.get(ours_source_job_id, [])
                )
                ours_pair_provenance = sorted(
                    {item.get("provenance") for item in ours_sources if item.get("provenance")}
                )
                rows.append(
                    {
                        "operator": first["operator"],
                        "rel_path": first["rel_path"],
                        "stage": stage,
                        "classification": classification,
                        "artifact_validation": artifact_validation,
                        "knowledge_evidence_tier": knowledge_evidence_tier,
                        "first_place": {
                            "speedup": first_speedup,
                            "passed_cases": first.get("passed_cases"),
                            "total_cases": first.get("total_cases"),
                            "job_id": source.get("job_id"),
                            "submission_id": source.get("submission_id"),
                            "display_user": source.get("display_user"),
                        },
                        "ours": {
                            "speedup": ours_speedup,
                            "passed_cases": None if ours_row is None else (ours_row.get("cases") or {}).get("passed"),
                            "total_cases": None if ours_row is None else (ours_row.get("cases") or {}).get("total"),
                            "job_id": None if ours_row is None else ours_row.get("job_id"),
                        },
                        "speedup_delta": None if not comparable else first_speedup - ours_speedup,
                        "first_place_artifact": artifact,
                        "ours_source_candidates": ours_sources,
                        "ours_source_lookup_job_id": ours_source_job_id,
                        "ours_source_pair_provenance": ours_pair_provenance,
                    }
                )

    counts = Counter(row["classification"] for row in rows)
    selected = [row for row in rows if row["classification"] == "strict_reference"]
    selected_validation = Counter(row["artifact_validation"] for row in selected)
    evidence_tiers = Counter(row["knowledge_evidence_tier"] for row in rows)
    selected_operators = {row["operator"] for row in selected}
    paired_operators = {
        row["operator"] for row in selected if row.get("ours_source_candidates")
    }
    partial_operators = {
        row["operator"]
        for row in selected
        if "candidate_inventory_partial_exact_file_sha"
        in row.get("ours_source_pair_provenance", [])
    }
    analysis = {
        "schema_version": 1,
        "comparison_scope": {
            "benchmark": "official-tasks",
            "hardware": "A3",
            "version": "1.1.0",
            "first_place_display_name": report["scheme"].get("public_tag"),
            "ours_display_name": ours.get("display_name"),
            "first_place_snapshot": report.get("snapshot"),
            "ours_snapshot_captured_at": ours.get("captured_at"),
            "selection_rule": "first_speedup > 1.0 and first_speedup > ours_speedup",
            "stage_selection_rule": "selection_rule and first-place stage passes every case",
            "knowledge_promotion_candidate_rule": (
                "stage_selection_rule and the same immutable artifact has a full hidden result"
            ),
            "causality_boundary": (
                "leaderboard association plus source inspection is not a controlled ablation; "
                "mechanisms remain evidence-backed hypotheses until independently reproduced"
            ),
        },
        "source_integrity": {
            "first_place_archive_path": str(first_place_zip),
            "first_place_archive_sha256": _sha256_file(first_place_zip),
            "first_place_manifest": manifest,
            "ours_snapshot_path": str(ours_snapshot_path),
            "ours_snapshot_sha256": _sha256_file(ours_snapshot_path),
        },
        "counts": {
            "paired_stage_records": len(rows),
            "strict_reference_stage_records": counts["strict_reference"],
            "strict_reference_operators": len({row["operator"] for row in selected}),
            "strict_reference_operators_with_local_exact_source": len(paired_operators),
            "strict_reference_operators_with_complete_local_source": len(
                paired_operators - partial_operators
            ),
            "strict_reference_operators_with_partial_local_source": len(partial_operators),
            "strict_reference_local_source_unresolved": sorted(
                selected_operators - paired_operators
            ),
            "strict_reference_local_source_partial": sorted(partial_operators),
            "hypothesis_only_stage_records": counts["hypothesis_only_incomplete_correctness"],
            "not_comparable_stage_records": counts["not_comparable_missing_ours_speedup"],
            "strict_reference_hidden_full": selected_validation["hidden_full"],
            "strict_reference_public_only": selected_validation[
                "public_full_hidden_unknown"
            ],
            "strict_reference_but_same_artifact_hidden_incomplete": selected_validation[
                "hidden_incomplete"
            ],
            "promotion_candidate_stage_records": evidence_tiers["promotion_candidate"],
            "promotion_candidate_operators": len(
                {
                    row["operator"]
                    for row in rows
                    if row["knowledge_evidence_tier"] == "promotion_candidate"
                }
            ),
            "supported_hypothesis_stage_records": evidence_tiers["supported_hypothesis"],
            "counterexample_or_rejected_stage_records": evidence_tiers[
                "counterexample_or_rejected"
            ],
        },
        "rows": rows,
    }

    _write_json(output_dir / "paired_operator_stage_metrics.json", analysis)
    _write_json(output_dir / "ours_live_leaderboard_snapshot.json", ours)
    with (output_dir / "paired_operator_stage_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "operator",
                "stage",
                "classification",
                "knowledge_evidence_tier",
                "artifact_validation",
                "first_speedup",
                "ours_speedup",
                "delta",
                "first_passed",
                "first_total",
                "first_submission_id",
                "first_job_id",
                "ours_job_id",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["operator"],
                    row["stage"],
                    row["classification"],
                    row["knowledge_evidence_tier"],
                    row["artifact_validation"],
                    row["first_place"]["speedup"],
                    row["ours"]["speedup"],
                    row["speedup_delta"],
                    row["first_place"]["passed_cases"],
                    row["first_place"]["total_cases"],
                    row["first_place"]["submission_id"],
                    row["first_place"]["job_id"],
                    row["ours"]["job_id"],
                ]
            )
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-place-zip", type=Path, required=True)
    parser.add_argument("--ours-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ours-evidence-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis = build_analysis(
        args.first_place_zip,
        args.ours_snapshot,
        args.output_dir,
        args.ours_evidence_root,
    )
    print(json.dumps(analysis["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
