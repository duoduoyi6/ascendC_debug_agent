#!/usr/bin/env python3
"""Request and archive CANNBench hidden evaluation through the official API."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HIDDEN_SCORE_THRESHOLD = 50.0
TERMINAL_STATUSES = {
    "succeeded",
    "validation_failed",
    "prepare_failed",
    "compile_failed",
    "correctness_failed",
    "performance_failed",
    "archive_failed",
    "timed_out",
    "infra_error",
    "canceled",
    "cancelled",
    "failed",
    "invalidated",
}
JOB_ID_RE = re.compile(r"^job_[A-Za-z0-9]+$")


class FollowupError(RuntimeError):
    """Raised when hidden follow-up cannot proceed safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def unwrap_job(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise FollowupError("job response must be a JSON object")
    job = payload.get("job", payload)
    if not isinstance(job, dict):
        raise FollowupError("job response does not contain a job object")
    return job


def result_score(job: dict[str, Any]) -> float | None:
    candidates = [
        job.get("result_score"),
        (job.get("results") or {}).get("overall_score")
        if isinstance(job.get("results"), dict)
        else None,
        ((job.get("results") or {}).get("summary") or {}).get("overall_score")
        if isinstance(job.get("results"), dict)
        and isinstance((job.get("results") or {}).get("summary"), dict)
        else None,
    ]
    for value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def selected_operators(job: dict[str, Any]) -> list[str]:
    raw = job.get("selected_operators") or job.get("operator_names") or []
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    for value in raw:
        item = str(value or "").strip()
        if item and item not in result:
            result.append(item)
    return result


def assess_hidden_eligibility(
    job: dict[str, Any],
    *,
    threshold: float = HIDDEN_SCORE_THRESHOLD,
) -> dict[str, Any]:
    score = result_score(job)
    status = str(job.get("status") or "").strip().lower()
    case_set = str(job.get("case_set") or "standard").strip().lower()
    submission_id = str(job.get("submission_id") or "").strip()
    operators = selected_operators(job)
    reasons: list[str] = []
    if status not in TERMINAL_STATUSES:
        reasons.append("standard job is not terminal")
    if case_set != "standard":
        reasons.append("source job is not a standard-case job")
    if score is None:
        reasons.append("standard job has no numeric score")
    elif score <= threshold:
        reasons.append(f"score must be strictly greater than {threshold:g}")
    if not submission_id:
        reasons.append("standard job has no submission_id")
    if not operators:
        reasons.append("standard job has no selected operators")
    return {
        "eligible": not reasons,
        "threshold_rule": f"score > {threshold:g}",
        "score": score,
        "status": status,
        "case_set": case_set,
        "submission_id": submission_id,
        "selected_operators": operators,
        "reasons": reasons,
    }


def validate_hidden_binding(
    standard_job: dict[str, Any],
    hidden_job: dict[str, Any],
) -> dict[str, Any]:
    standard_submission = str(standard_job.get("submission_id") or "").strip()
    hidden_submission = str(hidden_job.get("submission_id") or "").strip()
    standard_ops = set(selected_operators(standard_job))
    hidden_ops = set(selected_operators(hidden_job))
    errors: list[str] = []
    if str(hidden_job.get("case_set") or "").strip().lower() != "hidden":
        errors.append("follow-up job is not marked as hidden")
    if not standard_submission or hidden_submission != standard_submission:
        errors.append("hidden job is not bound to the standard submission")
    if not hidden_ops or hidden_ops != standard_ops:
        errors.append("hidden job operator set differs from the standard job")
    return {
        "passed": not errors,
        "standard_submission_id": standard_submission,
        "hidden_submission_id": hidden_submission,
        "standard_operators": sorted(standard_ops),
        "hidden_operators": sorted(hidden_ops),
        "errors": errors,
    }


class CANNBenchClient:
    def __init__(self, base_url: str, token: str, *, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}/api/{endpoint.lstrip('/')}"
        body = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise FollowupError(
                f"CANNBench API {method} {endpoint} failed with "
                f"HTTP {error.code}: {detail[:500]}"
            ) from error
        except urllib.error.URLError as error:
            raise FollowupError(
                f"CANNBench API {method} {endpoint} failed: {error.reason}"
            ) from error
        try:
            return json.loads(raw)
        except ValueError as error:
            raise FollowupError(
                f"CANNBench API {method} {endpoint} returned invalid JSON"
            ) from error

    def get_json(self, endpoint: str) -> Any:
        return self._request("GET", endpoint)

    def post_json(self, endpoint: str, payload: dict[str, Any]) -> Any:
        return self._request("POST", endpoint, payload)

    def download(self, download_url: str) -> bytes:
        if not download_url.startswith("/api/"):
            raise FollowupError("artifact download URL is outside the official API")
        url = f"{self.base_url}{download_url}"
        request = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self.token}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError) as error:
            raise FollowupError(f"artifact download failed: {download_url}") from error


def extract_requested_job_id(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise FollowupError("hidden request response must be a JSON object")
    nested = payload.get("job")
    candidates = [
        nested.get("id") if isinstance(nested, dict) else None,
        payload.get("job_id"),
        payload.get("id"),
    ]
    for value in candidates:
        job_id = str(value or "").strip()
        if JOB_ID_RE.fullmatch(job_id):
            return job_id
    raise FollowupError("hidden request response has no valid job ID")


def download_terminal_evidence(
    client: CANNBenchClient,
    job_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    logs: Any = {}
    artifacts: Any = {"artifacts": []}
    errors: list[str] = []
    try:
        logs = client.get_json(f"jobs/{job_id}/logs")
        write_json(output_dir / "logs.json", logs)
    except FollowupError as error:
        errors.append(str(error))
    try:
        artifacts = client.get_json(f"jobs/{job_id}/artifacts")
        write_json(output_dir / "artifacts.json", artifacts)
    except FollowupError as error:
        errors.append(str(error))

    downloaded: list[str] = []
    artifact_rows = artifacts.get("artifacts", []) if isinstance(artifacts, dict) else []
    if isinstance(artifact_rows, list):
        artifact_dir = output_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for row in artifact_rows:
            if not isinstance(row, dict):
                continue
            name = Path(str(row.get("name") or "")).name
            url = str(row.get("download_url") or "")
            if not name or not url.startswith("/api/"):
                continue
            try:
                (artifact_dir / name).write_bytes(client.download(url))
                downloaded.append(name)
            except FollowupError as error:
                errors.append(str(error))
    return {"downloaded_artifacts": sorted(downloaded), "errors": errors}


def follow_hidden_job(
    client: CANNBenchClient,
    hidden_job_id: str,
    output_dir: Path,
    *,
    poll: bool,
    poll_interval: float,
    max_wait_seconds: float,
) -> tuple[dict[str, Any], bool]:
    started = time.monotonic()
    while True:
        payload = client.get_json(f"jobs/{hidden_job_id}")
        write_json(output_dir / "job.json", payload)
        job = unwrap_job(payload)
        status = str(job.get("status") or "").strip().lower()
        print(f"job={hidden_job_id} status={status}", flush=True)
        if status in TERMINAL_STATUSES:
            return job, True
        if not poll:
            return job, False
        if time.monotonic() - started >= max_wait_seconds:
            return job, False
        time.sleep(poll_interval)


def build_summary(job: dict[str, Any]) -> dict[str, Any]:
    results = job.get("results") if isinstance(job.get("results"), dict) else {}
    operators: list[dict[str, Any]] = []
    for row in results.get("operators", []) if isinstance(results, dict) else []:
        if not isinstance(row, dict):
            continue
        operators.append({
            "operator": row.get("operator") or row.get("name"),
            "passed_cases": row.get("passed_cases"),
            "total_cases": row.get("total_cases"),
            "failed_cases": row.get("failed_cases"),
            "score": row.get("score"),
            "score_error_code": row.get("score_error_code"),
            "score_error": row.get("score_error"),
        })
    return {
        "job_id": job.get("id"),
        "status": job.get("status"),
        "case_set": job.get("case_set"),
        "submission_id": job.get("submission_id"),
        "selected_operators": selected_operators(job),
        "overall_score": result_score(job),
        "passed_cases": job.get("passed_cases", results.get("passed_cases")),
        "total_cases": job.get("total_cases", results.get("total_cases")),
        "error_code": job.get("error_code"),
        "error_message": job.get("error_message"),
        "operators": operators,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-job", required=True)
    parser.add_argument("--hidden-job")
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--request-hidden", action="store_true")
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--max-wait-seconds", type=float, default=21600.0)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BENCHSITE_URL", "https://cannbench.com"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("CANN_BENCH_TOKEN", "").strip()
    if not token:
        print("CANN_BENCH_TOKEN is unset.", file=sys.stderr)
        return 2
    if not JOB_ID_RE.fullmatch(args.standard_job):
        print("--standard-job is not a valid job ID.", file=sys.stderr)
        return 2
    if args.hidden_job and not JOB_ID_RE.fullmatch(args.hidden_job):
        print("--hidden-job is not a valid job ID.", file=sys.stderr)
        return 2
    if args.poll_interval <= 0 or args.max_wait_seconds <= 0:
        print("poll timing values must be positive.", file=sys.stderr)
        return 2

    evidence_root = args.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    client = CANNBenchClient(args.base_url, token)

    try:
        standard_payload = client.get_json(f"jobs/{args.standard_job}")
        write_json(evidence_root / "standard_job.json", standard_payload)
        standard_job = unwrap_job(standard_payload)
        eligibility = assess_hidden_eligibility(standard_job)
        eligibility["checked_at"] = utc_now()
        write_json(evidence_root / "hidden_eligibility.json", eligibility)
        if not eligibility["eligible"]:
            print(json.dumps(eligibility, indent=2, sort_keys=True))
            return 3

        id_file = evidence_root / "HIDDEN_JOB_ID"
        hidden_job_id = args.hidden_job
        if id_file.is_file():
            recorded = id_file.read_text(encoding="utf-8").strip()
            if not JOB_ID_RE.fullmatch(recorded):
                raise FollowupError("evidence root contains an invalid hidden job ID")
            if hidden_job_id and recorded != hidden_job_id:
                raise FollowupError(
                    "evidence root is already bound to a different hidden job")
            hidden_job_id = recorded

        request_payload: Any = None
        if hidden_job_id is None and args.request_hidden:
            request_payload = client.post_json(
                f"submissions/{eligibility['submission_id']}/rerun-hidden",
                {"selected_operators": eligibility["selected_operators"]},
            )
            write_json(evidence_root / "hidden_request_response.json", request_payload)
            hidden_job_id = extract_requested_job_id(request_payload)
        if hidden_job_id is not None and not id_file.exists():
            id_file.write_text(hidden_job_id + "\n", encoding="utf-8")

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "updated_at": utc_now(),
            "standard_job_id": args.standard_job,
            "standard_submission_id": eligibility["submission_id"],
            "threshold_rule": eligibility["threshold_rule"],
            "standard_score": eligibility["score"],
            "selected_operators": eligibility["selected_operators"],
            "hidden_job_id": hidden_job_id,
            "request_performed": request_payload is not None,
            "hidden_terminal": False,
        }
        if hidden_job_id is None:
            write_json(evidence_root / "hidden_followup_manifest.json", manifest)
            print(json.dumps(eligibility, indent=2, sort_keys=True))
            print("eligible: rerun with --request-hidden to submit the hidden job")
            return 0

        hidden_dir = evidence_root / "hidden_results" / hidden_job_id
        hidden_job, terminal = follow_hidden_job(
            client,
            hidden_job_id,
            hidden_dir,
            poll=args.poll,
            poll_interval=args.poll_interval,
            max_wait_seconds=args.max_wait_seconds,
        )
        binding = validate_hidden_binding(standard_job, hidden_job)
        write_json(hidden_dir / "binding.json", binding)
        if not binding["passed"]:
            raise FollowupError("hidden job binding validation failed")

        manifest["hidden_terminal"] = terminal
        manifest["hidden_status"] = hidden_job.get("status")
        if terminal:
            evidence = download_terminal_evidence(client, hidden_job_id, hidden_dir)
            summary = build_summary(hidden_job)
            summary["evidence_download"] = evidence
            write_json(hidden_dir / "summary.json", summary)
            manifest["hidden_summary"] = summary
        write_json(evidence_root / "hidden_followup_manifest.json", manifest)
        if args.poll and not terminal:
            print("hidden job did not reach a terminal state before the wait limit")
            return 4
        return 0
    except FollowupError as error:
        write_json(evidence_root / "hidden_followup_error.json", {
            "recorded_at": utc_now(),
            "error": str(error),
        })
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
