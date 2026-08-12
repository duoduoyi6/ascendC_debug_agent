"""Pure entry and Pareto gates for a post-correctness optimization stage."""
from __future__ import annotations

import statistics
from typing import Iterable


def _median(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("timing samples are empty")
    return statistics.median(numbers)


def _relative_mad(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    center = _median(numbers)
    if center <= 0:
        raise ValueError("timing samples must be positive")
    return statistics.median(abs(value - center) for value in numbers) / center


def optimization_entry_gate(baseline: dict) -> dict:
    checks = {
        "objective_success": baseline.get("objective_success") is True,
        "official_full_pass": baseline.get("official_full_pass") is True,
        "anti_cheat_clean": baseline.get("anti_cheat") == "CLEAN",
        "target_compile_pass": baseline.get("target_compile_pass") is True,
        "submission_ready": baseline.get("submission_ready") is True,
        "source_frozen": bool(baseline.get("source_sha256")),
        "case_set_frozen": bool(baseline.get("case_ids_sha256")),
        "environment_frozen": bool(baseline.get("environment_sha256")),
        "repeated_timing": len(baseline.get("timing_samples_us") or []) >= 3,
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
    }


def assess_optimization_candidate(
    baseline: dict,
    candidate: dict,
    *,
    minimum_gain: float = 0.03,
    noise_multiplier: float = 3.0,
    maximum_case_regression: float = 0.02,
) -> dict:
    entry = optimization_entry_gate(baseline)
    identity_checks = {
        "same_case_set": candidate.get("case_ids_sha256") == baseline.get("case_ids_sha256"),
        "same_environment": candidate.get("environment_sha256") == baseline.get("environment_sha256"),
        "source_changed": bool(candidate.get("source_sha256"))
        and candidate.get("source_sha256") != baseline.get("source_sha256"),
    }
    safety_checks = {
        "official_full_pass": candidate.get("official_full_pass") is True,
        "anti_cheat_clean": candidate.get("anti_cheat") == "CLEAN",
        "target_compile_pass": candidate.get("target_compile_pass") is True,
        "submission_ready": candidate.get("submission_ready") is True,
        "repeated_timing": len(candidate.get("timing_samples_us") or []) >= 3,
    }
    if not entry["eligible"] or not all(identity_checks.values()) or not all(safety_checks.values()):
        return {
            "accepted": False,
            "decision": "reject_before_performance_comparison",
            "entry_gate": entry,
            "identity_checks": identity_checks,
            "safety_checks": safety_checks,
        }

    baseline_time = _median(baseline["timing_samples_us"])
    candidate_time = _median(candidate["timing_samples_us"])
    relative_gain = (baseline_time - candidate_time) / baseline_time
    noise_floor = max(
        _relative_mad(baseline["timing_samples_us"]),
        _relative_mad(candidate["timing_samples_us"]),
    )
    required_gain = max(minimum_gain, noise_multiplier * noise_floor)

    baseline_cases = baseline.get("per_case_us") or {}
    candidate_cases = candidate.get("per_case_us") or {}
    same_case_keys = bool(baseline_cases) and set(baseline_cases) == set(candidate_cases)
    regressions = {}
    if same_case_keys:
        for case_id, baseline_value in baseline_cases.items():
            if float(baseline_value) <= 0:
                continue
            regressions[case_id] = (
                float(candidate_cases[case_id]) - float(baseline_value)
            ) / float(baseline_value)
    worst_case_regression = max(regressions.values(), default=0.0)
    case_gate = same_case_keys and worst_case_regression <= maximum_case_regression
    accepted = relative_gain >= required_gain and case_gate
    return {
        "accepted": accepted,
        "decision": "accept_pareto_improvement" if accepted else "rollback_candidate",
        "entry_gate": entry,
        "identity_checks": identity_checks,
        "safety_checks": safety_checks,
        "baseline_median_us": baseline_time,
        "candidate_median_us": candidate_time,
        "relative_gain": relative_gain,
        "noise_floor": noise_floor,
        "required_gain": required_gain,
        "same_per_case_set": same_case_keys,
        "worst_case_regression": worst_case_regression,
        "maximum_case_regression": maximum_case_regression,
    }
