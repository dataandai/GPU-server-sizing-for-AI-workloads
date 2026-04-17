from __future__ import annotations

import math
from typing import Any


def calibrated_stage_time_ms(
    *,
    anchor: dict[str, Any],
    size: float,
    output_units: float,
    current_gpu_bandwidth_gbps: float,
    current_model_params_b: float | None,
    current_stage_workers: int,
    runtime_family: str,
    execution_mode: str,
) -> tuple[float, dict[str, float | str | int]]:
    baseline = anchor.get("baseline") or {}
    scaling = anchor.get("scaling") or {}
    topology = anchor.get("reference_topology") or {}
    base_ms = float(baseline.get("base_ms", 1.0) or 1.0)
    input_power = float(scaling.get("input_scale_power", 1.0) or 1.0)
    output_power = float(scaling.get("output_scale_power", 1.0) or 1.0)
    input_term = float(baseline.get("ms_per_input_unit", 0.0) or 0.0) * (max(size, 0.0) ** input_power)
    output_term = float(baseline.get("ms_per_output_unit", 0.0) or 0.0) * (max(output_units, 0.0) ** output_power)
    base = base_ms + input_term + output_term

    ref_bandwidth = float(topology.get("reference_gpu_bandwidth_gbps", 0.0) or 0.0)
    bandwidth_exp = float(scaling.get("gpu_bandwidth_exponent", 0.0) or 0.0)
    if ref_bandwidth > 0 and current_gpu_bandwidth_gbps > 0 and bandwidth_exp > 0:
        bandwidth_factor = (ref_bandwidth / current_gpu_bandwidth_gbps) ** bandwidth_exp
    else:
        bandwidth_factor = 1.0

    ref_params = float(topology.get("reference_model_params_b", 0.0) or 0.0)
    model_exp = float(scaling.get("model_params_exponent", 0.0) or 0.0)
    if ref_params > 0 and current_model_params_b and model_exp > 0:
        model_factor = (max(current_model_params_b, 0.05) / ref_params) ** model_exp
    else:
        model_factor = 1.0

    ref_allocation = _reference_gpu_allocation(anchor)
    gpu_alloc_exp = float(scaling.get("gpu_allocation_exponent", 0.0) or 0.0)
    current_stage_workers = max(1, int(current_stage_workers or 1))
    if ref_allocation > 0 and gpu_alloc_exp > 0:
        gpu_allocation_factor = (ref_allocation / current_stage_workers) ** gpu_alloc_exp
    else:
        gpu_allocation_factor = 1.0

    runtime_factor = float((scaling.get("runtime_family_scalars") or {}).get(runtime_family, 1.0) or 1.0)
    execution_factor = float((scaling.get("execution_mode_scalars") or {}).get(execution_mode, 1.0) or 1.0)

    result = max(0.1, base * bandwidth_factor * model_factor * gpu_allocation_factor * runtime_factor * execution_factor)
    trace: dict[str, float | str | int] = {
        "anchor_id": str(anchor.get("anchor_id") or ""),
        "base_before_scaling_ms": round(base, 6),
        "bandwidth_factor": round(bandwidth_factor, 6),
        "model_factor": round(model_factor, 6),
        "gpu_allocation_factor": round(gpu_allocation_factor, 6),
        "runtime_factor": round(runtime_factor, 6),
        "execution_factor": round(execution_factor, 6),
        "reference_gpu_allocation": int(ref_allocation),
        "current_stage_workers": int(current_stage_workers),
        "calibrated_service_time_ms": round(result, 6),
    }
    return result, trace



def calibration_coverage_summary(stage_matches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = len(stage_matches)
    matched = len([m for m in stage_matches.values() if m.get("matched")])
    coverage_ratio = round(matched / total, 4) if total else 0.0
    policy = calibration_coverage_policy(coverage_ratio)
    return {
        "stage_count": total,
        "matched_stage_count": matched,
        "coverage_ratio": coverage_ratio,
        "matched_anchor_ids": [m.get("anchor_id") for m in stage_matches.values() if m.get("matched")],
        **policy,
    }


def calibration_coverage_policy(coverage_ratio: float) -> dict[str, Any]:
    ratio = max(0.0, min(1.0, float(coverage_ratio or 0.0)))
    if ratio <= 0.0:
        status = "uncalibrated"
        customer_label = "Heuristic estimate"
        severity = "warning"
        estimate_only = True
        summary_note = "No benchmark anchor match; the result should be interpreted as a heuristic sizing estimate."
    elif ratio < 0.5:
        status = "partial_low"
        customer_label = "Partially calibrated"
        severity = "warning"
        estimate_only = True
        summary_note = "Only a small part of the pipeline matched a benchmark anchor; interpret with caution."
    elif ratio < 0.8:
        status = "partial_medium"
        customer_label = "Mostly calibrated"
        severity = "info"
        estimate_only = True
        summary_note = "Multiple pipeline stages matched benchmark anchors, but coverage is not complete."
    else:
        status = "anchored"
        customer_label = "Benchmark-backed"
        severity = "ok"
        estimate_only = False
        summary_note = "A large part of the pipeline matched benchmark anchors; the result is a stronger basis for comparison and sizing."
    return {
        "coverage_status": status,
        "customer_label": customer_label,
        "severity": severity,
        "estimate_only": estimate_only,
        "summary_note": summary_note,
    }



def _reference_gpu_allocation(anchor: dict[str, Any]) -> int:
    topology = anchor.get("reference_topology") or {}
    allocations = topology.get("gpu_role_allocation") or {}
    stage_id = str(anchor.get("stage_id") or "")
    if stage_id and stage_id in allocations:
        return max(1, int(allocations.get(stage_id) or 1))
    stage_type = str(anchor.get("stage_type") or "")
    if stage_type and stage_type in allocations:
        return max(1, int(allocations.get(stage_type) or 1))
    gpu_count = int(topology.get("gpu_count", 1) or 1)
    return max(1, math.ceil(gpu_count / 2))
