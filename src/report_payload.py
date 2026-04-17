from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .catalog_loader import get_hardware_record, get_software_stack_record
from .reporting import result_to_dict


def build_customer_report_payload(results: list[Any]) -> dict[str, Any]:
    normalized = [_normalize_result(r) for r in results if r is not None]
    if not normalized:
        raise ValueError("No results available for report export.")
    workload_results = [r for r in normalized if r.get("result_type") == "workload_simulation"]
    if not workload_results:
        raise ValueError("Report export currently supports workload_simulation results.")

    primary = workload_results[0]
    comparisons = _build_comparisons(workload_results)
    report = {
        "report_type": "customer_sizing_report",
        "report_version": "1.0",
        "scenario_name": primary.get("scenario_name") or primary.get("workload_id") or "workload_report",
        "executive_summary": _executive_summary(primary),
        "workload_summary": _workload_summary(primary),
        "calibration_summary": _calibration_summary(primary),
        "recommended_configuration": _recommended_configuration(primary),
        "capacity_summary": _capacity_summary(primary),
        "bottleneck_analysis": _bottleneck_analysis(primary),
        "comparison_views": comparisons,
        "model_compatibility": _model_compatibility(primary),
        "override_impact": _override_impact(comparisons),
        "assumptions_and_limits": _assumptions(primary),
        "appendix": _appendix(primary),
        "results": workload_results,
    }
    return report



def _normalize_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if is_dataclass(result):
        try:
            return result_to_dict(result)
        except Exception:
            return asdict(result)
    return result_to_dict(result)



def _executive_summary(result: dict[str, Any]) -> dict[str, Any]:
    risk = result.get("qualitative_risk_level") or "UNKNOWN"
    fit = result.get("target_fit")
    fit_label = "Fit" if fit else "Gap"
    calibration = (result.get("calibration_trace") or {}).get("summary") or {}
    calibration_suffix = f" · {calibration.get('customer_label')}" if calibration.get("customer_label") else ""
    return {
        "headline": f"{result.get('scenario_name')}: {risk} risk, {result.get('procurement_band') or 'UNKNOWN'} procurement band{calibration_suffix}",
        "subheadline": result.get("recommended_action") or "The workload sizing summary is ready.",
        "top_metrics": [
            {"label": "Safe 24/7 capacity", "value": f"{float(result.get('safe_capacity_24x7_per_sec', 0.0)):.3f} req/s"},
            {"label": "Latency p95", "value": f"{float(result.get('latency_p95_ms', 0.0)):.0f} ms"},
            {"label": "GPU util p95", "value": f"{100*float(result.get('gpu_util_p95', 0.0)):.1f}%"},
            {"label": fit_label, "value": f"{float(result.get('capacity_gap_per_sec', 0.0)):+.3f} req/s"},
        ],
    }



def _workload_summary(result: dict[str, Any]) -> dict[str, Any]:
    template_inputs = result.get("template_inputs") or {}
    return {
        "workload_id": result.get("workload_id"),
        "workload_class": result.get("workload_class"),
        "workload_family": result.get("workload_family"),
        "configured_arrival_rate_per_sec": result.get("configured_arrival_rate_per_sec"),
        "planning_profile": result.get("planning_profile"),
        "template_inputs": template_inputs,
        "model_bindings": result.get("model_bindings") or {},
        "override_role_ids": result.get("override_role_ids") or [],
        "requested_model_overrides": result.get("requested_model_overrides") or {},
    }



def _calibration_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = (result.get("calibration_trace") or {}).get("summary") or {}
    return {
        "matched_stage_count": int(summary.get("matched_stage_count", 0) or 0),
        "stage_count": int(summary.get("stage_count", 0) or 0),
        "coverage_ratio": float(summary.get("coverage_ratio", 0.0) or 0.0),
        "coverage_status": summary.get("coverage_status") or "unknown",
        "customer_label": summary.get("customer_label") or "Unknown calibration status",
        "estimate_only": bool(summary.get("estimate_only", False)),
        "summary_note": summary.get("summary_note") or "",
        "matched_anchor_ids": summary.get("matched_anchor_ids") or [],
    }


def _recommended_configuration(result: dict[str, Any]) -> dict[str, Any]:
    hardware_id = str(result.get("hardware_catalog_id") or "")
    software_stack_id = str(result.get("software_stack_id") or "")
    hardware = get_hardware_record(hardware_id) if hardware_id else {}
    stack = get_software_stack_record(software_stack_id) if software_stack_id else {}
    return {
        "hardware_catalog_id": hardware_id,
        "hardware_display_name": hardware.get("display_name") or hardware_id,
        "gpu_count": result.get("gpu_count"),
        "runtime_family": result.get("runtime_family"),
        "runtime_mode": result.get("runtime_mode"),
        "software_stack": stack.get("framework") or software_stack_id,
        "execution_mode": result.get("execution_mode"),
        "mig_profile": result.get("mig_profile"),
        "planning_profile": result.get("planning_profile"),
    }



def _capacity_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "configured_arrival_rate_per_sec": float(result.get("configured_arrival_rate_per_sec", 0.0) or 0.0),
        "steady_state_capacity_per_sec": float(result.get("steady_state_capacity_per_sec", 0.0) or 0.0),
        "safe_capacity_24x7_per_sec": float(result.get("safe_capacity_24x7_per_sec", 0.0) or 0.0),
        "recommended_max_arrival_rate_per_sec": float(result.get("recommended_max_arrival_rate_per_sec", 0.0) or 0.0),
        "capacity_headroom_ratio": float(result.get("capacity_headroom_ratio", 0.0) or 0.0),
        "capacity_gap_per_sec": float(result.get("capacity_gap_per_sec", 0.0) or 0.0),
        "reserved_capacity_fraction": float(result.get("reserved_capacity_fraction", 0.0) or 0.0),
    }



def _bottleneck_analysis(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "bottleneck_stage": result.get("bottleneck_stage"),
        "breakdown": result.get("bottleneck_breakdown") or {},
        "stage_performance": result.get("stage_performance") or {},
        "calibration_trace": result.get("calibration_trace") or {},
    }



def _build_comparisons(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    views = []
    for result in results:
        bindings = result.get("model_bindings") or {}
        override_role_ids = [str(x) for x in (result.get("override_role_ids") or []) if str(x).strip()]
        has_override = bool(override_role_ids or (result.get("requested_model_overrides") or {}))
        views.append(
            {
                "scenario_name": result.get("scenario_name"),
                "planning_profile": result.get("planning_profile"),
                "hardware_catalog_id": result.get("hardware_catalog_id"),
                "safe_capacity_24x7_per_sec": result.get("safe_capacity_24x7_per_sec"),
                "latency_p95_ms": result.get("latency_p95_ms"),
                "gpu_util_p95": result.get("gpu_util_p95"),
                "procurement_band": result.get("procurement_band"),
                "language": _comparison_language(result),
                "override_active": bool(has_override),
                "override_role_ids": override_role_ids,
                "model_bindings": bindings,
            }
        )
    return views




def _comparison_language(result: dict[str, Any]) -> str:
    template_inputs = result.get("template_inputs") or {}
    explicit = str(template_inputs.get("_language_code") or "").strip().lower()
    if explicit:
        return explicit
    language_scaling = result.get("language_scaling") or {}
    if isinstance(language_scaling, dict):
        dominant = str(language_scaling.get("dominant_language") or "").strip().lower()
        if dominant:
            return dominant
    return "en"




def _model_compatibility(result: dict[str, Any]) -> dict[str, Any]:
    compat = result.get("model_compatibility") or {}
    return {
        "overall_status": compat.get("overall_status") or "unknown",
        "status_counts": compat.get("status_counts") or {},
        "blocking_roles": compat.get("blocking_roles") or [],
        "notes": compat.get("notes") or [],
        "serving_notes": compat.get("serving_notes") or [],
        "serving_plan": compat.get("serving_plan") or {},
        "checks": compat.get("checks") or {},
    }

def _override_impact(comparison_views: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next((row for row in comparison_views if row.get("planning_profile") == "baseline" and not row.get("override_active")), None)
    override = next((row for row in comparison_views if row.get("override_active")), None)
    safe = next((row for row in comparison_views if row.get("planning_profile") == "safe_24x7"), None)
    def _delta(a: dict[str, Any] | None, b: dict[str, Any] | None, key: str) -> float | None:
        if not a or not b:
            return None
        try:
            return float(b.get(key, 0.0)) - float(a.get(key, 0.0))
        except Exception:
            return None
    return {
        "baseline_vs_override": {
            "latency_p95_delta_ms": _delta(baseline, override, "latency_p95_ms"),
            "safe_capacity_delta_per_sec": _delta(baseline, override, "safe_capacity_24x7_per_sec"),
        },
        "baseline_vs_safe": {
            "latency_p95_delta_ms": _delta(baseline, safe, "latency_p95_ms"),
            "safe_capacity_delta_per_sec": _delta(baseline, safe, "safe_capacity_24x7_per_sec"),
        },
    }



def _assumptions(result: dict[str, Any]) -> dict[str, Any]:
    calibration_summary = (result.get("calibration_trace") or {}).get("summary") or {}
    notes = list(result.get("notes") or [])
    if calibration_summary.get("summary_note"):
        notes.append(str(calibration_summary.get("summary_note")))
    return {
        "notes": notes,
        "monte_carlo_trials": result.get("monte_carlo_trials"),
        "time_horizon_sec": result.get("time_horizon_sec"),
        "calibration_summary": calibration_summary,
    }



def _appendix(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage_performance": result.get("stage_performance") or {},
        "detailed_metrics": result.get("detailed_metrics") or {},
        "calibration_examples": (result.get("calibration_trace") or {}).get("examples") or {},
        "model_bindings": result.get("model_bindings") or {},
        "override_role_ids": result.get("override_role_ids") or [],
        "requested_model_overrides": result.get("requested_model_overrides") or {},
    }
