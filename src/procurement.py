from __future__ import annotations

from typing import Any


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def derive_workload_procurement_metrics(result: Any) -> dict[str, Any]:
    """Heuristic procurement-oriented metrics derived from workload simulation outputs.

    The goal is not low-level physical truth, but a stable, comparable sizing summary.
    """
    throughput = float(getattr(result, 'throughput_per_sec_mean', 0.0) or 0.0)
    throughput_p50 = float(getattr(result, 'throughput_per_sec_p50', throughput) or throughput)
    target_arrival = float(getattr(result, 'configured_arrival_rate_per_sec', 0.0) or 0.0)
    gpu_util_p95 = float(getattr(result, 'gpu_util_p95', 0.0) or 0.0)
    cpu_util_p95 = float(getattr(result, 'cpu_util_p95', 0.0) or 0.0)
    sla_mean = float(getattr(result, 'sla_hit_rate_mean', 1.0) or 1.0)
    drop_mean = float(getattr(result, 'drop_rate_mean', 0.0) or 0.0)
    risk = str(getattr(result, 'qualitative_risk_level', 'UNKNOWN') or 'UNKNOWN').upper()
    planning_profile = str(getattr(result, 'planning_profile', 'baseline') or 'baseline')
    reserved_fraction = float(getattr(result, 'reserved_capacity_fraction', 0.0) or 0.0)

    util_guard = max(gpu_util_p95, cpu_util_p95)
    headroom_ratio = _clamp(1.0 - util_guard, 0.0, 1.0)

    if planning_profile == 'safe_24x7':
        policy_factor = 0.82
    elif planning_profile == 'high_throughput':
        policy_factor = 0.97
    else:
        policy_factor = 0.9

    risk_factor = {
        'LOW': 0.95,
        'MEDIUM': 0.85,
        'MODERATE': 0.85,
        'HIGH': 0.72,
        'CRITICAL': 0.45,
    }.get(risk, 0.8)

    quality_factor = _clamp(sla_mean * (1.0 - 0.65 * drop_mean), 0.2, 1.0)
    steady_state_capacity = throughput_p50 * quality_factor
    safe_24x7_capacity = steady_state_capacity * policy_factor * risk_factor * (1.0 - reserved_fraction)
    recommended_max_arrival = safe_24x7_capacity * _clamp(0.92 + 0.3 * headroom_ratio, 0.7, 1.0)

    if target_arrival > 0:
        capacity_gap = safe_24x7_capacity - target_arrival
        target_fit = safe_24x7_capacity >= target_arrival
    else:
        capacity_gap = safe_24x7_capacity
        target_fit = None

    if safe_24x7_capacity <= 0:
        procurement_band = 'UNSUITABLE'
    elif target_arrival > 0 and safe_24x7_capacity < target_arrival * 0.85:
        procurement_band = 'UPSIZE_REQUIRED'
    elif risk in {'HIGH', 'CRITICAL'} or util_guard > 0.9:
        procurement_band = 'RISKY'
    elif planning_profile == 'safe_24x7' and safe_24x7_capacity >= target_arrival > 0:
        procurement_band = 'SAFE_24X7'
    elif target_arrival > 0 and safe_24x7_capacity >= target_arrival:
        procurement_band = 'FIT'
    else:
        procurement_band = 'TUNE_OR_SCALE'

    if procurement_band == 'SAFE_24X7':
        recommended_action = 'The current configuration is supportable for 24/7 operation with acceptable headroom.'
    elif procurement_band == 'FIT':
        recommended_action = 'The configuration is expected to serve the target load, but burst behavior and N+1 conditions should still be checked.'
    elif procurement_band == 'TUNE_OR_SCALE':
        recommended_action = 'The workload is probably runnable, but scheduler, batching, or deployment tuning is still needed.'
    elif procurement_band == 'UPSIZE_REQUIRED':
        recommended_action = 'Larger hardware or more GPUs are required for the target load.'
    elif procurement_band == 'RISKY':
        recommended_action = 'The configuration runs too close to its limit and is not stable for 24/7 or virtualized operation.'
    else:
        recommended_action = 'This configuration is not recommended for this workload profile.'

    return {
        'steady_state_capacity_per_sec': round(steady_state_capacity, 4),
        'safe_capacity_24x7_per_sec': round(safe_24x7_capacity, 4),
        'recommended_max_arrival_rate_per_sec': round(recommended_max_arrival, 4),
        'configured_arrival_rate_per_sec': round(target_arrival, 4),
        'capacity_gap_per_sec': round(capacity_gap, 4),
        'capacity_headroom_ratio': round(headroom_ratio, 4),
        'reserved_capacity_fraction': round(reserved_fraction, 4),
        'target_fit': target_fit,
        'procurement_band': procurement_band,
        'recommended_action': recommended_action,
    }


def summarize_workload_results_for_procurement(results: list[dict[str, Any]]) -> dict[str, Any]:
    workload_results = [r for r in results if r.get('result_type') == 'workload_simulation']
    if not workload_results:
        return {'count': 0, 'best_throughput': None, 'best_safe_24x7': None, 'lowest_latency': None, 'safest': None}

    def sort_key(metric: str):
        return lambda r: float(r.get(metric) or 0.0)

    best_throughput = max(workload_results, key=sort_key('throughput_per_sec_mean'))
    best_safe = max(workload_results, key=sort_key('safe_capacity_24x7_per_sec'))
    lowest_latency = min(workload_results, key=lambda r: float(r.get('latency_p95_ms') or float('inf')))
    risk_rank = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3, 'UNKNOWN': 4}
    safest = min(workload_results, key=lambda r: (risk_rank.get(str(r.get('qualitative_risk_level', 'UNKNOWN')).upper(), 4), -float(r.get('safe_capacity_24x7_per_sec') or 0.0)))
    return {
        'count': len(workload_results),
        'best_throughput': best_throughput,
        'best_safe_24x7': best_safe,
        'lowest_latency': lowest_latency,
        'safest': safest,
    }
