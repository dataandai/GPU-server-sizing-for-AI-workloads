"""
Output formatting and reporting.
Generates tables, JSON, and per-hardware engineering analysis.
"""

from __future__ import annotations
import json
from typing import Any
from tabulate import tabulate

from .config import SimulationResult
from .workload_simulation import WorkloadSimulationResult


ResultLike = SimulationResult | WorkloadSimulationResult


def result_to_dict(r: ResultLike) -> dict:
    """Convert a simulation result to a dictionary."""
    if isinstance(r, WorkloadSimulationResult):
        return {
            "result_type": r.result_type,
            "scenario_name": r.scenario_name,
            "workload_id": r.workload_id,
            "workload_class": r.workload_class,
            "workload_family": r.workload_family,
            "source_blueprint_id": r.source_blueprint_id,
            "source_template_id": r.source_template_id,
            "runtime_family": r.runtime_family,
            "runtime_mode": r.runtime_mode,
            "execution_mode": r.execution_mode,
            "hardware_catalog_id": r.hardware_catalog_id,
            "software_stack_id": r.software_stack_id,
            "planning_profile": r.planning_profile,
            "gpu_count": r.gpu_count,
            "mig_profile": r.mig_profile,
            "monte_carlo_trials": r.monte_carlo_trials,
            "time_horizon_sec": r.time_horizon_sec,
            "configured_arrival_rate_per_sec": round(r.configured_arrival_rate_per_sec, 4),
            "reserved_capacity_fraction": round(r.reserved_capacity_fraction, 4),
            "total_arrivals_mean": round(r.total_arrivals_mean, 2),
            "completed_mean": round(r.completed_mean, 2),
            "dropped_mean": round(r.dropped_mean, 2),
            "throughput_per_sec_mean": round(r.throughput_per_sec_mean, 4),
            "throughput_per_sec_p50": round(r.throughput_per_sec_p50, 4),
            "throughput_per_sec_p95": round(r.throughput_per_sec_p95, 4),
            "latency_p50_ms": round(r.latency_p50_ms, 2),
            "latency_p95_ms": round(r.latency_p95_ms, 2),
            "latency_p99_ms": round(r.latency_p99_ms, 2),
            "queue_wait_p95_ms": round(r.queue_wait_p95_ms, 2),
            "queue_wait_p99_ms": round(r.queue_wait_p99_ms, 2),
            "drop_rate_mean": round(r.drop_rate_mean, 4),
            "drop_rate_p95": round(r.drop_rate_p95, 4),
            "sla_hit_rate_mean": round(r.sla_hit_rate_mean, 4),
            "sla_hit_rate_p95": round(r.sla_hit_rate_p95, 4),
            "gpu_util_mean": round(r.gpu_util_mean, 4),
            "gpu_util_p95": round(r.gpu_util_p95, 4),
            "cpu_util_mean": round(r.cpu_util_mean, 4),
            "cpu_util_p95": round(r.cpu_util_p95, 4),
            "peak_inflight_p95": round(r.peak_inflight_p95, 2),
            "steady_state_capacity_per_sec": round(r.steady_state_capacity_per_sec, 4),
            "safe_capacity_24x7_per_sec": round(r.safe_capacity_24x7_per_sec, 4),
            "recommended_max_arrival_rate_per_sec": round(r.recommended_max_arrival_rate_per_sec, 4),
            "capacity_gap_per_sec": round(r.capacity_gap_per_sec, 4),
            "capacity_headroom_ratio": round(r.capacity_headroom_ratio, 4),
            "procurement_band": r.procurement_band,
            "recommended_action": r.recommended_action,
            "target_fit": r.target_fit,
            "bottleneck_stage": r.bottleneck_stage,
            "bottleneck_breakdown": r.bottleneck_breakdown,
            "stage_performance": r.stage_performance,
            "template_inputs": r.template_inputs,
            "detailed_metrics": r.detailed_metrics,
            "model_bindings": r.model_bindings,
            "language_scaling": r.language_scaling,
            "requested_model_overrides": r.requested_model_overrides,
            "override_role_ids": r.override_role_ids,
            "calibration_trace": r.calibration_trace,
            "model_compatibility": r.model_compatibility,
            "qualitative_risk_level": r.qualitative_risk_level,
            "notes": r.notes,
        }

    return {
        "result_type": "memory_simulation",
        "scenario_name": r.scenario_name,
        "model_name": r.model_name,
        "model_id": r.model_id,
        "model_source": r.model_source,
        "hardware_profile": r.hardware_profile,
        "hardware_catalog_id": r.hardware_catalog_id,
        "software_stack_id": r.software_stack_id,
        "deployment_profile_id": r.deployment_profile_id,
        "weight_precision": r.weight_precision,
        "kv_cache_precision": r.kv_cache_precision,
        "tensor_parallel_degree": r.tensor_parallel_degree,
        "num_gpus": r.num_gpus,
        "vram_per_gpu_gb": round(r.vram_per_gpu_gb, 1),
        "total_system_vram_gb": round(r.total_system_vram_gb, 1),
        "weight_memory_gb": round(r.weight_memory_gb, 2),
        "kv_cache_vram_mean_gb": round(r.kv_cache_vram_mean_gb, 2),
        "kv_cache_vram_p95_gb": round(r.kv_cache_vram_p95_gb, 2),
        "kv_cache_vram_p99_gb": round(r.kv_cache_vram_p99_gb, 2),
        "kv_cache_vram_max_gb": round(r.kv_cache_vram_max_gb, 2),
        "total_vram_mean_gb": round(r.total_vram_mean_gb, 2),
        "total_vram_p95_gb": round(r.total_vram_p95_gb, 2),
        "total_vram_p99_gb": round(r.total_vram_p99_gb, 2),
        "total_vram_max_gb": round(r.total_vram_max_gb, 2),
        "per_gpu_vram_mean_gb": round(r.per_gpu_vram_mean_gb, 2),
        "per_gpu_vram_p95_gb": round(r.per_gpu_vram_p95_gb, 2),
        "per_gpu_vram_p99_gb": round(r.per_gpu_vram_p99_gb, 2),
        "runtime_overhead_mean_gb": round(r.runtime_overhead_mean_gb, 2),
        "estimated_oom_probability": round(r.estimated_oom_probability, 4),
        "kv_cache_saturation_probability": round(r.kv_cache_saturation_probability, 4),
        "recommended_max_concurrency": r.recommended_max_concurrency,
        "qualitative_risk_level": r.qualitative_risk_level,
        "monte_carlo_iterations": r.monte_carlo_iterations,
        "total_concurrent_sequences": r.total_concurrent_sequences,
        "surface_status": "deprecated_legacy_memory_engine",
        "estimate_only": True,
        "notes": r.notes,
    }


def results_to_json(results: list[ResultLike], indent: int = 2) -> str:
    return json.dumps([result_to_dict(r) for r in results], indent=indent)


def format_summary_table(results: list[ResultLike]) -> str:
    if not results:
        return ""
    if all(isinstance(r, WorkloadSimulationResult) for r in results):
        return _format_workload_summary_table(results)
    return _format_memory_summary_table(results)  # type: ignore[arg-type]


def _format_workload_summary_table(results: list[WorkloadSimulationResult]) -> str:
    headers = [
        "Scenario", "Workload", "Runtime", "Deploy", "HW", "Thrpt/s\np50", "Thrpt/s\np95",
        "Lat\np50", "Lat\np95", "Queue\np95", "Drop\np95", "SLA\nmean", "GPU util\np95", "Risk",
    ]
    rows = []
    for r in results:
        rows.append([
            r.scenario_name[:28],
            r.workload_class[:18],
            r.runtime_family[:12],
            r.execution_mode[:18],
            r.hardware_catalog_id[:22],
            f"{r.throughput_per_sec_p50:.2f}",
            f"{r.throughput_per_sec_p95:.2f}",
            f"{r.latency_p50_ms:.0f}",
            f"{r.latency_p95_ms:.0f}",
            f"{r.queue_wait_p95_ms:.0f}",
            f"{r.drop_rate_p95:.1%}",
            f"{r.sla_hit_rate_mean:.1%}",
            f"{r.gpu_util_p95:.1%}",
            r.qualitative_risk_level,
        ])
    return tabulate(rows, headers=headers, tablefmt="grid")


def _format_memory_summary_table(results: list[SimulationResult]) -> str:
    headers = [
        "Scenario", "Hardware", "Weights\n(GB)", "KV Cache\nMean (GB)",
        "KV Cache\np95 (GB)", "Total VRAM\nMean (GB)", "Total VRAM\np95 (GB)",
        "Per-GPU\np95 (GB)", "GPU VRAM\n(GB)", "OOM\nProb",
        "Max\nConc.", "Risk",
    ]
    rows = []
    for r in results:
        short_name = r.scenario_name.split("__")[0] if "__" in r.scenario_name else r.scenario_name
        hw_short = r.hardware_profile.replace("NVIDIA ", "").replace(" Blackwell Server Edition", "")
        rows.append([
            short_name[:25], hw_short[:35], f"{r.weight_memory_gb:.1f}", f"{r.kv_cache_vram_mean_gb:.1f}",
            f"{r.kv_cache_vram_p95_gb:.1f}", f"{r.total_vram_mean_gb:.1f}", f"{r.total_vram_p95_gb:.1f}",
            f"{r.per_gpu_vram_p95_gb:.1f}", f"{r.vram_per_gpu_gb:.0f}", f"{r.estimated_oom_probability:.2%}",
            r.recommended_max_concurrency or "N/A", r.qualitative_risk_level,
        ])
    return tabulate(rows, headers=headers, tablefmt="grid")


def format_detailed_report(result: ResultLike) -> str:
    if isinstance(result, WorkloadSimulationResult):
        lines = [
            f"{'='*80}",
            f"  WORKLOAD SIMULATION REPORT: {result.scenario_name}",
            f"{'='*80}",
            "",
            "─── Deployment ───",
            f"  Workload ID:        {result.workload_id}",
            f"  Workload class:     {result.workload_class}",
            f"  Runtime:            {result.runtime_family}/{result.runtime_mode}",
            f"  Execution mode:     {result.execution_mode}",
            f"  Hardware catalog:   {result.hardware_catalog_id}",
            f"  GPU count:          {result.gpu_count}",
            f"  Software stack:     {result.software_stack_id or '-'}",
            f"  MIG profile:        {result.mig_profile or '-'}",
            "",
            "─── Throughput & Latency ───",
            f"  Throughput mean:    {result.throughput_per_sec_mean:.3f} items/s",
            f"  Throughput p50:     {result.throughput_per_sec_p50:.3f} items/s",
            f"  Throughput p95:     {result.throughput_per_sec_p95:.3f} items/s",
            f"  Latency p50:        {result.latency_p50_ms:.1f} ms",
            f"  Latency p95:        {result.latency_p95_ms:.1f} ms",
            f"  Latency p99:        {result.latency_p99_ms:.1f} ms",
            f"  Queue wait p95:     {result.queue_wait_p95_ms:.1f} ms",
            f"  Queue wait p99:     {result.queue_wait_p99_ms:.1f} ms",
            "",
            "─── Capacity & Risk ───",
            f"  Avg arrivals:       {result.total_arrivals_mean:.1f}",
            f"  Avg completed:      {result.completed_mean:.1f}",
            f"  Avg dropped:        {result.dropped_mean:.1f}",
            f"  Drop rate mean:     {result.drop_rate_mean:.2%}",
            f"  Drop rate p95:      {result.drop_rate_p95:.2%}",
            f"  SLA hit rate mean:  {result.sla_hit_rate_mean:.2%}",
            f"  SLA hit rate p95:   {result.sla_hit_rate_p95:.2%}",
            f"  GPU util mean:      {result.gpu_util_mean:.2%}",
            f"  GPU util p95:       {result.gpu_util_p95:.2%}",
            f"  CPU util mean:      {result.cpu_util_mean:.2%}",
            f"  CPU util p95:       {result.cpu_util_p95:.2%}",
            f"  Peak inflight p95:  {result.peak_inflight_p95:.1f}",
            f"  Bottleneck stage:   {result.bottleneck_stage or '-'}",
            f"  Risk level:         {result.qualitative_risk_level}",
            f"  MC trials:          {result.monte_carlo_trials}",
            "",
        ]
        if result.model_bindings:
            lines.append("─── Model bindings ───")
            for role_id, model_id in sorted(result.model_bindings.items()):
                lines.append(f"  • {role_id}: {model_id}")
            lines.append("")
        calibration_summary = (result.calibration_trace or {}).get("summary", {}) if isinstance(result.calibration_trace, dict) else {}
        if calibration_summary:
            lines.append("─── Calibration ───")
            lines.append(
                f"  Matched stages:     {int(calibration_summary.get('matched_stage_count', 0))}/{int(calibration_summary.get('stage_count', 0))}"
            )
            lines.append(f"  Coverage ratio:     {100*float(calibration_summary.get('coverage_ratio', 0.0)):.1f}%")
            lines.append(f"  Coverage status:    {calibration_summary.get('coverage_status', 'unknown')}")
            if calibration_summary.get('summary_note'):
                lines.append(f"  Note:               {calibration_summary.get('summary_note')}")
            anchor_ids = calibration_summary.get("matched_anchor_ids") or []
            if anchor_ids:
                lines.append(f"  Anchors:            {', '.join(str(x) for x in anchor_ids)}")
            lines.append("")
        if result.bottleneck_breakdown:
            lines.append("─── Bottleneck breakdown ───")
            for stage, share in result.bottleneck_breakdown.items():
                lines.append(f"  • {stage}: {share:.1%}")
            lines.append("")
        if result.notes:
            lines.append("─── Notes ───")
            for note in result.notes:
                lines.append(f"  • {note}")
            lines.append("")
        return "\n".join(lines)

    lines = [
        f"{'='*80}",
        f"  SIMULATION REPORT: {result.scenario_name}",
        f"{'='*80}",
        "",
        "─── Surface status ───",
        "  Status:             Deprecated legacy memory engine",
        "  Scope note:         Classic VRAM/KV-cache estimate; not the benchmark-calibrated workload simulator path.",
        "",
        "─── Hardware ───",
        f"  Platform:           {result.hardware_profile}",
        f"  Catalog ID:         {result.hardware_catalog_id or '-'}",
        f"  GPUs:               {result.num_gpus}",
        f"  VRAM per GPU:       {result.vram_per_gpu_gb:.0f} GB",
        f"  Total system VRAM:  {result.total_system_vram_gb:.0f} GB",
        f"  TP degree:          {result.tensor_parallel_degree}",
        "",
        "─── Model Weights ───",
        f"  Model:              {result.model_name or '-'}",
        f"  Model source:       {result.model_source}",
        f"  Model ID:           {result.model_id or '-'}",
        f"  Precision:          {result.weight_precision.upper()}",
        f"  Weight memory:      {result.weight_memory_gb:.2f} GB",
        f"  % of system VRAM:   {100*result.weight_memory_gb/result.total_system_vram_gb:.1f}%",
        "",
        "─── KV Cache ───",
        f"  Precision:          {result.kv_cache_precision.upper()}",
        f"  Concurrent seqs:    {result.total_concurrent_sequences}",
        f"  Mean:               {result.kv_cache_vram_mean_gb:.2f} GB",
        f"  p50:                {result.kv_cache_vram_p50_gb:.2f} GB",
        f"  p90:                {result.kv_cache_vram_p90_gb:.2f} GB",
        f"  p95:                {result.kv_cache_vram_p95_gb:.2f} GB",
        f"  p99:                {result.kv_cache_vram_p99_gb:.2f} GB",
        f"  Max:                {result.kv_cache_vram_max_gb:.2f} GB",
        "",
        "─── Total VRAM ───",
        f"  Mean:               {result.total_vram_mean_gb:.2f} GB",
        f"  p95:                {result.total_vram_p95_gb:.2f} GB",
        f"  p99:                {result.total_vram_p99_gb:.2f} GB",
        f"  Max:                {result.total_vram_max_gb:.2f} GB",
        "",
        "─── Per-GPU VRAM ───",
        f"  Mean:               {result.per_gpu_vram_mean_gb:.2f} GB",
        f"  p95:                {result.per_gpu_vram_p95_gb:.2f} GB",
        f"  p99:                {result.per_gpu_vram_p99_gb:.2f} GB",
        f"  Overhead:           {result.runtime_overhead_mean_gb:.2f} GB",
        "",
        "─── Risk Assessment ───",
        f"  OOM probability:    {result.estimated_oom_probability:.2%}",
        f"  KV saturation prob: {result.kv_cache_saturation_probability:.2%}",
        f"  Risk level:         {result.qualitative_risk_level}",
        f"  Max concurrency:    {result.recommended_max_concurrency}",
        f"  MC iterations:      {result.monte_carlo_iterations}",
        f"  Software stack:     {result.software_stack_id or '-'}",
        f"  Deployment mode:    {result.deployment_profile_id or '-'}",
        "",
    ]
    if result.notes:
        lines.append("─── Engineering Notes ───")
        for note in result.notes:
            lines.append(f"  • {note}")
        lines.append("")
    return "\n".join(lines)


def format_hardware_comparison(results: list[ResultLike]) -> str:
    if not results:
        return ""
    if all(isinstance(r, WorkloadSimulationResult) for r in results):
        by_hw: dict[str, list[WorkloadSimulationResult]] = {}
        for r in results:
            by_hw.setdefault(r.hardware_catalog_id, []).append(r)
        lines = [f"\n{'='*80}", "  WORKLOAD HARDWARE COMPARISON", f"{'='*80}\n"]
        for hw_id, hw_results in by_hw.items():
            lines.append(f"━━━ {hw_id} ━━━")
            lines.append(f"  Mean throughput: {sum(r.throughput_per_sec_mean for r in hw_results)/len(hw_results):.3f} items/s")
            lines.append(f"  Mean SLA hit:    {sum(r.sla_hit_rate_mean for r in hw_results)/len(hw_results):.2%}")
            lines.append(f"  Mean GPU util:   {sum(r.gpu_util_mean for r in hw_results)/len(hw_results):.2%}")
            worst = min(hw_results, key=lambda r: r.sla_hit_rate_mean - r.drop_rate_p95)
            lines.append(f"  Weakest scenario: {worst.scenario_name} ({worst.qualitative_risk_level})")
            lines.append("")
        return "\n".join(lines)

    by_hw: dict[str, list[SimulationResult]] = {}
    for r in results:
        if isinstance(r, SimulationResult):
            by_hw.setdefault(r.hardware_profile, []).append(r)
    lines = [f"\n{'='*80}", "  HARDWARE COMPARISON ANALYSIS", f"{'='*80}\n"]
    for hw_name, hw_results in by_hw.items():
        lines.append(f"━━━ {hw_name} ━━━")
        lines.append(f"  Total VRAM: {hw_results[0].total_system_vram_gb:.0f} GB")
        lines.append(f"  VRAM/GPU:   {hw_results[0].vram_per_gpu_gb:.0f} GB")
        lines.append(f"  TP degree:  {hw_results[0].tensor_parallel_degree}")
        lines.append("")
        wt = hw_results[0].weight_memory_gb
        wt_pct = 100 * wt / hw_results[0].total_system_vram_gb
        lines.append(f"  Weight baseline: {wt:.1f} GB ({wt_pct:.1f}% of system VRAM)")
        worst = max(hw_results, key=lambda r: r.estimated_oom_probability)
        lines.append(f"  Highest risk scenario: {worst.scenario_name.split('__')[0]}")
        lines.append(f"    OOM probability: {worst.estimated_oom_probability:.2%}")
        lines.append(f"    Risk level: {worst.qualitative_risk_level}")
        safe = [r for r in hw_results if r.qualitative_risk_level in ("LOW", "MODERATE")]
        lines.append(f"  Safe scenarios (LOW/MODERATE risk): {len(safe)}/{len(hw_results)}")
        kv_limited = [r for r in hw_results if r.kv_cache_vram_p95_gb > wt * 0.3]
        if kv_limited:
            lines.append(f"  ⚠ KV cache becomes dominant constraint in {len(kv_limited)} scenario(s)")
        lines.append("")
    return "\n".join(lines)
