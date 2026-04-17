from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .calibration_loader import describe_anchor_match, get_matching_calibration_anchor
from .calibration_rules import calibrated_stage_time_ms, calibration_coverage_summary
from .catalog_loader import get_hardware_record, get_gpu_record
from .model_loader import _extract_counts_from_name
from .procurement import derive_workload_procurement_metrics
from .language_policy import scalars_from_language_mix
from .workload_validation import (
    apply_selected_model_bindings,
    canonical_reserved_capacity_fraction,
    harmonize_deployment_policy,
    select_stage_model_id,
    validate_workload_simulation_spec,
)


@dataclass
class WorkloadSimulationResult:
    result_type: str = "workload_simulation"
    scenario_name: str = ""
    workload_id: str = ""
    workload_class: str = ""
    workload_family: str = ""
    source_blueprint_id: str | None = None
    source_template_id: str | None = None
    runtime_family: str = ""
    runtime_mode: str = ""
    execution_mode: str = ""
    hardware_catalog_id: str = ""
    software_stack_id: str | None = None
    planning_profile: str = "baseline"
    gpu_count: int = 0
    mig_profile: str | None = None
    monte_carlo_trials: int = 0
    time_horizon_sec: int = 0
    configured_arrival_rate_per_sec: float = 0.0
    reserved_capacity_fraction: float = 0.0
    total_arrivals_mean: float = 0.0
    completed_mean: float = 0.0
    dropped_mean: float = 0.0
    throughput_per_sec_mean: float = 0.0
    throughput_per_sec_p50: float = 0.0
    throughput_per_sec_p95: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    queue_wait_p95_ms: float = 0.0
    queue_wait_p99_ms: float = 0.0
    drop_rate_mean: float = 0.0
    drop_rate_p95: float = 0.0
    sla_hit_rate_mean: float = 0.0
    sla_hit_rate_p95: float = 0.0
    gpu_util_mean: float = 0.0
    gpu_util_p95: float = 0.0
    cpu_util_mean: float = 0.0
    cpu_util_p95: float = 0.0
    peak_inflight_p95: float = 0.0
    steady_state_capacity_per_sec: float = 0.0
    safe_capacity_24x7_per_sec: float = 0.0
    recommended_max_arrival_rate_per_sec: float = 0.0
    capacity_gap_per_sec: float = 0.0
    capacity_headroom_ratio: float = 0.0
    procurement_band: str = "UNKNOWN"
    recommended_action: str = ""
    target_fit: bool | None = None
    bottleneck_stage: str = ""
    bottleneck_breakdown: dict[str, float] = field(default_factory=dict)
    stage_performance: dict[str, dict[str, float]] = field(default_factory=dict)
    template_inputs: dict[str, Any] = field(default_factory=dict)
    detailed_metrics: dict[str, Any] = field(default_factory=dict)
    model_bindings: dict[str, str] = field(default_factory=dict)
    language_scaling: dict[str, Any] = field(default_factory=dict)
    requested_model_overrides: dict[str, str] = field(default_factory=dict)
    override_role_ids: list[str] = field(default_factory=list)
    calibration_trace: dict[str, Any] = field(default_factory=dict)
    model_compatibility: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    qualitative_risk_level: str = "UNKNOWN"


def is_workload_simulation_payload(data: Any) -> bool:
    return isinstance(data, dict) and "workload_definition" in data and "runtime_profile" in data and "deployment_profile" in data and "simulation_profile" in data


def is_workload_simulation_file(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return is_workload_simulation_payload(data)


def load_workload_simulation(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not is_workload_simulation_payload(data):
        raise ValueError(f"'{path}' is not a workload simulation scenario.")
    return data




def _language_mix_scalars(work_item_profile: dict[str, Any]) -> tuple[float, float, float]:
    mix = (work_item_profile or {}).get("language_mix") or []
    return scalars_from_language_mix(mix)

class WorkloadSimulationEngine:
    def __init__(self, spec: dict[str, Any]):
        normalized = harmonize_deployment_policy(apply_selected_model_bindings(spec))
        errors, warnings = validate_workload_simulation_spec(normalized, strict_catalog=True)
        if errors:
            raise ValueError("Workload simulation validation failed: " + "; ".join(errors))
        self.validation_warnings = warnings
        self.spec = copy.deepcopy(normalized)
        self.workload = self.spec["workload_definition"]
        self.runtime = self.spec["runtime_profile"]
        self.deployment = self.spec["deployment_profile"]
        self.simulation = self.spec["simulation_profile"]
        self.hardware_binding = self.deployment.get("hardware_binding", {})
        self.hardware_catalog_id = str(self.hardware_binding.get("hardware_catalog_id", ""))
        if not self.hardware_catalog_id:
            raise ValueError("deployment_profile.hardware_binding.hardware_catalog_id is required")
        self.hardware_record = get_hardware_record(self.hardware_catalog_id)
        self.gpu_record = get_gpu_record(self.hardware_record["gpu_id"])
        self.metadata = self.spec.get("metadata", {}) or {}
        self._stage_calibration_anchors: dict[str, dict[str, Any]] = {}
        self._stage_calibration_matches: dict[str, dict[str, Any]] = {}
        self._stage_calibration_examples: dict[str, dict[str, Any]] = {}
        runtime_family = str(self.runtime.get("runtime_family", "custom"))
        execution_mode = str(self.deployment.get("execution_mode", "bare_metal"))
        for stage in self.workload.get("pipeline", {}).get("stages", []) or []:
            stage_id = str(stage.get("stage_id") or "")
            if not stage_id:
                continue
            anchor = get_matching_calibration_anchor(
                metadata=self.metadata,
                stage=stage,
                runtime_family=runtime_family,
                execution_mode=execution_mode,
                hardware_record=self.hardware_record,
                gpu_record=self.gpu_record,
            )
            self._stage_calibration_anchors[stage_id] = anchor or {}
            self._stage_calibration_matches[stage_id] = describe_anchor_match(anchor, stage=stage)

    def run(self) -> WorkloadSimulationResult:
        trials = int(self.simulation.get("monte_carlo_trials", 100))
        seed = int(self.simulation.get("random_seed", 42))

        metadata = self.metadata
        template_inputs = {str(k): v for k, v in (metadata.get("template_inputs") or {}).items()}
        throughput_samples: list[float] = []
        latency_p50_samples: list[float] = []
        latency_p95_samples: list[float] = []
        latency_p99_samples: list[float] = []
        queue_p95_samples: list[float] = []
        queue_p99_samples: list[float] = []
        drop_rate_samples: list[float] = []
        sla_hit_rate_samples: list[float] = []
        gpu_util_samples: list[float] = []
        cpu_util_samples: list[float] = []
        arrivals_samples: list[int] = []
        completed_samples: list[int] = []
        dropped_samples: list[int] = []
        peak_inflight_samples: list[int] = []
        bottleneck_counts: dict[str, int] = {}
        stage_metric_samples: dict[str, dict[str, list[float]]] = {}
        model_bindings = self._resolve_model_bindings()

        for i in range(trials):
            rng = np.random.default_rng(seed + i)
            trial_spec = copy.deepcopy(self.spec)
            self._apply_random_variables(trial_spec, rng)
            trial_metrics = self._run_trial(trial_spec, rng)
            throughput_samples.append(trial_metrics["throughput_per_sec"])
            latency_p50_samples.append(trial_metrics["latency_p50_ms"])
            latency_p95_samples.append(trial_metrics["latency_p95_ms"])
            latency_p99_samples.append(trial_metrics["latency_p99_ms"])
            queue_p95_samples.append(trial_metrics["queue_wait_p95_ms"])
            queue_p99_samples.append(trial_metrics["queue_wait_p99_ms"])
            drop_rate_samples.append(trial_metrics["drop_rate"])
            sla_hit_rate_samples.append(trial_metrics["sla_hit_rate"])
            gpu_util_samples.append(trial_metrics["gpu_util"])
            cpu_util_samples.append(trial_metrics["cpu_util"])
            arrivals_samples.append(trial_metrics["total_arrivals"])
            completed_samples.append(trial_metrics["completed"])
            dropped_samples.append(trial_metrics["dropped"])
            peak_inflight_samples.append(trial_metrics["peak_inflight"])
            bottleneck = trial_metrics["bottleneck_stage"]
            if bottleneck:
                bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
            for stage_id, metrics in (trial_metrics.get("stage_metrics") or {}).items():
                bucket = stage_metric_samples.setdefault(str(stage_id), {})
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        bucket.setdefault(str(metric_name), []).append(float(metric_value))

        bottleneck_breakdown = {
            stage: count / trials for stage, count in sorted(bottleneck_counts.items(), key=lambda x: (-x[1], x[0]))
        }
        top_bottleneck = max(bottleneck_breakdown, key=bottleneck_breakdown.get) if bottleneck_breakdown else ""

        result = WorkloadSimulationResult(
            scenario_name=str(self.spec.get("name") or self.workload.get("workload_id") or "workload_simulation"),
            workload_id=str(self.workload.get("workload_id", "")),
            workload_class=str(self.workload.get("workload_class", "general_pipeline")),
            workload_family=str(metadata.get("workload_family") or self.workload.get("workload_class", "general_pipeline")),
            source_blueprint_id=str(metadata.get("source_blueprint_id")) if metadata.get("source_blueprint_id") is not None else None,
            source_template_id=str(metadata.get("source_template_id")) if metadata.get("source_template_id") is not None else None,
            runtime_family=str(self.runtime.get("runtime_family", "custom")),
            runtime_mode=str(self.runtime.get("scheduler_model", {}).get("mode", "custom")),
            execution_mode=str(self.deployment.get("execution_mode", "bare_metal")),
            hardware_catalog_id=self.hardware_catalog_id,
            software_stack_id=self.hardware_binding.get("software_stack_id"),
            planning_profile=str(self.deployment.get("operational_policy", {}).get("planning_profile", "baseline")),
            gpu_count=int(self.hardware_binding.get("gpu_count", self.hardware_record.get("gpu_count", 1))),
            mig_profile=self.deployment.get("partitioning", {}).get("mig_profile"),
            monte_carlo_trials=trials,
            time_horizon_sec=int(self.simulation.get("time_horizon_sec", 3600)),
            configured_arrival_rate_per_sec=float(self.workload.get("input_profile", {}).get("arrival_pattern", {}).get("mean_arrival_rate_per_sec", 0.0) or 0.0),
            reserved_capacity_fraction=float(self.deployment.get("operational_policy", {}).get("reserved_capacity_fraction", 0.0) or 0.0),
            total_arrivals_mean=float(np.mean(arrivals_samples)),
            completed_mean=float(np.mean(completed_samples)),
            dropped_mean=float(np.mean(dropped_samples)),
            throughput_per_sec_mean=float(np.mean(throughput_samples)),
            throughput_per_sec_p50=float(np.percentile(throughput_samples, 50)),
            throughput_per_sec_p95=float(np.percentile(throughput_samples, 95)),
            latency_p50_ms=float(np.median(latency_p50_samples)),
            latency_p95_ms=float(np.percentile(latency_p95_samples, 95)),
            latency_p99_ms=float(np.percentile(latency_p99_samples, 99)),
            queue_wait_p95_ms=float(np.percentile(queue_p95_samples, 95)),
            queue_wait_p99_ms=float(np.percentile(queue_p99_samples, 99)),
            drop_rate_mean=float(np.mean(drop_rate_samples)),
            drop_rate_p95=float(np.percentile(drop_rate_samples, 95)),
            sla_hit_rate_mean=float(np.mean(sla_hit_rate_samples)),
            sla_hit_rate_p95=float(np.percentile(sla_hit_rate_samples, 95)),
            gpu_util_mean=float(np.mean(gpu_util_samples)),
            gpu_util_p95=float(np.percentile(gpu_util_samples, 95)),
            cpu_util_mean=float(np.mean(cpu_util_samples)),
            cpu_util_p95=float(np.percentile(cpu_util_samples, 95)),
            peak_inflight_p95=float(np.percentile(peak_inflight_samples, 95)),
            bottleneck_stage=top_bottleneck,
            bottleneck_breakdown=bottleneck_breakdown,
            stage_performance=self._aggregate_stage_performance(stage_metric_samples),
            template_inputs=template_inputs,
            model_bindings=model_bindings,
            language_scaling=copy.deepcopy(metadata.get("language_scaling") or {}),
            requested_model_overrides=copy.deepcopy(metadata.get("requested_model_overrides") or {}),
            override_role_ids=list(metadata.get("override_role_ids") or []),
            calibration_trace={
                "stages": copy.deepcopy(self._stage_calibration_matches),
                "examples": copy.deepcopy(self._stage_calibration_examples),
                "summary": calibration_coverage_summary(self._stage_calibration_matches),
            },
            model_compatibility=copy.deepcopy(metadata.get("model_compatibility") or {}),
        )
        result.qualitative_risk_level = self._assess_risk(result)
        procurement = derive_workload_procurement_metrics(result)
        result.steady_state_capacity_per_sec = procurement["steady_state_capacity_per_sec"]
        result.safe_capacity_24x7_per_sec = procurement["safe_capacity_24x7_per_sec"]
        result.recommended_max_arrival_rate_per_sec = procurement["recommended_max_arrival_rate_per_sec"]
        result.capacity_gap_per_sec = procurement["capacity_gap_per_sec"]
        result.capacity_headroom_ratio = procurement["capacity_headroom_ratio"]
        result.procurement_band = procurement["procurement_band"]
        result.recommended_action = procurement["recommended_action"]
        result.target_fit = procurement["target_fit"]
        result.detailed_metrics = self._build_detailed_metrics(result)
        result.notes = self._build_notes(result)
        return result



    def _aggregate_stage_performance(self, stage_metric_samples: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
        stage_order = [str(stage.get("stage_id", "stage")) for stage in self.workload.get("pipeline", {}).get("stages", [])]
        ordered_stage_ids = stage_order + [sid for sid in stage_metric_samples.keys() if sid not in stage_order]
        aggregated: dict[str, dict[str, float]] = {}
        for stage_id in ordered_stage_ids:
            metric_samples = stage_metric_samples.get(stage_id) or {}
            if not metric_samples:
                continue
            stage_summary: dict[str, float] = {}
            for metric_name, samples in metric_samples.items():
                if not samples:
                    continue
                stage_summary[f"{metric_name}_mean"] = float(np.mean(samples))
                stage_summary[f"{metric_name}_p95"] = float(np.percentile(samples, 95))
            aggregated[stage_id] = stage_summary
        return aggregated

    def _build_detailed_metrics(self, result: WorkloadSimulationResult) -> dict[str, Any]:
        family = str(result.workload_family or result.workload_class or "")
        inputs = result.template_inputs or {}
        detail = {
            "family_display_name": _workload_family_label(family),
            "primary_metrics": [],
            "secondary_metrics": [],
            "decision_notes": [],
        }

        def add_primary(label: str, value: Any, unit: str | None = None, display: str | None = None) -> None:
            detail["primary_metrics"].append(_detail_metric(label, value, unit=unit, display=display))

        def add_secondary(label: str, value: Any, unit: str | None = None, display: str | None = None) -> None:
            detail["secondary_metrics"].append(_detail_metric(label, value, unit=unit, display=display))

        top_stage_ids = sorted(
            result.stage_performance.keys(),
            key=lambda sid: float((result.stage_performance.get(sid) or {}).get("pressure_score_mean", 0.0)),
            reverse=True,
        )[:3]
        if top_stage_ids:
            detail["decision_notes"].append("Top pressure stages: " + ", ".join(top_stage_ids))

        if family == "realtime_voice_call_processing":
            concurrent_calls = max(1.0, float(inputs.get("concurrent_calls", 24) or 24))
            chunk_ms = max(80.0, float(inputs.get("audio_chunk_ms", 320) or 320))
            chunk_rate_per_call = 1000.0 / chunk_ms
            safe_calls = result.safe_capacity_24x7_per_sec / chunk_rate_per_call if chunk_rate_per_call > 0 else 0.0
            safe_audio_seconds_per_sec = result.safe_capacity_24x7_per_sec * chunk_ms / 1000.0
            add_primary("Target concurrent calls", concurrent_calls, unit="call")
            add_primary("24/7 safe concurrent calls", safe_calls, unit="call")
            add_primary("Safe audio throughput", safe_audio_seconds_per_sec, unit="audio-sec/s")
            add_primary("Chunk cadence", chunk_rate_per_call, unit="chunk/call/s")
            add_secondary("Chunk size", chunk_ms, unit="ms")
            add_secondary("Diarization", True, display=_bool_text(bool(inputs.get("diarization_enabled", True))))
            add_secondary("Live translation", True, display=_bool_text(bool(inputs.get("live_translation_enabled", False))))
            add_secondary("Agent assist", True, display=_bool_text(bool(inputs.get("agent_assist_enabled", True))))
            add_secondary("TTS response", True, display=_bool_text(bool(inputs.get("tts_response_enabled", False))))
            gap = safe_calls - concurrent_calls
            if gap >= 0:
                detail["decision_notes"].append(f"Safe 24/7 capacity provides roughly {gap:.1f} additional concurrent calls of headroom.")
            else:
                detail["decision_notes"].append(f"Target concurrency exceeds the safe 24/7 range by about {abs(gap):.1f} calls.")
        elif family == "route_optimization_decisioning":
            requests_per_min = max(0.1, float(inputs.get("solve_requests_per_minute", 6) or 6))
            avg_stops = max(1.0, float(inputs.get("avg_stops_per_job", 180) or 180))
            avg_vehicles = max(1.0, float(inputs.get("avg_vehicle_count", 48) or 48))
            constraints = max(0.0, float(inputs.get("constraint_count", 40) or 40))
            safe_solves_per_min = result.safe_capacity_24x7_per_sec * 60.0
            add_primary("Target solves / min", requests_per_min, unit="solve/min")
            add_primary("24/7 safe solve / perc", safe_solves_per_min, unit="solve/min")
            add_primary("Stop volumen", requests_per_min * avg_stops, unit="stop/min")
            add_primary("Safe stop volumen", safe_solves_per_min * avg_stops, unit="stop/min")
            add_secondary("Average stops / job", avg_stops, unit="stop")
            add_secondary("Average vehicle count", avg_vehicles, unit="vehicle")
            add_secondary("Constraints", constraints, unit="constraint")
            add_secondary("Dynamic replanning", True, display=_bool_text(bool(inputs.get("dynamic_replans_enabled", True))))
            detail["decision_notes"].append("Here, solver capacity should be interpreted in terms of optimization runs and problem size, not token throughput.")
        elif family == "predictive_forecasting_analytics":
            assets = max(1.0, float(inputs.get("monitored_assets", 250) or 250))
            metrics_per_asset = max(1.0, float(inputs.get("metrics_per_asset", 24) or 24))
            refresh_sec = max(1.0, float(inputs.get("refresh_interval_sec", 30) or 30))
            lookback = max(1.0, float(inputs.get("lookback_window_min", 180) or 180))
            horizon = max(1.0, float(inputs.get("forecast_horizon_min", 60) or 60))
            safe_assets = result.safe_capacity_24x7_per_sec * refresh_sec
            add_primary("Target asset coverage", assets, unit="asset")
            add_primary("24/7 safe asset coverage", safe_assets, unit="asset")
            add_primary("Telemetry points / s", (assets * metrics_per_asset) / refresh_sec, unit="metric/s")
            add_primary("Safe telemetry points / s", (safe_assets * metrics_per_asset) / refresh_sec, unit="metric/s")
            add_secondary("Metrics / asset", metrics_per_asset, unit="metric")
            add_secondary("Refresh interval", refresh_sec, unit="s")
            add_secondary("Lookback window", lookback, unit="min")
            add_secondary("Forecast horizon", horizon, unit="min")
            add_secondary("Anomaly detection", True, display=_bool_text(bool(inputs.get("anomaly_detection_enabled", True))))
            detail["decision_notes"].append("The safe asset count shows how many assets can be sustained in 24/7 operation at the specified refresh interval.")
        elif family == "recommendation_ranking_service":
            req_per_sec = max(0.1, float(inputs.get("recommendation_requests_per_sec", 8) or 8))
            candidate_pool = max(1.0, float(inputs.get("candidate_pool_size", 120) or 120))
            filters = max(0.0, float(inputs.get("catalog_filter_count", 6) or 6))
            safe_candidates_per_sec = result.safe_capacity_24x7_per_sec * candidate_pool
            add_primary("Target requests / s", req_per_sec, unit="req/s")
            add_primary("24/7 safe requests / s", result.safe_capacity_24x7_per_sec, unit="req/s")
            add_primary("Candidate volume", req_per_sec * candidate_pool, unit="candidate/s")
            add_primary("Safe candidate volume", safe_candidates_per_sec, unit="candidate/s")
            add_secondary("Candidate pool", candidate_pool, unit="candidate")
            add_secondary("Catalog filters", filters, unit="filter")
            add_secondary("Reranking", True, display=_bool_text(bool(inputs.get("reranking_enabled", True))))
            add_secondary("Personalization", True, display=_bool_text(bool(inputs.get("personalization_enabled", True))))
            add_secondary("Image grounding", True, display=_bool_text(bool(inputs.get("image_grounding_enabled", False))))
            detail["decision_notes"].append("Candidate volume shows how many candidates per second actually load the ranking stack.")
        elif family == "fraud_anomaly_detection":
            events_per_sec = max(1.0, float(inputs.get("transaction_events_per_sec", 150) or 150))
            avg_entities = max(1.0, float(inputs.get("avg_entities_per_case", 6) or 6))
            fp_sensitivity = max(0.1, float(inputs.get("false_positive_sensitivity", 0.7) or 0.7))
            add_primary("Target transactions / s", events_per_sec, unit="txn/s")
            add_primary("24/7 safe transactions / s", result.safe_capacity_24x7_per_sec, unit="txn/s")
            add_primary("Entity fan-out", events_per_sec * avg_entities, unit="entity/s")
            add_primary("Safe entity fan-out", result.safe_capacity_24x7_per_sec * avg_entities, unit="entity/s")
            add_secondary("Entities / case", avg_entities, unit="entity")
            add_secondary("False positive sensitivity", fp_sensitivity, unit="score")
            add_secondary("Graph features", True, display=_bool_text(bool(inputs.get("graph_features_enabled", True))))
            add_secondary("Realtime blocking", True, display=_bool_text(bool(inputs.get("realtime_blocking_enabled", True))))
            add_secondary("Explainability", True, display=_bool_text(bool(inputs.get("explainability_enabled", True))))
            detail["decision_notes"].append("In realtime blocking mode, latency p95 and queue wait matter more than maximum raw throughput.")
        else:
            add_primary("Target load", result.configured_arrival_rate_per_sec, unit="item/s")
            add_primary("24/7 safe capacity", result.safe_capacity_24x7_per_sec, unit="item/s")
            add_secondary("Workload family", family or result.workload_class)
        if result.bottleneck_stage:
            detail["decision_notes"].append(f"Dominant bottleneck: {result.bottleneck_stage}.")
        return detail
    def _resolve_model_bindings(self) -> dict[str, str]:
        bindings: dict[str, str] = {}
        for stage in self.workload.get("pipeline", {}).get("stages", []):
            binding = stage.get("model_binding") or {}
            role_id = binding.get("role_id")
            if not role_id:
                continue
            selected = str(binding.get("selected_model_id") or select_stage_model_id(stage, self.workload) or "").strip()
            if selected:
                bindings[str(role_id)] = selected
        return bindings

    def _run_trial(self, spec: dict[str, Any], rng: np.random.Generator) -> dict[str, float]:
        workload = spec["workload_definition"]
        runtime = spec["runtime_profile"]
        deployment = spec["deployment_profile"]
        horizon_ms = int(spec["simulation_profile"].get("time_horizon_sec", 3600)) * 1000
        warmup_ms = int(spec["simulation_profile"].get("warmup_sec", 0)) * 1000

        items = self._generate_work_items(workload, horizon_ms, rng)
        if not items:
            return {
                "throughput_per_sec": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0,
                "queue_wait_p95_ms": 0.0,
                "queue_wait_p99_ms": 0.0,
                "drop_rate": 0.0,
                "sla_hit_rate": 1.0,
                "gpu_util": 0.0,
                "cpu_util": 0.0,
                "total_arrivals": 0,
                "completed": 0,
                "dropped": 0,
                "peak_inflight": 0,
                "bottleneck_stage": "",
                "stage_metrics": {},
            }

        stages = workload.get("pipeline", {}).get("stages", []) or []
        if not stages:
            raise ValueError("workload_definition.pipeline.stages is empty")

        stage_metrics: dict[str, dict[str, float]] = {}
        for stage in stages:
            items, metrics = self._run_stage(items, stage, runtime, deployment, horizon_ms, rng)
            stage_metrics[str(stage.get("stage_id", "stage"))] = metrics

        latency_target = float(workload.get("sla_policy", {}).get("latency_target_ms_p95", math.inf))
        deadline_ms = float(workload.get("sla_policy", {}).get("deadline_per_item_ms", math.inf))
        completed_items = [item for item in items if not item.get("dropped") and item.get("completed_time_ms") is not None]
        post_warmup = [item for item in completed_items if float(item["arrival_time_ms"]) >= warmup_ms]
        post_warmup_latencies = [float(item["completed_time_ms"]) - float(item["arrival_time_ms"]) for item in post_warmup]
        post_warmup_queue = [float(item.get("queue_wait_total_ms", 0.0)) for item in post_warmup]
        if post_warmup_latencies:
            lat_p50 = float(np.percentile(post_warmup_latencies, 50))
            lat_p95 = float(np.percentile(post_warmup_latencies, 95))
            lat_p99 = float(np.percentile(post_warmup_latencies, 99))
            queue_p95 = float(np.percentile(post_warmup_queue, 95))
            queue_p99 = float(np.percentile(post_warmup_queue, 99))
        else:
            lat_p50 = lat_p95 = lat_p99 = queue_p95 = queue_p99 = 0.0

        arrivals = len([item for item in items if float(item["arrival_time_ms"]) >= warmup_ms])
        completed = len(post_warmup)
        dropped = len([item for item in items if item.get("dropped") and float(item["arrival_time_ms"]) >= warmup_ms])
        drop_rate = dropped / arrivals if arrivals else 0.0
        sla_hits = 0
        for item in post_warmup:
            total_latency = float(item["completed_time_ms"]) - float(item["arrival_time_ms"])
            if total_latency <= min(latency_target, deadline_ms):
                sla_hits += 1
        sla_hit_rate = sla_hits / arrivals if arrivals else 1.0
        effective_duration_sec = max(1.0, (horizon_ms - warmup_ms) / 1000.0)
        throughput = completed / effective_duration_sec
        peak_inflight = self._estimate_peak_inflight(items, warmup_ms)

        gpu_util = min(1.0, sum(m.get("gpu_busy_ms", 0.0) for m in stage_metrics.values()) / max(1.0, self._effective_gpu_workers(deployment) * horizon_ms))
        cpu_util = min(1.0, sum(m.get("cpu_busy_core_ms", 0.0) for m in stage_metrics.values()) / max(1.0, self._effective_cpu_workers(deployment) * horizon_ms))
        bottleneck_stage = max(stage_metrics.items(), key=lambda kv: kv[1].get("pressure_score", 0.0))[0] if stage_metrics else ""

        return {
            "throughput_per_sec": throughput,
            "latency_p50_ms": lat_p50,
            "latency_p95_ms": lat_p95,
            "latency_p99_ms": lat_p99,
            "queue_wait_p95_ms": queue_p95,
            "queue_wait_p99_ms": queue_p99,
            "drop_rate": drop_rate,
            "sla_hit_rate": sla_hit_rate,
            "gpu_util": gpu_util,
            "cpu_util": cpu_util,
            "total_arrivals": arrivals,
            "completed": completed,
            "dropped": dropped,
            "peak_inflight": peak_inflight,
            "bottleneck_stage": bottleneck_stage,
            "stage_metrics": stage_metrics,
        }

    def _generate_work_items(self, workload: dict[str, Any], horizon_ms: int, rng: np.random.Generator) -> list[dict[str, Any]]:
        input_profile = workload.get("input_profile", {})
        arrival_pattern = input_profile.get("arrival_pattern", {})
        mode = str(arrival_pattern.get("mode", "poisson"))
        base_rate = float(arrival_pattern.get("mean_arrival_rate_per_sec", 1.0) or 1.0)
        base_rate *= self._business_cycle_multiplier(arrival_pattern, rng)
        if mode == "bursty_poisson":
            p95_mult = float(arrival_pattern.get("burst_multiplier_p95", 2.0) or 2.0)
            sigma = max(0.05, math.log(max(p95_mult, 1.01)) / 1.645)
            base_rate *= float(rng.lognormal(mean=0.0, sigma=sigma))
        arrival_times = self._sample_arrival_times(mode, base_rate, horizon_ms, rng)
        items = []
        work_item_profile = input_profile.get("work_item", {})
        prompt_mult, output_mult, context_mult = _language_mix_scalars(work_item_profile)
        for idx, arrival_ms in enumerate(arrival_times):
            size = self._sample_distribution(work_item_profile.get("size_distribution", {}), rng) * prompt_mult
            complexity = max(0.2, self._sample_distribution(work_item_profile.get("complexity_distribution", {"distribution": "fixed", "value": 1.0}), rng))
            output_units = max(32.0, size * (0.18 + 0.12 * complexity) * max(1.0, output_mult / max(prompt_mult, 1e-6)))
            items.append({
                "item_id": idx,
                "arrival_time_ms": float(arrival_ms),
                "ready_time_ms": float(arrival_ms),
                "size": float(size),
                "complexity": float(complexity),
                "output_units": float(output_units),
                "queue_wait_total_ms": 0.0,
                "dropped": False,
                "completed_time_ms": None,
            })
        return items

    def _run_stage(
        self,
        items: list[dict[str, Any]],
        stage: dict[str, Any],
        runtime: dict[str, Any],
        deployment: dict[str, Any],
        horizon_ms: int,
        rng: np.random.Generator,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        stage_id = str(stage.get("stage_id", "stage"))
        ready_items = [item for item in items if not item.get("dropped")]
        ready_items.sort(key=lambda x: float(x.get("ready_time_ms", 0.0)))
        if not ready_items:
            return items, {
                "pressure_score": 0.0,
                "gpu_busy_ms": 0.0,
                "cpu_busy_core_ms": 0.0,
                "queue_wait_p95_ms": 0.0,
                "queue_wait_p99_ms": 0.0,
                "drop_share": 0.0,
                "completion_share": 0.0,
                "gpu_busy_share": 0.0,
                "cpu_busy_share": 0.0,
                "item_count": 0.0,
                "worker_count": 0.0,
            }

        batching = stage.get("batching_profile", {})
        batching_mode = str(batching.get("batching_mode", "none"))
        max_batch = int(batching.get("max_batch_size", 1) or 1)
        timeout_ms = float(batching.get("batch_timeout_ms", 0.0) or 0.0)
        worker_count = self._stage_worker_count(stage, deployment)
        worker_available = [0.0 for _ in range(worker_count)]
        gpu_busy_ms = 0.0
        cpu_busy_ms = 0.0
        queue_waits: list[float] = []
        dropped = 0
        completed = 0

        pending = list(ready_items)
        while pending:
            worker_idx = min(range(worker_count), key=lambda i: worker_available[i])
            worker_free = worker_available[worker_idx]
            first = pending.pop(0)
            if first.get("dropped"):
                continue
            batch = [first]
            if batching_mode != "none" and max_batch > 1:
                batch = self._assemble_batch(first, pending, worker_free, max_batch, timeout_ms)
                for chosen in batch[1:]:
                    pending.remove(chosen)

            batch_start = max(worker_free, float(batch[0].get("ready_time_ms", 0.0)))
            if batching_mode != "none" and timeout_ms > 0 and len(batch) < max_batch:
                batch_start = max(batch_start, float(batch[0].get("ready_time_ms", 0.0)) + timeout_ms)
            elif len(batch) > 1:
                batch_start = max(batch_start, max(float(x.get("ready_time_ms", 0.0)) for x in batch))

            queue_wait_batch = [max(0.0, batch_start - float(item.get("ready_time_ms", 0.0))) for item in batch]
            queue_waits.extend(queue_wait_batch)
            memory_pressure = self._batch_memory_pressure(stage, batch, deployment)
            service_times = [self._stage_service_time_ms(stage, item, runtime, deployment) for item in batch]
            batch_service_ms = self._batch_service_time_ms(service_times, batch, stage, runtime, deployment)
            batch_service_ms *= self._memory_pressure_penalty(memory_pressure)
            batch_service_ms += self._deployment_static_latency_overhead(deployment)
            batch_service_ms += max(0.0, float(rng.normal(0.0, self._deployment_latency_jitter(deployment))))
            batch_end = batch_start + batch_service_ms
            worker_available[worker_idx] = batch_end

            resource_profile = stage.get("resource_profile", {})
            primary_device = str(resource_profile.get("primary_device", "gpu"))
            if primary_device in ("gpu", "hybrid"):
                gpu_busy_ms += batch_service_ms * max(1, self._gpu_cost_multiplier(stage, deployment))
            if primary_device in ("cpu", "hybrid"):
                cpu_busy_ms += batch_service_ms * max(1.0, float(resource_profile.get("cpu_cores_per_work_item", 1.0))) * len(batch)

            drop_on_overload = bool(stage.get("failure_policy", {}).get("drop_on_overload", False))
            max_queue_wait_ms = float(self.workload.get("sla_policy", {}).get("max_queue_wait_ms", math.inf))
            deadline_ms = float(self.workload.get("sla_policy", {}).get("deadline_per_item_ms", math.inf))
            for item, wait_ms in zip(batch, queue_wait_batch):
                item["queue_wait_total_ms"] = float(item.get("queue_wait_total_ms", 0.0)) + wait_ms
                total_latency_if_completed = batch_end - float(item.get("arrival_time_ms", 0.0))
                overload = memory_pressure > 1.0 or wait_ms > max_queue_wait_ms or total_latency_if_completed > deadline_ms * 1.2
                if drop_on_overload and overload:
                    item["dropped"] = True
                    item["completed_time_ms"] = None
                    dropped += 1
                else:
                    item["ready_time_ms"] = batch_end
                    item["completed_time_ms"] = batch_end
                    completed += 1

        queue_wait_p95_ms = float(np.percentile(queue_waits, 95)) if queue_waits else 0.0
        queue_wait_p99_ms = float(np.percentile(queue_waits, 99)) if queue_waits else 0.0
        gpu_busy_share = gpu_busy_ms / max(1.0, self._effective_gpu_workers(deployment) * horizon_ms)
        cpu_busy_share = cpu_busy_ms / max(1.0, self._effective_cpu_workers(deployment) * horizon_ms)
        drop_share = dropped / max(1, len(ready_items))
        completion_share = completed / max(1, len(ready_items))
        pressure_score = queue_wait_p95_ms
        pressure_score += 1000.0 * max(0.0, drop_share)
        pressure_score += 200.0 * max(0.0, gpu_busy_share - 0.8)
        return items, {
            "pressure_score": pressure_score,
            "gpu_busy_ms": gpu_busy_ms,
            "cpu_busy_core_ms": cpu_busy_ms,
            "queue_wait_p95_ms": queue_wait_p95_ms,
            "queue_wait_p99_ms": queue_wait_p99_ms,
            "drop_share": drop_share,
            "completion_share": completion_share,
            "gpu_busy_share": gpu_busy_share,
            "cpu_busy_share": cpu_busy_share,
            "item_count": float(len(ready_items)),
            "worker_count": float(worker_count),
            "dropped": dropped,
            "completed": completed,
        }

    def _assemble_batch(self, first: dict[str, Any], pending: list[dict[str, Any]], worker_free: float, max_batch: int, timeout_ms: float) -> list[dict[str, Any]]:
        batch = [first]
        first_ready = float(first.get("ready_time_ms", 0.0))
        cutoff = max(worker_free, first_ready) + timeout_ms
        for item in pending:
            if len(batch) >= max_batch:
                break
            if float(item.get("ready_time_ms", 0.0)) <= cutoff:
                batch.append(item)
        return batch

    def _stage_service_time_ms(self, stage: dict[str, Any], item: dict[str, Any], runtime: dict[str, Any], deployment: dict[str, Any]) -> float:
        resource_profile = stage.get("resource_profile", {})
        model = resource_profile.get("service_time_model", {})
        mode = str(model.get("mode", "constant"))
        stage_id = str(stage.get("stage_id") or "")
        size = float(item.get("size", 1.0)) * float(item.get("complexity", 1.0))
        output_units = float(item.get("output_units", 0.0))
        anchor = self._stage_calibration_anchors.get(stage_id) or None
        if anchor:
            stage_binding = stage.get("model_binding") or {}
            stage_model_id = str(stage_binding.get("selected_model_id") or select_stage_model_id(stage, self.workload) or "")
            params_b = _estimate_model_size_billions(stage_model_id)
            base, trace = calibrated_stage_time_ms(
                anchor=anchor,
                size=size,
                output_units=output_units,
                current_gpu_bandwidth_gbps=float(self.gpu_record.get("memory_bandwidth_gbps") or 1500.0),
                current_model_params_b=params_b,
                current_stage_workers=self._stage_worker_count(stage, deployment),
                runtime_family=str(runtime.get("runtime_family", "custom")),
                execution_mode=str(deployment.get("execution_mode", "bare_metal")),
            )
            if stage_id and stage_id not in self._stage_calibration_examples:
                self._stage_calibration_examples[stage_id] = trace
        elif mode == "constant":
            base = float(model.get("base_ms", 1.0))
        elif mode == "linear":
            base = (
                float(model.get("base_ms", 0.0))
                + float(model.get("ms_per_input_unit", 0.0)) * size
                + float(model.get("ms_per_output_unit", 0.0)) * output_units
            )
        elif mode == "reference_scaled":
            reference = _REFERENCE_SERVICE_PROFILES.get(str(model.get("reference_profile_id", "")), _REFERENCE_SERVICE_PROFILES["default_llm_stage"])
            stage_binding = stage.get("model_binding") or {}
            stage_model_id = str(stage_binding.get("selected_model_id") or select_stage_model_id(stage, self.workload) or "")
            params_b = _estimate_model_size_billions(stage_model_id) or reference["reference_model_params_b"]
            gpu_bandwidth = float(self.gpu_record.get("memory_bandwidth_gbps") or 1500.0)
            hardware_factor = max(0.4, reference["reference_gpu_bandwidth_gbps"] / max(gpu_bandwidth, 1.0))
            model_factor = max(0.35, params_b / max(reference["reference_model_params_b"], 1e-6))
            base = (
                reference["base_ms"]
                + reference["ms_per_input_unit"] * size
                + reference["ms_per_output_unit"] * output_units
            ) * hardware_factor * model_factor
        else:
            base = float(model.get("base_ms", 1.0))
        base *= self._runtime_scalar(runtime, stage, item)
        base *= self._deployment_penalty_scalar(deployment)
        return max(0.1, base)

    def _batch_service_time_ms(
        self,
        service_times: list[float],
        batch: list[dict[str, Any]],
        stage: dict[str, Any],
        runtime: dict[str, Any],
        deployment: dict[str, Any],
    ) -> float:
        if not service_times:
            return 0.0
        batching = stage.get("batching_profile", {})
        mode = str(batching.get("batching_mode", "none"))
        n = len(service_times)
        if n == 1 or mode == "none":
            batch_time = service_times[0]
        else:
            base = max(service_times)
            logn = math.log2(n + 1)
            if mode in ("dynamic", "static"):
                batch_time = base * (1.0 + 0.12 * logn)
            elif mode in ("continuous", "inflight"):
                batch_time = base * (1.0 + 0.08 * logn)
            else:
                batch_time = base * (1.0 + 0.15 * logn)
            lengths = np.array([float(item.get("size", 1.0)) for item in batch], dtype=float)
            if len(lengths) > 1 and float(lengths.mean()) > 0:
                cv = float(lengths.std() / lengths.mean())
                if batching.get("allow_mixed_lengths", False):
                    batch_time *= 1.0 + max(0.0, cv - 0.25) * 0.10
                else:
                    batch_time *= 1.0 + max(0.0, cv) * 0.20
            penalties = runtime.get("runtime_penalties", {})
            if n <= 2:
                batch_time *= float(penalties.get("small_batch_efficiency_penalty", 1.0) or 1.0)
            if cv > 0.5:
                batch_time *= float(penalties.get("mixed_length_batch_penalty", 1.0) or 1.0)
        if mode in ("continuous", "inflight"):
            max_tokens = batching.get("max_batched_tokens") or runtime.get("scheduler_model", {}).get("max_num_batched_tokens")
            if max_tokens:
                total_tokens = sum(float(item.get("size", 0.0)) + float(item.get("output_units", 0.0)) for item in batch)
                if total_tokens > float(max_tokens):
                    batch_time *= 1.0 + (total_tokens / float(max_tokens) - 1.0) * 0.25
        return batch_time

    def _runtime_scalar(self, runtime: dict[str, Any], stage: dict[str, Any], item: dict[str, Any]) -> float:
        scheduler = runtime.get("scheduler_model", {})
        mode = str(scheduler.get("mode", "custom"))
        scalar = 1.0
        stage_type = str(stage.get("stage_type", "custom"))
        if mode == "vllm_continuous_batching":
            if stage_type in ("generator", "decoder"):
                if scheduler.get("prefix_cache_enabled"):
                    hit_rate = float(scheduler.get("prefix_cache_hit_rate", 0.0) or 0.0)
                    scalar *= max(0.72, 1.0 - hit_rate * 0.25)
                if scheduler.get("chunked_prefill"):
                    scalar *= 0.94
                if scheduler.get("speculative_decode_enabled"):
                    scalar *= 0.88
            if scheduler.get("decode_priority") and stage_type in ("generator", "decoder"):
                scalar *= 0.96
        elif mode == "trtllm_inflight_batching":
            if stage_type in ("generator", "decoder"):
                scalar *= 0.92 if scheduler.get("inflight_batching", True) else 1.0
                if scheduler.get("speculative_decode_enabled"):
                    scalar *= 0.9
            if scheduler.get("scheduler_policy") == "MAX_UTILIZATION":
                scalar *= 0.96
            elif scheduler.get("scheduler_policy") == "GUARANTEED_NO_EVICT":
                scalar *= 1.04
        elif mode == "triton_dynamic_batching":
            queue_delay_us = float(scheduler.get("queue_delay_us") or scheduler.get("max_queue_delay_microseconds") or 0)
            scalar *= 1.0 + min(queue_delay_us / 100000.0, 0.08)
        return scalar

    def _deployment_penalty_scalar(self, deployment: dict[str, Any]) -> float:
        mode = str(deployment.get("execution_mode", "bare_metal"))
        base = {
            "bare_metal": 1.0,
            "container_on_bare_metal": 1.01,
            "vm_passthrough": 1.05,
            "vmware_vgpu_time_sliced": 1.18,
            "vmware_vgpu_mig_backed": 1.12,
            "bare_metal_mig": 1.08,
        }.get(mode, 1.0)
        partition = deployment.get("partitioning", {})
        sm_fraction = float(partition.get("sm_fraction", 1.0) or 1.0)
        if sm_fraction < 1.0:
            base *= 1.0 / max(sm_fraction, 0.1)
        tp_factor = float(deployment.get("overhead_model", {}).get("throughput_penalty_factor", 1.0) or 1.0)
        if tp_factor > 0:
            base *= 1.0 / tp_factor
        return base

    def _deployment_static_latency_overhead(self, deployment: dict[str, Any]) -> float:
        return float(deployment.get("overhead_model", {}).get("static_latency_overhead_ms", 0.0) or 0.0)

    def _deployment_latency_jitter(self, deployment: dict[str, Any]) -> float:
        base = float(deployment.get("overhead_model", {}).get("latency_jitter_stddev_ms", 0.0) or 0.0)
        return base

    def _stage_worker_count(self, stage: dict[str, Any], deployment: dict[str, Any]) -> int:
        primary = str(stage.get("resource_profile", {}).get("primary_device", "gpu"))
        if primary in ("gpu", "hybrid"):
            gpu_workers = self._effective_gpu_workers(deployment)
            tp_size = int(deployment.get("hardware_binding", {}).get("tp_size", 1) or 1)
            if str(stage.get("stage_type", "")) in ("generator", "decoder", "encoder"):
                return max(1, gpu_workers // max(tp_size, 1))
            return max(1, gpu_workers)
        return max(1, self._effective_cpu_workers(deployment))

    def _effective_gpu_workers(self, deployment: dict[str, Any]) -> int:
        gpu_count = int(deployment.get("hardware_binding", {}).get("gpu_count", self.hardware_record.get("gpu_count", 1)) or 1)
        partition = deployment.get("partitioning", {})
        mode = str(partition.get("mode", "none"))
        reserved_fraction = canonical_reserved_capacity_fraction(deployment)
        usable_gpu = max(1, math.floor(gpu_count * max(0.1, 1.0 - reserved_fraction)))
        if mode in ("mig", "mig_backed_vgpu"):
            slices = int(partition.get("virtual_gpus_per_device", 0) or 0)
            if slices <= 0:
                sm_fraction = float(partition.get("sm_fraction", 1.0) or 1.0)
                slices = max(1, round(1.0 / max(sm_fraction, 0.125)))
            return max(1, usable_gpu * slices)
        return usable_gpu

    def _effective_cpu_workers(self, deployment: dict[str, Any]) -> int:
        gpu_count = int(deployment.get("hardware_binding", {}).get("gpu_count", self.hardware_record.get("gpu_count", 1)) or 1)
        return max(4, gpu_count * 4)

    def _gpu_cost_multiplier(self, stage: dict[str, Any], deployment: dict[str, Any]) -> int:
        tp_size = int(deployment.get("hardware_binding", {}).get("tp_size", 1) or 1)
        if str(stage.get("stage_type", "")) in ("generator", "decoder", "encoder"):
            return max(1, tp_size)
        return 1

    def _batch_memory_pressure(self, stage: dict[str, Any], batch: list[dict[str, Any]], deployment: dict[str, Any]) -> float:
        resource = stage.get("resource_profile", {})
        gb_per_item = float(resource.get("gpu_memory_gb_per_work_item", 0.0) or 0.0)
        if gb_per_item <= 0:
            return 0.0
        avg_complexity = float(np.mean([item.get("complexity", 1.0) for item in batch]))
        avg_size_factor = float(np.mean([item.get("size", 1.0) for item in batch])) / 2000.0
        batch_gb = gb_per_item * len(batch) * max(0.5, avg_complexity) * max(0.5, avg_size_factor)
        memory_gb = float(self.gpu_record.get("memory_gb", 0.0) or 0.0)
        partition = deployment.get("partitioning", {})
        memory_fraction = float(partition.get("memory_fraction", 1.0) or 1.0)
        if partition.get("mig_profile") and memory_fraction == 1.0:
            memory_fraction = _infer_memory_fraction_from_mig_profile(str(partition.get("mig_profile")))
        effective_mem = max(1.0, memory_gb * memory_fraction)
        return batch_gb / effective_mem

    def _memory_pressure_penalty(self, pressure: float) -> float:
        if pressure <= 0.75:
            return 1.0
        if pressure <= 1.0:
            return 1.0 + (pressure - 0.75) * 0.6
        return 1.15 + (pressure - 1.0) * 2.0

    def _estimate_peak_inflight(self, items: list[dict[str, Any]], warmup_ms: int) -> int:
        events: list[tuple[float, int]] = []
        for item in items:
            if item.get("dropped"):
                continue
            start = float(item.get("arrival_time_ms", 0.0))
            end = float(item.get("completed_time_ms", start))
            if start < warmup_ms:
                start = warmup_ms
            if end <= start:
                continue
            events.append((start, 1))
            events.append((end, -1))
        events.sort(key=lambda x: (x[0], x[1]))
        inflight = 0
        peak = 0
        for _, delta in events:
            inflight += delta
            peak = max(peak, inflight)
        return peak

    def _apply_random_variables(self, spec: dict[str, Any], rng: np.random.Generator) -> None:
        for variable in spec.get("simulation_profile", {}).get("random_variables", []) or []:
            target = str(variable.get("target_path", "") or "")
            if not target:
                continue
            sampled = self._sample_random_variable(variable, rng)
            parent, last_key = _resolve_parent(spec, target)
            if parent is None:
                continue
            existing = parent.get(last_key)
            if isinstance(existing, dict) and isinstance(sampled, (int, float)):
                existing["noise_multiplier"] = float(sampled)
            else:
                parent[last_key] = sampled

    def _sample_random_variable(self, variable: dict[str, Any], rng: np.random.Generator) -> Any:
        dist = str(variable.get("distribution", "fixed"))
        params = variable.get("parameters", {}) or {}
        if dist == "fixed":
            return params.get("value", 1.0)
        if dist == "normal":
            return float(rng.normal(float(params.get("mean", 0.0)), float(params.get("stddev", params.get("std", 1.0)))))
        if dist == "lognormal":
            mean = float(params.get("mean", 0.0))
            sigma = float(params.get("sigma", params.get("stddev", 0.25)))
            return float(rng.lognormal(mean=mean, sigma=sigma))
        if dist == "uniform":
            return float(rng.uniform(float(params.get("low", 0.0)), float(params.get("high", 1.0))))
        if dist == "bernoulli":
            return bool(rng.random() < float(params.get("p", 0.5)))
        if dist == "triangular":
            low = float(params.get("low", 0.8))
            mode = float(params.get("mode", 1.0))
            high = float(params.get("high", 1.2))
            return float(rng.triangular(low, mode, high))
        if dist == "empirical":
            samples = params.get("samples") or [1.0]
            return float(rng.choice(samples))
        return 1.0

    def _sample_arrival_times(self, mode: str, rate_per_sec: float, horizon_ms: int, rng: np.random.Generator) -> list[float]:
        horizon_sec = horizon_ms / 1000.0
        rate_per_sec = max(0.0001, rate_per_sec)
        if mode in ("poisson", "bursty_poisson"):
            t = 0.0
            arrivals = []
            while t < horizon_sec:
                t += float(rng.exponential(1.0 / rate_per_sec))
                if t < horizon_sec:
                    arrivals.append(t * 1000.0)
            return arrivals
        if mode in ("scheduled_batch", "streaming_fixed_rate"):
            step = 1.0 / rate_per_sec
            t = 0.0
            arrivals = []
            while t < horizon_sec:
                arrivals.append(t * 1000.0)
                t += step
            return arrivals
        # piecewise / fallback
        return self._sample_arrival_times("poisson", rate_per_sec, horizon_ms, rng)

    def _business_cycle_multiplier(self, arrival_pattern: dict[str, Any], rng: np.random.Generator) -> float:
        business = arrival_pattern.get("business_cycle", {}) or {}
        if not business.get("enabled"):
            return 1.0
        daily = float(business.get("daily_peak_multiplier", 1.0) or 1.0)
        weekly = float(business.get("weekly_peak_multiplier", 1.0) or 1.0)
        return float(rng.uniform(1.0, daily) * rng.uniform(1.0, weekly))

    def _sample_distribution(self, dist: dict[str, Any], rng: np.random.Generator) -> float:
        distribution = str(dist.get("distribution", "fixed"))
        noise_multiplier = float(dist.get("noise_multiplier", 1.0) or 1.0)
        if distribution == "fixed":
            value = float(dist.get("value", dist.get("mean", dist.get("p50", 1.0))))
            return max(0.0, value * noise_multiplier)
        if distribution == "normal":
            value = float(rng.normal(float(dist.get("mean", 1.0)), float(dist.get("stddev", dist.get("std", 0.2)))))
            return max(0.0, value * noise_multiplier)
        if distribution == "uniform":
            low = float(dist.get("low", dist.get("mean", 1.0) * 0.8))
            high = float(dist.get("high", dist.get("mean", 1.0) * 1.2))
            return max(0.0, float(rng.uniform(low, high)) * noise_multiplier)
        if distribution == "lognormal":
            if dist.get("p50") and dist.get("p95"):
                p50 = float(dist.get("p50"))
                p95 = float(dist.get("p95"))
                mu = math.log(max(p50, 1e-6))
                sigma = max(0.05, (math.log(max(p95, 1e-6)) - mu) / 1.645)
            else:
                mean = float(dist.get("mean", 1.0))
                sigma = float(dist.get("sigma", 0.4))
                mu = math.log(max(mean, 1e-6)) - 0.5 * sigma * sigma
            return max(0.0, float(rng.lognormal(mu, sigma)) * noise_multiplier)
        if distribution == "empirical":
            samples = dist.get("samples") or [dist.get("mean", 1.0)]
            return max(0.0, float(rng.choice(samples)) * noise_multiplier)
        if distribution == "triangular":
            low = float(dist.get("low", dist.get("mean", 1.0) * 0.6))
            mode = float(dist.get("mode", dist.get("p50", dist.get("mean", 1.0))))
            high = float(dist.get("high", dist.get("p95", dist.get("mean", 1.0) * 1.8)))
            if high < low:
                low, high = high, low
            mode = min(max(mode, low), high)
            return max(0.0, float(rng.triangular(low, mode, high)) * noise_multiplier)
        return max(0.0, float(dist.get("mean", 1.0)) * noise_multiplier)

    def _assess_risk(self, result: WorkloadSimulationResult) -> str:
        if result.drop_rate_p95 > 0.05 or result.sla_hit_rate_mean < 0.90 or result.gpu_util_p95 > 0.95:
            return "CRITICAL"
        if result.drop_rate_p95 > 0.02 or result.sla_hit_rate_mean < 0.95 or result.gpu_util_p95 > 0.88:
            return "HIGH"
        if result.drop_rate_mean > 0.005 or result.sla_hit_rate_mean < 0.98 or result.gpu_util_p95 > 0.78:
            return "MEDIUM"
        return "LOW"

    def _build_notes(self, result: WorkloadSimulationResult) -> list[str]:
        notes = [
            f"Runtime: {result.runtime_family}/{result.runtime_mode}",
            f"Deployment: {result.execution_mode}",
            f"Hardware: {result.hardware_catalog_id} ({result.gpu_count} GPU worker domain)",
        ]
        if getattr(self, "validation_warnings", None):
            notes.append("Validation warnings: " + " | ".join(self.validation_warnings))
        if result.mig_profile:
            notes.append(f"MIG profile active: {result.mig_profile}")
        if result.bottleneck_stage:
            notes.append(f"Most frequent bottleneck stage: {result.bottleneck_stage}")
        if result.model_bindings:
            notes.append("Role bindings: " + ", ".join(f"{k}={v}" for k, v in sorted(result.model_bindings.items())))
        calibration_summary = (result.calibration_trace or {}).get("summary", {}) if isinstance(result.calibration_trace, dict) else {}
        if calibration_summary.get("stage_count"):
            notes.append(
                f"Calibration coverage: {int(calibration_summary.get('matched_stage_count', 0))}/{int(calibration_summary.get('stage_count', 0))} stages matched to benchmark anchors."
            )
        compatibility = result.model_compatibility or {}
        if compatibility:
            counts = compatibility.get("status_counts") or {}
            notes.append(
                f"Model compatibility: {compatibility.get('overall_status', 'unknown')} (ok={counts.get('ok', 0)}, warning={counts.get('warning', 0)}, error={counts.get('error', 0)})."
            )
        if result.execution_mode == "vmware_vgpu_time_sliced":
            notes.append("VMware time-sliced vGPU adds scheduler jitter and lowers large-batch predictability.")
        if result.execution_mode in ("bare_metal_mig", "vmware_vgpu_mig_backed"):
            notes.append("MIG isolation increases determinism, but reduces per-slice memory/compute budget.")
        if getattr(result, "procurement_band", ""):
            notes.append(f"Procurement band: {result.procurement_band}. {result.recommended_action}")
        if getattr(result, "configured_arrival_rate_per_sec", 0.0) > 0:
            notes.append(
                f"Configured arrival rate: {result.configured_arrival_rate_per_sec:.2f}/s, safe 24/7 capacity: {getattr(result, 'safe_capacity_24x7_per_sec', 0.0):.2f}/s."
            )
        return notes



def _detail_metric(label: str, value: object, *, unit: str | None = None, display: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {"label": label, "value": value}
    if unit:
        payload["unit"] = unit
    if display is not None:
        payload["display"] = display
    return payload


def _bool_text(value: bool) -> str:
    return "Yes" if value else "No"


def _workload_family_label(family: str) -> str:
    return {
        "realtime_voice_call_processing": "Real-time voice call processing",
        "route_optimization_decisioning": "Dynamic route optimization",
        "predictive_forecasting_analytics": "Predictive maintenance and forecasting",
        "recommendation_ranking_service": "Recommendation and ranking",
        "fraud_anomaly_detection": "Fraud and anomaly detection",
        "enterprise_rag": "Enterprise RAG",
        "video_search_and_summarization": "Video search and summarization",
        "document_to_audio": "Document-to-audio",
        "retail_multimodal_assistant": "Retail multimodal assistant",
        "agentic_research": "Agentic research",
        "biomedical_agentic_research": "Biomedical research agent",
    }.get(family, family.replace("_", " ").title() if family else "Workload")

_REFERENCE_SERVICE_PROFILES: dict[str, dict[str, float]] = {
    "rag_blueprint_llm_stage": {
        "base_ms": 45.0,
        "ms_per_input_unit": 0.0014,
        "ms_per_output_unit": 0.020,
        "reference_model_params_b": 49.0,
        "reference_gpu_bandwidth_gbps": 3000.0,
    },
    "default_llm_stage": {
        "base_ms": 40.0,
        "ms_per_input_unit": 0.0012,
        "ms_per_output_unit": 0.018,
        "reference_model_params_b": 32.0,
        "reference_gpu_bandwidth_gbps": 3000.0,
    },
}


def _estimate_model_size_billions(model_id: str) -> float | None:
    if not model_id:
        return None
    total, active = _extract_counts_from_name(model_id)
    return active or total


def _resolve_parent(root: dict[str, Any], dotted_path: str) -> tuple[dict[str, Any] | None, str | None]:
    parts = dotted_path.split(".")
    cursor: Any = root
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            return None, None
        cursor = cursor.get(part)
        if cursor is None:
            return None, None
    if not isinstance(cursor, dict):
        return None, None
    return cursor, parts[-1]


def _infer_memory_fraction_from_mig_profile(profile: str) -> float:
    # Examples: 1g.35gb, 3g.71gb, 7g.141gb
    text = str(profile or "")
    parts = text.split("g.")
    try:
        gi = float(parts[0])
    except Exception:
        return 1.0
    return max(0.1, min(1.0, gi / 7.0))
