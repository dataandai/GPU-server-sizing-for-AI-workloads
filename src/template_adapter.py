from __future__ import annotations

import copy
from typing import Any

import yaml

from .blueprint_adapter import (
    _build_deployment_profile,
    _build_runtime_profile,
    _build_stage,
    _slug,
    _as_bool,
    _as_float,
    _normalize_template_inputs,
    _remove_stage_by_id,
    _dominant_language,
)
from .language_policy import context_multiplier, generation_multiplier, language_scaling_metadata, prompt_multiplier
from .catalog_loader import (
    get_advanced_workload_template_record,
    get_deployment_profile_record,
    get_hardware_record,
    get_software_stack_record,
    list_advanced_workload_template_records,
)
from .workload_validation import (
    apply_selected_model_bindings,
    harmonize_deployment_policy,
    validate_workload_simulation_spec,
)
from .model_compatibility import assess_template_model_compatibility, build_role_model_map


TEMPLATE_FAMILY_DEFAULTS: dict[str, dict[str, Any]] = {
    "retrieval_rerank_generation": {
        "workload_class": "retrieval_augmented",
        "unit_type": "request",
        "modalities": ["text"],
        "arrival_mode": "bursty_poisson",
        "mean_arrival_rate_per_sec": 2.5,
        "size_distribution": {"distribution": "lognormal", "p50": 2400, "p95": 12000},
        "complexity_distribution": {"distribution": "triangular", "mean": 1.15, "stddev": 0.25, "p95": 1.8},
        "sla": {"latency_target_ms_p95": 2200, "throughput_target_per_sec": 2.0, "deadline_per_item_ms": 4500, "availability_target": 0.999, "max_drop_rate": 0.01, "max_queue_wait_ms": 700},
    },
    "multi_stage_agentic_workflow": {
        "workload_class": "multistage_agentic",
        "unit_type": "workflow_run",
        "modalities": ["text", "document"],
        "arrival_mode": "bursty_poisson",
        "mean_arrival_rate_per_sec": 0.12,
        "size_distribution": {"distribution": "lognormal", "p50": 9000, "p95": 54000},
        "complexity_distribution": {"distribution": "triangular", "mean": 1.7, "stddev": 0.35, "p95": 2.6},
        "sla": {"latency_target_ms_p95": 180000, "throughput_target_per_sec": 0.08, "deadline_per_item_ms": 300000, "availability_target": 0.995, "max_drop_rate": 0.03, "max_queue_wait_ms": 12000},
    },
    "batch_document_pipeline": {
        "workload_class": "document_processing",
        "unit_type": "document",
        "modalities": ["document", "image", "text"],
        "arrival_mode": "scheduled_batch",
        "mean_arrival_rate_per_sec": 0.03,
        "size_distribution": {"distribution": "lognormal", "p50": 24, "p95": 180},
        "complexity_distribution": {"distribution": "triangular", "mean": 1.3, "stddev": 0.3, "p95": 2.0},
        "sla": {"latency_target_ms_p95": 60000, "throughput_target_per_sec": 0.03, "deadline_per_item_ms": 120000, "availability_target": 0.995, "max_drop_rate": 0.03, "max_queue_wait_ms": 6000},
    },
    "streaming_queue_heavy_inference": {
        "workload_class": "stream_processing",
        "unit_type": "stream_chunk",
        "modalities": ["audio", "video", "text"],
        "arrival_mode": "streaming_fixed_rate",
        "mean_arrival_rate_per_sec": 60.0,
        "size_distribution": {"distribution": "fixed", "value": 1.0},
        "complexity_distribution": {"distribution": "triangular", "mean": 1.0, "stddev": 0.2, "p95": 1.5},
        "sla": {"latency_target_ms_p95": 500, "throughput_target_per_sec": 55.0, "deadline_per_item_ms": 1500, "availability_target": 0.999, "max_drop_rate": 0.01, "max_queue_wait_ms": 120},
    },
    "hybrid_multimodal_pipeline": {
        "workload_class": "vision_processing",
        "unit_type": "request",
        "modalities": ["text", "image", "document", "video"],
        "arrival_mode": "bursty_poisson",
        "mean_arrival_rate_per_sec": 1.2,
        "size_distribution": {"distribution": "lognormal", "p50": 3400, "p95": 18000},
        "complexity_distribution": {"distribution": "triangular", "mean": 1.25, "stddev": 0.3, "p95": 1.95},
        "sla": {"latency_target_ms_p95": 3200, "throughput_target_per_sec": 1.0, "deadline_per_item_ms": 6500, "availability_target": 0.999, "max_drop_rate": 0.015, "max_queue_wait_ms": 900},
    },
}


def list_workload_template_records() -> list[dict[str, Any]]:
    return list_advanced_workload_template_records()


def _language_prompt_multiplier(language_code: str, language_share: float = 1.0) -> float:
    return prompt_multiplier(language_code, language_share)


def _language_generation_multiplier(language_code: str, language_share: float = 1.0) -> float:
    return generation_multiplier(language_code, language_share)


def _language_context_multiplier(language_code: str, language_share: float = 1.0) -> float:
    return context_multiplier(language_code, language_share)


def build_workload_from_template(
    *,
    template_id: str,
    scenario_name: str,
    hardware_catalog_id: str,
    software_stack_id: str,
    deployment_profile_id: str,
    language_code: str = "hu",
    language_share: float = 0.8,
    model_overrides: dict[str, str] | None = None,
    monte_carlo_trials: int = 40,
    time_horizon_sec: int = 300,
    planning_profile: str = "baseline",
    template_inputs: dict[str, Any] | None = None,
    sla_latency_p95_ms: float | None = None,
) -> dict[str, Any]:
    model_overrides = model_overrides or {}
    template = get_advanced_workload_template_record(template_id)
    hardware = get_hardware_record(hardware_catalog_id)
    stack = get_software_stack_record(software_stack_id)
    deployment_seed = get_deployment_profile_record(deployment_profile_id)
    family = str(template.get("template_family") or template.get("workload_family") or "retrieval_rerank_generation")
    defaults = copy.deepcopy(TEMPLATE_FAMILY_DEFAULTS[family])
    template_inputs = _normalize_template_inputs(template_inputs)

    workload = {
        "workload_id": _slug(template_id),
        "workload_class": defaults["workload_class"],
        "input_profile": {
            "arrival_pattern": {
                "mode": defaults["arrival_mode"],
                "mean_arrival_rate_per_sec": float(defaults["mean_arrival_rate_per_sec"]),
                "burst_multiplier_p95": 1.8,
                "business_cycle": {"enabled": True, "daily_peak_multiplier": 1.4, "weekly_peak_multiplier": 1.1},
            },
            "work_item": {
                "unit_type": defaults["unit_type"],
                "modalities": list(defaults["modalities"]),
                "size_distribution": copy.deepcopy(defaults["size_distribution"]),
                "complexity_distribution": copy.deepcopy(defaults["complexity_distribution"]),
                "language_mix": [
                    {"language": language_code, "share": float(language_share)},
                    {"language": "en", "share": float(max(0.0, 1.0 - float(language_share)))},
                ],
            },
        },
        "pipeline": {
            "topology": "linear",
            "stages": [_build_stage(seed, model_overrides, language_code) for seed in template.get("stages", [])],
        },
        "sla_policy": copy.deepcopy(defaults["sla"]),
    }
    workload = _apply_template_inputs(workload, template, template_inputs)
    dominant_language, dominant_share = _dominant_language(workload)
    if sla_latency_p95_ms is not None:
        workload["sla_policy"]["latency_target_ms_p95"] = float(sla_latency_p95_ms)
    deployment_profile = _build_deployment_profile(hardware, software_stack_id, deployment_seed, planning_profile=planning_profile)
    runtime_profile = _build_runtime_profile(stack, template)
    if planning_profile == "high_throughput":
        runtime_profile.setdefault("runtime_penalties", {})["small_batch_efficiency_penalty"] = float(runtime_profile.get("runtime_penalties", {}).get("small_batch_efficiency_penalty", 1.0)) * 0.96
    elif planning_profile == "safe_24x7":
        runtime_profile.setdefault("runtime_penalties", {})["queue_timeout_penalty"] = float(runtime_profile.get("runtime_penalties", {}).get("queue_timeout_penalty", 1.0)) * 1.02

    spec = {
        "schema_version": "1.0",
        "name": scenario_name,
        "description": f"Advanced workload template: {template.get('display_name')}",
        "metadata": {
            "source_template_id": template_id,
            "source_template_library_id": template_id,
            "template_source": "workload_library",
            "workload_family": family,
            "template_display_name": template.get("display_name"),
            "planning_profile": planning_profile,
            "template_inputs": template_inputs,
            "language_scaling": language_scaling_metadata(language_code=dominant_language, language_share=dominant_share),
            "requested_model_overrides": dict(model_overrides),
            "override_role_ids": sorted(model_overrides.keys()),
        },
        "workload_definition": workload,
        "runtime_profile": runtime_profile,
        "deployment_profile": deployment_profile,
        "simulation_profile": {
            "monte_carlo_trials": int(monte_carlo_trials),
            "warmup_sec": 30,
            "time_horizon_sec": int(time_horizon_sec),
            "time_step_ms": 10,
            "random_variables": [
                {
                    "name": "arrival_noise",
                    "applies_to": "input",
                    "distribution": "lognormal",
                    "parameters": {"mean": 0, "sigma": 0.2},
                    "target_path": "workload_definition.input_profile.arrival_pattern.mean_arrival_rate_per_sec",
                },
                {
                    "name": "size_noise",
                    "applies_to": "input",
                    "distribution": "lognormal",
                    "parameters": {"sigma": 0.25},
                    "target_path": "workload_definition.input_profile.work_item.size_distribution",
                },
            ],
        },
    }
    spec = harmonize_deployment_policy(apply_selected_model_bindings(spec))
    role_ids = []
    for stage in spec.get("workload_definition", {}).get("pipeline", {}).get("stages", []) or []:
        role_id = str(stage.get("role_id") or (stage.get("model_binding") or {}).get("role_id") or "").strip()
        if role_id and role_id not in role_ids:
            role_ids.append(role_id)
    compatibility = assess_template_model_compatibility(
        role_model_map=build_role_model_map(role_ids, model_overrides),
        hardware_catalog_id=hardware_catalog_id,
        software_stack_id=software_stack_id,
        deployment_profile_id=deployment_profile_id,
        template_family=family,
        template_inputs=template_inputs,
    )
    spec.setdefault("metadata", {})["model_compatibility"] = compatibility
    errors, warnings = validate_workload_simulation_spec(spec, strict_catalog=True)
    warnings = list(warnings)
    warnings.extend(compatibility.get("notes") or [])
    if compatibility.get("overall_status") == "error":
        errors.append("Model compatibility failed: " + "; ".join(compatibility.get("notes") or compatibility.get("blocking_roles") or ["incompatible selections"]))
    if errors:
        raise ValueError("Template workload validation failed: " + "; ".join(errors))
    if warnings:
        spec.setdefault("metadata", {})["validation_warnings"] = warnings
    return spec



def build_template_workload_yaml(**kwargs: Any) -> str:
    return yaml.safe_dump(build_workload_from_template(**kwargs), sort_keys=False, allow_unicode=True)



def _apply_template_inputs(workload: dict[str, Any], template: dict[str, Any], template_inputs: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(workload)
    family = str(template.get("template_family") or "")
    arrival = normalized.setdefault("input_profile", {}).setdefault("arrival_pattern", {})
    work_item = normalized.setdefault("input_profile", {}).setdefault("work_item", {})
    stages = normalized.setdefault("pipeline", {}).setdefault("stages", [])
    sla = normalized.setdefault("sla_policy", {})
    dominant_language, dominant_share = _dominant_language(normalized)
    prompt_mult = _language_prompt_multiplier(dominant_language, dominant_share)
    gen_mult = _language_generation_multiplier(dominant_language, dominant_share)
    ctx_mult = _language_context_multiplier(dominant_language, dominant_share)

    if family == "retrieval_rerank_generation":
        request_rate = max(0.05, _as_float(template_inputs.get("request_rate_per_sec"), 2.5))
        context_docs = max(1.0, _as_float(template_inputs.get("avg_context_docs"), 8.0))
        prompt_tokens = max(128.0, _as_float(template_inputs.get("avg_prompt_tokens"), 2400.0)) * prompt_mult
        output_tokens = max(32.0, _as_float(template_inputs.get("avg_output_tokens"), 420.0)) * gen_mult
        rerank_top_k = max(1.0, _as_float(template_inputs.get("rerank_top_k"), 24.0))
        guardrail_enabled = _as_bool(template_inputs.get("guardrail_enabled"), True)
        arrival["mean_arrival_rate_per_sec"] = round(request_rate, 4)
        work_item["size_distribution"] = {"distribution": "lognormal", "p50": round(prompt_tokens * max(1.0, ctx_mult / max(prompt_mult, 1e-6)), 2), "p95": round(prompt_tokens * (1.6 + context_docs / 20.0) * max(1.0, ctx_mult / max(prompt_mult, 1e-6)), 2)}
        mean_complexity = min(4.0, 0.85 + context_docs / 14.0 + rerank_top_k / 80.0 + output_tokens / 1600.0)
        work_item["complexity_distribution"] = {"distribution": "triangular", "mean": round(mean_complexity, 3), "stddev": 0.22, "p95": round(mean_complexity * 1.35, 3)}
        sla["latency_target_ms_p95"] = round(1400.0 + context_docs * 40.0 + output_tokens * 1.2, 2)
        sla["throughput_target_per_sec"] = round(max(0.05, request_rate * 0.85), 4)
        if not guardrail_enabled:
            stages = _remove_stage_by_id(stages, "optional_guardrail")
        normalized["pipeline"]["stages"] = stages
    elif family == "multi_stage_agentic_workflow":
        concurrent_runs = max(1.0, _as_float(template_inputs.get("concurrent_runs"), 4.0))
        avg_steps = max(1.0, _as_float(template_inputs.get("avg_steps_per_run"), 6.0))
        avg_tool_calls = max(0.0, _as_float(template_inputs.get("avg_tool_calls_per_run"), 3.0))
        avg_docs = max(0.0, _as_float(template_inputs.get("avg_docs_per_step"), 4.0))
        report_len = max(128.0, _as_float(template_inputs.get("report_length_tokens"), 2200.0)) * gen_mult
        human_gate = _as_bool(template_inputs.get("human_gate_enabled"), False)
        arrival["mean_arrival_rate_per_sec"] = round(max(0.01, concurrent_runs / max(60.0, avg_steps * 25.0)), 4)
        work_item["size_distribution"] = {"distribution": "lognormal", "p50": round(report_len + avg_docs * 600, 2), "p95": round((report_len + avg_docs * 600) * (1.6 + avg_steps / 10.0), 2)}
        mean_complexity = min(4.0, 1.1 + avg_steps / 4.5 + avg_tool_calls / 6.0 + avg_docs / 10.0 + report_len / 4000.0)
        work_item["complexity_distribution"] = {"distribution": "triangular", "mean": round(mean_complexity, 3), "stddev": 0.3, "p95": round(mean_complexity * 1.4, 3)}
        sla["latency_target_ms_p95"] = round(45000.0 + avg_steps * 12000.0 + avg_tool_calls * 4000.0 + report_len * 3.2, 2)
        sla["deadline_per_item_ms"] = round(sla["latency_target_ms_p95"] * 1.7, 2)
        if not human_gate:
            stages = _remove_stage_by_id(stages, "optional_review")
        normalized["pipeline"]["stages"] = stages
    elif family == "batch_document_pipeline":
        docs_per_hour = max(1.0, _as_float(template_inputs.get("docs_per_hour"), 120.0))
        pages = max(1.0, _as_float(template_inputs.get("avg_pages_per_doc"), 18.0))
        image_share = max(0.0, min(1.0, _as_float(template_inputs.get("image_heavy_share"), 0.35)))
        table_share = max(0.0, min(1.0, _as_float(template_inputs.get("table_heavy_share"), 0.25)))
        batch_window = max(1.0, _as_float(template_inputs.get("batch_window_sec"), 60.0))
        retry_rate = max(0.0, min(1.0, _as_float(template_inputs.get("retry_rate"), 0.05)))
        arrival["mode"] = "scheduled_batch"
        arrival["mean_arrival_rate_per_sec"] = round(max(0.001, docs_per_hour / 3600.0), 5)
        work_item["size_distribution"] = {"distribution": "lognormal", "p50": round(pages, 2), "p95": round(pages * (2.4 + image_share + table_share), 2)}
        mean_complexity = min(4.0, 0.9 + pages / 24.0 + image_share * 0.8 + table_share * 0.6 + retry_rate * 0.8)
        work_item["complexity_distribution"] = {"distribution": "triangular", "mean": round(mean_complexity, 3), "stddev": 0.28, "p95": round(mean_complexity * 1.4, 3)}
        sla["latency_target_ms_p95"] = round(max(15000.0, batch_window * 1000.0 * 0.85) * max(1.0, 0.6 * gen_mult + 0.4), 2)
        sla["deadline_per_item_ms"] = round(batch_window * 1000.0 * 1.5, 2)
    elif family == "streaming_queue_heavy_inference":
        streams = max(1.0, _as_float(template_inputs.get("streams"), 24.0))
        frames_per_sec = max(0.1, _as_float(template_inputs.get("frames_per_sec"), 10.0))
        burst_p95 = max(1.0, _as_float(template_inputs.get("burst_multiplier_p95"), 1.8))
        queue_budget = max(1.0, _as_float(template_inputs.get("queue_budget_ms"), 120.0))
        strict_ordering = _as_bool(template_inputs.get("strict_ordering"), False)
        max_inflight = max(1.0, _as_float(template_inputs.get("max_inflight"), 64.0))
        arrival["mode"] = "streaming_fixed_rate"
        arrival["mean_arrival_rate_per_sec"] = round(streams * frames_per_sec, 4)
        arrival["burst_multiplier_p95"] = round(burst_p95, 3)
        work_item["size_distribution"] = {"distribution": "fixed", "value": 1.0}
        mean_complexity = min(4.0, 0.75 + burst_p95 / 2.0 + streams / 80.0 + max_inflight / 160.0)
        work_item["complexity_distribution"] = {"distribution": "triangular", "mean": round(mean_complexity, 3), "stddev": 0.2, "p95": round(mean_complexity * 1.25, 3)}
        sla["max_queue_wait_ms"] = round(queue_budget, 2)
        sla["latency_target_ms_p95"] = round(queue_budget * (2.8 if strict_ordering else 2.2), 2)
        sla["deadline_per_item_ms"] = round(queue_budget * 6.0, 2)
        if not _as_bool(template_inputs.get("strict_ordering"), False):
            stages = _remove_stage_by_id(stages, "optional_publish")
        normalized["pipeline"]["stages"] = stages
    elif family == "hybrid_multimodal_pipeline":
        request_rate = max(0.01, _as_float(template_inputs.get("request_rate_per_sec"), 1.4))
        image_share = max(0.0, min(1.0, _as_float(template_inputs.get("image_share"), 0.35)))
        document_share = max(0.0, min(1.0, _as_float(template_inputs.get("document_share"), 0.4)))
        video_share = max(0.0, min(1.0, _as_float(template_inputs.get("video_clip_share"), 0.15)))
        context_docs = max(1.0, _as_float(template_inputs.get("avg_context_docs"), 6.0))
        output_tokens = max(32.0, _as_float(template_inputs.get("avg_output_tokens"), 560.0)) * gen_mult
        arrival["mean_arrival_rate_per_sec"] = round(request_rate, 4)
        work_item["modalities"] = ["text"] + (["image"] if image_share > 0 else []) + (["document"] if document_share > 0 else []) + (["video"] if video_share > 0 else [])
        work_item["size_distribution"] = {"distribution": "lognormal", "p50": round((1800 + context_docs * 260 + video_share * 1800) * max(1.0, ctx_mult), 2), "p95": round((1800 + context_docs * 260 + video_share * 1800) * (1.8 + image_share + document_share) * max(1.0, ctx_mult), 2)}
        mean_complexity = min(4.0, 0.95 + context_docs / 8.0 + image_share * 0.7 + document_share * 0.6 + video_share * 1.1 + output_tokens / 2400.0)
        work_item["complexity_distribution"] = {"distribution": "triangular", "mean": round(mean_complexity, 3), "stddev": 0.26, "p95": round(mean_complexity * 1.35, 3)}
        sla["latency_target_ms_p95"] = round(1800.0 + image_share * 300 + document_share * 250 + video_share * 650 + output_tokens * 1.3, 2)
        if image_share <= 0 and video_share <= 0:
            stages = _remove_stage_by_id(stages, "vlm_analyze")
        if document_share <= 0.05:
            stages = _remove_stage_by_id(stages, "optional_parse")
        if context_docs < 2:
            stages = _remove_stage_by_id(stages, "optional_rerank")
        normalized["pipeline"]["stages"] = stages
    return normalized
