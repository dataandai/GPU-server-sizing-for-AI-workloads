from __future__ import annotations

import math
import re
from typing import Any

from .catalog_loader import (
    default_tensor_parallel_degree_for_hardware,
    get_deployment_profile_record,
    get_gpu_record,
    get_hardware_record,
    get_model_profile_record,
    get_software_stack_record,
)
from .model_role_registry import resolve_role_defaults
from .language_policy import scalars_for_language

_PARAM_RE = re.compile(r"(?<!\d)(\d{1,3})(?:\.(\d))?B(?!\w)", re.IGNORECASE)

ROLE_HINTS: dict[str, dict[str, Any]] = {
    "llm_role": {"model_kind": "llm"},
    "planner_role": {"model_kind": "llm"},
    "reasoning_role": {"model_kind": "llm"},
    "writer_role": {"model_kind": "llm"},
    "infer_role": {"model_kind": "llm"},
    "embedding_role": {"model_kind": "embedding"},
    "retrieval_role": {"model_kind": "embedding"},
    "reranker_role": {"model_kind": "reranker"},
    "guardrail_role": {"model_kind": "guardrail"},
    "review_role": {"model_kind": "guardrail"},
    "vlm_role": {"model_kind": "vlm"},
    "ocr_role": {"model_kind": "ocr"},
    "parse_role": {"model_kind": "service"},
    "tool_role": {"model_kind": "service"},
    "publish_role": {"model_kind": "service"},
    "ingest_role": {"model_kind": "service"},
    "preprocess_role": {"model_kind": "service"},
    "postprocess_role": {"model_kind": "service"},
    "rag_role": {"model_kind": "service"},
}

STACK_ENGINE_RULES: dict[str, dict[str, Any]] = {
    "nvidia_vllm_cuda": {
        "engine_id": "vllm",
        "engine_label": "vLLM continuous batching",
        "weight_precision_order": ["fp8", "bf16", "fp16"],
        "kv_cache_precision_order": ["fp8", "fp16"],
        "supports_prefix_caching": True,
        "supports_chunked_prefill": True,
        "supports_dynamic_batching": True,
        "notes": "General on-prem LLM serving path; a good baseline for long context and mixed batch workloads.",
    },
    "nvidia_tensorrt_llm": {
        "engine_id": "tensorrt_llm",
        "engine_label": "TensorRT-LLM",
        "weight_precision_order": ["fp8", "bf16", "fp16"],
        "kv_cache_precision_order": ["fp8", "fp16"],
        "supports_prefix_caching": True,
        "supports_chunked_prefill": True,
        "supports_dynamic_batching": True,
        "notes": "NVIDIA-specific serving path; primary choice for high throughput and low jitter.",
    },
    "nvidia_triton_inference": {
        "engine_id": "triton",
        "engine_label": "Triton dynamic batching",
        "weight_precision_order": ["bf16", "fp16", "fp8"],
        "kv_cache_precision_order": ["fp16", "fp8"],
        "supports_prefix_caching": False,
        "supports_chunked_prefill": False,
        "supports_dynamic_batching": True,
        "notes": "Primary for non-LLM or queue-heavy inference workloads; less ideal for long generative LLMs.",
    },
    "nvidia_nim_llm": {
        "engine_id": "nim",
        "engine_label": "NVIDIA NIM",
        "weight_precision_order": ["fp8", "bf16", "fp16"],
        "kv_cache_precision_order": ["fp8", "fp16"],
        "supports_prefix_caching": True,
        "supports_chunked_prefill": True,
        "supports_dynamic_batching": True,
        "notes": "Profile-based NVIDIA on-prem serving; operationally simple, but the support matrix is decisive.",
    },
    "nvidia_dgx_enterprise_stack": {
        "engine_id": "dgx_reference",
        "engine_label": "DGX reference stack",
        "weight_precision_order": ["bf16", "fp16", "fp8"],
        "kv_cache_precision_order": ["fp16", "fp8"],
        "supports_prefix_caching": True,
        "supports_chunked_prefill": True,
        "supports_dynamic_batching": True,
        "notes": "Reference DGX stack; a specific serving engine should be selected based on the workload family.",
    },
}

TEMPLATE_WORKLOAD_HEURISTICS: dict[str, dict[str, Any]] = {
    "retrieval_rerank_generation": {"prompt": 2400, "output": 420, "concurrency": 4, "queue_budget_ms": 700},
    "multi_stage_agentic_workflow": {"prompt": 6400, "output": 2200, "concurrency": 4, "queue_budget_ms": 12000},
    "batch_document_pipeline": {"prompt": 3200, "output": 800, "concurrency": 8, "queue_budget_ms": 6000},
    "streaming_queue_heavy_inference": {"prompt": 384, "output": 96, "concurrency": 64, "queue_budget_ms": 120},
    "hybrid_multimodal_pipeline": {"prompt": 3600, "output": 560, "concurrency": 3, "queue_budget_ms": 900},
    "enterprise_rag": {"prompt": 2200, "output": 320, "concurrency": 3, "queue_budget_ms": 900},
    "agentic_research": {"prompt": 6400, "output": 2200, "concurrency": 3, "queue_budget_ms": 12000},
    "biomedical_agentic_research": {"prompt": 7200, "output": 2400, "concurrency": 2, "queue_budget_ms": 14000},
    "multimodal_video_summary": {"prompt": 5200, "output": 1800, "concurrency": 2, "queue_budget_ms": 5000},
    "streaming_video_detection": {"prompt": 384, "output": 96, "concurrency": 24, "queue_budget_ms": 150},
    "multi_camera_3d_video_analytics": {"prompt": 512, "output": 96, "concurrency": 16, "queue_budget_ms": 150},
    "agentic_video_analytics": {"prompt": 3600, "output": 1100, "concurrency": 6, "queue_budget_ms": 3000},
    "document_to_audio": {"prompt": 4200, "output": 1800, "concurrency": 3, "queue_budget_ms": 4000},
    "retail_multimodal_assistant": {"prompt": 3400, "output": 520, "concurrency": 4, "queue_budget_ms": 1200},
    "realtime_voice_call_processing": {"prompt": 256, "output": 64, "concurrency": 32, "queue_budget_ms": 80},
    "route_optimization_decisioning": {"prompt": 1024, "output": 256, "concurrency": 2, "queue_budget_ms": 4000},
    "predictive_forecasting_analytics": {"prompt": 1024, "output": 128, "concurrency": 12, "queue_budget_ms": 1200},
    "recommendation_ranking_service": {"prompt": 1400, "output": 180, "concurrency": 10, "queue_budget_ms": 450},
    "fraud_anomaly_detection": {"prompt": 512, "output": 96, "concurrency": 24, "queue_budget_ms": 180},
}


def assess_role_model_compatibility(
    *,
    role_id: str,
    model_id: str,
    hardware_catalog_id: str,
    software_stack_id: str,
    deployment_profile_id: str,
    template_family: str | None = None,
    template_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hardware = get_hardware_record(hardware_catalog_id)
    gpu = get_gpu_record(str(hardware.get("gpu_id") or ""))
    stack = get_software_stack_record(software_stack_id)
    deployment = get_deployment_profile_record(deployment_profile_id)
    profile = _resolve_model_profile(role_id, model_id)

    gpu_mem = float(gpu.get("memory_gb") or 0.0)
    gpu_count = int(hardware.get("gpu_count") or 0)
    stack_id = str(stack.get("id") or software_stack_id)
    deployment_id = str(deployment.get("id") or deployment_profile_id)
    messages: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    supported_stacks = set(str(x) for x in profile.get("supported_stack_ids") or [])
    if supported_stacks and stack_id not in supported_stacks:
        errors.append(f"A(z) {model_id} is not marked as compatible with this stack: {stack_id}.")

    if bool(deployment.get("is_virtualized")):
        errors.append("The advanced template flow expects on-prem, non-virtualized deployment.")

    estimated_min_vram = profile.get("estimated_min_vram_gb_per_gpu")
    estimated_recommended_vram = profile.get("estimated_recommended_vram_gb_per_gpu")
    recommended_min_gpu_count = int(profile.get("recommended_min_gpu_count") or 1)

    if estimated_min_vram is not None and gpu_mem and gpu_mem + 1e-6 < float(estimated_min_vram):
        errors.append(
            f"Estimated minimum VRAM/GPU ({float(estimated_min_vram):.0f} GB) is larger than the selected GPU memory ({gpu_mem:.0f} GB)."
        )
    elif estimated_recommended_vram is not None and gpu_mem and gpu_mem + 1e-6 < float(estimated_recommended_vram):
        warnings.append(
            f"The model’s recommended VRAM/GPU is about {float(estimated_recommended_vram):.0f} GB, while the selected GPU has {gpu_mem:.0f} GB. It may fit, but with limited headroom."
        )

    if gpu_count < recommended_min_gpu_count:
        errors.append(
            f"The model recommends at least {recommended_min_gpu_count} GPUs, while the selected hardware has {gpu_count}."
        )

    validation_level = str(profile.get("validation_level") or "estimated").lower()
    if validation_level == "heuristic":
        warnings.append("Compatibility is derived from an estimated profile, not a cataloged model record.")
    elif validation_level == "advisory":
        warnings.append("Service-reference style role: compatibility is advisory only.")

    serving = build_serving_recommendation(
        role_id=role_id,
        model_id=model_id,
        model_profile=profile,
        hardware=hardware,
        gpu=gpu,
        stack=stack,
        workload_family=template_family,
        workload_inputs=template_inputs,
    )
    engine_notes = serving.get("notes") or []
    if engine_notes:
        messages.append(engine_notes[0])
    if serving.get("status") == "warning":
        warnings.extend(serving.get("warnings") or [])
    if serving.get("status") == "error":
        errors.extend(serving.get("errors") or [])

    if str(profile.get("model_kind") or "") == "llm" and gpu_count == 1 and float(profile.get("param_size_b") or 0.0) >= 60:
        warnings.append("Large LLM on a single GPU: runnability is uncertain and throughput/latency risk is high.")

    if not errors:
        messages.append("The selected hardware/stack/deployment combination is broadly aligned with the role.")
    status = "error" if errors else "warning" if warnings else "ok"
    return {
        "role_id": role_id,
        "model_id": model_id,
        "status": status,
        "hardware_catalog_id": hardware_catalog_id,
        "software_stack_id": software_stack_id,
        "deployment_profile_id": deployment_id,
        "gpu_memory_gb": gpu_mem,
        "gpu_count": gpu_count,
        "estimated_model_profile": {
            "model_kind": profile.get("model_kind"),
            "param_size_b": profile.get("param_size_b"),
            "estimated_min_vram_gb_per_gpu": estimated_min_vram,
            "estimated_recommended_vram_gb_per_gpu": estimated_recommended_vram,
            "recommended_min_gpu_count": recommended_min_gpu_count,
            "validation_level": validation_level,
        },
        "serving_recommendation": serving,
        "messages": _dedupe(messages),
        "warnings": _dedupe(warnings),
        "errors": _dedupe(errors),
        "notes": [note for note in [profile.get("notes")] if note],
    }


def assess_workload_model_compatibility(
    *,
    role_model_map: dict[str, str],
    hardware_catalog_id: str,
    software_stack_id: str,
    deployment_profile_id: str,
    workload_family: str | None = None,
    workload_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks = {
        role_id: assess_role_model_compatibility(
            role_id=role_id,
            model_id=model_id,
            hardware_catalog_id=hardware_catalog_id,
            software_stack_id=software_stack_id,
            deployment_profile_id=deployment_profile_id,
            template_family=workload_family,
            template_inputs=workload_inputs,
        )
        for role_id, model_id in role_model_map.items()
        if str(model_id).strip()
    }
    return summarize_compatibility_checks(checks)


def assess_template_model_compatibility(
    *,
    role_model_map: dict[str, str],
    hardware_catalog_id: str,
    software_stack_id: str,
    deployment_profile_id: str,
    template_family: str | None = None,
    template_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return assess_workload_model_compatibility(
        role_model_map=role_model_map,
        hardware_catalog_id=hardware_catalog_id,
        software_stack_id=software_stack_id,
        deployment_profile_id=deployment_profile_id,
        workload_family=template_family,
        workload_inputs=template_inputs,
    )


def assess_blueprint_model_compatibility(
    *,
    role_model_map: dict[str, str],
    hardware_catalog_id: str,
    software_stack_id: str,
    deployment_profile_id: str,
    workload_family: str | None = None,
    template_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    compatibility = assess_workload_model_compatibility(
        role_model_map=role_model_map,
        hardware_catalog_id=hardware_catalog_id,
        software_stack_id=software_stack_id,
        deployment_profile_id=deployment_profile_id,
        workload_family=workload_family,
        workload_inputs=template_inputs,
    )
    relaxed_checks: dict[str, dict[str, Any]] = {}
    for role_id, check in (compatibility.get("checks") or {}).items():
        cloned = dict(check)
        errors = list(cloned.get("errors") or [])
        warnings = list(cloned.get("warnings") or [])
        profile = cloned.get("estimated_model_profile") or {}
        role_kind = str(profile.get("model_kind") or ROLE_HINTS.get(role_id, {}).get("model_kind") or "generic").lower()
        demoted: list[str] = []
        remaining: list[str] = []
        for err in errors:
            if "is not marked as compatible with this stack" in err:
                demoted.append(err + " In blueprint flow, this is downgraded to an advisory warning because the pipeline may contain multiple serving role types.")
            else:
                remaining.append(err)
        errors = remaining
        warnings.extend(demoted)
        cloned["errors"] = _dedupe(errors)
        cloned["warnings"] = _dedupe(warnings)
        cloned["status"] = "error" if cloned["errors"] else "warning" if cloned["warnings"] else "ok"
        relaxed_checks[role_id] = cloned
    return summarize_compatibility_checks(relaxed_checks)

def summarize_compatibility_checks(checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    status_counts = {"ok": 0, "warning": 0, "error": 0}
    blocking_roles: list[str] = []
    note_lines: list[str] = []
    serving_notes: list[str] = []
    recommendation_lines: list[str] = []
    preferred_engine = None
    for role_id, check in checks.items():
        status = str(check.get("status") or "ok")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "error":
            blocking_roles.append(role_id)
        first_issue = (check.get("errors") or check.get("warnings") or check.get("messages") or [None])[0]
        if first_issue:
            note_lines.append(f"{role_id}: {first_issue}")
        serving = check.get("serving_recommendation") or {}
        if serving.get("summary_line"):
            recommendation_lines.append(f"{role_id}: {serving['summary_line']}")
        if serving.get("notes"):
            serving_notes.extend(serving.get("notes") or [])
        engine_id = serving.get("engine_id")
        if engine_id and preferred_engine is None:
            preferred_engine = engine_id
    overall_status = "error" if status_counts.get("error") else "warning" if status_counts.get("warning") else "ok"
    return {
        "overall_status": overall_status,
        "status_counts": status_counts,
        "blocking_roles": blocking_roles,
        "checks": checks,
        "notes": _dedupe(note_lines),
        "serving_notes": _dedupe(serving_notes),
        "serving_plan": {
            "preferred_engine": preferred_engine,
            "recommendation_lines": _dedupe(recommendation_lines),
        },
    }



def build_role_model_map(role_ids: list[str], model_overrides: dict[str, str] | None = None) -> dict[str, str]:
    overrides = {str(k): str(v).strip() for k, v in (model_overrides or {}).items() if str(v).strip()}
    role_map: dict[str, str] = {}
    for role_id in role_ids:
        defaults = resolve_role_defaults(role_id)
        role_map[role_id] = overrides.get(role_id) or str(defaults.get("default_model_id") or "")
    return role_map



def build_serving_recommendation(
    *,
    role_id: str,
    model_id: str,
    model_profile: dict[str, Any],
    hardware: dict[str, Any],
    gpu: dict[str, Any],
    stack: dict[str, Any],
    workload_family: str | None = None,
    workload_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stack_id = str(stack.get("id") or "")
    engine = STACK_ENGINE_RULES.get(stack_id, {
        "engine_id": stack_id or "custom",
        "engine_label": stack.get("framework") or stack_id or "custom runtime",
        "weight_precision_order": ["bf16", "fp16"],
        "kv_cache_precision_order": ["fp16"],
        "supports_prefix_caching": False,
        "supports_chunked_prefill": False,
        "supports_dynamic_batching": True,
        "notes": "Custom serving stack; recommendations should be interpreted as estimates.",
    })
    role_kind = str(model_profile.get("model_kind") or ROLE_HINTS.get(role_id, {}).get("model_kind") or "generic").lower()
    gpu_mem = float(gpu.get("memory_gb") or 0.0)
    gpu_count = int(hardware.get("gpu_count") or 1)
    params_b = float(model_profile.get("param_size_b") or 0.0)
    workload = infer_workload_characteristics(template_family=workload_family, template_inputs=workload_inputs, role_id=role_id, model_kind=role_kind)

    preferred_precision = choose_weight_precision(role_kind=role_kind, params_b=params_b, gpu=gpu, stack_id=stack_id, engine=engine)
    kv_cache_precision = choose_kv_cache_precision(role_kind=role_kind, preferred_precision=preferred_precision, engine=engine)
    recommended_tp = recommend_tensor_parallel_degree(
        params_b=params_b,
        role_kind=role_kind,
        hardware=hardware,
        gpu_mem=gpu_mem,
        model_profile=model_profile,
        preferred_precision=preferred_precision,
    )
    effective_context = recommend_context_window_tokens(
        role_kind=role_kind,
        params_b=params_b,
        engine=engine,
        workload=workload,
    )
    kv_est = estimate_kv_cache(
        role_kind=role_kind,
        params_b=params_b,
        context_tokens=effective_context,
        kv_cache_precision=kv_cache_precision,
        tp_degree=recommended_tp,
        gpu_mem=gpu_mem,
        model_profile=model_profile,
        concurrency=max(1, int(workload.get("effective_concurrency") or 1)),
    )
    recommended_batch = recommend_max_batch_size(
        role_kind=role_kind,
        kv_estimate=kv_est,
        gpu_mem=gpu_mem,
        tp_degree=recommended_tp,
        model_profile=model_profile,
        engine=engine,
        params_b=params_b,
        workload=workload,
    )

    warnings: list[str] = []
    errors: list[str] = []
    notes: list[str] = []
    status = "ok"

    if role_kind in {"llm", "vlm"} and engine["engine_id"] == "triton" and effective_context >= 8192:
        warnings.append("For longer-context generative LLM/VLM workloads, the generic Triton stack is not the primary choice; vLLM or TensorRT-LLM may be better.")
    if role_kind in {"embedding", "reranker", "guardrail", "ocr", "service", "custom_service"} and engine["engine_id"] in {"vllm", "tensorrt_llm"}:
        warnings.append("This role is more service- or encoder-like; Triton/NIM-side serving is often simpler and more predictable.")
    if role_kind in {"llm", "vlm"} and recommended_tp >= gpu_count and gpu_count > 1:
        warnings.append("The estimate uses the full GPU domain, leaving little isolated capacity for separate roles.")
    if role_kind in {"llm", "vlm"} and recommended_batch <= 1 and workload.get("effective_concurrency", 1) > 2:
        warnings.append("The recommended max batch is small relative to workload concurrency; throughput limits are likely.")
    if role_kind in {"llm", "vlm"} and kv_est.get("fits_with_headroom") is False:
        status = "error"
        errors.append("Estimated context + KV cache does not fit with safe headroom on the selected GPU/TP combination.")
    elif warnings:
        status = "warning"

    notes.extend([
        engine.get("notes") or "",
        f"Recommended engine: {engine.get('engine_label')}, precision: {preferred_precision.upper()}, KV cache: {kv_cache_precision.upper()}, TP: {recommended_tp}.",
    ])
    if role_kind in {"llm", "vlm"}:
        notes.append(
            f"Estimated context target: {effective_context} token, recommended max batch: {recommended_batch}, KV cache requirement ~{kv_est.get('kv_cache_gb_total', 0.0):.1f} GB total."
        )
    summary = f"{engine.get('engine_label')} · {preferred_precision.upper()} · TP={recommended_tp} · ctx≈{effective_context} · batch≈{recommended_batch}"
    return {
        "status": status,
        "engine_id": engine.get("engine_id"),
        "engine_label": engine.get("engine_label"),
        "recommended_weight_precision": preferred_precision,
        "recommended_kv_cache_precision": kv_cache_precision,
        "recommended_tensor_parallel_degree": recommended_tp,
        "recommended_context_window_tokens": effective_context,
        "recommended_max_batch_size": recommended_batch,
        "workload_characteristics": workload,
        "kv_cache_estimate": kv_est,
        "warnings": _dedupe(warnings),
        "errors": _dedupe(errors),
        "notes": [note for note in _dedupe(notes) if note],
        "summary_line": summary,
    }





def _language_scalars_from_inputs(template_inputs: dict[str, Any] | None) -> tuple[float, float, float]:
    inputs = template_inputs or {}
    code = inputs.get("_language_code") or inputs.get("language_code") or ""
    share = inputs.get("_language_share", inputs.get("language_share", 1.0))
    return scalars_for_language(code, share)

def infer_workload_characteristics(
    *,
    template_family: str | None,
    template_inputs: dict[str, Any] | None,
    role_id: str,
    model_kind: str,
) -> dict[str, Any]:
    family = str(template_family or "")
    payload = dict(TEMPLATE_WORKLOAD_HEURISTICS.get(family, {}))
    inputs = template_inputs or {}
    if template_family == "retrieval_rerank_generation":
        payload["prompt"] = int(_num(inputs.get("avg_prompt_tokens"), payload.get("prompt", 2400)))
        payload["output"] = int(_num(inputs.get("avg_output_tokens"), payload.get("output", 420)))
        payload["effective_concurrency"] = max(1, math.ceil(_num(inputs.get("request_rate_per_sec"), 2.5) * 1.8))
        payload["queue_budget_ms"] = 700
    elif template_family == "multi_stage_agentic_workflow":
        payload["prompt"] = int(max(1200, _num(inputs.get("avg_docs_per_step"), 4) * 850))
        payload["output"] = int(_num(inputs.get("report_length_tokens"), payload.get("output", 2200)))
        payload["effective_concurrency"] = max(1, int(round(_num(inputs.get("concurrent_runs"), 4))))
        payload["queue_budget_ms"] = 12000
    elif template_family == "batch_document_pipeline":
        pages = _num(inputs.get("avg_pages_per_doc"), 18)
        payload["prompt"] = int(max(800, pages * 180))
        payload["output"] = int(400 + pages * 12)
        payload["effective_concurrency"] = max(1, min(32, math.ceil(_num(inputs.get("docs_per_hour"), 120) / 12.0)))
        payload["queue_budget_ms"] = max(1000, int(_num(inputs.get("batch_window_sec"), 60) * 100))
    elif template_family == "streaming_queue_heavy_inference":
        payload["prompt"] = int(max(64, _num(inputs.get("frames_per_sec"), 10) * 24))
        payload["output"] = int(64)
        payload["effective_concurrency"] = max(1, int(_num(inputs.get("max_inflight"), 64)))
        payload["queue_budget_ms"] = int(_num(inputs.get("queue_budget_ms"), 120))
    elif template_family == "hybrid_multimodal_pipeline":
        payload["prompt"] = int(1600 + _num(inputs.get("avg_context_docs"), 6) * 260 + _num(inputs.get("video_clip_share"), 0.15) * 1800)
        payload["output"] = int(_num(inputs.get("avg_output_tokens"), 560))
        payload["effective_concurrency"] = max(1, math.ceil(_num(inputs.get("request_rate_per_sec"), 1.2) * 1.8))
        payload["queue_budget_ms"] = 900
    elif family == "enterprise_rag":
        payload["prompt"] = int(max(1024, _num(inputs.get("query_prompt_tokens"), payload.get("prompt", 2200)) + _num(inputs.get("avg_context_docs"), 8) * 180))
        payload["output"] = int(_num(inputs.get("answer_tokens"), payload.get("output", 320)))
        payload["effective_concurrency"] = max(1, math.ceil(_num(inputs.get("query_rate_per_sec"), 1.5) * 1.8))
        payload["queue_budget_ms"] = 900
    elif family in {"agentic_research", "biomedical_agentic_research", "agentic_video_analytics"}:
        payload["prompt"] = int(max(2400, payload.get("prompt", 6400)))
        payload["output"] = int(max(1200, payload.get("output", 2200)))
        payload["effective_concurrency"] = max(1, int(round(_num(inputs.get("concurrent_runs"), payload.get("concurrency", 3)))))
        payload["queue_budget_ms"] = int(payload.get("queue_budget_ms", 12000))
    elif family == "multimodal_video_summary":
        payload["prompt"] = int(max(4096, payload.get("prompt", 5200)))
        payload["output"] = int(max(1024, payload.get("output", 1800)))
        payload["effective_concurrency"] = max(1, int(round(payload.get("concurrency", 2))))
        payload["queue_budget_ms"] = 5000
    elif family in {"streaming_video_detection", "multi_camera_3d_video_analytics"}:
        payload["prompt"] = int(max(128, payload.get("prompt", 384)))
        payload["output"] = int(max(32, payload.get("output", 96)))
        payload["effective_concurrency"] = max(8, int(round(payload.get("concurrency", 24))))
        payload["queue_budget_ms"] = 150
    elif family == "document_to_audio":
        payload["prompt"] = int(max(2400, payload.get("prompt", 4200)))
        payload["output"] = int(max(600, payload.get("output", 1800)))
        payload["effective_concurrency"] = max(1, int(round(payload.get("concurrency", 3))))
        payload["queue_budget_ms"] = 4000
    elif family == "retail_multimodal_assistant":
        payload["prompt"] = int(max(1800, payload.get("prompt", 3400)))
        payload["output"] = int(max(300, payload.get("output", 520)))
        payload["effective_concurrency"] = max(1, int(round(payload.get("concurrency", 4))))
        payload["queue_budget_ms"] = 1200
    else:
        payload.setdefault("prompt", 2048)
        payload.setdefault("output", 384)
        payload.setdefault("effective_concurrency", int(payload.get("concurrency", 2) or 2))
        payload.setdefault("queue_budget_ms", 1000)

    prompt_mult, gen_mult, ctx_mult = _language_scalars_from_inputs(inputs)
    prompt = int(max(1, round((payload.get("prompt") or 0) * prompt_mult)))
    output = int(max(1, round((payload.get("output") or 0) * gen_mult)))
    effective_concurrency = int(payload.get("effective_concurrency") or 1)
    if model_kind in {"embedding", "reranker", "guardrail", "ocr", "service", "custom_service"}:
        effective_context = max(256, int(round(prompt * ctx_mult)))
    else:
        effective_context = max(512, int(round((prompt + min(output, max(128, output // 2))) * ctx_mult)))
    return {
        "template_family": family,
        "prompt_tokens": prompt,
        "output_tokens": output,
        "effective_concurrency": effective_concurrency,
        "effective_context_tokens": effective_context,
        "queue_budget_ms": int(payload.get("queue_budget_ms") or 1000),
    }



def choose_weight_precision(*, role_kind: str, params_b: float, gpu: dict[str, Any], stack_id: str, engine: dict[str, Any]) -> str:
    supported = [str(x).lower() for x in (gpu.get("default_precision_targets") or [])]
    order = [str(x).lower() for x in engine.get("weight_precision_order") or ["bf16", "fp16"]]
    if role_kind not in {"llm", "vlm"}:
        return "bf16" if "bf16" in supported else "fp16"
    for candidate in order:
        if candidate == "fp8" and params_b and params_b < 7:
            continue
        if candidate in supported:
            return candidate
    return "bf16"



def choose_kv_cache_precision(*, role_kind: str, preferred_precision: str, engine: dict[str, Any]) -> str:
    if role_kind not in {"llm", "vlm"}:
        return "fp16"
    order = [str(x).lower() for x in engine.get("kv_cache_precision_order") or ["fp16"]]
    for candidate in order:
        if candidate == "fp8" and preferred_precision in {"fp8", "int8"}:
            return "fp8"
        if candidate == "fp16":
            return "fp16"
    return "fp16"



def recommend_tensor_parallel_degree(
    *,
    params_b: float,
    role_kind: str,
    hardware: dict[str, Any],
    gpu_mem: float,
    model_profile: dict[str, Any],
    preferred_precision: str,
) -> int:
    gpu_count = int(hardware.get("gpu_count") or 1)
    if role_kind not in {"llm", "vlm"}:
        return 1
    recommended_tp_values = [int(v) for v in (hardware.get("recommended_tp_values") or []) if int(v) > 0]
    if 1 not in recommended_tp_values:
        recommended_tp_values = sorted(set([1] + recommended_tp_values + [gpu_count]))
    else:
        recommended_tp_values = sorted(set(recommended_tp_values + [gpu_count]))
    budget_per_gpu = max(1.0, gpu_mem * (0.88 if preferred_precision in {"fp8", "int8"} else 0.82))
    baseline_vram = float(model_profile.get("estimated_recommended_vram_gb_per_gpu") or model_profile.get("estimated_min_vram_gb_per_gpu") or 0.0)
    if baseline_vram <= 0 and params_b > 0:
        bytes_per_param = 1.0 if preferred_precision == "fp8" else 2.0
        baseline_vram = (params_b * bytes_per_param * 1e9) / (1024**3)
    min_tp = max(1, math.ceil(baseline_vram / max(1.0, budget_per_gpu)))
    if params_b >= 60 and gpu_count >= 4:
        min_tp = max(min_tp, 2)
    if params_b >= 100 and gpu_count >= 8:
        min_tp = max(min_tp, 4)
    for candidate in recommended_tp_values:
        if candidate >= min_tp and candidate <= gpu_count:
            return candidate
    return min(gpu_count, max(1, default_tensor_parallel_degree_for_hardware(str(hardware.get("id") or ""))))



def recommend_context_window_tokens(*, role_kind: str, params_b: float, engine: dict[str, Any], workload: dict[str, Any]) -> int:
    base = int(workload.get("effective_context_tokens") or 2048)
    if role_kind not in {"llm", "vlm"}:
        return min(8192, max(512, base))
    floor = 4096
    if params_b >= 60:
        cap = 16384
    elif params_b >= 24:
        cap = 32768
    else:
        cap = 65536
    if engine.get("engine_id") == "triton":
        cap = min(cap, 8192)
    if engine.get("engine_id") == "tensorrt_llm":
        cap = min(cap, 32768)
    target = max(floor, min(cap, int(math.ceil(base / 512.0) * 512)))
    return target



def estimate_kv_cache(
    *,
    role_kind: str,
    params_b: float,
    context_tokens: int,
    kv_cache_precision: str,
    tp_degree: int,
    gpu_mem: float,
    model_profile: dict[str, Any],
    concurrency: int,
) -> dict[str, Any]:
    if role_kind not in {"llm", "vlm"}:
        return {
            "kv_cache_gb_total": 0.0,
            "kv_cache_gb_per_gpu": 0.0,
            "fits_with_headroom": True,
            "estimated_total_active_tokens": 0,
            "estimated_kv_gb_per_1k_tokens_total": 0.0,
        }
    scalar = 0.025 * math.sqrt(max(params_b, 1.0)) + 0.005
    if role_kind == "vlm":
        scalar *= 1.2
    if kv_cache_precision == "fp8":
        scalar *= 0.55
    total_active_tokens = max(1, int(context_tokens)) * max(1, int(concurrency))
    kv_per_1k_tokens_total = scalar
    kv_total = (total_active_tokens / 1000.0) * kv_per_1k_tokens_total
    kv_per_gpu = kv_total / max(1, tp_degree)
    base_model_load = float(model_profile.get("estimated_recommended_vram_gb_per_gpu") or model_profile.get("estimated_min_vram_gb_per_gpu") or 0.0) / max(1, tp_degree)
    headroom_limit = gpu_mem * 0.90
    fits = (base_model_load + kv_per_gpu) <= headroom_limit + 1e-6
    return {
        "kv_cache_gb_total": round(kv_total, 3),
        "kv_cache_gb_per_gpu": round(kv_per_gpu, 3),
        "fits_with_headroom": fits,
        "estimated_total_active_tokens": total_active_tokens,
        "estimated_kv_gb_per_1k_tokens_total": round(kv_per_1k_tokens_total, 4),
        "headroom_limit_gb_per_gpu": round(headroom_limit, 3),
    }



def recommend_max_batch_size(
    *,
    role_kind: str,
    kv_estimate: dict[str, Any],
    gpu_mem: float,
    tp_degree: int,
    model_profile: dict[str, Any],
    engine: dict[str, Any],
    params_b: float,
    workload: dict[str, Any],
) -> int:
    if role_kind not in {"llm", "vlm"}:
        concurrency = max(1, int(workload.get("effective_concurrency") or 1))
        return min(256, max(1, concurrency * (4 if engine.get("supports_dynamic_batching") else 2)))
    base_model_load = float(model_profile.get("estimated_recommended_vram_gb_per_gpu") or model_profile.get("estimated_min_vram_gb_per_gpu") or 0.0) / max(1, tp_degree)
    kv_per_seq = float(kv_estimate.get("kv_cache_gb_total") or 0.0) / max(1, int(workload.get("effective_concurrency") or 1))
    if kv_per_seq <= 0:
        return 1
    kv_budget_total = max(0.0, (gpu_mem * 0.90 - base_model_load) * max(1, tp_degree) * 0.78)
    capacity = max(1, int(kv_budget_total / max(kv_per_seq, 1e-6)))
    if engine.get("engine_id") == "triton":
        capacity = min(capacity, 16 if params_b >= 24 else 32)
    elif engine.get("engine_id") == "tensorrt_llm":
        capacity = min(capacity, 64)
    else:
        capacity = min(capacity, 48)
    return max(1, capacity)



def _resolve_model_profile(role_id: str, model_id: str) -> dict[str, Any]:
    model_id = str(model_id or "").strip()
    if not model_id:
        return {"model_kind": "unknown", "validation_level": "heuristic"}
    try:
        profile = dict(get_model_profile_record(model_id))
        profile.setdefault("validation_level", "catalog")
        return profile
    except Exception:
        return _heuristic_profile(role_id, model_id)



def _heuristic_profile(role_id: str, model_id: str) -> dict[str, Any]:
    defaults = resolve_role_defaults(role_id)
    hint = ROLE_HINTS.get(role_id, {})
    model_kind = hint.get("model_kind") or defaults.get("role_type") or "generic"
    if model_id.startswith("custom/") or model_id.startswith("nvidia/") and model_id.endswith("reference"):
        return {
            "id": model_id,
            "model_kind": "custom_service",
            "validation_level": "advisory",
            "supported_stack_ids": ["nvidia_vllm_cuda", "nvidia_tensorrt_llm", "nvidia_nim_llm", "nvidia_triton_inference", "nvidia_dgx_enterprise_stack"],
            "notes": "Custom or service-reference override; precise VRAM validation is not possible from the name alone.",
        }
    params = _extract_params_b(model_id)
    if params is None:
        return {
            "id": model_id,
            "model_kind": model_kind,
            "validation_level": "heuristic",
            "supported_stack_ids": ["nvidia_vllm_cuda", "nvidia_tensorrt_llm", "nvidia_nim_llm", "nvidia_triton_inference"],
            "notes": "Model size could not be derived clearly from the name; only a stack/deployment-level estimate was possible.",
        }
    min_vram, rec_vram, min_gpu = _estimate_requirements(model_kind, params)
    return {
        "id": model_id,
        "model_kind": model_kind,
        "param_size_b": params,
        "estimated_min_vram_gb_per_gpu": min_vram,
        "estimated_recommended_vram_gb_per_gpu": rec_vram,
        "recommended_min_gpu_count": min_gpu,
        "validation_level": "heuristic",
        "supported_stack_ids": ["nvidia_vllm_cuda", "nvidia_tensorrt_llm", "nvidia_nim_llm", "nvidia_triton_inference"],
        "notes": "The model profile was built from a parameter-size estimate derived from the model name.",
    }



def _extract_params_b(model_id: str) -> float | None:
    match = _PARAM_RE.search(model_id)
    if not match:
        return None
    whole = int(match.group(1))
    frac = match.group(2)
    return float(f"{whole}.{frac}" if frac is not None else whole)



def _estimate_requirements(model_kind: str, params_b: float) -> tuple[float, float, int]:
    kind = str(model_kind or "").lower()
    if kind in {"embedding", "reranker"}:
        return (6.0, 8.0 if params_b <= 1 else 12.0, 1)
    if kind in {"guardrail", "ocr"}:
        if params_b <= 1:
            return (4.0, 6.0, 1)
        if params_b <= 8:
            return (12.0, 18.0, 1)
        return (20.0, 28.0, 1)
    if kind == "vlm":
        if params_b <= 8:
            return (18.0, 28.0, 1)
        if params_b <= 24:
            return (40.0, 56.0, 1)
        return (90.0, 120.0, 2)
    if kind in {"service", "custom_service"}:
        return (4.0, 8.0, 1)
    # llm / generic
    if params_b <= 8:
        return (12.0, 16.0, 1)
    if params_b <= 16:
        return (20.0, 32.0, 1)
    if params_b <= 32:
        return (44.0, 64.0, 1)
    if params_b <= 48:
        return (64.0, 80.0, 1)
    return (90.0, 120.0, 2)



def _num(value: Any, default: float) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)



def _dedupe(lines: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        text = str(line or "").strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out
