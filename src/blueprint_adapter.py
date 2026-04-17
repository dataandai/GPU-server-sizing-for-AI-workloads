from __future__ import annotations

import copy
from typing import Any

import yaml

from .catalog_loader import (
    get_blueprint_bundle,
    get_deployment_profile_record,
    get_hardware_record,
    get_software_stack_record,
)
from .workload_validation import (
    apply_selected_model_bindings,
    harmonize_deployment_policy,
    validate_workload_simulation_spec,
)
from .model_role_registry import resolve_role_defaults
from .model_compatibility import assess_blueprint_model_compatibility, build_role_model_map
from .language_policy import context_multiplier, dominant_language, generation_multiplier, language_scaling_metadata, prompt_multiplier

RESOURCE_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "io_cpu_decode": {"stage_type": "preprocess", "resource_profile": {"primary_device": "cpu", "cpu_cores_per_work_item": 1.5, "service_time_model": {"mode": "linear", "base_ms": 10, "ms_per_input_unit": 0.05}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 8, "batch_timeout_ms": 20, "allow_mixed_lengths": True}},
    "cpu_decode_gpu_preprocess": {"stage_type": "preprocess", "resource_profile": {"primary_device": "hybrid", "cpu_cores_per_work_item": 1.2, "gpu_memory_gb_per_work_item": 0.12, "service_time_model": {"mode": "linear", "base_ms": 5, "ms_per_input_unit": 0.03}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 16, "batch_timeout_ms": 12, "allow_mixed_lengths": True}},
    "cpu_gpu_mixed": {"stage_type": "aggregation", "resource_profile": {"primary_device": "hybrid", "cpu_cores_per_work_item": 1.0, "gpu_memory_gb_per_work_item": 0.08, "service_time_model": {"mode": "linear", "base_ms": 7, "ms_per_input_unit": 0.02}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 16, "batch_timeout_ms": 10, "allow_mixed_lengths": True}},
    "gpu_embedding_batchable": {"stage_type": "embedding", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 0.12, "service_time_model": {"mode": "linear", "base_ms": 5, "ms_per_input_unit": 0.003}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 32, "batch_timeout_ms": 8, "allow_mixed_lengths": True}},
    "gpu_rerank_batchable": {"stage_type": "rerank", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 0.22, "service_time_model": {"mode": "linear", "base_ms": 10, "ms_per_input_unit": 0.002}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 16, "batch_timeout_ms": 10, "allow_mixed_lengths": True}},
    "gpu_parse_bursty": {"stage_type": "ocr", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 0.4, "service_time_model": {"mode": "linear", "base_ms": 18, "ms_per_input_unit": 0.006}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 8, "batch_timeout_ms": 25, "allow_mixed_lengths": True}},
    "gpu_vlm_optional": {"stage_type": "encoder", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 1.0, "service_time_model": {"mode": "reference_scaled", "reference_profile_id": "default_vlm_stage"}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 8, "batch_timeout_ms": 20, "allow_mixed_lengths": True}},
    "gpu_vlm_batchable": {"stage_type": "encoder", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 0.8, "service_time_model": {"mode": "reference_scaled", "reference_profile_id": "default_vlm_stage"}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 12, "batch_timeout_ms": 15, "allow_mixed_lengths": True}},
    "gpu_vlm_bursty": {"stage_type": "encoder", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 1.4, "service_time_model": {"mode": "reference_scaled", "reference_profile_id": "default_vlm_stage"}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 6, "batch_timeout_ms": 30, "allow_mixed_lengths": True}},
    "gpu_vision_realtime": {"stage_type": "detector", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 0.3, "service_time_model": {"mode": "linear", "base_ms": 14, "ms_per_input_unit": 0.02}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 8, "batch_timeout_ms": 6, "allow_mixed_lengths": False}},
    "gpu_multicam_heavy": {"stage_type": "detector", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 0.9, "service_time_model": {"mode": "linear", "base_ms": 30, "ms_per_input_unit": 0.04}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 4, "batch_timeout_ms": 8, "allow_mixed_lengths": False}},
    "gpu_or_cpu_retrieval_optional": {"stage_type": "retrieval", "resource_profile": {"primary_device": "hybrid", "cpu_cores_per_work_item": 0.7, "gpu_memory_gb_per_work_item": 0.1, "service_time_model": {"mode": "linear", "base_ms": 8, "ms_per_input_unit": 0.002}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 24, "batch_timeout_ms": 10, "allow_mixed_lengths": True}},
    "gpu_llm_prefill_decode": {"stage_type": "generator", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 1.8, "service_time_model": {"mode": "reference_scaled", "reference_profile_id": "default_llm_stage"}}, "batching_profile": {"batching_mode": "continuous", "max_batch_size": 64, "batch_timeout_ms": 15, "max_batched_tokens": 32768, "allow_mixed_lengths": True}},
    "gpu_llm_interactive": {"stage_type": "generator", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 1.2, "service_time_model": {"mode": "reference_scaled", "reference_profile_id": "interactive_llm_stage"}}, "batching_profile": {"batching_mode": "continuous", "max_batch_size": 24, "batch_timeout_ms": 10, "max_batched_tokens": 12288, "allow_mixed_lengths": True}},
    "gpu_llm_bursty": {"stage_type": "generator", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 1.4, "service_time_model": {"mode": "reference_scaled", "reference_profile_id": "default_llm_stage"}}, "batching_profile": {"batching_mode": "inflight", "max_batch_size": 32, "batch_timeout_ms": 25, "max_batched_tokens": 16384, "allow_mixed_lengths": True}},
    "gpu_llm_long_context": {"stage_type": "generator", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 2.4, "service_time_model": {"mode": "reference_scaled", "reference_profile_id": "long_context_llm_stage"}}, "batching_profile": {"batching_mode": "continuous", "max_batch_size": 8, "batch_timeout_ms": 20, "max_batched_tokens": 65536, "allow_mixed_lengths": True}},
    "gpu_llm_optional": {"stage_type": "generator", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 0.8, "service_time_model": {"mode": "reference_scaled", "reference_profile_id": "interactive_llm_stage"}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 8, "batch_timeout_ms": 25, "allow_mixed_lengths": True}},
    "componentized_rag_optional": {"stage_type": "retrieval", "resource_profile": {"primary_device": "hybrid", "cpu_cores_per_work_item": 0.9, "gpu_memory_gb_per_work_item": 0.35, "service_time_model": {"mode": "linear", "base_ms": 15, "ms_per_input_unit": 0.004, "ms_per_output_unit": 0.001}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 16, "batch_timeout_ms": 18, "allow_mixed_lengths": True}},
    "gpu_or_cpu_guardrail": {"stage_type": "classifier", "resource_profile": {"primary_device": "hybrid", "cpu_cores_per_work_item": 0.4, "gpu_memory_gb_per_work_item": 0.08, "service_time_model": {"mode": "linear", "base_ms": 6, "ms_per_input_unit": 0.0015}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 32, "batch_timeout_ms": 8, "allow_mixed_lengths": True}},
    "gpu_tts_streaming_or_batch": {"stage_type": "postprocess", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 0.35, "service_time_model": {"mode": "linear", "base_ms": 40, "ms_per_input_unit": 0.02, "ms_per_output_unit": 0.04}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 12, "batch_timeout_ms": 15, "allow_mixed_lengths": True}},
    "external_tooling_latency_bound": {"stage_type": "tool_call", "resource_profile": {"primary_device": "cpu", "cpu_cores_per_work_item": 0.25, "service_time_model": {"mode": "linear", "base_ms": 120, "ms_per_input_unit": 0.004}}, "batching_profile": {"batching_mode": "none", "max_batch_size": 1, "batch_timeout_ms": 0, "allow_mixed_lengths": False}},
    "gpu_asr_streaming": {"stage_type": "asr", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 0.45, "service_time_model": {"mode": "linear", "base_ms": 18, "ms_per_input_unit": 0.012}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 24, "batch_timeout_ms": 4, "allow_mixed_lengths": True}},
    "gpu_audio_diarization_realtime": {"stage_type": "classifier", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 0.18, "service_time_model": {"mode": "linear", "base_ms": 10, "ms_per_input_unit": 0.006}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 24, "batch_timeout_ms": 4, "allow_mixed_lengths": True}},
    "gpu_translation_streaming": {"stage_type": "generator", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 0.65, "service_time_model": {"mode": "linear", "base_ms": 16, "ms_per_input_unit": 0.01, "ms_per_output_unit": 0.012}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 16, "batch_timeout_ms": 6, "allow_mixed_lengths": True}},
    "cpu_route_problem_build": {"stage_type": "preprocess", "resource_profile": {"primary_device": "cpu", "cpu_cores_per_work_item": 0.6, "service_time_model": {"mode": "linear", "base_ms": 25, "ms_per_input_unit": 0.03}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 8, "batch_timeout_ms": 15, "allow_mixed_lengths": True}},
    "gpu_optimization_solver": {"stage_type": "tool_call", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 0.3, "service_time_model": {"mode": "linear", "base_ms": 150, "ms_per_input_unit": 0.025, "ms_per_output_unit": 0.004}}, "batching_profile": {"batching_mode": "none", "max_batch_size": 1, "batch_timeout_ms": 0, "allow_mixed_lengths": False}},
    "cpu_publish_dispatch": {"stage_type": "postprocess", "resource_profile": {"primary_device": "cpu", "cpu_cores_per_work_item": 0.15, "service_time_model": {"mode": "linear", "base_ms": 12, "ms_per_input_unit": 0.003}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 16, "batch_timeout_ms": 10, "allow_mixed_lengths": True}},
    "cpu_sensor_ingest_stream": {"stage_type": "preprocess", "resource_profile": {"primary_device": "cpu", "cpu_cores_per_work_item": 0.35, "service_time_model": {"mode": "linear", "base_ms": 8, "ms_per_input_unit": 0.004}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 32, "batch_timeout_ms": 6, "allow_mixed_lengths": True}},
    "gpu_timeseries_forecast_batchable": {"stage_type": "classifier", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 0.2, "service_time_model": {"mode": "linear", "base_ms": 14, "ms_per_input_unit": 0.006}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 32, "batch_timeout_ms": 8, "allow_mixed_lengths": True}},
    "gpu_tabular_anomaly_batchable": {"stage_type": "classifier", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 0.16, "service_time_model": {"mode": "linear", "base_ms": 10, "ms_per_input_unit": 0.004}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 48, "batch_timeout_ms": 6, "allow_mixed_lengths": True}},
    "cpu_event_publish": {"stage_type": "postprocess", "resource_profile": {"primary_device": "cpu", "cpu_cores_per_work_item": 0.18, "service_time_model": {"mode": "linear", "base_ms": 12, "ms_per_input_unit": 0.002}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 24, "batch_timeout_ms": 8, "allow_mixed_lengths": True}},
    "cpu_feature_enrichment": {"stage_type": "preprocess", "resource_profile": {"primary_device": "cpu", "cpu_cores_per_work_item": 0.3, "service_time_model": {"mode": "linear", "base_ms": 9, "ms_per_input_unit": 0.003}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 32, "batch_timeout_ms": 8, "allow_mixed_lengths": True}},
    "cpu_merge_results": {"stage_type": "aggregation", "resource_profile": {"primary_device": "cpu", "cpu_cores_per_work_item": 0.22, "service_time_model": {"mode": "linear", "base_ms": 7, "ms_per_input_unit": 0.002}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 32, "batch_timeout_ms": 6, "allow_mixed_lengths": True}},
    "cpu_transaction_enrichment": {"stage_type": "preprocess", "resource_profile": {"primary_device": "cpu", "cpu_cores_per_work_item": 0.28, "service_time_model": {"mode": "linear", "base_ms": 6, "ms_per_input_unit": 0.003}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 48, "batch_timeout_ms": 4, "allow_mixed_lengths": True}},
    "gpu_graph_risk_scoring": {"stage_type": "classifier", "resource_profile": {"primary_device": "gpu", "gpu_memory_gb_per_work_item": 0.24, "service_time_model": {"mode": "linear", "base_ms": 12, "ms_per_input_unit": 0.008}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 32, "batch_timeout_ms": 5, "allow_mixed_lengths": True}},
    "cpu_case_explainability": {"stage_type": "postprocess", "resource_profile": {"primary_device": "cpu", "cpu_cores_per_work_item": 0.25, "service_time_model": {"mode": "linear", "base_ms": 18, "ms_per_input_unit": 0.004}}, "batching_profile": {"batching_mode": "dynamic", "max_batch_size": 24, "batch_timeout_ms": 8, "allow_mixed_lengths": True}},
}

ROLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "llm_role": {"default_model_id": "Qwen/Qwen2.5-32B-Instruct", "override_model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "model_source": "huggingface", "language_sensitive": True},
    "orchestrator_llm_role": {"default_model_id": "Qwen/Qwen2.5-32B-Instruct", "override_model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "model_source": "huggingface", "language_sensitive": True},
    "deep_research_role": {"default_model_id": "Qwen/Qwen2.5-72B-Instruct", "override_model_id": "Qwen/Qwen2.5-32B-Instruct", "model_source": "huggingface", "language_sensitive": True},
    "shallow_research_role": {"default_model_id": "Qwen/Qwen2.5-14B-Instruct", "override_model_id": "mistralai/Mistral-7B-Instruct-v0.3", "model_source": "huggingface", "language_sensitive": True},
    "embedding_role": {"default_model_id": "nvidia/nv-embedqa-e5-v5", "override_model_id": "intfloat/multilingual-e5-large", "model_source": "huggingface", "language_sensitive": True},
    "reranker_role": {"default_model_id": "nvidia/llama-3.2-nv-rerankqa-1b-v2", "override_model_id": "BAAI/bge-reranker-v2-m3", "model_source": "huggingface", "language_sensitive": True},
    "vlm_role": {"default_model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "override_model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "model_source": "huggingface", "language_sensitive": True},
    "detector_role": {"default_model_id": "nvidia/visual-detector-reference", "override_model_id": "nvidia/visual-detector-reference", "model_source": "catalog_seed", "language_sensitive": False},
    "tracking_role": {"default_model_id": "nvidia/tracking-analytics-reference", "override_model_id": "nvidia/tracking-analytics-reference", "model_source": "catalog_seed", "language_sensitive": False},
    "reporting_role": {"default_model_id": "Qwen/Qwen2.5-14B-Instruct", "override_model_id": "mistralai/Mistral-7B-Instruct-v0.3", "model_source": "huggingface", "language_sensitive": True},
    "event_reasoning_role": {"default_model_id": "Qwen/Qwen2.5-14B-Instruct", "override_model_id": "mistralai/Mistral-7B-Instruct-v0.3", "model_source": "huggingface", "language_sensitive": True},
    "rag_role": {"default_model_id": "nvidia/rag-service-reference", "override_model_id": "nvidia/rag-service-reference", "model_source": "catalog_seed", "language_sensitive": True},
    "parse_role": {"default_model_id": "nvidia/parse-service-reference", "override_model_id": "nvidia/parse-service-reference", "model_source": "catalog_seed", "language_sensitive": True},
    "multicam_perception_role": {"default_model_id": "nvidia/sparse4d-reference", "override_model_id": "nvidia/sparse4d-reference", "model_source": "catalog_seed", "language_sensitive": False},
    "video_io_role": {"default_model_id": "VIOS", "override_model_id": "VIOS", "model_source": "service", "language_sensitive": False},
    "retrieval_role": {"default_model_id": "nvidia/nv-embedqa-e5-v5", "override_model_id": "intfloat/multilingual-e5-large", "model_source": "huggingface", "language_sensitive": True},
    "assistant_llm_role": {"default_model_id": "meta-llama/Llama-3.1-70B-Instruct", "override_model_id": "Qwen/Qwen2.5-32B-Instruct", "model_source": "huggingface", "language_sensitive": True},
    "script_llm_role": {"default_model_id": "meta-llama/Llama-3.1-70B-Instruct", "override_model_id": "Qwen/Qwen2.5-32B-Instruct", "model_source": "huggingface", "language_sensitive": True},
    "guardrail_role": {"default_model_id": "nvidia/nemoguard-reference", "override_model_id": "nvidia/nemoguard-reference", "model_source": "catalog_seed", "language_sensitive": True},
    "image_embedding_role": {"default_model_id": "nvidia/nvclip-reference", "override_model_id": "nvidia/nvclip-reference", "model_source": "catalog_seed", "language_sensitive": False},
    "reasoning_role": {"default_model_id": "Qwen/Qwen2.5-72B-Instruct", "override_model_id": "Qwen/Qwen2.5-32B-Instruct", "model_source": "huggingface", "language_sensitive": True},
    "report_llm_role": {"default_model_id": "Qwen/Qwen2.5-32B-Instruct", "override_model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "model_source": "huggingface", "language_sensitive": True},
    "biomedical_rag_role": {"default_model_id": "nvidia/biomedical-rag-reference", "override_model_id": "nvidia/biomedical-rag-reference", "model_source": "catalog_seed", "language_sensitive": True},
    "tool_agent_role": {"default_model_id": "nvidia/tool-router-reference", "override_model_id": "nvidia/tool-router-reference", "model_source": "catalog_seed", "language_sensitive": False},
    "tts_role": {"default_model_id": "nvidia/tts-reference", "override_model_id": "nvidia/tts-reference", "model_source": "catalog_seed", "language_sensitive": True},
    "asr_role": {"default_model_id": "nvidia/streaming-asr-reference", "override_model_id": "nvidia/streaming-asr-reference", "model_source": "catalog_seed", "language_sensitive": True},
    "diarization_role": {"default_model_id": "nvidia/diarization-reference", "override_model_id": "nvidia/diarization-reference", "model_source": "catalog_seed", "language_sensitive": False},
    "translation_role": {"default_model_id": "nvidia/translation-reference", "override_model_id": "nvidia/translation-reference", "model_source": "catalog_seed", "language_sensitive": True},
    "agent_assist_role": {"default_model_id": "Qwen/Qwen2.5-14B-Instruct", "override_model_id": "mistralai/Mistral-7B-Instruct-v0.3", "model_source": "huggingface", "language_sensitive": True},
    "route_preprocess_role": {"default_model_id": "nvidia/route-problem-builder", "override_model_id": "nvidia/route-problem-builder", "model_source": "catalog_seed", "language_sensitive": False},
    "optimization_solver_role": {"default_model_id": "nvidia/cuopt", "override_model_id": "nvidia/cuopt", "model_source": "catalog_seed", "language_sensitive": False},
    "dispatch_publish_role": {"default_model_id": "nvidia/dispatch-publisher", "override_model_id": "nvidia/dispatch-publisher", "model_source": "catalog_seed", "language_sensitive": False},
    "sensor_ingest_role": {"default_model_id": "nvidia/sensor-ingest-service", "override_model_id": "nvidia/sensor-ingest-service", "model_source": "service", "language_sensitive": False},
    "forecast_model_role": {"default_model_id": "nvidia/timeseries-forecast-reference", "override_model_id": "nvidia/timeseries-forecast-reference", "model_source": "catalog_seed", "language_sensitive": False},
    "anomaly_model_role": {"default_model_id": "nvidia/anomaly-detection-reference", "override_model_id": "nvidia/anomaly-detection-reference", "model_source": "catalog_seed", "language_sensitive": False},
    "alert_publish_role": {"default_model_id": "nvidia/alert-publisher", "override_model_id": "nvidia/alert-publisher", "model_source": "service", "language_sensitive": False},
    "candidate_retrieval_role": {"default_model_id": "nvidia/nv-embedqa-e5-v5", "override_model_id": "intfloat/multilingual-e5-large", "model_source": "huggingface", "language_sensitive": True},
    "ranking_model_role": {"default_model_id": "nvidia/llama-nemotron-rerank-1b-v2", "override_model_id": "nvidia/llama-3.2-nv-rerankqa-1b-v2", "model_source": "catalog_seed", "language_sensitive": True},
    "blending_rules_role": {"default_model_id": "nvidia/recommendation-blender", "override_model_id": "nvidia/recommendation-blender", "model_source": "service", "language_sensitive": False},
    "transaction_feature_role": {"default_model_id": "nvidia/transaction-feature-pipeline", "override_model_id": "nvidia/transaction-feature-pipeline", "model_source": "service", "language_sensitive": False},
    "graph_risk_role": {"default_model_id": "nvidia/financial-fraud-gnn", "override_model_id": "nvidia/financial-fraud-gnn", "model_source": "catalog_seed", "language_sensitive": False},
    "fraud_classifier_role": {"default_model_id": "nvidia/transaction-foundation-model", "override_model_id": "nvidia/transaction-foundation-model", "model_source": "catalog_seed", "language_sensitive": False},
    "case_explain_role": {"default_model_id": "nvidia/explainability-service", "override_model_id": "nvidia/explainability-service", "model_source": "service", "language_sensitive": False},
    "case_publish_role": {"default_model_id": "nvidia/case-management-publisher", "override_model_id": "nvidia/case-management-publisher", "model_source": "service", "language_sensitive": False},
}

FAMILY_DEFAULTS: dict[str, dict[str, Any]] = {
    "enterprise_rag": {"workload_class": "retrieval_augmented", "unit_type": "request", "modalities": ["text"], "arrival_mode": "bursty_poisson", "mean_arrival_rate_per_sec": 3.0, "size_distribution": {"distribution": "lognormal", "p50": 1800, "p95": 12000}, "complexity_distribution": {"distribution": "triangular", "mean": 1.0, "stddev": 0.25, "p95": 1.5}, "sla": {"latency_target_ms_p95": 2200, "throughput_target_per_sec": 2.5, "deadline_per_item_ms": 4000, "availability_target": 0.999, "max_drop_rate": 0.01, "max_queue_wait_ms": 800}},
    "agentic_research": {"workload_class": "multistage_agentic", "unit_type": "agent_task", "modalities": ["text"], "arrival_mode": "bursty_poisson", "mean_arrival_rate_per_sec": 0.6, "size_distribution": {"distribution": "lognormal", "p50": 6000, "p95": 48000}, "complexity_distribution": {"distribution": "triangular", "mean": 1.4, "stddev": 0.35, "p95": 2.2}, "sla": {"latency_target_ms_p95": 180000, "throughput_target_per_sec": 0.4, "deadline_per_item_ms": 300000, "availability_target": 0.995, "max_drop_rate": 0.03, "max_queue_wait_ms": 15000}},
    "multimodal_video_summary": {"workload_class": "vision_processing", "unit_type": "video_chunk", "modalities": ["video", "text"], "arrival_mode": "scheduled_batch", "mean_arrival_rate_per_sec": 0.08, "size_distribution": {"distribution": "lognormal", "p50": 4000, "p95": 20000}, "complexity_distribution": {"distribution": "triangular", "mean": 1.2, "stddev": 0.3, "p95": 1.8}, "sla": {"latency_target_ms_p95": 45000, "throughput_target_per_sec": 0.05, "deadline_per_item_ms": 90000, "availability_target": 0.99, "max_drop_rate": 0.05, "max_queue_wait_ms": 5000}},
    "streaming_video_detection": {"workload_class": "stream_processing", "unit_type": "image", "modalities": ["video"], "arrival_mode": "streaming_fixed_rate", "mean_arrival_rate_per_sec": 12.0, "size_distribution": {"distribution": "fixed", "value": 1.0}, "complexity_distribution": {"distribution": "triangular", "mean": 1.0, "stddev": 0.2, "p95": 1.4}, "sla": {"latency_target_ms_p95": 250, "throughput_target_per_sec": 10.0, "deadline_per_item_ms": 800, "availability_target": 0.999, "max_drop_rate": 0.01, "max_queue_wait_ms": 120}},
    "agentic_video_analytics": {"workload_class": "multistage_agentic", "unit_type": "event", "modalities": ["video", "text"], "arrival_mode": "bursty_poisson", "mean_arrival_rate_per_sec": 1.2, "size_distribution": {"distribution": "lognormal", "p50": 900, "p95": 8000}, "complexity_distribution": {"distribution": "triangular", "mean": 1.1, "stddev": 0.25, "p95": 1.7}, "sla": {"latency_target_ms_p95": 3500, "throughput_target_per_sec": 0.9, "deadline_per_item_ms": 7000, "availability_target": 0.999, "max_drop_rate": 0.02, "max_queue_wait_ms": 1000}},
    "multi_camera_3d_video_analytics": {"workload_class": "stream_processing", "unit_type": "image", "modalities": ["video"], "arrival_mode": "streaming_fixed_rate", "mean_arrival_rate_per_sec": 4.0, "size_distribution": {"distribution": "fixed", "value": 1.2}, "complexity_distribution": {"distribution": "triangular", "mean": 1.4, "stddev": 0.3, "p95": 2.0}, "sla": {"latency_target_ms_p95": 600, "throughput_target_per_sec": 3.5, "deadline_per_item_ms": 1500, "availability_target": 0.999, "max_drop_rate": 0.02, "max_queue_wait_ms": 180}},
    "document_to_audio": {"workload_class": "document_processing", "unit_type": "document", "modalities": ["text", "audio"], "arrival_mode": "scheduled_batch", "mean_arrival_rate_per_sec": 0.03, "size_distribution": {"distribution": "lognormal", "p50": 30, "p95": 200}, "complexity_distribution": {"distribution": "triangular", "mean": 1.3, "stddev": 0.35, "p95": 2.1}, "sla": {"latency_target_ms_p95": 180000, "throughput_target_per_sec": 0.02, "deadline_per_item_ms": 360000, "availability_target": 0.99, "max_drop_rate": 0.05, "max_queue_wait_ms": 12000}},
    "retail_multimodal_assistant": {"workload_class": "retrieval_augmented", "unit_type": "request", "modalities": ["text", "image"], "arrival_mode": "bursty_poisson", "mean_arrival_rate_per_sec": 2.0, "size_distribution": {"distribution": "lognormal", "p50": 2200, "p95": 16000}, "complexity_distribution": {"distribution": "triangular", "mean": 1.15, "stddev": 0.28, "p95": 1.8}, "sla": {"latency_target_ms_p95": 2500, "throughput_target_per_sec": 1.8, "deadline_per_item_ms": 5000, "availability_target": 0.999, "max_drop_rate": 0.01, "max_queue_wait_ms": 900}},
    "biomedical_agentic_research": {"workload_class": "multistage_agentic", "unit_type": "research_run", "modalities": ["text", "image", "document"], "arrival_mode": "bursty_poisson", "mean_arrival_rate_per_sec": 0.15, "size_distribution": {"distribution": "lognormal", "p50": 12000, "p95": 96000}, "complexity_distribution": {"distribution": "triangular", "mean": 1.8, "stddev": 0.4, "p95": 2.8}, "sla": {"latency_target_ms_p95": 300000, "throughput_target_per_sec": 0.08, "deadline_per_item_ms": 480000, "availability_target": 0.995, "max_drop_rate": 0.03, "max_queue_wait_ms": 20000}},
    "realtime_voice_call_processing": {"workload_class": "speech_processing", "unit_type": "audio_chunk", "modalities": ["audio", "text"], "arrival_mode": "streaming_fixed_rate", "mean_arrival_rate_per_sec": 60.0, "size_distribution": {"distribution": "fixed", "value": 320}, "complexity_distribution": {"distribution": "triangular", "mean": 1.0, "stddev": 0.18, "p95": 1.5}, "sla": {"latency_target_ms_p95": 800, "throughput_target_per_sec": 55.0, "deadline_per_item_ms": 1500, "availability_target": 0.9995, "max_drop_rate": 0.005, "max_queue_wait_ms": 120}},
    "route_optimization_decisioning": {"workload_class": "batch_processing", "unit_type": "optimization_job", "modalities": ["structured_data"], "arrival_mode": "bursty_poisson", "mean_arrival_rate_per_sec": 0.1, "size_distribution": {"distribution": "lognormal", "p50": 180, "p95": 600}, "complexity_distribution": {"distribution": "triangular", "mean": 1.3, "stddev": 0.35, "p95": 2.4}, "sla": {"latency_target_ms_p95": 4000, "throughput_target_per_sec": 0.08, "deadline_per_item_ms": 10000, "availability_target": 0.995, "max_drop_rate": 0.02, "max_queue_wait_ms": 1500}},
    "predictive_forecasting_analytics": {"workload_class": "stream_processing", "unit_type": "telemetry_window", "modalities": ["structured_data", "timeseries"], "arrival_mode": "streaming_fixed_rate", "mean_arrival_rate_per_sec": 8.0, "size_distribution": {"distribution": "lognormal", "p50": 24, "p95": 120}, "complexity_distribution": {"distribution": "triangular", "mean": 1.0, "stddev": 0.24, "p95": 1.7}, "sla": {"latency_target_ms_p95": 2500, "throughput_target_per_sec": 7.0, "deadline_per_item_ms": 6000, "availability_target": 0.999, "max_drop_rate": 0.01, "max_queue_wait_ms": 300}},
    "recommendation_ranking_service": {"workload_class": "retrieval_augmented", "unit_type": "request", "modalities": ["text", "structured_data"], "arrival_mode": "bursty_poisson", "mean_arrival_rate_per_sec": 8.0, "size_distribution": {"distribution": "lognormal", "p50": 120, "p95": 420}, "complexity_distribution": {"distribution": "triangular", "mean": 1.0, "stddev": 0.22, "p95": 1.6}, "sla": {"latency_target_ms_p95": 1200, "throughput_target_per_sec": 7.0, "deadline_per_item_ms": 2500, "availability_target": 0.999, "max_drop_rate": 0.01, "max_queue_wait_ms": 250}},
    "fraud_anomaly_detection": {"workload_class": "stream_processing", "unit_type": "transaction_event", "modalities": ["structured_data"], "arrival_mode": "streaming_fixed_rate", "mean_arrival_rate_per_sec": 150.0, "size_distribution": {"distribution": "lognormal", "p50": 6, "p95": 24}, "complexity_distribution": {"distribution": "triangular", "mean": 1.2, "stddev": 0.26, "p95": 1.9}, "sla": {"latency_target_ms_p95": 250, "throughput_target_per_sec": 135.0, "deadline_per_item_ms": 1200, "availability_target": 0.9995, "max_drop_rate": 0.005, "max_queue_wait_ms": 120}},
}

RUNTIME_PRESETS: dict[str, dict[str, Any]] = {
    "vllm": {"runtime_family": "vllm", "scheduler_model": {"mode": "vllm_continuous_batching", "chunked_prefill": True, "decode_priority": True, "max_num_batched_tokens": 32768, "max_num_seqs": 64, "prefix_cache_enabled": True, "prefix_cache_hit_rate": 0.25, "preemption_penalty_ms": 12, "speculative_decode_enabled": False}, "runtime_penalties": {"small_batch_efficiency_penalty": 1.08, "mixed_length_batch_penalty": 1.06, "queue_timeout_penalty": 1.03}},
    "tensorrt_llm": {"runtime_family": "tensorrt_llm", "scheduler_model": {"mode": "trtllm_inflight_batching", "inflight_batching": True, "chunked_context": True, "scheduler_policy": "MAX_UTILIZATION", "speculative_decode_enabled": False, "max_num_batched_tokens": 32768}, "runtime_penalties": {"small_batch_efficiency_penalty": 1.05, "mixed_length_batch_penalty": 1.04, "queue_timeout_penalty": 1.02}},
    "triton": {"runtime_family": "triton", "scheduler_model": {"mode": "triton_dynamic_batching", "max_queue_delay_microseconds": 8000, "preferred_batch_sizes": [1, 2, 4, 8, 16], "priority_levels": 2, "preserve_ordering": False}, "runtime_penalties": {"small_batch_efficiency_penalty": 1.12, "mixed_length_batch_penalty": 1.08, "queue_timeout_penalty": 1.05}},
    "speech_streaming": {"runtime_family": "speech_streaming", "scheduler_model": {"mode": "triton_dynamic_batching", "max_queue_delay_microseconds": 2000, "preferred_batch_sizes": [1, 2, 4, 8], "priority_levels": 3, "preserve_ordering": True}, "runtime_penalties": {"small_batch_efficiency_penalty": 1.04, "mixed_length_batch_penalty": 1.03, "queue_timeout_penalty": 1.01}},
    "solver_service": {"runtime_family": "solver_service", "scheduler_model": {"mode": "custom_solver_queue", "max_inflight_jobs": 8, "queue_policy": "fifo"}, "runtime_penalties": {"small_batch_efficiency_penalty": 1.0, "mixed_length_batch_penalty": 1.02, "queue_timeout_penalty": 1.02}},
}


PLANNING_PROFILES: dict[str, dict[str, Any]] = {
    "baseline": {
        "reserved_capacity_fraction": 0.08,
        "target_max_gpu_utilization": 0.82,
        "target_max_cpu_utilization": 0.80,
        "n_plus_one_enabled": False,
        "throughput_tuning_bias": 1.0,
    },
    "safe_24x7": {
        "reserved_capacity_fraction": 0.20,
        "target_max_gpu_utilization": 0.70,
        "target_max_cpu_utilization": 0.72,
        "n_plus_one_enabled": True,
        "throughput_tuning_bias": 0.9,
    },
    "high_throughput": {
        "reserved_capacity_fraction": 0.03,
        "target_max_gpu_utilization": 0.90,
        "target_max_cpu_utilization": 0.88,
        "n_plus_one_enabled": False,
        "throughput_tuning_bias": 1.08,
    },
}


def _canonical_execution_mode(deployment_seed: dict[str, Any]) -> str:
    mode = str(deployment_seed.get("mode") or deployment_seed.get("id") or "bare_metal")
    gpu_partitioning = str(deployment_seed.get("gpu_partitioning") or "none")
    if mode == "kubernetes_on_bare_metal":
        return "container_on_bare_metal"
    if mode == "virtual_machine_passthrough":
        return "vm_passthrough"
    if mode == "virtual_machine_vgpu":
        if gpu_partitioning == "time_sliced":
            return "vmware_vgpu_time_sliced"
        if gpu_partitioning == "mig_backed":
            return "vmware_vgpu_mig_backed"
    if mode == "bare_metal" and gpu_partitioning == "mig":
        return "bare_metal_mig"
    if mode == "bare_metal" and "container" in str(deployment_seed.get("id") or ""):
        return "container_on_bare_metal"
    return mode


def list_bindable_roles(blueprint_id: str) -> list[dict[str, Any]]:
    return copy.deepcopy(get_blueprint_bundle(blueprint_id)["blueprint"].get("pipeline_roles", []))


def build_workload_from_blueprint(*, blueprint_id: str, template_id: str, scenario_name: str, hardware_catalog_id: str, software_stack_id: str, deployment_profile_id: str, language_code: str = "hu", language_share: float = 0.8, model_overrides: dict[str, str] | None = None, mean_arrival_rate_per_sec: float | None = None, sla_latency_p95_ms: float | None = None, monte_carlo_trials: int = 40, time_horizon_sec: int = 300, planning_profile: str = "baseline", template_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    model_overrides = model_overrides or {}
    bundle = get_blueprint_bundle(blueprint_id)
    blueprint = bundle["blueprint"]
    template = next((t for t in bundle["templates"] if t.get("template_id") == template_id), None)
    if template is None:
        raise KeyError(f"Template not found in blueprint bundle: {template_id}")
    hardware = get_hardware_record(hardware_catalog_id)
    stack = get_software_stack_record(software_stack_id)
    deployment_seed = get_deployment_profile_record(deployment_profile_id)
    family_defaults = copy.deepcopy(FAMILY_DEFAULTS.get(template.get("workload_family"), FAMILY_DEFAULTS["enterprise_rag"]))
    template_inputs = _normalize_template_inputs(template_inputs)
    workload = {
        "workload_id": _slug(template_id),
        "workload_class": family_defaults["workload_class"],
        "input_profile": {
            "arrival_pattern": {"mode": family_defaults["arrival_mode"], "mean_arrival_rate_per_sec": float(mean_arrival_rate_per_sec or family_defaults["mean_arrival_rate_per_sec"]), "burst_multiplier_p95": 2.0, "business_cycle": {"enabled": True, "daily_peak_multiplier": 1.5, "weekly_peak_multiplier": 1.15}},
            "work_item": {"unit_type": family_defaults["unit_type"], "modalities": family_defaults["modalities"], "size_distribution": family_defaults["size_distribution"], "complexity_distribution": family_defaults["complexity_distribution"], "language_mix": [{"language": language_code, "share": float(language_share)}, {"language": "en", "share": float(max(0.0, 1.0 - float(language_share)))}]},
        },
        "pipeline": {"topology": "linear", "stages": [_build_stage(s, model_overrides, language_code) for s in template.get("stages", [])]},
        "sla_policy": copy.deepcopy(family_defaults["sla"]),
    }
    if template_inputs:
        workload = _apply_template_inputs_to_workload(workload, template, template_inputs)
    dominant_language, dominant_share = _dominant_language(workload)
    if sla_latency_p95_ms is not None:
        workload["sla_policy"]["latency_target_ms_p95"] = float(sla_latency_p95_ms)
    deployment_profile = _build_deployment_profile(hardware, software_stack_id, deployment_seed, planning_profile=planning_profile)
    runtime_profile = _build_runtime_profile(stack, template)
    if planning_profile == "high_throughput":
        runtime_profile["runtime_penalties"]["small_batch_efficiency_penalty"] *= 0.96
    elif planning_profile == "safe_24x7":
        runtime_profile["runtime_penalties"]["queue_timeout_penalty"] *= 1.02
    spec = {
        "schema_version": "1.0",
        "name": scenario_name,
        "description": f"Blueprint-based workload simulation: {blueprint.get('name')} / {template.get('template_id')}",
        "metadata": {"source_blueprint_id": blueprint_id, "source_template_id": template_id, "workload_family": template.get("workload_family"), "blueprint_name": blueprint.get("name"), "reference_perf_anchor": template.get("reference_perf_anchor"), "supports_model_substitution": blueprint.get("supports_model_substitution", False), "visualization_defaults": template.get("visualization_defaults", {}), "planning_profile": planning_profile, "template_inputs": template_inputs, "language_scaling": language_scaling_metadata(language_code=dominant_language, language_share=dominant_share), "requested_model_overrides": dict(model_overrides), "override_role_ids": sorted(model_overrides.keys())},
        "workload_definition": workload,
        "runtime_profile": runtime_profile,
        "deployment_profile": deployment_profile,
        "simulation_profile": {"monte_carlo_trials": int(monte_carlo_trials), "warmup_sec": 30, "time_horizon_sec": int(time_horizon_sec), "time_step_ms": 10, "random_variables": [{"name": "arrival_noise", "applies_to": "input", "distribution": "lognormal", "parameters": {"mean": 0, "sigma": 0.22}, "target_path": "workload_definition.input_profile.arrival_pattern.mean_arrival_rate_per_sec"}, {"name": "size_noise", "applies_to": "input", "distribution": "lognormal", "parameters": {"sigma": 0.30}, "target_path": "workload_definition.input_profile.work_item.size_distribution"}, {"name": "runtime_jitter", "applies_to": "deployment", "distribution": "normal", "parameters": {"mean": deployment_profile["overhead_model"]["latency_jitter_stddev_ms"], "stddev": max(1.0, deployment_profile["overhead_model"]["latency_jitter_stddev_ms"] * 0.25)}, "target_path": "deployment_profile.overhead_model.latency_jitter_stddev_ms"}]},
    }
    spec = harmonize_deployment_policy(apply_selected_model_bindings(spec))
    role_ids: list[str] = []
    for stage in spec.get("workload_definition", {}).get("pipeline", {}).get("stages", []) or []:
        role_id = str(stage.get("role_id") or (stage.get("model_binding") or {}).get("role_id") or "").strip()
        if role_id and role_id not in role_ids:
            role_ids.append(role_id)
    compatibility = assess_blueprint_model_compatibility(
        role_model_map=build_role_model_map(role_ids, model_overrides),
        hardware_catalog_id=hardware_catalog_id,
        software_stack_id=software_stack_id,
        deployment_profile_id=deployment_profile_id,
        workload_family=str(template.get("workload_family") or "").strip() or None,
        template_inputs=template_inputs,
    )
    spec.setdefault("metadata", {})["model_compatibility"] = compatibility
    errors, warnings = validate_workload_simulation_spec(spec, strict_catalog=True)
    warnings = list(warnings)
    warnings.extend(compatibility.get("notes") or [])
    if compatibility.get("overall_status") == "error":
        errors.append("Model compatibility failed: " + "; ".join(compatibility.get("notes") or compatibility.get("blocking_roles") or ["incompatible selections"]))
    if errors:
        raise ValueError("Blueprint workload validation failed: " + "; ".join(errors))
    if warnings:
        spec.setdefault("metadata", {})["validation_warnings"] = warnings
    return spec





def _language_generation_multiplier(language_code: str, language_share: float = 1.0) -> float:
    return generation_multiplier(language_code, language_share)


def _language_prompt_multiplier(language_code: str, language_share: float = 1.0) -> float:
    return prompt_multiplier(language_code, language_share)


def _language_context_multiplier(language_code: str, language_share: float = 1.0) -> float:
    return context_multiplier(language_code, language_share)


def _dominant_language(workload: dict[str, Any]) -> tuple[str, float]:
    mix = (((workload or {}).get('input_profile') or {}).get('work_item') or {}).get('language_mix') or []
    return dominant_language(mix)


def _normalize_template_inputs(template_inputs: dict[str, Any] | None) -> dict[str, Any]:
    return {str(k): v for k, v in (template_inputs or {}).items()}


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "igen"}


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _remove_stage_by_id(stages: list[dict[str, Any]], stage_id: str) -> list[dict[str, Any]]:
    return [stage for stage in stages if str(stage.get("stage_id") or "") != stage_id]


def _apply_template_inputs_to_workload(workload: dict[str, Any], template: dict[str, Any], template_inputs: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(workload)
    family = str(template.get("workload_family") or "")
    arrival = normalized.setdefault("input_profile", {}).setdefault("arrival_pattern", {})
    work_item = normalized.setdefault("input_profile", {}).setdefault("work_item", {})
    stages = normalized.setdefault("pipeline", {}).setdefault("stages", [])
    sla = normalized.setdefault("sla_policy", {})
    dominant_language, dominant_share = _dominant_language(normalized)
    prompt_mult = _language_prompt_multiplier(dominant_language, dominant_share)
    gen_mult = _language_generation_multiplier(dominant_language, dominant_share)

    if family == "realtime_voice_call_processing":
        concurrent_calls = max(1.0, _as_float(template_inputs.get("concurrent_calls"), 24.0))
        chunk_ms = max(80.0, _as_float(template_inputs.get("audio_chunk_ms"), 320.0))
        diarization_enabled = _as_bool(template_inputs.get("diarization_enabled"), True)
        translation_enabled = _as_bool(template_inputs.get("live_translation_enabled"), False)
        agent_assist_enabled = _as_bool(template_inputs.get("agent_assist_enabled"), True)
        tts_enabled = _as_bool(template_inputs.get("tts_response_enabled"), False)
        mean_rate = concurrent_calls * (1000.0 / chunk_ms)
        arrival["mode"] = "streaming_fixed_rate"
        arrival["mean_arrival_rate_per_sec"] = round(mean_rate, 4)
        arrival["burst_multiplier_p95"] = 1.15
        work_item["unit_type"] = "audio_chunk"
        work_item["modalities"] = ["audio", "text"]
        work_item["size_distribution"] = {"distribution": "fixed", "value": round(chunk_ms, 2)}
        complexity = 0.95 + (0.10 if diarization_enabled else 0.0) + (0.18 if translation_enabled else 0.0) + (0.22 if agent_assist_enabled else 0.0) + (0.12 if tts_enabled else 0.0)
        work_item["complexity_distribution"] = {"distribution": "triangular", "mean": round(complexity, 3), "stddev": 0.18, "p95": round(complexity * 1.35, 3)}
        if not diarization_enabled:
            stages = _remove_stage_by_id(stages, "speaker_diarization")
        if not translation_enabled:
            stages = _remove_stage_by_id(stages, "optional_live_translation")
        if not agent_assist_enabled:
            stages = _remove_stage_by_id(stages, "optional_agent_assist")
        if not tts_enabled:
            stages = _remove_stage_by_id(stages, "optional_tts_response")
        normalized["pipeline"]["stages"] = stages
    elif family == "route_optimization_decisioning":
        requests_per_min = max(0.1, _as_float(template_inputs.get("solve_requests_per_minute"), 6.0))
        avg_stops = max(10.0, _as_float(template_inputs.get("avg_stops_per_job"), 180.0))
        avg_vehicles = max(1.0, _as_float(template_inputs.get("avg_vehicle_count"), 48.0))
        constraint_count = max(0.0, _as_float(template_inputs.get("constraint_count"), 40.0))
        dynamic_replans = _as_bool(template_inputs.get("dynamic_replans_enabled"), True)
        eta_publish_enabled = _as_bool(template_inputs.get("eta_publish_enabled"), True)
        solve_sla_ms = max(500.0, _as_float(template_inputs.get("solve_sla_ms"), 4000.0))
        arrival["mode"] = "bursty_poisson"
        arrival["mean_arrival_rate_per_sec"] = round(requests_per_min / 60.0, 4)
        arrival["burst_multiplier_p95"] = 2.4 if dynamic_replans else 1.7
        work_item["unit_type"] = "optimization_job"
        work_item["modalities"] = ["structured_data"]
        work_item["size_distribution"] = {"distribution": "lognormal", "p50": round(avg_stops, 2), "p95": round(avg_stops * (2.1 if dynamic_replans else 1.7), 2)}
        mean_complexity = min(4.5, 0.65 + (avg_stops / 320.0) + (avg_vehicles / 220.0) + (constraint_count / 140.0) + (0.35 if dynamic_replans else 0.0))
        work_item["complexity_distribution"] = {"distribution": "triangular", "mean": round(mean_complexity, 3), "stddev": 0.35, "p95": round(mean_complexity * 1.45, 3)}
        sla["latency_target_ms_p95"] = round(solve_sla_ms, 2)
        sla["deadline_per_item_ms"] = round(solve_sla_ms * 2.2, 2)
        if not eta_publish_enabled:
            stages = _remove_stage_by_id(stages, "optional_publish_dispatch")
        normalized["pipeline"]["stages"] = stages
    elif family == "predictive_forecasting_analytics":
        monitored_assets = max(1.0, _as_float(template_inputs.get("monitored_assets"), 250.0))
        metrics_per_asset = max(1.0, _as_float(template_inputs.get("metrics_per_asset"), 24.0))
        refresh_interval_sec = max(1.0, _as_float(template_inputs.get("refresh_interval_sec"), 30.0))
        lookback_window_min = max(5.0, _as_float(template_inputs.get("lookback_window_min"), 180.0))
        forecast_horizon_min = max(5.0, _as_float(template_inputs.get("forecast_horizon_min"), 60.0))
        anomaly_enabled = _as_bool(template_inputs.get("anomaly_detection_enabled"), True)
        alert_publish_enabled = _as_bool(template_inputs.get("alert_publish_enabled"), True)
        mean_rate = monitored_assets / refresh_interval_sec
        arrival["mode"] = "streaming_fixed_rate"
        arrival["mean_arrival_rate_per_sec"] = round(mean_rate, 4)
        arrival["burst_multiplier_p95"] = 1.25
        work_item["unit_type"] = "telemetry_window"
        work_item["modalities"] = ["structured_data", "timeseries"]
        work_item["size_distribution"] = {"distribution": "lognormal", "p50": round(metrics_per_asset * max(1.0, lookback_window_min / 30.0), 2), "p95": round(metrics_per_asset * max(2.0, lookback_window_min / 12.0), 2)}
        mean_complexity = min(4.0, 0.55 + (metrics_per_asset / 24.0) + (lookback_window_min / 240.0) + (forecast_horizon_min / 180.0) + (0.25 if anomaly_enabled else 0.0))
        work_item["complexity_distribution"] = {"distribution": "triangular", "mean": round(mean_complexity, 3), "stddev": 0.28, "p95": round(mean_complexity * 1.4, 3)}
        sla["latency_target_ms_p95"] = round(min(max(refresh_interval_sec * 400.0, 1200.0), 12000.0), 2)
        sla["deadline_per_item_ms"] = round(sla["latency_target_ms_p95"] * 2.0, 2)
        if not anomaly_enabled:
            stages = _remove_stage_by_id(stages, "optional_detect_anomalies")
        if not alert_publish_enabled:
            stages = _remove_stage_by_id(stages, "optional_publish_alerts")
        normalized["pipeline"]["stages"] = stages
    elif family == "recommendation_ranking_service":
        requests_per_sec = max(0.1, _as_float(template_inputs.get("recommendation_requests_per_sec"), 8.0))
        candidate_pool_size = max(10.0, _as_float(template_inputs.get("candidate_pool_size"), 120.0))
        catalog_filter_count = max(0.0, _as_float(template_inputs.get("catalog_filter_count"), 6.0))
        reranking_enabled = _as_bool(template_inputs.get("reranking_enabled"), True)
        personalization_enabled = _as_bool(template_inputs.get("personalization_enabled"), True)
        image_grounding_enabled = _as_bool(template_inputs.get("image_grounding_enabled"), False)
        arrival["mode"] = "bursty_poisson"
        arrival["mean_arrival_rate_per_sec"] = round(requests_per_sec, 4)
        arrival["burst_multiplier_p95"] = 2.0 if personalization_enabled else 1.7
        work_item["unit_type"] = "request"
        work_item["modalities"] = ["text", "structured_data"] + (["image"] if image_grounding_enabled else [])
        work_item["size_distribution"] = {"distribution": "lognormal", "p50": round(candidate_pool_size * prompt_mult, 2), "p95": round(candidate_pool_size * (3.0 if reranking_enabled else 2.0) * max(1.0, prompt_mult), 2)}
        mean_complexity = min(4.0, 0.45 + (candidate_pool_size / 160.0) + (catalog_filter_count / 18.0) + (0.4 if reranking_enabled else 0.0) + (0.16 if personalization_enabled else 0.0) + (0.22 if image_grounding_enabled else 0.0))
        work_item["complexity_distribution"] = {"distribution": "triangular", "mean": round(mean_complexity, 3), "stddev": 0.24, "p95": round(mean_complexity * 1.35, 3)}
        base_sla = (850.0 + (180.0 if reranking_enabled else 0.0) + (140.0 if image_grounding_enabled else 0.0)) * max(1.0, gen_mult)
        sla["latency_target_ms_p95"] = round(base_sla, 2)
        sla["deadline_per_item_ms"] = round(base_sla * 2.2, 2)
        if not personalization_enabled:
            stages = _remove_stage_by_id(stages, "optional_personalization_features")
        if not reranking_enabled:
            stages = _remove_stage_by_id(stages, "rerank_candidates")
        if not image_grounding_enabled:
            stages = _remove_stage_by_id(stages, "optional_image_grounding")
        normalized["pipeline"]["stages"] = stages
    elif family == "fraud_anomaly_detection":
        events_per_sec = max(1.0, _as_float(template_inputs.get("transaction_events_per_sec"), 150.0))
        avg_entities = max(1.0, _as_float(template_inputs.get("avg_entities_per_case"), 6.0))
        false_positive_sensitivity = max(0.1, _as_float(template_inputs.get("false_positive_sensitivity"), 0.7))
        graph_features_enabled = _as_bool(template_inputs.get("graph_features_enabled"), True)
        realtime_blocking_enabled = _as_bool(template_inputs.get("realtime_blocking_enabled"), True)
        explainability_enabled = _as_bool(template_inputs.get("explainability_enabled"), True)
        arrival["mode"] = "streaming_fixed_rate"
        arrival["mean_arrival_rate_per_sec"] = round(events_per_sec, 4)
        arrival["burst_multiplier_p95"] = 1.45 if realtime_blocking_enabled else 1.8
        work_item["unit_type"] = "transaction_event"
        work_item["modalities"] = ["structured_data"]
        work_item["size_distribution"] = {"distribution": "lognormal", "p50": round(avg_entities, 2), "p95": round(avg_entities * (3.0 if graph_features_enabled else 2.0), 2)}
        mean_complexity = min(4.0, 0.7 + (avg_entities / 10.0) + (false_positive_sensitivity * 0.45) + (0.28 if graph_features_enabled else 0.0) + (0.18 if explainability_enabled else 0.0))
        work_item["complexity_distribution"] = {"distribution": "triangular", "mean": round(mean_complexity, 3), "stddev": 0.25, "p95": round(mean_complexity * 1.38, 3)}
        target_sla = 250.0 if realtime_blocking_enabled else 900.0
        sla["latency_target_ms_p95"] = round(target_sla, 2)
        sla["deadline_per_item_ms"] = round(target_sla * 3.0, 2)
        sla["max_queue_wait_ms"] = 80 if realtime_blocking_enabled else 220
        if not graph_features_enabled:
            stages = _remove_stage_by_id(stages, "optional_graph_risk")
        if not explainability_enabled:
            stages = _remove_stage_by_id(stages, "optional_case_explainability")
        normalized["pipeline"]["stages"] = stages
    return normalized

def build_blueprint_workload_yaml(**kwargs: Any) -> str:
    return yaml.safe_dump(build_workload_from_blueprint(**kwargs), sort_keys=False, allow_unicode=True)


def _build_runtime_profile(stack: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    framework = str(stack.get("framework") or "").lower()
    family = str(stack.get("runtime_family") or stack.get("framework") or "vllm").lower()
    hint = str(template.get("runtime_preset_hint") or "").strip().lower()
    if hint in RUNTIME_PRESETS:
        preset = copy.deepcopy(RUNTIME_PRESETS[hint])
    elif "cuopt" in framework or "cuopt" in family or "solver" in family:
        preset = copy.deepcopy(RUNTIME_PRESETS["solver_service"])
    elif "riva" in framework or "speech" in framework or "riva" in family:
        preset = copy.deepcopy(RUNTIME_PRESETS["speech_streaming"])
    elif "trt" in family or "tensorrt" in family:
        preset = copy.deepcopy(RUNTIME_PRESETS["tensorrt_llm"])
    elif "triton" in family:
        preset = copy.deepcopy(RUNTIME_PRESETS["triton"])
    else:
        preset = copy.deepcopy(RUNTIME_PRESETS["vllm"])
    batching_mode = str(template.get("batching_mode", ""))
    if "interactive" in batching_mode:
        preset["scheduler_model"]["max_num_batched_tokens"] = min(16384, int(preset["scheduler_model"].get("max_num_batched_tokens", 16384)))
    if "asynchronous" in batching_mode:
        preset["runtime_penalties"]["queue_timeout_penalty"] = 1.06
    if "streaming" in batching_mode and preset["scheduler_model"].get("mode") == "triton_dynamic_batching":
        preset["scheduler_model"]["max_queue_delay_microseconds"] = min(2000, int(preset["scheduler_model"].get("max_queue_delay_microseconds", 2000)))
    preset["runtime_id"] = f"{preset['runtime_family']}_{_slug(template.get('template_id', 'template'))}"
    return preset


def _build_deployment_profile(hardware: dict[str, Any], software_stack_id: str, deployment_seed: dict[str, Any], *, planning_profile: str = "baseline") -> dict[str, Any]:
    execution_mode = _canonical_execution_mode(deployment_seed)
    throughput_hint = float(deployment_seed.get("throughput_adjustment_hint") or 1.0)
    is_virtualized = bool(deployment_seed.get("is_virtualized", False))
    gpu_partitioning = str(deployment_seed.get("gpu_partitioning") or "none")
    latency_jitter = 4.0 + (8.0 if is_virtualized else 0.0) + (10.0 if "time" in gpu_partitioning else 4.0 if "mig" in gpu_partitioning else 0.0)
    throughput_penalty = max(0.5, 1.0 / throughput_hint) if throughput_hint > 0 else 1.0
    if "time" in gpu_partitioning:
        throughput_penalty *= 1.18
    elif "mig" in gpu_partitioning:
        throughput_penalty *= 1.08
    planning = copy.deepcopy(PLANNING_PROFILES.get(planning_profile, PLANNING_PROFILES["baseline"]))

    partition_mode = {
        "none": "none",
        "mig": "mig",
        "time_sliced": "time_sliced_vgpu",
        "mig_backed": "mig_backed_vgpu",
        "dedicated_full_gpu": "none",
    }.get(gpu_partitioning, "none")

    deployment_profile = {
        "deployment_id": deployment_seed.get("id", execution_mode),
        "execution_mode": execution_mode,
        "hardware_binding": {
            "hardware_catalog_id": hardware["id"],
            "gpu_count": int(hardware.get("gpu_count", 1)),
            "tp_size": max(1, min(int(hardware.get("gpu_count", 1)), max((hardware.get("recommended_tp_values") or [1])))),
            "software_stack_id": software_stack_id,
        },
        "partitioning": {
            "mode": partition_mode,
            "mig_profile": deployment_seed.get("default_mig_profile") if partition_mode in {"mig", "mig_backed_vgpu"} else None,
            "virtual_gpus_per_device": 1 if partition_mode in {"time_sliced_vgpu", "mig_backed_vgpu"} else None,
            "isolation_level": "hard" if partition_mode in {"mig", "mig_backed_vgpu"} or not is_virtualized else "soft",
        },
        "overhead_model": {
            "static_latency_overhead_ms": 4.0 if is_virtualized else 0.0,
            "throughput_penalty_factor": throughput_penalty / max(0.5, planning.get("throughput_tuning_bias", 1.0)),
            "latency_jitter_stddev_ms": latency_jitter,
        },
        "availability": {
            "reserved_capacity_fraction": planning.get("reserved_capacity_fraction", 0.08),
            "reserved_capacity_percent": round(planning.get("reserved_capacity_fraction", 0.08) * 100, 4),
            "n_plus_one_enabled": bool(planning.get("n_plus_one_enabled", False)),
            "max_safe_gpu_utilization": planning.get("target_max_gpu_utilization", 0.82),
        },
        "operational_policy": {
            "planning_profile": planning_profile,
            "reserved_capacity_fraction": planning.get("reserved_capacity_fraction", 0.08),
            "target_max_gpu_utilization": planning.get("target_max_gpu_utilization", 0.82),
            "target_max_cpu_utilization": planning.get("target_max_cpu_utilization", 0.80),
            "n_plus_one_enabled": bool(planning.get("n_plus_one_enabled", False)),
        },
    }
    return deployment_profile


def _build_stage(stage_seed: dict[str, Any], model_overrides: dict[str, str], language_code: str) -> dict[str, Any]:
    preset = copy.deepcopy(RESOURCE_PROFILE_PRESETS.get(stage_seed.get("resource_profile"), RESOURCE_PROFILE_PRESETS["gpu_llm_prefill_decode"]))
    role_id = str(stage_seed.get("role_id") or stage_seed.get("stage_id"))
    role_defaults = resolve_role_defaults(role_id)
    stage = {"stage_id": stage_seed.get("stage_id"), "display_name": str(stage_seed.get("stage_id", "stage")).replace("_", " ").title(), "stage_type": preset["stage_type"], "language_sensitive": bool(role_defaults.get("language_sensitive", False)), "model_binding": {"role_id": role_id, "default_model_id": role_defaults.get("default_model_id"), "override_model_id": model_overrides.get(role_id) or role_defaults.get("override_model_id"), "model_source": role_defaults.get("model_source", "huggingface")}, "resource_profile": preset["resource_profile"], "batching_profile": preset["batching_profile"]}
    if stage["stage_type"] in {"generator", "decoder"}:
        stage["failure_policy"] = {"retry_on_failure": False, "drop_on_overload": True}
    if stage_seed.get("stage_id", "").startswith("optional_"):
        stage["optional"] = True
    if stage["language_sensitive"] and language_code.lower().startswith("hu"):
        stage["model_binding"]["requires_manual_validation"] = True
    return stage


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
