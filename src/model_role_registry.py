from __future__ import annotations

import copy
from typing import Any

ROLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "llm_role": {"default_model_id": "Qwen/Qwen2.5-32B-Instruct", "override_model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "model_source": "huggingface", "language_sensitive": True, "role_type": "llm_generation", "notes": "General on-prem LLM generation role."},
    "orchestrator_llm_role": {"default_model_id": "Qwen/Qwen2.5-32B-Instruct", "override_model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "model_source": "huggingface", "language_sensitive": True, "role_type": "llm_orchestration"},
    "deep_research_role": {"default_model_id": "Qwen/Qwen2.5-72B-Instruct", "override_model_id": "Qwen/Qwen2.5-32B-Instruct", "model_source": "huggingface", "language_sensitive": True, "role_type": "llm_reasoning"},
    "shallow_research_role": {"default_model_id": "Qwen/Qwen2.5-14B-Instruct", "override_model_id": "mistralai/Mistral-7B-Instruct-v0.3", "model_source": "huggingface", "language_sensitive": True, "role_type": "llm_research"},
    "embedding_role": {"default_model_id": "nvidia/nv-embedqa-e5-v5", "override_model_id": "intfloat/multilingual-e5-large", "model_source": "huggingface", "language_sensitive": True, "role_type": "embedding"},
    "reranker_role": {"default_model_id": "nvidia/llama-3.2-nv-rerankqa-1b-v2", "override_model_id": "BAAI/bge-reranker-v2-m3", "model_source": "huggingface", "language_sensitive": True, "role_type": "reranker"},
    "vlm_role": {"default_model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "override_model_id": "OpenGVLab/InternVL2_5-8B", "model_source": "huggingface", "language_sensitive": True, "role_type": "vlm"},
    "detector_role": {"default_model_id": "nvidia/visual-detector-reference", "override_model_id": "nvidia/visual-detector-reference", "model_source": "catalog_seed", "language_sensitive": False, "role_type": "vision_detector"},
    "tracking_role": {"default_model_id": "nvidia/tracking-analytics-reference", "override_model_id": "nvidia/tracking-analytics-reference", "model_source": "catalog_seed", "language_sensitive": False, "role_type": "tracking"},
    "reporting_role": {"default_model_id": "Qwen/Qwen2.5-14B-Instruct", "override_model_id": "mistralai/Mistral-7B-Instruct-v0.3", "model_source": "huggingface", "language_sensitive": True, "role_type": "llm_reporting"},
    "event_reasoning_role": {"default_model_id": "Qwen/Qwen2.5-14B-Instruct", "override_model_id": "mistralai/Mistral-7B-Instruct-v0.3", "model_source": "huggingface", "language_sensitive": True, "role_type": "llm_reasoning"},
    "rag_role": {"default_model_id": "nvidia/rag-service-reference", "override_model_id": "nvidia/rag-service-reference", "model_source": "catalog_seed", "language_sensitive": True, "role_type": "rag_service"},
    "parse_role": {"default_model_id": "nvidia/parse-service-reference", "override_model_id": "nvidia/nv-ingest-doc-splitter", "model_source": "catalog_seed", "language_sensitive": True, "role_type": "document_parse"},
    "multicam_perception_role": {"default_model_id": "nvidia/sparse4d-reference", "override_model_id": "nvidia/sparse4d-reference", "model_source": "catalog_seed", "language_sensitive": False, "role_type": "perception"},
    "video_io_role": {"default_model_id": "VIOS", "override_model_id": "VIOS", "model_source": "service", "language_sensitive": False, "role_type": "video_io"},
    "retrieval_role": {"default_model_id": "nvidia/nv-embedqa-e5-v5", "override_model_id": "intfloat/multilingual-e5-large", "model_source": "huggingface", "language_sensitive": True, "role_type": "retrieval"},
    "assistant_llm_role": {"default_model_id": "meta-llama/Llama-3.1-70B-Instruct", "override_model_id": "Qwen/Qwen2.5-32B-Instruct", "model_source": "huggingface", "language_sensitive": True, "role_type": "llm_assistant"},
    "script_llm_role": {"default_model_id": "meta-llama/Llama-3.1-70B-Instruct", "override_model_id": "Qwen/Qwen2.5-32B-Instruct", "model_source": "huggingface", "language_sensitive": True, "role_type": "llm_script"},
    "guardrail_role": {"default_model_id": "nvidia/nemoguard-reference", "override_model_id": "meta-llama/Llama-Guard-3-8B", "model_source": "catalog_seed", "language_sensitive": True, "role_type": "guardrail"},
    "image_embedding_role": {"default_model_id": "nvidia/nvclip-reference", "override_model_id": "openai/clip-vit-large-patch14", "model_source": "catalog_seed", "language_sensitive": False, "role_type": "image_embedding"},
    "reasoning_role": {"default_model_id": "Qwen/Qwen2.5-72B-Instruct", "override_model_id": "Qwen/Qwen2.5-32B-Instruct", "model_source": "huggingface", "language_sensitive": True, "role_type": "llm_reasoning"},
    "report_llm_role": {"default_model_id": "Qwen/Qwen2.5-32B-Instruct", "override_model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "model_source": "huggingface", "language_sensitive": True, "role_type": "llm_reporting"},
    "biomedical_rag_role": {"default_model_id": "nvidia/biomedical-rag-reference", "override_model_id": "nvidia/biomedical-rag-reference", "model_source": "catalog_seed", "language_sensitive": True, "role_type": "rag_service"},
    "tool_agent_role": {"default_model_id": "nvidia/tool-router-reference", "override_model_id": "nvidia/tool-router-reference", "model_source": "catalog_seed", "language_sensitive": False, "role_type": "tooling"},
    "tts_role": {"default_model_id": "nvidia/tts-reference", "override_model_id": "nvidia/tts-reference", "model_source": "catalog_seed", "language_sensitive": True, "role_type": "tts"},
    "asr_role": {"default_model_id": "nvidia/streaming-asr-reference", "override_model_id": "nvidia/streaming-asr-reference", "model_source": "catalog_seed", "language_sensitive": True, "role_type": "asr"},
    "diarization_role": {"default_model_id": "nvidia/diarization-reference", "override_model_id": "nvidia/diarization-reference", "model_source": "catalog_seed", "language_sensitive": False, "role_type": "diarization"},
    "translation_role": {"default_model_id": "nvidia/translation-reference", "override_model_id": "nvidia/translation-reference", "model_source": "catalog_seed", "language_sensitive": True, "role_type": "translation"},
    "agent_assist_role": {"default_model_id": "Qwen/Qwen2.5-14B-Instruct", "override_model_id": "mistralai/Mistral-7B-Instruct-v0.3", "model_source": "huggingface", "language_sensitive": True, "role_type": "llm_assistant"},
    "route_preprocess_role": {"default_model_id": "nvidia/route-problem-builder", "override_model_id": "nvidia/route-problem-builder", "model_source": "catalog_seed", "language_sensitive": False, "role_type": "optimizer_preprocess"},
    "optimization_solver_role": {"default_model_id": "nvidia/cuopt", "override_model_id": "nvidia/cuopt", "model_source": "catalog_seed", "language_sensitive": False, "role_type": "solver"},
    "dispatch_publish_role": {"default_model_id": "nvidia/dispatch-publisher", "override_model_id": "nvidia/dispatch-publisher", "model_source": "catalog_seed", "language_sensitive": False, "role_type": "publisher"},
    "sensor_ingest_role": {"default_model_id": "nvidia/sensor-ingest-service", "override_model_id": "nvidia/sensor-ingest-service", "model_source": "service", "language_sensitive": False, "role_type": "ingest"},
    "forecast_model_role": {"default_model_id": "nvidia/timeseries-forecast-reference", "override_model_id": "nvidia/timeseries-forecast-reference", "model_source": "catalog_seed", "language_sensitive": False, "role_type": "forecast"},
    "anomaly_model_role": {"default_model_id": "nvidia/anomaly-detection-reference", "override_model_id": "nvidia/anomaly-detection-reference", "model_source": "catalog_seed", "language_sensitive": False, "role_type": "anomaly"},
    "alert_publish_role": {"default_model_id": "nvidia/alert-publisher", "override_model_id": "nvidia/alert-publisher", "model_source": "service", "language_sensitive": False, "role_type": "publisher"},
    "candidate_retrieval_role": {"default_model_id": "nvidia/nv-embedqa-e5-v5", "override_model_id": "intfloat/multilingual-e5-large", "model_source": "huggingface", "language_sensitive": True, "role_type": "retrieval"},
    "ranking_model_role": {"default_model_id": "nvidia/llama-nemotron-rerank-1b-v2", "override_model_id": "nvidia/llama-3.2-nv-rerankqa-1b-v2", "model_source": "catalog_seed", "language_sensitive": True, "role_type": "reranker"},
    "blending_rules_role": {"default_model_id": "nvidia/recommendation-blender", "override_model_id": "nvidia/recommendation-blender", "model_source": "service", "language_sensitive": False, "role_type": "recommender"},
    "transaction_feature_role": {"default_model_id": "nvidia/transaction-feature-pipeline", "override_model_id": "nvidia/transaction-feature-pipeline", "model_source": "service", "language_sensitive": False, "role_type": "feature_pipeline"},
    "graph_risk_role": {"default_model_id": "nvidia/financial-fraud-gnn", "override_model_id": "nvidia/financial-fraud-gnn", "model_source": "catalog_seed", "language_sensitive": False, "role_type": "graph_model"},
    "fraud_classifier_role": {"default_model_id": "nvidia/transaction-foundation-model", "override_model_id": "nvidia/transaction-foundation-model", "model_source": "catalog_seed", "language_sensitive": False, "role_type": "classifier"},
    "case_explain_role": {"default_model_id": "nvidia/explainability-service", "override_model_id": "nvidia/explainability-service", "model_source": "service", "language_sensitive": False, "role_type": "explainer"},
    "case_publish_role": {"default_model_id": "nvidia/case-management-publisher", "override_model_id": "nvidia/case-management-publisher", "model_source": "service", "language_sensitive": False, "role_type": "publisher"},
    # advanced template specific roles
    "planner_role": {"default_model_id": "Qwen/Qwen2.5-14B-Instruct", "override_model_id": "mistralai/Mistral-7B-Instruct-v0.3", "model_source": "huggingface", "language_sensitive": True, "role_type": "llm_planner", "notes": "Agentic planner / workflow routing role."},
    "writer_role": {"default_model_id": "Qwen/Qwen2.5-32B-Instruct", "override_model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "model_source": "huggingface", "language_sensitive": True, "role_type": "llm_writer"},
    "review_role": {"default_model_id": "meta-llama/Llama-Guard-3-8B", "override_model_id": "nvidia/nemoguard-reference", "model_source": "huggingface", "language_sensitive": True, "role_type": "review_guardrail"},
    "tool_role": {"default_model_id": "nvidia/tool-router-reference", "override_model_id": "nvidia/tool-router-reference", "model_source": "service", "language_sensitive": False, "role_type": "tool_router"},
    "ocr_role": {"default_model_id": "microsoft/trocr-base-printed", "override_model_id": "nvidia/ocr-service-reference", "model_source": "huggingface", "language_sensitive": True, "role_type": "ocr"},
    "extract_role": {"default_model_id": "nvidia/nv-ingest-structured-extract", "override_model_id": "nvidia/llama-3.1-nemotron-extract", "model_source": "catalog_seed", "language_sensitive": True, "role_type": "document_extract"},
    "publish_role": {"default_model_id": "nvidia/result-publisher", "override_model_id": "nvidia/result-publisher", "model_source": "service", "language_sensitive": False, "role_type": "publisher"},
    "ingest_role": {"default_model_id": "nvidia/ingest-gateway", "override_model_id": "nvidia/ingest-gateway", "model_source": "service", "language_sensitive": False, "role_type": "ingest"},
    "preprocess_role": {"default_model_id": "nvidia/preprocess-service-reference", "override_model_id": "nvidia/preprocess-service-reference", "model_source": "service", "language_sensitive": False, "role_type": "preprocess"},
    "infer_role": {"default_model_id": "Qwen/Qwen2.5-14B-Instruct", "override_model_id": "mistralai/Mistral-7B-Instruct-v0.3", "model_source": "huggingface", "language_sensitive": True, "role_type": "inference"},
    "postprocess_role": {"default_model_id": "nvidia/postprocess-service-reference", "override_model_id": "nvidia/postprocess-service-reference", "model_source": "service", "language_sensitive": False, "role_type": "postprocess"},
}

ROLE_CANDIDATES: dict[str, list[dict[str, Any]]] = {
    "embedding_role": [
        {"id": "nvidia/nv-embedqa-e5-v5", "label": "NV-EmbedQA E5 v5", "source": "huggingface", "on_prem": True},
        {"id": "intfloat/multilingual-e5-large", "label": "Multilingual E5 Large", "source": "huggingface", "on_prem": True},
        {"id": "BAAI/bge-m3", "label": "BGE-M3", "source": "huggingface", "on_prem": True},
    ],
    "retrieval_role": [
        {"id": "nvidia/nv-embedqa-e5-v5", "label": "NV-EmbedQA E5 v5", "source": "huggingface", "on_prem": True},
        {"id": "intfloat/multilingual-e5-large", "label": "Multilingual E5 Large", "source": "huggingface", "on_prem": True},
        {"id": "BAAI/bge-m3", "label": "BGE-M3", "source": "huggingface", "on_prem": True},
    ],
    "reranker_role": [
        {"id": "nvidia/llama-3.2-nv-rerankqa-1b-v2", "label": "NV RerankQA 1B v2", "source": "huggingface", "on_prem": True},
        {"id": "BAAI/bge-reranker-v2-m3", "label": "BGE Reranker v2 m3", "source": "huggingface", "on_prem": True},
        {"id": "nvidia/llama-nemotron-rerank-1b-v2", "label": "Llama Nemotron Rerank 1B v2", "source": "huggingface", "on_prem": True},
    ],
    "llm_role": [
        {"id": "Qwen/Qwen2.5-32B-Instruct", "label": "Qwen2.5 32B Instruct", "source": "huggingface", "on_prem": True},
        {"id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "label": "Mistral Small 3.1 24B", "source": "huggingface", "on_prem": True},
        {"id": "meta-llama/Llama-3.1-70B-Instruct", "label": "Llama 3.1 70B Instruct", "source": "huggingface", "on_prem": True},
        {"id": "Qwen/Qwen2.5-14B-Instruct", "label": "Qwen2.5 14B Instruct", "source": "huggingface", "on_prem": True},
    ],
    "planner_role": [
        {"id": "Qwen/Qwen2.5-14B-Instruct", "label": "Qwen2.5 14B Planner", "source": "huggingface", "on_prem": True},
        {"id": "mistralai/Mistral-7B-Instruct-v0.3", "label": "Mistral 7B Planner", "source": "huggingface", "on_prem": True},
        {"id": "Qwen/Qwen2.5-32B-Instruct", "label": "Qwen2.5 32B Planner", "source": "huggingface", "on_prem": True},
    ],
    "reasoning_role": [
        {"id": "Qwen/Qwen2.5-72B-Instruct", "label": "Qwen2.5 72B Reasoning", "source": "huggingface", "on_prem": True},
        {"id": "Qwen/Qwen2.5-32B-Instruct", "label": "Qwen2.5 32B Reasoning", "source": "huggingface", "on_prem": True},
        {"id": "meta-llama/Llama-3.1-70B-Instruct", "label": "Llama 3.1 70B Reasoning", "source": "huggingface", "on_prem": True},
    ],
    "writer_role": [
        {"id": "Qwen/Qwen2.5-32B-Instruct", "label": "Qwen2.5 32B Writer", "source": "huggingface", "on_prem": True},
        {"id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "label": "Mistral Small 3.1 Writer", "source": "huggingface", "on_prem": True},
        {"id": "Qwen/Qwen2.5-14B-Instruct", "label": "Qwen2.5 14B Writer", "source": "huggingface", "on_prem": True},
    ],
    "infer_role": [
        {"id": "Qwen/Qwen2.5-14B-Instruct", "label": "Qwen2.5 14B Inference", "source": "huggingface", "on_prem": True},
        {"id": "mistralai/Mistral-7B-Instruct-v0.3", "label": "Mistral 7B Inference", "source": "huggingface", "on_prem": True},
        {"id": "Qwen/Qwen2.5-32B-Instruct", "label": "Qwen2.5 32B Inference", "source": "huggingface", "on_prem": True},
    ],
    "guardrail_role": [
        {"id": "nvidia/nemoguard-reference", "label": "NVIDIA NeMo Guard", "source": "catalog_seed", "on_prem": True},
        {"id": "meta-llama/Llama-Guard-3-8B", "label": "Llama Guard 3 8B", "source": "huggingface", "on_prem": True},
        {"id": "ProtectAI/deberta-v3-base-prompt-injection-v2", "label": "Prompt Injection Guard", "source": "huggingface", "on_prem": True},
    ],
    "review_role": [
        {"id": "meta-llama/Llama-Guard-3-8B", "label": "Llama Guard 3 8B", "source": "huggingface", "on_prem": True},
        {"id": "nvidia/nemoguard-reference", "label": "NVIDIA NeMo Guard", "source": "catalog_seed", "on_prem": True},
        {"id": "Qwen/Qwen2.5-14B-Instruct", "label": "Qwen2.5 14B Reviewer", "source": "huggingface", "on_prem": True},
    ],
    "vlm_role": [
        {"id": "Qwen/Qwen2.5-VL-7B-Instruct", "label": "Qwen2.5 VL 7B", "source": "huggingface", "on_prem": True},
        {"id": "OpenGVLab/InternVL2_5-8B", "label": "InternVL2.5 8B", "source": "huggingface", "on_prem": True},
        {"id": "llava-hf/llava-1.5-7b-hf", "label": "LLaVA 1.5 7B", "source": "huggingface", "on_prem": True},
    ],
    "parse_role": [
        {"id": "nvidia/parse-service-reference", "label": "NVIDIA Parse Service", "source": "catalog_seed", "on_prem": True},
        {"id": "nvidia/nv-ingest-doc-splitter", "label": "NVIDIA Doc Splitter", "source": "catalog_seed", "on_prem": True},
        {"id": "custom/onprem-doc-parser", "label": "Custom on-prem parser", "source": "custom", "on_prem": True},
    ],
    "ocr_role": [
        {"id": "microsoft/trocr-base-printed", "label": "TrOCR Base Printed", "source": "huggingface", "on_prem": True},
        {"id": "nvidia/ocr-service-reference", "label": "NVIDIA OCR Service", "source": "catalog_seed", "on_prem": True},
        {"id": "custom/onprem-ocr", "label": "Custom on-prem OCR", "source": "custom", "on_prem": True},
    ],
    "extract_role": [
        {"id": "nvidia/nv-ingest-structured-extract", "label": "NV Ingest Structured Extract", "source": "catalog_seed", "on_prem": True},
        {"id": "nvidia/llama-3.1-nemotron-extract", "label": "Nemotron Extract", "source": "catalog_seed", "on_prem": True},
        {"id": "custom/onprem-extractor", "label": "Custom on-prem extract", "source": "custom", "on_prem": True},
    ],
    "tool_role": [
        {"id": "nvidia/tool-router-reference", "label": "NVIDIA Tool Router", "source": "service", "on_prem": True},
        {"id": "custom/onprem-tool-router", "label": "Custom on-prem tool router", "source": "custom", "on_prem": True},
    ],
    "publish_role": [
        {"id": "nvidia/result-publisher", "label": "NVIDIA Result Publisher", "source": "service", "on_prem": True},
        {"id": "custom/onprem-publisher", "label": "Custom on-prem publisher", "source": "custom", "on_prem": True},
    ],
    "ingest_role": [
        {"id": "nvidia/ingest-gateway", "label": "NVIDIA Ingest Gateway", "source": "service", "on_prem": True},
        {"id": "custom/onprem-ingest", "label": "Custom on-prem ingest", "source": "custom", "on_prem": True},
    ],
    "preprocess_role": [
        {"id": "nvidia/preprocess-service-reference", "label": "NVIDIA Preprocess Service", "source": "service", "on_prem": True},
        {"id": "custom/onprem-preprocess", "label": "Custom on-prem preprocess", "source": "custom", "on_prem": True},
    ],
    "postprocess_role": [
        {"id": "nvidia/postprocess-service-reference", "label": "NVIDIA Postprocess Service", "source": "service", "on_prem": True},
        {"id": "custom/onprem-postprocess", "label": "Custom on-prem postprocess", "source": "custom", "on_prem": True},
    ],
    "rag_role": [
        {"id": "nvidia/rag-service-reference", "label": "NVIDIA RAG Service", "source": "catalog_seed", "on_prem": True},
        {"id": "custom/onprem-rag-service", "label": "Custom on-prem RAG service", "source": "custom", "on_prem": True},
    ],
}

ROLE_LABELS = {
    "llm_role": "LLM",
    "embedding_role": "Embedding",
    "reranker_role": "Reranker",
    "guardrail_role": "Guardrail",
    "planner_role": "Planner",
    "reasoning_role": "Reasoning",
    "writer_role": "Writer",
    "review_role": "Review",
    "tool_role": "Tool router",
    "parse_role": "Parser",
    "ocr_role": "OCR",
    "extract_role": "Extractor",
    "publish_role": "Publisher",
    "ingest_role": "Ingest",
    "preprocess_role": "Preprocess",
    "infer_role": "Inference",
    "postprocess_role": "Postprocess",
    "retrieval_role": "Retrieval",
    "vlm_role": "VLM",
    "rag_role": "RAG service",
}


def resolve_role_defaults(role_id: str) -> dict[str, Any]:
    defaults = ROLE_DEFAULTS.get(
        role_id,
        {
            "default_model_id": f"reference/{role_id}",
            "override_model_id": f"reference/{role_id}",
            "model_source": "catalog_seed",
            "language_sensitive": False,
            "role_type": "generic",
            "notes": "Generic on-prem reference role.",
        },
    )
    return copy.deepcopy(defaults)



def build_role_ui_record(role_id: str, *, replaceable: bool = True, notes: str | None = None) -> dict[str, Any]:
    defaults = resolve_role_defaults(role_id)
    default_id = str(defaults.get("default_model_id") or "")
    override_id = str(defaults.get("override_model_id") or default_id)
    candidates = []
    seen: set[str] = set()
    for candidate_id in [default_id, override_id]:
        if candidate_id and candidate_id not in seen:
            candidates.append({"id": candidate_id, "label": candidate_id, "source": defaults.get("model_source", "catalog_seed"), "on_prem": True})
            seen.add(candidate_id)
    for candidate in ROLE_CANDIDATES.get(role_id, []):
        cid = str(candidate.get("id") or "").strip()
        if not cid or cid in seen:
            continue
        candidates.append(copy.deepcopy(candidate))
        seen.add(cid)
    return {
        "role_id": role_id,
        "role_label": ROLE_LABELS.get(role_id, role_id.replace("_", " ").title()),
        "role_type": defaults.get("role_type", "generic"),
        "replaceable": replaceable,
        "language_sensitive": bool(defaults.get("language_sensitive", False)),
        "default_model_reference": default_id,
        "suggested_override_model": override_id,
        "model_source": defaults.get("model_source", "catalog_seed"),
        "on_prem_primary": True,
        "candidate_models": candidates,
        "notes": notes or defaults.get("notes") or "Primary on-prem role; you can also use your own self-hosted model or service reference.",
    }
