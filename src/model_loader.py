"""Model loading helpers for built-in, inline, and Hugging Face sourced LLM configs."""

from __future__ import annotations

import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import ModelConfig

try:  # pragma: no cover - import presence depends on environment
    from huggingface_hub import HfApi, hf_hub_download
except Exception:  # pragma: no cover
    HfApi = None
    hf_hub_download = None


def model_config_from_mapping(data: dict[str, Any], default_name: str = "custom-model") -> ModelConfig:
    """Create a ModelConfig from a plain mapping.

    Missing fields fall back to neutral values when safe, or are derived from
    related architecture fields.
    """
    name = str(data.get("name") or data.get("model_id") or default_name)
    hidden_size = int(data.get("hidden_size", 4096))
    num_attention_heads = int(data.get("num_attention_heads", 32))
    head_dim = data.get("head_dim")
    if head_dim is None:
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Cannot derive head_dim: hidden_size={hidden_size} is not divisible by "
                f"num_attention_heads={num_attention_heads}."
            )
        head_dim = hidden_size // num_attention_heads

    num_experts = _first_present_int(
        data,
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
        "moe_num_experts",
        "n_experts",
    ) or 0
    num_experts_per_tok = _first_present_int(
        data,
        "num_experts_per_tok",
        "num_experts_per_token",
        "experts_per_token",
        "router_topk",
        "top_k_experts",
    ) or 0
    moe_intermediate_size = _first_present_int(
        data,
        "moe_intermediate_size",
        "expert_intermediate_size",
        "ffn_config.moe_intermediate_size",
    ) or 0

    architecture_hints = " ".join(str(x) for x in (data.get("architectures") or []))
    model_type = str(data.get("model_type") or "")
    inferred_is_moe = (
        num_experts > 0
        or num_experts_per_tok > 0
        or moe_intermediate_size > 0
        or "mixtral" in model_type.lower()
        or "deepseek" in model_type.lower()
        or "moe" in architecture_hints.lower()
    )
    is_moe = bool(data.get("is_moe", inferred_is_moe))

    total_params = data.get("total_params_billions")
    active_params = data.get("active_params_billions")

    cfg = ModelConfig(
        name=name,
        total_params_billions=float(total_params) if total_params is not None else 0.0,
        active_params_billions=float(active_params) if active_params is not None else 0.0,
        hidden_size=hidden_size,
        head_dim=int(head_dim),
        num_hidden_layers=int(data.get("num_hidden_layers", data.get("n_layer", 32))),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=int(
            data.get("num_key_value_heads", data.get("multi_query_group_num", num_attention_heads))
        ),
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        intermediate_size=int(data.get("intermediate_size", data.get("ffn_dim", 0) or 0)),
        vocab_size=int(data.get("vocab_size", 0) or 0),
        max_position_embeddings=int(data.get("max_position_embeddings", data.get("max_seq_len", 0) or 0)),
        is_moe=is_moe,
    )

    if cfg.total_params_billions <= 0:
        dense_estimate = estimate_dense_total_params_billions(data)
        if dense_estimate is not None and not is_moe:
            cfg.total_params_billions = dense_estimate

    if cfg.active_params_billions <= 0:
        cfg.active_params_billions = cfg.total_params_billions

    return cfg


def estimate_dense_total_params_billions(config: dict[str, Any]) -> float | None:
    """Estimate dense model parameter count from architecture fields.

    This is intentionally approximate. It is used only when the model card does
    not expose a parameter count and the model is not MoE.
    """
    hidden_size = _as_int(config.get("hidden_size"))
    num_layers = _as_int(config.get("num_hidden_layers") or config.get("n_layer"))
    num_heads = _as_int(config.get("num_attention_heads") or config.get("n_head"))
    vocab_size = _as_int(config.get("vocab_size"))
    intermediate_size = _as_int(config.get("intermediate_size") or config.get("ffn_dim"))

    if not all([hidden_size, num_layers, num_heads, vocab_size, intermediate_size]):
        return None

    head_dim = _as_int(config.get("head_dim"))
    if not head_dim:
        if hidden_size % num_heads != 0:
            return None
        head_dim = hidden_size // num_heads

    num_kv_heads = _as_int(
        config.get("num_key_value_heads") or config.get("multi_query_group_num") or num_heads
    )
    kv_proj_size = num_kv_heads * head_dim

    q_proj = hidden_size * hidden_size
    o_proj = hidden_size * hidden_size
    k_proj = hidden_size * kv_proj_size
    v_proj = hidden_size * kv_proj_size
    attention_params = q_proj + k_proj + v_proj + o_proj

    gated_mlp = bool(intermediate_size and config.get("hidden_act", "silu") not in {"relu"})
    mlp_multiplier = 3 if gated_mlp else 2
    mlp_params = mlp_multiplier * hidden_size * intermediate_size
    layer_params = attention_params + mlp_params + (2 * hidden_size)  # norms

    token_embedding = vocab_size * hidden_size
    lm_head = 0 if config.get("tie_word_embeddings", False) else vocab_size * hidden_size
    total_params = token_embedding + lm_head + num_layers * layer_params
    return total_params / 1e9


@lru_cache(maxsize=64)
def load_model_config_from_huggingface(
    model_id: str,
    revision: str | None = None,
    local_files_only: bool = False,
) -> ModelConfig:
    """Load and normalize model architecture details from Hugging Face.

    Required fields come primarily from `config.json`. Parameter counts are taken
    from model card metadata or text when present; dense models fall back to an
    architecture-based estimate.
    """
    if HfApi is None or hf_hub_download is None:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is not installed. Add it to requirements.txt or install it manually."
        )

    model_id = _normalize_model_id(model_id)

    info = _fetch_model_info(model_id, revision=revision, local_files_only=local_files_only)

    config_data = _download_json_file(
        model_id,
        "config.json",
        revision=revision,
        local_files_only=local_files_only,
    )
    generation_config = _download_json_file(
        model_id,
        "generation_config.json",
        revision=revision,
        local_files_only=local_files_only,
        required=False,
    ) or {}
    readme_text = _download_text_file(
        model_id,
        "README.md",
        revision=revision,
        local_files_only=local_files_only,
        required=False,
    ) or ""

    card_data = _normalize_card_data(getattr(info, "cardData", None) if info is not None else None)
    normalized = dict(config_data)
    normalized["name"] = getattr(info, "id", None) if info is not None else model_id
    normalized["name"] = normalized["name"] or model_id
    normalized["model_id"] = model_id
    if "max_position_embeddings" not in normalized and generation_config.get("max_length") is not None:
        normalized["max_position_embeddings"] = generation_config.get("max_length")

    total_params, active_params = _extract_param_counts(model_id, readme_text, card_data)
    if total_params is not None:
        normalized["total_params_billions"] = total_params
    if active_params is not None:
        normalized["active_params_billions"] = active_params

    model_cfg = model_config_from_mapping(normalized, default_name=model_id)
    model_cfg.name = (getattr(info, "id", None) if info is not None else None) or model_cfg.name

    if model_cfg.total_params_billions <= 0:
        raise ValueError(
            f"Could not determine total parameter count for '{model_id}'. "
            "Add 'model_config' overrides in the scenario YAML."
        )

    if model_cfg.is_moe and model_cfg.active_params_billions <= 0:
        raise ValueError(
            f"Could not determine active parameter count for MoE model '{model_id}'. "
            "Add 'active_params_billions' under 'model_config' in the scenario YAML."
        )

    return model_cfg


def _normalize_model_id(model_id: str) -> str:
    text = str(model_id).strip()
    text = re.sub(r"^https?://huggingface\.co/", "", text)
    text = text.strip("/")
    if "/tree/" in text:
        text = text.split("/tree/", 1)[0]
    if "/resolve/" in text:
        text = text.split("/resolve/", 1)[0]
    return text


def _fetch_model_info(
    model_id: str,
    revision: str | None = None,
    local_files_only: bool = False,
):
    if local_files_only:
        return None
    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    try:
        return api.model_info(model_id, revision=revision, files_metadata=False)
    except Exception:
        return None


@lru_cache(maxsize=64)
def _download_json_file(
    model_id: str,
    filename: str,
    revision: str | None = None,
    local_files_only: bool = False,
    required: bool = True,
) -> dict[str, Any] | None:
    text = _download_text_file(
        model_id,
        filename,
        revision=revision,
        local_files_only=local_files_only,
        required=required,
    )
    if text is None:
        return None
    return json.loads(text)


@lru_cache(maxsize=128)
def _download_text_file(
    model_id: str,
    filename: str,
    revision: str | None = None,
    local_files_only: bool = False,
    required: bool = True,
) -> str | None:
    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            local_files_only=local_files_only,
        )
        return Path(path).read_text(encoding="utf-8")
    except Exception as exc:
        if required:
            mode = "local cache only" if local_files_only else "online/cache"
            raise RuntimeError(
                f"Failed to load '{filename}' for Hugging Face model '{model_id}' in {mode} mode. "
                "Check the model id, network access, authentication for gated/private models, or provide inline model_config overrides."
            ) from exc
        return None


def _normalize_card_data(card_data: Any) -> dict[str, Any]:
    if card_data is None:
        return {}
    if isinstance(card_data, dict):
        return card_data
    if hasattr(card_data, "to_dict"):
        return card_data.to_dict()
    if hasattr(card_data, "__dict__"):
        return dict(card_data.__dict__)
    return {}


def _extract_param_counts(
    model_id: str,
    readme_text: str,
    card_data: dict[str, Any],
) -> tuple[float | None, float | None]:
    total = _extract_count_from_card_data(card_data, keys=(
        "total_params_billions", "params", "parameter_count", "model_size", "num_parameters"
    ))
    active = _extract_count_from_card_data(card_data, keys=(
        "active_params_billions", "activated_params", "active_parameters"
    ))

    total_from_name, active_from_name = _extract_counts_from_name(model_id)
    total = total or total_from_name
    active = active or active_from_name

    total_from_readme, active_from_readme = _extract_counts_from_text(readme_text)
    total = total or total_from_readme
    active = active or active_from_readme

    return total, active


def _extract_count_from_card_data(card_data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = card_data.get(key)
        if value is None:
            continue
        parsed = _parse_billion_value(value)
        if parsed is not None:
            return parsed
    return None


def _extract_counts_from_name(name: str) -> tuple[float | None, float | None]:
    active = None
    total = None

    active_match = re.search(r"A(\d+(?:\.\d+)?)B", name, flags=re.IGNORECASE)
    if active_match:
        active = float(active_match.group(1))

    total_match = re.search(r"(\d+(?:\.\d+)?)B", name, flags=re.IGNORECASE)
    if total_match:
        total = float(total_match.group(1))

    return total, active


def _extract_counts_from_text(text: str) -> tuple[float | None, float | None]:
    if not text:
        return None, None

    compact = re.sub(r"\s+", " ", text)

    active_patterns = [
        r"(\d+(?:\.\d+)?)\s*B(?:illion)?\s+(?:active|activated)\s+parameters",
        r"active(?:d)?\s+parameters[^\d]{0,20}(\d+(?:\.\d+)?)\s*B",
        r"A(\d+(?:\.\d+)?)B",
    ]
    total_patterns = [
        r"(\d+(?:\.\d+)?)\s*B(?:illion)?\s+(?:total\s+)?parameters",
        r"total\s+parameters[^\d]{0,20}(\d+(?:\.\d+)?)\s*B",
        r"(\d+(?:\.\d+)?)B(?:\s|-)(?:[Aa]\d+(?:\.\d+)?)?B",
    ]

    active = _search_first_billion(compact, active_patterns)
    total = _search_first_billion(compact, total_patterns)
    return total, active


def _search_first_billion(text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _parse_billion_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 10_000_000:
            return numeric / 1e9
        return numeric

    text = str(value).strip()
    if not text:
        return None

    match = re.search(r"(\d+(?:\.\d+)?)\s*B", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))

    digits = re.sub(r"[,_ ]", "", text)
    if digits.isdigit():
        numeric = float(digits)
        if numeric > 10_000_000:
            return numeric / 1e9
        return numeric
    return None



def _first_present_int(data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if "." in key:
            value: Any = data
            ok = True
            for part in key.split("."):
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    ok = False
                    break
            if not ok:
                continue
        else:
            value = data.get(key)
        parsed = _as_int(value)
        if parsed is not None:
            return parsed
    return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None
