from __future__ import annotations

from typing import Any


DEFAULT_SCALARS = {
    "prompt": 1.0,
    "output": 1.0,
    "context": 1.0,
}

LANGUAGE_RULES: dict[str, dict[str, float]] = {
    "hu": {"prompt": 0.35, "output": 1.4, "context": 1.0},
}


def clamp_share(share: Any, default: float = 0.0) -> float:
    try:
        value = float(share)
    except (TypeError, ValueError):
        value = float(default)
    return max(0.0, min(1.0, value))


def normalize_language_code(language_code: Any, default: str = "en") -> str:
    code = str(language_code or default).strip().lower()
    return code or default


def scalars_for_language(language_code: Any, language_share: Any = 1.0) -> tuple[float, float, float]:
    code = normalize_language_code(language_code)
    share = clamp_share(language_share, 1.0)
    rule = next((payload for prefix, payload in LANGUAGE_RULES.items() if code.startswith(prefix)), None)
    if not rule:
        return (1.0, 1.0, 1.0)
    return (
        1.0 + float(rule.get("prompt", 0.0)) * share,
        1.0 + float(rule.get("output", 0.0)) * share,
        1.0 + float(rule.get("context", 0.0)) * share,
    )


def prompt_multiplier(language_code: Any, language_share: Any = 1.0) -> float:
    return scalars_for_language(language_code, language_share)[0]


def generation_multiplier(language_code: Any, language_share: Any = 1.0) -> float:
    return scalars_for_language(language_code, language_share)[1]


def context_multiplier(language_code: Any, language_share: Any = 1.0) -> float:
    return scalars_for_language(language_code, language_share)[2]


def dominant_language(language_mix: list[dict[str, Any]] | None) -> tuple[str, float]:
    best_lang, best_share = "en", 0.0
    for item in language_mix or []:
        share = clamp_share(item.get("share", 0.0), 0.0)
        if share >= best_share:
            best_lang = normalize_language_code(item.get("language") or "en")
            best_share = share
    return best_lang, best_share


def scalars_from_language_mix(language_mix: list[dict[str, Any]] | None) -> tuple[float, float, float]:
    prompt = 1.0
    output = 1.0
    context = 1.0
    for item in language_mix or []:
        lang = normalize_language_code(item.get("language") or "en")
        share = clamp_share(item.get("share", 0.0), 0.0)
        p_mult, o_mult, c_mult = scalars_for_language(lang, share)
        prompt += max(0.0, p_mult - 1.0)
        output += max(0.0, o_mult - 1.0)
        context += max(0.0, c_mult - 1.0)
    return prompt, output, context


def language_scaling_metadata(*, language_code: Any | None = None, language_share: Any = 1.0, language_mix: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if language_mix is not None:
        dominant_code, dominant_share = dominant_language(language_mix)
    else:
        dominant_code = normalize_language_code(language_code)
        dominant_share = clamp_share(language_share, 1.0)
    prompt, output, context = scalars_for_language(dominant_code, dominant_share)
    return {
        "dominant_language": dominant_code,
        "dominant_share": round(dominant_share, 4),
        "prompt_multiplier": round(prompt, 4),
        "generation_multiplier": round(output, 4),
        "context_multiplier": round(context, 4),
    }
