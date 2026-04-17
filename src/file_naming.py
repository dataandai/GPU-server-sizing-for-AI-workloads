"""Helpers for human-readable output file names."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .config import SimulationConfig, SimulationResult


def slugify(text: str | None, max_length: int = 40) -> str:
    value = (text or "").strip().lower()
    value = value.replace("/", "-")
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-._")
    if not value:
        value = "item"
    return value[:max_length].rstrip("-._") or "item"


def _model_label_from_config(config: SimulationConfig) -> str:
    if config.model_id:
        return config.model_id.split("/")[-1]
    if config.model_config and getattr(config.model_config, "name", None):
        return str(config.model_config.name)
    return config.model_source or "builtin"


def _hardware_label_from_config(config: SimulationConfig) -> str:
    if config.hardware_catalog_id:
        return config.hardware_catalog_id
    hp = getattr(config.hardware_profile, "value", config.hardware_profile)
    return str(hp or "hardware")


def build_base_name_from_config(config: SimulationConfig, when: datetime | None = None) -> str:
    ts = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    model = slugify(_model_label_from_config(config), 36)
    hardware = slugify(_hardware_label_from_config(config), 28)
    return f"{model}_{hardware}_{ts}"


def _model_label_from_result(result: SimulationResult) -> str:
    if result.model_id:
        return result.model_id.split("/")[-1]
    if result.model_name:
        return result.model_name
    return result.model_source or "builtin"


def _hardware_label_from_result(result: SimulationResult) -> str:
    if result.hardware_catalog_id:
        return result.hardware_catalog_id
    return result.hardware_profile or "hardware"


def build_base_name_from_results(results: Iterable[SimulationResult], when: datetime | None = None) -> str:
    results = list(results)
    ts = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    if len(results) == 1:
        r = results[0]
        model = slugify(_model_label_from_result(r), 36)
        hardware = slugify(_hardware_label_from_result(r), 28)
        return f"{model}_{hardware}_{ts}"
    if results:
        scenario = slugify(results[0].scenario_name or "batch", 24)
        return f"{scenario}_batch_{len(results)}runs_{ts}"
    return f"results_{ts}"


def default_json_output_path(results: Iterable[SimulationResult], when: datetime | None = None) -> Path:
    return Path("output") / "results" / f"{build_base_name_from_results(results, when=when)}.json"
