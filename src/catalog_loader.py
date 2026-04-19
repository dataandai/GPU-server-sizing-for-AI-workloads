"""Helpers for loading curated hardware/software/blueprint seed catalogs."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import HardwareProfile
from .model_role_registry import ROLE_DEFAULTS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = PROJECT_ROOT / "catalog"

CATALOG_PATHS: dict[str, tuple[str, ...]] = {
    "gpu_catalog": (
        "hardware/gpu_catalog.json",
    ),
    "hardware_catalog": (
        "hardware/hardware_catalog.json",
    ),
    "software_stacks": (
        "hardware/software_stacks.json",
    ),
    "deployment_profiles": (
        "hardware/deployment_profiles.json",
    ),
    "hf_model_ingest_contract": (
        "models/hf_model_ingest_contract.json",
    ),
    "model_profiles": (
        "models/model_profiles.json",
    ),
    "source_manifest": (
        "sources/source_manifest.md",
    ),
    "nvidia_blueprints_catalog": (
        "blueprints/nvidia_blueprints_catalog.json",
    ),
    "nvidia_blueprints_catalog_schema": (
        "blueprints/nvidia_blueprints_catalog.schema.json",
    ),
    "nvidia_blueprint_templates": (
        "blueprints/nvidia_blueprint_templates.json",
    ),
    "workload_simulation_schema": (
        "simulation/workload_simulation.schema.json",
    ),
    "benchmark_observations": (
        "calibration/benchmark_observations.json",
    ),
    "calibration_anchors": (
        "calibration/calibration_anchors.json",
    ),
    "calibration_anchors_schema": (
        "calibration/calibration_anchors.schema.json",
    ),
    "advanced_workload_templates": (
        "workload_templates/advanced_workload_templates.json",
    ),
    "ui_help_tooltips_hu": (
        "ui_help/tooltips_hu.json",
    ),
    "ui_help_tooltips_en": (
        "ui_help/tooltips_en.json",
    ),
}


def _catalog_path(logical_name: str) -> Path:
    candidates = CATALOG_PATHS.get(logical_name, (logical_name,))
    for relative in candidates:
        path = CATALOG_DIR / relative
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Catalog file not found for '{logical_name}'. Looked in: "
        + ", ".join(str(CATALOG_DIR / rel) for rel in candidates)
    )


@lru_cache(maxsize=None)
def _load_catalog(logical_name: str) -> Any:
    path = _catalog_path(logical_name)
    if path.suffix.lower() == ".md":
        return path.read_text(encoding="utf-8")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        for key in ("records", "templates", "items", "entries", "blueprints"):
            value = payload.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    raise ValueError("Catalog payload is not a supported record container.")


@lru_cache(maxsize=None)
def _record_index(logical_name: str, id_field: str = "id") -> dict[str, dict[str, Any]]:
    data = _load_catalog(logical_name)
    index: dict[str, dict[str, Any]] = {}
    for record in _extract_records(data):
        record_id = record.get(id_field)
        if record_id is None:
            continue
        index[str(record_id)] = record
    return index


def _records(logical_name: str, id_field: str = "id") -> list[dict[str, Any]]:
    return list(_record_index(logical_name, id_field=id_field).values())


def _get_record(logical_name: str, record_id: str, id_field: str = "id") -> dict[str, Any]:
    try:
        return _record_index(logical_name, id_field=id_field)[record_id]
    except KeyError as exc:
        raise KeyError(f"Unknown {logical_name} id: {record_id}") from exc


def list_gpu_records() -> list[dict[str, Any]]:
    return _records("gpu_catalog")


def list_hardware_records() -> list[dict[str, Any]]:
    return _records("hardware_catalog")


def list_software_stack_records() -> list[dict[str, Any]]:
    return _records("software_stacks")


def list_deployment_profile_records() -> list[dict[str, Any]]:
    return _records("deployment_profiles")


def list_blueprint_records() -> list[dict[str, Any]]:
    return _records("nvidia_blueprints_catalog", id_field="blueprint_id")


def list_blueprint_template_records() -> list[dict[str, Any]]:
    return _records("nvidia_blueprint_templates", id_field="template_id")


def get_gpu_record(gpu_id: str) -> dict[str, Any]:
    return _get_record("gpu_catalog", gpu_id)


def get_hardware_record(hardware_id: str) -> dict[str, Any]:
    return _get_record("hardware_catalog", hardware_id)


def get_software_stack_record(stack_id: str) -> dict[str, Any]:
    return _get_record("software_stacks", stack_id)


def get_deployment_profile_record(profile_id: str) -> dict[str, Any]:
    return _get_record("deployment_profiles", profile_id)


def list_model_profile_records() -> list[dict[str, Any]]:
    return _records("model_profiles")


def get_model_profile_record(model_id: str) -> dict[str, Any]:
    return _get_record("model_profiles", model_id)


def get_blueprint_record(blueprint_id: str) -> dict[str, Any]:
    return _get_record("nvidia_blueprints_catalog", blueprint_id, id_field="blueprint_id")


def get_blueprint_template_record(template_id: str) -> dict[str, Any]:
    return _get_record("nvidia_blueprint_templates", template_id, id_field="template_id")


def get_blueprint_bundle(blueprint_id: str) -> dict[str, Any]:
    blueprint = get_blueprint_record(blueprint_id)
    templates = [
        template
        for template in list_blueprint_template_records()
        if template.get("blueprint_id") == blueprint_id
    ]
    return {
        "blueprint": blueprint,
        "templates": templates,
    }


def _infer_interconnect(topology_class: str) -> str:
    topology = (topology_class or "").lower()
    if "nvswitch" in topology or "nvlink" in topology:
        return "nvlink"
    return "pcie5"


def default_tensor_parallel_degree_for_hardware(hardware_id: str) -> int:
    record = get_hardware_record(hardware_id)
    recommended = [int(v) for v in record.get("recommended_tp_values", []) if v]
    if recommended:
        return max(recommended)
    return int(record.get("gpu_count", 1))


def validate_catalog_selection(
    hardware_id: str,
    software_stack_id: str | None = None,
    deployment_profile_id: str | None = None,
) -> list[str]:
    record = get_hardware_record(hardware_id)
    warnings: list[str] = []

    if software_stack_id and software_stack_id not in record.get("software_stack_ids", []):
        warnings.append(
            f"Software stack '{software_stack_id}' is not listed as a preferred stack for '{hardware_id}'."
        )

    if deployment_profile_id and deployment_profile_id not in record.get("deployment_profile_ids", []):
        warnings.append(
            f"Deployment profile '{deployment_profile_id}' is not listed as a preferred mode for '{hardware_id}'."
        )

    return warnings


def resolve_hardware_profile(hardware_id: str) -> HardwareProfile:
    record = get_hardware_record(hardware_id)
    gpu = get_gpu_record(record["gpu_id"])

    notes = [record.get("notes", "").strip()]
    topology = record.get("topology_class")
    if topology:
        notes.append(f"Topology: {topology}.")
    if record.get("software_stack_ids"):
        notes.append(
            "Preferred stacks: " + ", ".join(record.get("software_stack_ids", [])) + "."
        )

    return HardwareProfile(
        name=record.get("display_name") or hardware_id,
        num_gpus=int(record.get("gpu_count", 1)),
        vram_per_gpu_gb=float(gpu.get("memory_gb", 0.0)),
        memory_bandwidth_gbps=float(gpu.get("memory_bandwidth_gbps") or 0.0),
        interconnect=_infer_interconnect(str(topology or "")),
        notes=" ".join(n for n in notes if n),
        catalog_id=hardware_id,
        vendor=record.get("vendor"),
        gpu_id=record.get("gpu_id"),
        topology_class=topology,
        software_stack_ids=list(record.get("software_stack_ids", [])),
        deployment_profile_ids=list(record.get("deployment_profile_ids", [])),
        source_urls=list(record.get("source_urls", [])),
    )






def list_ui_help_records(language: str = "hu") -> list[dict[str, Any]]:
    """Load UI help records in the specified language (default: Hungarian).
    
    Args:
        language: Language code ("hu" for Hungarian, "en" for English).
    
    Returns:
        List of UI help records in the specified language.
    """
    language = language.lower().strip()
    if language not in ("hu", "en", "de"):
        language = "hu"  # Default to Hungarian
    logical_name = f"ui_help_tooltips_{language}"
    return _records(logical_name, id_field="field_id")


def get_ui_help_record(field_id: str) -> dict[str, Any]:
    return _get_record("ui_help_tooltips_hu", field_id, id_field="field_id")

def list_benchmark_observations() -> list[dict[str, Any]]:
    return _records("benchmark_observations", id_field="observation_id")


def list_calibration_anchor_records() -> list[dict[str, Any]]:
    return _records("calibration_anchors", id_field="anchor_id")


def get_calibration_anchor_record(anchor_id: str) -> dict[str, Any]:
    return _get_record("calibration_anchors", anchor_id, id_field="anchor_id")


def list_advanced_workload_template_records() -> list[dict[str, Any]]:
    return _records("advanced_workload_templates", id_field="template_id")


def get_advanced_workload_template_record(template_id: str) -> dict[str, Any]:
    return _get_record("advanced_workload_templates", template_id, id_field="template_id")

def get_workload_simulation_schema() -> dict[str, Any]:
    payload = _load_catalog("workload_simulation_schema")
    if not isinstance(payload, dict):
        raise ValueError("Workload simulation schema payload must be a JSON object.")
    return payload


def validate_catalog_integrity() -> tuple[list[str], list[str]]:
    """Semantic cross-reference validation for the curated catalogs."""
    errors: list[str] = []
    warnings: list[str] = []

    gpu_ids = {str(r.get("id") or "") for r in list_gpu_records()}
    stack_ids = {str(r.get("id") or "") for r in list_software_stack_records()}
    deployment_ids = {str(r.get("id") or "") for r in list_deployment_profile_records()}
    blueprint_ids = {str(r.get("blueprint_id") or "") for r in list_blueprint_records()}
    known_role_ids = set(ROLE_DEFAULTS.keys())

    for record in list_hardware_records():
        hardware_id = str(record.get("id") or "")
        gpu_id = str(record.get("gpu_id") or "")
        if gpu_id not in gpu_ids:
            errors.append(f"Hardware '{hardware_id}' references unknown GPU '{gpu_id}'.")
        for stack_id in record.get("software_stack_ids", []) or []:
            if str(stack_id) not in stack_ids:
                errors.append(f"Hardware '{hardware_id}' references unknown software stack '{stack_id}'.")
        for profile_id in record.get("deployment_profile_ids", []) or []:
            if str(profile_id) not in deployment_ids:
                errors.append(f"Hardware '{hardware_id}' references unknown deployment profile '{profile_id}'.")

    for profile in list_model_profile_records():
        model_id = str(profile.get("id") or "")
        supported_stacks = profile.get("supported_stack_ids", []) or []
        if not supported_stacks:
            warnings.append(f"Model profile '{model_id}' has no supported_stack_ids.")
        for stack_id in supported_stacks:
            if str(stack_id) not in stack_ids:
                errors.append(f"Model profile '{model_id}' references unknown software stack '{stack_id}'.")

    blueprint_role_map: dict[str, set[str]] = {}
    for blueprint in list_blueprint_records():
        blueprint_id = str(blueprint.get("blueprint_id") or "")
        role_ids: set[str] = set()
        for role in blueprint.get("pipeline_roles", []) or []:
            role_id = str((role or {}).get("role_id") or "").strip()
            if not role_id:
                errors.append(f"Blueprint '{blueprint_id}' has a pipeline role without role_id.")
                continue
            role_ids.add(role_id)
            if role_id not in known_role_ids:
                warnings.append(f"Blueprint '{blueprint_id}' uses role '{role_id}' without explicit registry defaults.")
        blueprint_role_map[blueprint_id] = role_ids

    for template in list_blueprint_template_records():
        template_id = str(template.get("template_id") or "")
        blueprint_id = str(template.get("blueprint_id") or "")
        if blueprint_id not in blueprint_ids:
            errors.append(f"Blueprint template '{template_id}' references unknown blueprint '{blueprint_id}'.")
            continue
        blueprint_roles = blueprint_role_map.get(blueprint_id, set())
        for stage in template.get("stages", []) or []:
            stage_id = str((stage or {}).get("stage_id") or "").strip()
            role_id = str((stage or {}).get("role_id") or "").strip()
            if not stage_id:
                errors.append(f"Blueprint template '{template_id}' contains a stage without stage_id.")
            if not role_id:
                errors.append(f"Blueprint template '{template_id}' contains stage '{stage_id or '-'}' without role_id.")
                continue
            if blueprint_roles and role_id not in blueprint_roles:
                errors.append(f"Blueprint template '{template_id}' stage '{stage_id}' references role '{role_id}' that is not declared on blueprint '{blueprint_id}'.")

    for template in list_advanced_workload_template_records():
        template_id = str(template.get("template_id") or "")
        for stage in template.get("stages", []) or []:
            stage_id = str((stage or {}).get("stage_id") or "").strip()
            role_id = str((stage or {}).get("role_id") or "").strip()
            if not stage_id:
                errors.append(f"Advanced template '{template_id}' contains a stage without stage_id.")
            if not role_id:
                errors.append(f"Advanced template '{template_id}' contains stage '{stage_id or '-'}' without role_id.")
                continue
            if role_id not in known_role_ids:
                errors.append(f"Advanced template '{template_id}' uses unknown role '{role_id}'.")

    return errors, warnings
