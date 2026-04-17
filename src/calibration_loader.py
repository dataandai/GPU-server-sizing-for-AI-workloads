from __future__ import annotations

import copy
from typing import Any

from .catalog_loader import list_benchmark_observations, list_calibration_anchor_records


def list_calibration_anchors() -> list[dict[str, Any]]:
    return copy.deepcopy(list_calibration_anchor_records())


def list_benchmark_records() -> list[dict[str, Any]]:
    return copy.deepcopy(list_benchmark_observations())


def get_matching_calibration_anchor(
    *,
    metadata: dict[str, Any],
    stage: dict[str, Any],
    runtime_family: str,
    execution_mode: str,
    hardware_record: dict[str, Any],
    gpu_record: dict[str, Any],
) -> dict[str, Any] | None:
    candidates: list[tuple[int, dict[str, Any]]] = []
    stage_id = str(stage.get("stage_id") or "")
    stage_type = str(stage.get("stage_type") or "")
    service_time_model = stage.get("resource_profile", {}).get("service_time_model", {}) or {}
    reference_profile_id = str(service_time_model.get("reference_profile_id") or "")

    for anchor in list_calibration_anchor_records():
        score = _score_anchor(
            anchor=anchor,
            metadata=metadata,
            stage_id=stage_id,
            stage_type=stage_type,
            reference_profile_id=reference_profile_id,
            runtime_family=runtime_family,
            execution_mode=execution_mode,
            hardware_record=hardware_record,
            gpu_record=gpu_record,
        )
        if score > 0:
            candidates.append((score, anchor))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], str(item[1].get("anchor_id") or "")))
    return copy.deepcopy(candidates[0][1])


def describe_anchor_match(anchor: dict[str, Any] | None, *, stage: dict[str, Any]) -> dict[str, Any]:
    stage_id = str(stage.get("stage_id") or "")
    stage_type = str(stage.get("stage_type") or "")
    if not anchor:
        return {
            "stage_id": stage_id,
            "stage_type": stage_type,
            "matched": False,
        }
    return {
        "stage_id": stage_id,
        "stage_type": stage_type,
        "matched": True,
        "anchor_id": anchor.get("anchor_id"),
        "blueprint_id": anchor.get("blueprint_id"),
        "template_id": anchor.get("template_id"),
        "workload_family": anchor.get("workload_family"),
        "reference_profile_id": anchor.get("reference_profile_id"),
        "notes": anchor.get("notes"),
    }



def _score_anchor(
    *,
    anchor: dict[str, Any],
    metadata: dict[str, Any],
    stage_id: str,
    stage_type: str,
    reference_profile_id: str,
    runtime_family: str,
    execution_mode: str,
    hardware_record: dict[str, Any],
    gpu_record: dict[str, Any],
) -> int:
    score = 0
    if str(anchor.get("stage_id") or "") == stage_id:
        score += 60
    elif str(anchor.get("stage_type") or "") == stage_type and stage_type:
        score += 25
    elif reference_profile_id and str(anchor.get("reference_profile_id") or "") == reference_profile_id:
        score += 20
    else:
        return 0

    if _matches_exact(anchor.get("blueprint_id"), metadata.get("source_blueprint_id")):
        score += 25
    elif anchor.get("blueprint_id"):
        return 0

    if _matches_exact(anchor.get("template_id"), metadata.get("source_template_id")):
        score += 18
    elif anchor.get("template_id"):
        return 0

    if _matches_exact(anchor.get("workload_family"), metadata.get("workload_family")):
        score += 12
    elif anchor.get("workload_family"):
        return 0

    applies = anchor.get("applies_if") or {}
    runtime_families = {str(x) for x in applies.get("runtime_family", []) if x}
    if runtime_families:
        if runtime_family not in runtime_families:
            return 0
        score += 8

    execution_modes = {str(x) for x in applies.get("execution_mode", []) if x}
    if execution_modes:
        if execution_mode not in execution_modes:
            return 0
        score += 6

    gpu_vendor = str(applies.get("gpu_vendor") or "").strip().lower()
    if gpu_vendor:
        if str(gpu_record.get("vendor") or hardware_record.get("vendor") or "").strip().lower() != gpu_vendor:
            return 0
        score += 4

    gpu_ids = {str(x) for x in applies.get("gpu_ids", []) if x}
    if gpu_ids:
        if str(hardware_record.get("gpu_id") or "") not in gpu_ids:
            return 0
        score += 4

    return score



def _matches_exact(anchor_value: Any, metadata_value: Any) -> bool:
    if anchor_value in (None, ""):
        return False
    return str(anchor_value) == str(metadata_value or "")
