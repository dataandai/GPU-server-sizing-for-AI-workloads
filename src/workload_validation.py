from __future__ import annotations

import copy
from typing import Any

from .catalog_loader import (
    get_deployment_profile_record,
    get_gpu_record,
    get_hardware_record,
    get_software_stack_record,
    validate_catalog_selection,
)


def canonical_reserved_capacity_fraction(deployment: dict[str, Any]) -> float:
    """Single source of truth for reserved capacity.

    Prefer operational_policy.reserved_capacity_fraction. Fall back to availability
    percent/fraction variants only for backward compatibility.
    """
    operational = deployment.get("operational_policy", {}) or {}
    availability = deployment.get("availability", {}) or {}

    if operational.get("reserved_capacity_fraction") is not None:
        value = float(operational.get("reserved_capacity_fraction") or 0.0)
        return max(0.0, min(0.95, value))

    if availability.get("reserved_capacity_fraction") is not None:
        value = float(availability.get("reserved_capacity_fraction") or 0.0)
        return max(0.0, min(0.95, value))

    if availability.get("reserved_capacity_percent") is not None:
        value = float(availability.get("reserved_capacity_percent") or 0.0) / 100.0
        return max(0.0, min(0.95, value))

    return 0.0


def harmonize_deployment_policy(spec: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(spec)
    deployment = normalized.get("deployment_profile", {})
    operational = deployment.setdefault("operational_policy", {})
    availability = deployment.setdefault("availability", {})
    fraction = canonical_reserved_capacity_fraction(deployment)
    operational["reserved_capacity_fraction"] = fraction
    availability["reserved_capacity_fraction"] = fraction
    availability["reserved_capacity_percent"] = round(fraction * 100.0, 4)
    return normalized


def select_stage_model_id(stage: dict[str, Any], workload_definition: dict[str, Any]) -> str:
    binding = stage.get("model_binding") or {}
    selected = str(binding.get("default_model_id") or "").strip()
    override = str(binding.get("override_model_id") or "").strip()
    if not override:
        return selected

    language_mix = (
        workload_definition.get("input_profile", {})
        .get("work_item", {})
        .get("language_mix", [])
        or []
    )
    hu_share = sum(
        float(item.get("share", 0.0) or 0.0)
        for item in language_mix
        if str(item.get("language", "")).lower().startswith("hu")
    )
    if stage.get("language_sensitive") and hu_share >= 0.5:
        return override
    return selected or override


def apply_selected_model_bindings(spec: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(spec)
    workload = normalized.get("workload_definition", {})
    stages = workload.get("pipeline", {}).get("stages", []) or []
    for stage in stages:
        binding = stage.setdefault("model_binding", {})
        selected = select_stage_model_id(stage, workload)
        if selected:
            binding["selected_model_id"] = selected
    return normalized


def _runtime_family_allows_mode(runtime_family: str, mode: str) -> bool:
    runtime_family = (runtime_family or "").lower()
    mode = (mode or "").lower()
    if mode == "vllm_continuous_batching":
        return "vllm" in runtime_family
    if mode == "trtllm_inflight_batching":
        return "tensorrt" in runtime_family or "trtllm" in runtime_family
    if mode == "triton_dynamic_batching":
        return "triton" in runtime_family or "speech" in runtime_family or "riva" in runtime_family
    return True


def validate_workload_simulation_spec(spec: dict[str, Any], *, strict_catalog: bool = True) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    workload = spec.get("workload_definition", {})
    runtime = spec.get("runtime_profile", {})
    deployment = spec.get("deployment_profile", {})
    partition = deployment.get("partitioning", {}) or {}
    availability = deployment.get("availability", {}) or {}
    operational = deployment.get("operational_policy", {}) or {}
    scheduler = runtime.get("scheduler_model", {}) or {}

    execution_mode = str(deployment.get("execution_mode", ""))
    partition_mode = str(partition.get("mode", "none") or "none")
    runtime_family = str(runtime.get("runtime_family", ""))
    scheduler_mode = str(scheduler.get("mode", ""))

    if not _runtime_family_allows_mode(runtime_family, scheduler_mode):
        errors.append(
            f"Runtime family '{runtime_family}' is not aligned with scheduler mode: '{scheduler_mode}'."
        )

    queue_delay_value = scheduler.get("queue_delay_us")
    if queue_delay_value in (None, ""):
        queue_delay_value = scheduler.get("max_queue_delay_microseconds")
    if scheduler_mode != "triton_dynamic_batching" and queue_delay_value not in (None, 0, 0.0):
        errors.append("Triton queue delay fields are only relevant for Triton dynamic batching runtimes.")

    if partition_mode == "none":
        forbidden = [key for key in ("mig_profile", "sm_fraction", "memory_fraction", "virtual_gpus_per_device") if partition.get(key) not in (None, "", 0, 0.0, 1, 1.0)]
        if forbidden:
            errors.append(
                "The following partitioning fields can only be used in MIG/vGPU mode: " + ", ".join(forbidden)
            )

    if partition_mode == "mig":
        if execution_mode not in {"bare_metal_mig", "bare_metal", "container_on_bare_metal"}:
            errors.append("MIG partitioning is only valid for bare-metal style execution.")
        if not partition.get("mig_profile"):
            errors.append("In MIG mode, 'partitioning.mig_profile' is required.")

    if partition_mode == "time_sliced_vgpu" and execution_mode != "vmware_vgpu_time_sliced":
        errors.append("Time-sliced vGPU partitioning requires 'vmware_vgpu_time_sliced' execution_mode.")

    if partition_mode == "mig_backed_vgpu":
        if execution_mode != "vmware_vgpu_mig_backed":
            errors.append("MIG-backed vGPU partitioning requires 'vmware_vgpu_mig_backed' execution_mode.")
        if not partition.get("mig_profile"):
            errors.append("In MIG-backed vGPU mode, 'partitioning.mig_profile' is required.")

    reserved_fraction = canonical_reserved_capacity_fraction(deployment)
    op_fraction = operational.get("reserved_capacity_fraction")
    av_percent = availability.get("reserved_capacity_percent")
    if op_fraction is not None and av_percent is not None:
        if abs(float(op_fraction) - (float(av_percent) / 100.0)) > 1e-6:
            errors.append(
                "Reserved capacity differs between operational_policy and availability. Use a single canonical value."
            )

    if reserved_fraction >= 0.95:
        errors.append("Reserved capacity is too high; usable capacity would drop close to zero.")

    hardware_binding = deployment.get("hardware_binding", {}) or {}
    hardware_id = str(hardware_binding.get("hardware_catalog_id") or "")
    stack_id = str(hardware_binding.get("software_stack_id") or "")
    if hardware_id:
        try:
            hardware = get_hardware_record(hardware_id)
            gpu = get_gpu_record(hardware["gpu_id"])
        except Exception as exc:
            errors.append(str(exc))
            hardware = None
            gpu = None
        if hardware and stack_id:
            try:
                stack = get_software_stack_record(stack_id)
            except Exception as exc:
                errors.append(str(exc))
                stack = None
            deployment_id = str(deployment.get("deployment_id") or "").strip()
            try:
                deployment_record = get_deployment_profile_record(deployment_id) if deployment_id else None
            except Exception:
                deployment_record = None
            selection_warnings = validate_catalog_selection(
                hardware_id,
                stack_id,
                deployment_id if deployment_record is not None else None,
            )
            if strict_catalog and selection_warnings:
                errors.extend(selection_warnings)
            else:
                warnings.extend(selection_warnings)
            if stack and gpu:
                vendor_focus = str(stack.get("vendor_focus") or "").lower()
                gpu_vendor = str(gpu.get("vendor") or hardware.get("vendor") or "").lower()
                if vendor_focus and vendor_focus not in {"generic", "mixed"} and vendor_focus.lower() not in gpu_vendor:
                    errors.append(
                        f"A '{stack_id}' stack vendor focus ({stack.get('vendor_focus')}) does not match the hardware ({hardware.get('vendor')})."
                    )
                supports_mig = bool(stack.get("supports_mig", False))
                supports_time = bool(stack.get("supports_time_sliced_vgpu", False))
                if partition_mode in {"mig", "mig_backed_vgpu"} and not supports_mig:
                    errors.append(f"A '{stack_id}' stack is not marked as MIG-compatible.")
                if partition_mode == "time_sliced_vgpu" and not supports_time:
                    errors.append(f"A '{stack_id}' stack is not marked as time-sliced vGPU-compatible.")

    for stage in workload.get("pipeline", {}).get("stages", []) or []:
        stage_type = str(stage.get("stage_type", ""))
        binding = stage.get("model_binding") or {}
        batching = stage.get("batching_profile", {}) or {}
        if stage_type in {"generator", "decoder", "encoder", "rerank", "embedding", "classifier", "detector", "segmenter"} and not binding.get("role_id"):
            errors.append(f"'{stage.get('stage_id', 'stage')}' stage is missing model_binding.role_id.")
        batching_mode = str(batching.get("batching_mode", "none"))
        if batching_mode == "none":
            irrelevant = [k for k in ("batch_timeout_ms", "max_batched_tokens") if batching.get(k) not in (None, 0, 0.0, "")]
            if irrelevant:
                errors.append(
                    f"'{stage.get('stage_id', 'stage')}' stage has batching_mode='none', so these fields are not relevant: {', '.join(irrelevant)}."
                )
        if stage_type not in {"generator", "decoder"} and batching.get("max_batched_tokens") not in (None, ""):
            warnings.append(
                f"'{stage.get('stage_id', 'stage')}' stage is not a generator/decoder, so the max_batched_tokens field is probably not relevant."
            )

    return errors, warnings
