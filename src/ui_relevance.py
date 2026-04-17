from __future__ import annotations

from typing import Any

from .blueprint_adapter import list_bindable_roles
from .model_role_registry import build_role_ui_record
from .catalog_loader import (
    get_advanced_workload_template_record,
    get_blueprint_bundle,
    get_deployment_profile_record,
    get_gpu_record,
    get_hardware_record,
    get_software_stack_record,
    list_deployment_profile_records,
    list_hardware_records,
    list_software_stack_records,
)


def _hardware_allowed_for_blueprint(record: dict[str, Any], blueprint: dict[str, Any]) -> bool:
    if not blueprint:
        return True
    try:
        gpu = get_gpu_record(record["gpu_id"])
    except Exception:
        gpu = {}
    gpu_vendor = str(gpu.get("vendor") or record.get("vendor") or "").lower()
    blueprint_vendor = str(blueprint.get("vendor") or "").lower()
    if blueprint_vendor == "nvidia" and "nvidia" not in gpu_vendor:
        return False
    if blueprint.get("supports_large_server_deployment") and int(record.get("gpu_count") or 0) < 4:
        return False
    return True


def _stack_allowed_for_context(stack: dict[str, Any], blueprint: dict[str, Any], deployment: dict[str, Any] | None) -> bool:
    vendor_focus = str(stack.get("vendor_focus") or "").lower()
    blueprint_vendor = str(blueprint.get("vendor") or "").lower()
    if blueprint_vendor and vendor_focus not in {"", "generic", "mixed", blueprint_vendor}:
        return False

    gpu_partitioning = str((deployment or {}).get("gpu_partitioning") or "none")
    if gpu_partitioning in {"mig", "mig_backed"} and not bool(stack.get("supports_mig", False)):
        return False
    if gpu_partitioning == "time_sliced" and not bool(stack.get("supports_time_sliced_vgpu", False)):
        return False
    return True


def _deployment_allowed_for_context(deployment: dict[str, Any], stack: dict[str, Any] | None) -> bool:
    if stack is None:
        return True
    gpu_partitioning = str(deployment.get("gpu_partitioning") or "none")
    if gpu_partitioning in {"mig", "mig_backed"} and not bool(stack.get("supports_mig", False)):
        return False
    if gpu_partitioning == "time_sliced" and not bool(stack.get("supports_time_sliced_vgpu", False)):
        return False
    return True


def _blueprint_role_records(blueprint_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in list_bindable_roles(blueprint_id):
        role_id = str(raw.get("role_id") or "").strip()
        if not role_id:
            continue
        ui_record = build_role_ui_record(role_id, replaceable=bool(raw.get("replaceable", True)))
        if raw.get("role_type"):
            ui_record["role_type"] = raw.get("role_type")
        if raw.get("default_model_reference"):
            ui_record["default_model_reference"] = raw.get("default_model_reference")
        if raw.get("language_sensitive") is not None:
            ui_record["language_sensitive"] = bool(raw.get("language_sensitive"))
        notes = [str(raw.get("notes") or "").strip(), str(ui_record.get("notes") or "").strip()]
        ui_record["notes"] = " ".join([n for n in notes if n]).strip()
        records.append(ui_record)
    return records


def build_blueprint_ui_context(
    blueprint_id: str,
    *,
    template_id: str | None = None,
    hardware_catalog_id: str | None = None,
    software_stack_id: str | None = None,
    deployment_profile_id: str | None = None,
) -> dict[str, Any]:
    bundle = get_blueprint_bundle(blueprint_id)
    blueprint = bundle["blueprint"]
    template = next((t for t in bundle["templates"] if t.get("template_id") == template_id), bundle["templates"][0] if bundle["templates"] else None)
    roles = _blueprint_role_records(blueprint_id)
    replaceable_roles = [r for r in roles if r.get("replaceable")]
    fixed_roles = [r for r in roles if not r.get("replaceable")]
    show_language_controls = bool(
        (template or {}).get("language_override_strategy", {}).get("supported")
        or any(bool(r.get("language_sensitive")) for r in replaceable_roles)
    )

    allowed_hardware = [r for r in list_hardware_records() if _hardware_allowed_for_blueprint(r, blueprint)]
    allowed_hardware_ids = [r["id"] for r in allowed_hardware]

    selected_hardware = None
    if hardware_catalog_id and hardware_catalog_id in allowed_hardware_ids:
        selected_hardware = get_hardware_record(hardware_catalog_id)
    elif allowed_hardware:
        selected_hardware = allowed_hardware[0]

    hardware_stack_ids = list((selected_hardware or {}).get("software_stack_ids") or [])
    selected_deployment = None
    if deployment_profile_id:
        try:
            selected_deployment = get_deployment_profile_record(deployment_profile_id)
        except Exception:
            selected_deployment = None

    allowed_software_stack_ids: list[str] = []
    for stack_id in hardware_stack_ids:
        stack = get_software_stack_record(stack_id)
        if _stack_allowed_for_context(stack, blueprint, selected_deployment):
            allowed_software_stack_ids.append(stack_id)

    selected_stack = None
    if software_stack_id and software_stack_id in allowed_software_stack_ids:
        selected_stack = get_software_stack_record(software_stack_id)
    elif allowed_software_stack_ids:
        selected_stack = get_software_stack_record(allowed_software_stack_ids[0])

    hardware_deployment_ids = list((selected_hardware or {}).get("deployment_profile_ids") or [])
    allowed_deployment_profile_ids: list[str] = []
    for dep_id in hardware_deployment_ids:
        deployment = get_deployment_profile_record(dep_id)
        if _deployment_allowed_for_context(deployment, selected_stack):
            allowed_deployment_profile_ids.append(dep_id)

    if selected_deployment is None and allowed_deployment_profile_ids:
        selected_deployment = get_deployment_profile_record(allowed_deployment_profile_ids[0])

    notes: list[str] = []
    if blueprint.get("supports_large_server_deployment"):
        notes.append("The UI narrows the selected NVIDIA blueprint to large-machine configurations with at least 4 GPUs.")
    if fixed_roles:
        notes.append(f"{len(fixed_roles)} system or service roles remain fixed; these cannot be changed in the UI.")
    if show_language_controls:
        notes.append("Language fields are shown because the selected blueprint contains language-sensitive, replaceable model roles.")
    if selected_stack is not None:
        if not bool(selected_stack.get("supports_mig", False)):
            notes.append("The selected stack is not MIG-compatible, so MIG deployment options remain hidden.")
        if not bool(selected_stack.get("supports_time_sliced_vgpu", False)):
            notes.append("The selected stack is not compatible with time-sliced vGPU, so those deployment options remain hidden.")

    return {
        "blueprint_id": blueprint_id,
        "template_id": template.get("template_id") if template else None,
        "workload_family": (template or {}).get("workload_family"),
        "show_language_controls": show_language_controls,
        "show_model_binding": bool(replaceable_roles),
        "allowed_hardware_ids": allowed_hardware_ids,
        "allowed_software_stack_ids": allowed_software_stack_ids,
        "allowed_deployment_profile_ids": allowed_deployment_profile_ids,
        "replaceable_roles": replaceable_roles,
        "fixed_roles": fixed_roles,
        "template_contract": {
            "required": list((template or {}).get("input_profile_schema", {}).get("required") or []),
            "optional": list((template or {}).get("input_profile_schema", {}).get("optional") or []),
        },
        "template_fields": list((template or {}).get("parameter_definitions") or []),
        "ui_defaults": (template or {}).get("ui_defaults") or {},
        "language_override_strategy": (template or {}).get("language_override_strategy") or {},
        "field_visibility": {
            "bp_language_code": show_language_controls,
            "bp_language_share": show_language_controls,
            "bp_model_binding": bool(replaceable_roles),
            "bp_fixed_roles": bool(fixed_roles),
            "bp_template_contract": bool(template),
            "bp_template_inputs": bool((template or {}).get("parameter_definitions")),
        },
        "notes": notes,
    }



def _stack_allowed_for_template(stack: dict[str, Any], template: dict[str, Any], deployment: dict[str, Any] | None) -> bool:
    if str(stack.get("vendor_focus") or "").lower() != "nvidia":
        return False
    framework = str(stack.get("framework") or "").lower()
    hint = str(template.get("runtime_preset_hint") or "").lower()
    if hint == "vllm" and "vllm" not in framework:
        return False
    if hint == "tensorrt_llm" and "tensorrt" not in framework:
        return False
    if hint == "triton" and "triton" not in framework:
        return False
    return _stack_allowed_for_context(stack, {"vendor": "nvidia"}, deployment)


def _template_role_records(template: dict[str, Any]) -> list[dict[str, Any]]:
    role_ids: list[str] = []
    for stage in template.get("stages", []) or []:
        role_id = str(stage.get("role_id") or stage.get("stage_id") or "").strip()
        if role_id and role_id not in role_ids:
            role_ids.append(role_id)
    return [
        build_role_ui_record(
            role_id,
            replaceable=True,
            notes="Primary on-prem AI role in the advanced template library; the selected value can also be replaced with your own self-hosted model or service reference.",
        )
        for role_id in role_ids
    ]


def build_template_ui_context(
    template_id: str,
    *,
    hardware_catalog_id: str | None = None,
    software_stack_id: str | None = None,
    deployment_profile_id: str | None = None,
) -> dict[str, Any]:
    template = get_advanced_workload_template_record(template_id)
    roles = _template_role_records(template)
    show_language_controls = any(bool(r.get("language_sensitive")) for r in roles)

    allowed_hardware = [r for r in list_hardware_records() if int(r.get("gpu_count") or 0) >= 1]
    allowed_hardware_ids = [r["id"] for r in allowed_hardware]

    selected_hardware = None
    if hardware_catalog_id and hardware_catalog_id in allowed_hardware_ids:
        selected_hardware = get_hardware_record(hardware_catalog_id)
    elif allowed_hardware:
        selected_hardware = next((r for r in allowed_hardware if r.get("id") == "nvidia_dgx_h200_8gpu"), allowed_hardware[0])

    selected_deployment = None
    if deployment_profile_id:
        try:
            selected_deployment = get_deployment_profile_record(deployment_profile_id)
        except Exception:
            selected_deployment = None

    hardware_stack_ids = list((selected_hardware or {}).get("software_stack_ids") or [])
    allowed_software_stack_ids: list[str] = []
    for stack_id in hardware_stack_ids:
        stack = get_software_stack_record(stack_id)
        if _stack_allowed_for_template(stack, template, selected_deployment):
            allowed_software_stack_ids.append(stack_id)

    if not allowed_software_stack_ids:
        for stack in list_software_stack_records():
            if _stack_allowed_for_template(stack, template, selected_deployment):
                allowed_software_stack_ids.append(str(stack.get("id")))

    selected_stack = None
    if software_stack_id and software_stack_id in allowed_software_stack_ids:
        selected_stack = get_software_stack_record(software_stack_id)
    elif allowed_software_stack_ids:
        preferred_stack_id = allowed_software_stack_ids[0]
        hint = str(template.get("runtime_preset_hint") or "").lower()
        if hint == "vllm":
            preferred_stack_id = next((sid for sid in allowed_software_stack_ids if sid == "nvidia_vllm_cuda"), preferred_stack_id)
        elif hint == "tensorrt_llm":
            preferred_stack_id = next((sid for sid in allowed_software_stack_ids if sid == "nvidia_tensorrt_llm"), preferred_stack_id)
        elif hint == "triton":
            preferred_stack_id = next((sid for sid in allowed_software_stack_ids if sid == "nvidia_triton_inference"), preferred_stack_id)
        selected_stack = get_software_stack_record(preferred_stack_id)

    hardware_deployment_ids = list((selected_hardware or {}).get("deployment_profile_ids") or [])
    allowed_deployment_profile_ids: list[str] = []
    for dep_id in hardware_deployment_ids:
        deployment = get_deployment_profile_record(dep_id)
        if bool(deployment.get("is_virtualized")):
            continue
        if _deployment_allowed_for_context(deployment, selected_stack):
            allowed_deployment_profile_ids.append(dep_id)

    if not allowed_deployment_profile_ids:
        for deployment in list_deployment_profile_records():
            if bool(deployment.get("is_virtualized")):
                continue
            if _deployment_allowed_for_context(deployment, selected_stack):
                allowed_deployment_profile_ids.append(str(deployment.get("id")))

    notes = [
        "In the advanced template flow, on-prem AI is primary: NVIDIA-focused stacks and non-virtualized deployments are shown by default.",
        "For every advanced template role, you can choose a recommended model or your own self-hosted reference.",
    ]
    if show_language_controls:
        notes.append("Language fields are shown because the selected template also contains a language-sensitive role.")

    return {
        "template_id": template_id,
        "template_family": template.get("template_family") or template.get("workload_family"),
        "runtime_preset_hint": template.get("runtime_preset_hint"),
        "show_language_controls": show_language_controls,
        "allowed_hardware_ids": allowed_hardware_ids,
        "allowed_software_stack_ids": allowed_software_stack_ids,
        "allowed_deployment_profile_ids": allowed_deployment_profile_ids,
        "replaceable_roles": roles,
        "fixed_roles": [],
        "template_fields": list(template.get("parameter_definitions") or []),
        "field_visibility": {
            "tpl_language_code": show_language_controls,
            "tpl_language_share": show_language_controls,
            "tpl_model_binding": bool(roles),
            "tpl_template_inputs": bool(template.get("parameter_definitions")),
        },
        "notes": notes,
    }
