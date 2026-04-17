"""Workload generators that produce SimulationConfig objects from YAML scenario files."""

from __future__ import annotations

import copy
from typing import Any

import yaml

from .catalog_loader import (
    default_tensor_parallel_degree_for_hardware,
    resolve_hardware_profile,
    validate_catalog_selection,
)
from .config import (
    AgenticWorkload,
    DistributionSpec,
    DistributionType,
    FixedBatchWorkload,
    HardwareProfileName,
    HybridWorkload,
    HumanInTheLoopWorkload,
    KVCachePrecision,
    MixedBatchClass,
    MixedBatchWorkload,
    ModelConfig,
    SimulationConfig,
    WeightPrecision,
    WorkloadType,
)
from .model_loader import load_model_config_from_huggingface, model_config_from_mapping


def _parse_distribution(data: Any) -> DistributionSpec:
    """Parse a distribution specification from YAML data."""
    if isinstance(data, (int, float)):
        return DistributionSpec.fixed(float(data))

    if isinstance(data, dict):
        dtype = DistributionType(data.get("type", "fixed"))
        spec = DistributionSpec(type=dtype)

        if dtype == DistributionType.FIXED:
            spec.value = float(data.get("value", 0))
        elif dtype == DistributionType.UNIFORM:
            spec.low = float(data.get("low", 0))
            spec.high = float(data.get("high", 0))
        elif dtype == DistributionType.NORMAL:
            spec.mean = float(data.get("mean", 0))
            spec.std = float(data.get("std", 1))
        elif dtype == DistributionType.LOGNORMAL:
            spec.mu = float(data.get("mu", 0))
            spec.sigma = float(data.get("sigma", 1))
        elif dtype == DistributionType.POISSON:
            spec.lam = float(data.get("lam", 1))
        elif dtype == DistributionType.EMPIRICAL:
            spec.values = [float(v) for v in data.get("values", [])]
            spec.probabilities = [float(p) for p in data.get("probabilities", [])]

        spec.min_clip = data.get("min_clip")
        spec.max_clip = data.get("max_clip")
        return spec

    return DistributionSpec.fixed(0)


def _parse_fixed_batch(data: dict) -> FixedBatchWorkload:
    return FixedBatchWorkload(
        prompt_tokens=_parse_distribution(data.get("prompt_tokens", 1200)),
        max_output_tokens=_parse_distribution(data.get("max_output_tokens", 500)),
        num_parallel_jobs=int(data.get("num_parallel_jobs", 10)),
    )


def _parse_mixed_batch(data: dict) -> MixedBatchWorkload:
    classes = []
    for cls_data in data.get("classes", []):
        classes.append(MixedBatchClass(
            label=cls_data.get("label", "default"),
            prompt_tokens=_parse_distribution(cls_data.get("prompt_tokens", 1024)),
            max_output_tokens=_parse_distribution(cls_data.get("max_output_tokens", 512)),
            count=int(cls_data.get("count", 5)),
        ))
    return MixedBatchWorkload(classes=classes)


def _parse_agentic(data: dict) -> AgenticWorkload:
    return AgenticWorkload(
        initial_prompt_tokens=_parse_distribution(data.get("initial_prompt_tokens",
            {"type": "lognormal", "mu": 7.0, "sigma": 0.3})),
        max_turns_per_run=int(data.get("max_turns_per_run", 10)),
        tool_call_probability=float(data.get("tool_call_probability", 0.7)),
        tool_result_tokens=_parse_distribution(data.get("tool_result_tokens",
            {"type": "lognormal", "mu": 6.0, "sigma": 0.7})),
        assistant_response_tokens=_parse_distribution(data.get("assistant_response_tokens",
            {"type": "lognormal", "mu": 5.5, "sigma": 0.5})),
        summarization_probability=float(data.get("summarization_probability", 0.0)),
        summarization_compression_ratio=float(data.get("summarization_compression_ratio", 0.3)),
        stop_probability_per_turn=float(data.get("stop_probability_per_turn", 0.15)),
        max_total_tokens_per_run=int(data.get("max_total_tokens_per_run", 32768)),
        num_parallel_agents=int(data.get("num_parallel_agents", 10)),
        enable_context_reset=bool(data.get("enable_context_reset", False)),
        context_reset_threshold=int(data.get("context_reset_threshold", 16384)),
        enable_max_context_cap=bool(data.get("enable_max_context_cap", True)),
        max_context_cap=int(data.get("max_context_cap", 32768)),
    )


def _parse_human_in_the_loop(data: dict) -> HumanInTheLoopWorkload:
    return HumanInTheLoopWorkload(
        system_prompt_tokens=_parse_distribution(data.get("system_prompt_tokens",
            {"type": "lognormal", "mu": 7.6, "sigma": 0.4})),
        num_concurrent_sessions=int(data.get("num_concurrent_sessions", 5)),
        min_rounds=int(data.get("min_rounds", 5)),
        max_rounds=int(data.get("max_rounds", 30)),
        rounds_distribution=_parse_distribution(data.get("rounds_distribution",
            {"type": "lognormal", "mu": 2.5, "sigma": 0.5})),
        user_message_tokens=_parse_distribution(data.get("user_message_tokens",
            {"type": "lognormal", "mu": 5.5, "sigma": 0.8})),
        assistant_response_tokens=_parse_distribution(data.get("assistant_response_tokens",
            {"type": "lognormal", "mu": 6.0, "sigma": 0.6})),
        tool_call_probability=float(data.get("tool_call_probability", 0.6)),
        num_tool_calls_per_round=_parse_distribution(data.get("num_tool_calls_per_round",
            {"type": "poisson", "lam": 2.0, "min_clip": 1})),
        tool_output_tokens=_parse_distribution(data.get("tool_output_tokens",
            {"type": "lognormal", "mu": 6.5, "sigma": 0.7})),
        enable_context_summarization=bool(data.get("enable_context_summarization", False)),
        summarization_trigger_tokens=int(data.get("summarization_trigger_tokens", 16384)),
        summarization_compression_ratio=float(data.get("summarization_compression_ratio", 0.25)),
        enable_old_turn_pruning=bool(data.get("enable_old_turn_pruning", False)),
        pruning_keep_last_n_turns=int(data.get("pruning_keep_last_n_turns", 10)),
        pruning_estimated_tokens_per_turn=int(data.get("pruning_estimated_tokens_per_turn", 1200)),
        enable_tool_output_truncation=bool(data.get("enable_tool_output_truncation", False)),
        tool_output_max_tokens=int(data.get("tool_output_max_tokens", 2000)),
        enable_hard_context_cap=bool(data.get("enable_hard_context_cap", True)),
        hard_context_cap=int(data.get("hard_context_cap", 32768)),
        session_stop_probability_per_round=float(data.get("session_stop_probability_per_round", 0.05)),
    )


def _parse_hybrid(data: dict) -> HybridWorkload:
    fixed = None
    agentic = None
    if "fixed_batch" in data:
        fixed = _parse_fixed_batch(data["fixed_batch"])
    if "agentic" in data:
        agentic = _parse_agentic(data["agentic"])
    return HybridWorkload(fixed_batch=fixed, agentic=agentic)


def _parse_inline_model_config(data: dict[str, Any], scenario_name: str) -> ModelConfig:
    return model_config_from_mapping(data, default_name=scenario_name)


def _resolve_model_config(data: dict[str, Any], scenario_name: str) -> tuple[str, str | None, str | None, bool, ModelConfig | None]:
    model_source = str(data.get("model_source", "builtin"))
    model_id = data.get("model_id")
    model_revision = data.get("model_revision")
    model_local_files_only = bool(data.get("model_local_files_only", False))

    inline_model = data.get("model_config")
    if isinstance(inline_model, dict):
        return model_source, model_id, model_revision, model_local_files_only, _parse_inline_model_config(
            inline_model, scenario_name
        )

    if model_source == "huggingface":
        if not model_id:
            raise ValueError("Scenario sets model_source=huggingface but does not provide model_id.")
        return (
            model_source,
            str(model_id),
            str(model_revision) if model_revision is not None else None,
            model_local_files_only,
            load_model_config_from_huggingface(
                str(model_id),
                revision=str(model_revision) if model_revision is not None else None,
                local_files_only=model_local_files_only,
            ),
        )

    return model_source, str(model_id) if model_id is not None else None, str(model_revision) if model_revision is not None else None, model_local_files_only, None


def load_scenario(path: str) -> SimulationConfig:
    """Load a simulation scenario from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    scenario_name = data.get("scenario_name", "unnamed")
    hardware_catalog_id = data.get("hardware_catalog_id")
    hardware_config = None
    selection_warnings: list[str] = []

    if hardware_catalog_id:
        hardware_config = resolve_hardware_profile(str(hardware_catalog_id))
        selection_warnings.extend(
            validate_catalog_selection(
                str(hardware_catalog_id),
                data.get("software_stack_id"),
                data.get("deployment_profile_id"),
            )
        )

    tp_value = data.get("tensor_parallel_degree")
    if tp_value is None and hardware_catalog_id:
        tensor_parallel_degree = default_tensor_parallel_degree_for_hardware(str(hardware_catalog_id))
    else:
        tensor_parallel_degree = int(tp_value if tp_value is not None else 8)

    model_source, model_id, model_revision, model_local_files_only, model_config = _resolve_model_config(
        data, scenario_name
    )

    hardware_profile = HardwareProfileName(data.get("hardware_profile", "h200_8gpu"))
    if hardware_catalog_id and not data.get("hardware_profile"):
        hardware_profile = HardwareProfileName.H200_8GPU

    config = SimulationConfig(
        scenario_name=scenario_name,
        hardware_profile=hardware_profile,
        hardware_config=hardware_config,
        hardware_catalog_id=str(hardware_catalog_id) if hardware_catalog_id is not None else None,
        weight_precision=WeightPrecision(data.get("weight_precision", "int8")),
        kv_cache_precision=KVCachePrecision(data.get("kv_cache_precision", "fp16")),
        tensor_parallel_degree=tensor_parallel_degree,
        runtime_overhead_ratio=float(data.get("runtime_overhead_ratio", 0.10)),
        max_vram_utilization_target=float(data.get("max_vram_utilization_target", 0.95)),
        safety_headroom_ratio=float(data.get("safety_headroom_ratio", 0.05)),
        workload_type=WorkloadType(data.get("workload_type", "fixed_batch")),
        monte_carlo_enabled=bool(data.get("monte_carlo_enabled", False)),
        monte_carlo_iterations=int(data.get("monte_carlo_iterations", 10000)),
        random_seed=data.get("random_seed", 42),
        percentile_outputs=data.get("percentile_outputs", [50, 90, 95, 99]),
        model_source=model_source,
        model_id=model_id,
        model_revision=model_revision,
        model_local_files_only=model_local_files_only,
        model_config=model_config,
        software_stack_id=data.get("software_stack_id"),
        deployment_profile_id=data.get("deployment_profile_id"),
        selection_warnings=selection_warnings,
    )

    wl = data.get("workload", {})
    if config.workload_type == WorkloadType.FIXED_BATCH:
        config.fixed_batch = _parse_fixed_batch(wl)
    elif config.workload_type == WorkloadType.MIXED_BATCH:
        config.mixed_batch = _parse_mixed_batch(wl)
    elif config.workload_type == WorkloadType.AGENTIC:
        config.agentic = _parse_agentic(wl)
    elif config.workload_type == WorkloadType.HYBRID:
        config.hybrid = _parse_hybrid(wl)
    elif config.workload_type == WorkloadType.HUMAN_IN_THE_LOOP:
        config.human_in_the_loop = _parse_human_in_the_loop(wl)

    return config


def load_scenarios_multi_hardware(path: str) -> list[SimulationConfig]:
    """Load a scenario and create copies for the legacy built-in hardware profiles.

    If the scenario already selects a curated catalog hardware entry, return a
    single resolved config instead of exploding it across the legacy enum set.
    """
    base = load_scenario(path)
    if base.hardware_catalog_id:
        return [base]

    configs = []
    for hw in HardwareProfileName:
        cfg = copy.deepcopy(base)
        cfg.hardware_profile = hw
        if hw.value.endswith("8gpu"):
            cfg.tensor_parallel_degree = 8
        else:
            cfg.tensor_parallel_degree = 4
        cfg.scenario_name = f"{base.scenario_name}__{hw.value}"
        configs.append(cfg)

    return configs
