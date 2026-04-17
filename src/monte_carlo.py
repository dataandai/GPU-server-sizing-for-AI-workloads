"""
Monte Carlo simulation engine.

Runs N iterations of the workload simulation, collecting VRAM usage distributions.
Computes percentiles, OOM probability, and KV cache saturation probability.

Monte Carlo Design:
- Each iteration samples fresh token counts from the configured distributions
- Sequences are INDEPENDENT across concurrent requests
- Within agentic runs, turns are SEQUENTIALLY DEPENDENT (KV cache accumulates)
- Tool call count and tool output length are CONDITIONALLY INDEPENDENT given the turn
- OOM probability = count(per_gpu_vram > limit) / total_iterations
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from .config import (
    SimulationConfig, SimulationResult, WorkloadType,
    ModelConfig, HardwareProfile,
)
from .model_params import get_default_model
from .hardware import get_hardware_profile
from .memory_calc import (
    calc_weight_memory_gb, calc_total_kv_cache_gb,
    calc_total_vram_gb, calc_per_gpu_vram_gb, check_oom,
    calc_available_kv_cache_gb, calc_max_tokens_from_budget,
    calc_kv_bytes_per_token_gb,
)
from .distributions import DistributionSampler
from .kv_cache_model import KVCacheModel


class MonteCarloEngine:
    """Monte Carlo simulation engine for VRAM estimation."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.model = config.model_config or get_default_model()
        self.hardware = config.hardware_config or get_hardware_profile(config.hardware_profile)

    def run(self) -> SimulationResult:
        """Run the simulation (deterministic or Monte Carlo)."""
        if self.config.monte_carlo_enabled:
            return self._run_monte_carlo()
        else:
            return self._run_deterministic()

    def _run_deterministic(self) -> SimulationResult:
        """Single deterministic run with fixed values."""
        seed = self.config.random_seed if self.config.random_seed is not None else 42
        rng = np.random.default_rng(seed)
        sampler = DistributionSampler(rng)
        kv_model = KVCacheModel(sampler)

        # Simulate workload
        token_counts = self._simulate_workload(kv_model)
        total_tokens = int(np.sum(token_counts))

        # Calculate memory
        weight_gb = calc_weight_memory_gb(self.model, self.config.weight_precision)
        kv_gb = calc_total_kv_cache_gb(self.model, self.config.kv_cache_precision, total_tokens)
        total_gb = calc_total_vram_gb(weight_gb, kv_gb, self.config.runtime_overhead_ratio)
        per_gpu_gb = calc_per_gpu_vram_gb(total_gb, self.config.tensor_parallel_degree)
        overhead_gb = total_gb - weight_gb - kv_gb

        is_oom, limit = check_oom(
            per_gpu_gb, self.hardware,
            self.config.max_vram_utilization_target,
            self.config.safety_headroom_ratio,
        )

        result = SimulationResult(
            scenario_name=self.config.scenario_name,
            model_name=self.model.name,
            model_id=self.config.model_id,
            model_source=self.config.model_source,
            hardware_profile=self.hardware.name,
            hardware_catalog_id=self.hardware.catalog_id or self.config.hardware_catalog_id,
            software_stack_id=self.config.software_stack_id,
            deployment_profile_id=self.config.deployment_profile_id,
            weight_precision=self.config.weight_precision.value,
            kv_cache_precision=self.config.kv_cache_precision.value,
            tensor_parallel_degree=self.config.tensor_parallel_degree,
            num_gpus=self.hardware.num_gpus,
            vram_per_gpu_gb=self.hardware.vram_per_gpu_gb,
            total_system_vram_gb=self.hardware.total_vram_gb,
            weight_memory_gb=weight_gb,
            kv_cache_vram_mean_gb=kv_gb,
            kv_cache_vram_p50_gb=kv_gb,
            kv_cache_vram_p90_gb=kv_gb,
            kv_cache_vram_p95_gb=kv_gb,
            kv_cache_vram_p99_gb=kv_gb,
            kv_cache_vram_max_gb=kv_gb,
            total_vram_mean_gb=total_gb,
            total_vram_p50_gb=total_gb,
            total_vram_p90_gb=total_gb,
            total_vram_p95_gb=total_gb,
            total_vram_p99_gb=total_gb,
            total_vram_max_gb=total_gb,
            per_gpu_vram_mean_gb=per_gpu_gb,
            per_gpu_vram_p95_gb=per_gpu_gb,
            per_gpu_vram_p99_gb=per_gpu_gb,
            runtime_overhead_mean_gb=overhead_gb,
            estimated_oom_probability=1.0 if is_oom else 0.0,
            kv_cache_saturation_probability=1.0 if is_oom else 0.0,
            vram_exceedance_probability=1.0 if is_oom else 0.0,
            monte_carlo_iterations=1,
            total_concurrent_sequences=len(token_counts),
        )
        result.qualitative_risk_level = self._assess_risk(result)
        result.recommended_max_concurrency = self._estimate_max_concurrency()
        result.notes = self._generate_notes(result)
        return result

    def _run_monte_carlo(self) -> SimulationResult:
        """Run Monte Carlo simulation with N iterations."""
        n_iter = self.config.monte_carlo_iterations
        seed = self.config.random_seed if self.config.random_seed is not None else 42

        weight_gb = calc_weight_memory_gb(self.model, self.config.weight_precision)

        kv_gb_samples = np.zeros(n_iter)
        total_vram_samples = np.zeros(n_iter)
        per_gpu_samples = np.zeros(n_iter)
        oom_count = 0

        _, effective_limit = check_oom(
            0, self.hardware,
            self.config.max_vram_utilization_target,
            self.config.safety_headroom_ratio,
        )

        # Available KV budget for saturation calculation
        available_kv = calc_available_kv_cache_gb(
            self.hardware, self.model, self.config.weight_precision,
            self.config.tensor_parallel_degree,
            self.config.runtime_overhead_ratio,
            self.config.max_vram_utilization_target,
            self.config.safety_headroom_ratio,
        )
        kv_saturation_count = 0

        for i in range(n_iter):
            rng = np.random.default_rng(seed + i)
            sampler = DistributionSampler(rng)
            kv_model = KVCacheModel(sampler)

            token_counts = self._simulate_workload(kv_model)
            total_tokens = int(np.sum(token_counts))
            num_sequences = len(token_counts)

            kv_gb = calc_total_kv_cache_gb(
                self.model, self.config.kv_cache_precision, total_tokens)
            total_gb = calc_total_vram_gb(
                weight_gb, kv_gb, self.config.runtime_overhead_ratio)
            per_gpu_gb = calc_per_gpu_vram_gb(
                total_gb, self.config.tensor_parallel_degree)

            kv_gb_samples[i] = kv_gb
            total_vram_samples[i] = total_gb
            per_gpu_samples[i] = per_gpu_gb

            if per_gpu_gb > effective_limit:
                oom_count += 1

            # KV cache saturation: raw KV exceeds available budget
            kv_with_overhead = kv_gb * (1 + self.config.runtime_overhead_ratio)
            if kv_with_overhead > available_kv:
                kv_saturation_count += 1

        # Compute percentiles
        pcts = self.config.percentile_outputs
        kv_pcts = np.percentile(kv_gb_samples, pcts)
        total_pcts = np.percentile(total_vram_samples, pcts)
        per_gpu_pcts = np.percentile(per_gpu_samples, pcts)

        pct_map = {p: i for i, p in enumerate(pcts)}

        result = SimulationResult(
            scenario_name=self.config.scenario_name,
            model_name=self.model.name,
            model_id=self.config.model_id,
            model_source=self.config.model_source,
            hardware_profile=self.hardware.name,
            hardware_catalog_id=self.hardware.catalog_id or self.config.hardware_catalog_id,
            software_stack_id=self.config.software_stack_id,
            deployment_profile_id=self.config.deployment_profile_id,
            weight_precision=self.config.weight_precision.value,
            kv_cache_precision=self.config.kv_cache_precision.value,
            tensor_parallel_degree=self.config.tensor_parallel_degree,
            num_gpus=self.hardware.num_gpus,
            vram_per_gpu_gb=self.hardware.vram_per_gpu_gb,
            total_system_vram_gb=self.hardware.total_vram_gb,
            weight_memory_gb=weight_gb,
            kv_cache_vram_mean_gb=float(np.mean(kv_gb_samples)),
            kv_cache_vram_p50_gb=float(kv_pcts[pct_map.get(50, 0)]) if 50 in pct_map else 0,
            kv_cache_vram_p90_gb=float(kv_pcts[pct_map.get(90, 0)]) if 90 in pct_map else 0,
            kv_cache_vram_p95_gb=float(kv_pcts[pct_map.get(95, 0)]) if 95 in pct_map else 0,
            kv_cache_vram_p99_gb=float(kv_pcts[pct_map.get(99, 0)]) if 99 in pct_map else 0,
            kv_cache_vram_max_gb=float(np.max(kv_gb_samples)),
            total_vram_mean_gb=float(np.mean(total_vram_samples)),
            total_vram_p50_gb=float(total_pcts[pct_map.get(50, 0)]) if 50 in pct_map else 0,
            total_vram_p90_gb=float(total_pcts[pct_map.get(90, 0)]) if 90 in pct_map else 0,
            total_vram_p95_gb=float(total_pcts[pct_map.get(95, 0)]) if 95 in pct_map else 0,
            total_vram_p99_gb=float(total_pcts[pct_map.get(99, 0)]) if 99 in pct_map else 0,
            total_vram_max_gb=float(np.max(total_vram_samples)),
            per_gpu_vram_mean_gb=float(np.mean(per_gpu_samples)),
            per_gpu_vram_p95_gb=float(np.percentile(per_gpu_samples, 95)),
            per_gpu_vram_p99_gb=float(np.percentile(per_gpu_samples, 99)),
            runtime_overhead_mean_gb=float(np.mean(total_vram_samples) - weight_gb - np.mean(kv_gb_samples)),
            estimated_oom_probability=oom_count / n_iter,
            kv_cache_saturation_probability=kv_saturation_count / n_iter,
            vram_exceedance_probability=oom_count / n_iter,
            monte_carlo_iterations=n_iter,
            total_concurrent_sequences=self._get_total_sequences(),
        )
        result.qualitative_risk_level = self._assess_risk(result)
        result.recommended_max_concurrency = self._estimate_max_concurrency()
        result.notes = self._generate_notes(result)
        return result

    def _simulate_workload(self, kv_model: KVCacheModel) -> np.ndarray:
        """Simulate the configured workload and return per-sequence token counts."""
        wtype = self.config.workload_type

        if wtype == WorkloadType.FIXED_BATCH and self.config.fixed_batch:
            return kv_model.simulate_fixed_batch(self.config.fixed_batch)
        elif wtype == WorkloadType.MIXED_BATCH and self.config.mixed_batch:
            return kv_model.simulate_mixed_batch(self.config.mixed_batch)
        elif wtype == WorkloadType.AGENTIC and self.config.agentic:
            return kv_model.simulate_agentic(self.config.agentic)
        elif wtype == WorkloadType.HYBRID and self.config.hybrid:
            return kv_model.simulate_hybrid(self.config.hybrid)
        elif wtype == WorkloadType.HUMAN_IN_THE_LOOP and self.config.human_in_the_loop:
            return kv_model.simulate_human_in_the_loop(self.config.human_in_the_loop)
        else:
            return np.array([0])

    def _get_total_sequences(self) -> int:
        """Get total concurrent sequences for this workload."""
        wtype = self.config.workload_type
        if wtype == WorkloadType.FIXED_BATCH and self.config.fixed_batch:
            return self.config.fixed_batch.num_parallel_jobs
        elif wtype == WorkloadType.MIXED_BATCH and self.config.mixed_batch:
            return self.config.mixed_batch.total_jobs
        elif wtype == WorkloadType.AGENTIC and self.config.agentic:
            return self.config.agentic.num_parallel_agents
        elif wtype == WorkloadType.HYBRID and self.config.hybrid:
            n = 0
            if self.config.hybrid.fixed_batch:
                n += self.config.hybrid.fixed_batch.num_parallel_jobs
            if self.config.hybrid.agentic:
                n += self.config.hybrid.agentic.num_parallel_agents
            return n
        elif wtype == WorkloadType.HUMAN_IN_THE_LOOP and self.config.human_in_the_loop:
            return self.config.human_in_the_loop.num_concurrent_sessions
        return 0

    def _assess_risk(self, result: SimulationResult) -> str:
        """Qualitative risk assessment based on OOM probability."""
        p = result.estimated_oom_probability
        if p == 0:
            # Check headroom
            utilization = result.per_gpu_vram_p95_gb / result.vram_per_gpu_gb
            if utilization < 0.7:
                return "LOW"
            elif utilization < 0.85:
                return "MODERATE"
            else:
                return "HIGH"
        elif p < 0.01:
            return "MODERATE"
        elif p < 0.05:
            return "HIGH"
        else:
            return "CRITICAL"

    def _estimate_max_concurrency(self) -> int:
        """Estimate maximum safe concurrent sequences."""
        available_kv = calc_available_kv_cache_gb(
            self.hardware, self.model, self.config.weight_precision,
            self.config.tensor_parallel_degree,
            self.config.runtime_overhead_ratio,
            self.config.max_vram_utilization_target,
            self.config.safety_headroom_ratio,
        )
        max_tokens = calc_max_tokens_from_budget(
            available_kv, self.model, self.config.kv_cache_precision,
            self.config.runtime_overhead_ratio,
        )

        # Estimate average tokens per sequence based on workload type
        avg_tokens_per_seq = 2000  # Conservative default
        wtype = self.config.workload_type
        if wtype == WorkloadType.FIXED_BATCH and self.config.fixed_batch:
            fb = self.config.fixed_batch
            avg_tokens_per_seq = int(fb.prompt_tokens.value + fb.max_output_tokens.value)
        elif wtype == WorkloadType.AGENTIC and self.config.agentic:
            # Conservative estimate: initial prompt + 4 turns of tool calling
            avg_tokens_per_seq = 8000
        elif wtype == WorkloadType.HUMAN_IN_THE_LOOP and self.config.human_in_the_loop:
            # HITL sessions accumulate context heavily, conservative estimate 15k
            avg_tokens_per_seq = 15000

        if avg_tokens_per_seq <= 0:
            return 0
        return max(1, max_tokens // avg_tokens_per_seq)

    def _generate_notes(self, result: SimulationResult) -> list[str]:
        """Generate engineering notes about the simulation result."""
        notes = ["⚠ Legacy memory simulation surface: this path is the classic VRAM/KV-cache estimator, separate from the benchmark-aware workload simulator."]
        hw = self.hardware

        # Weight memory analysis
        weight_pct = (result.weight_memory_gb / result.total_system_vram_gb) * 100
        notes.append(f"Model weights occupy {weight_pct:.1f}% of total system VRAM")

        # KV cache budget
        available_kv = calc_available_kv_cache_gb(
            hw, self.model, self.config.weight_precision,
            self.config.tensor_parallel_degree,
            self.config.runtime_overhead_ratio,
            self.config.max_vram_utilization_target,
            self.config.safety_headroom_ratio,
        )
        notes.append(f"Available KV cache budget: {available_kv:.1f} GB")

        max_tokens = calc_max_tokens_from_budget(
            available_kv, self.model, self.config.kv_cache_precision,
            self.config.runtime_overhead_ratio,
        )
        notes.append(f"Max total tokens in KV cache: {max_tokens:,}")

        # Catalog / runtime selection notes
        if self.config.model_source != "builtin":
            source_label = self.config.model_source
            if self.config.model_id:
                notes.append(f"Model source: {source_label} ({self.config.model_id})")
            else:
                notes.append(f"Model source: {source_label}")

        if hw.catalog_id:
            notes.append(f"Catalog hardware ID: {hw.catalog_id}")

        if self.config.software_stack_id:
            notes.append(f"Software stack: {self.config.software_stack_id}")

        if self.config.deployment_profile_id:
            notes.append(f"Deployment profile: {self.config.deployment_profile_id}")

        for warning in self.config.selection_warnings:
            notes.append(f"⚠ {warning}")

        # Interconnect note
        if hw.interconnect == "pcie5":
            notes.append("⚠ PCIe 5.0 interconnect — TP communication overhead "
                        "will reduce effective batch throughput vs NVLink")

        # HITL specific notes
        if self.config.workload_type == WorkloadType.HUMAN_IN_THE_LOOP and self.config.human_in_the_loop:
            notes.append("ℹ Human-in-the-loop: Cache grows per-session continuously across user/assistant/tool rounds.")
            if not self.config.human_in_the_loop.enable_context_summarization \
               and not self.config.human_in_the_loop.enable_old_turn_pruning:
                notes.append("⚠ No context governance enabled (summarization/pruning) — high risk of unchecked KV growth.")

        # Tight fit warning
        if result.per_gpu_vram_p95_gb > hw.vram_per_gpu_gb * 0.85:
            notes.append("⚠ Per-GPU VRAM utilization >85% at p95 — high OOM risk under load spikes")

        return notes
