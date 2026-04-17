"""
Qwen3-235B-A22B VRAM & KV Cache Monte Carlo Simulator
Configuration dataclasses for all simulation parameters.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any


# ─── Enums ───────────────────────────────────────────────────────────────────

class WeightPrecision(Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    NVFP4 = "nvfp4"

    @property
    def bytes_per_param(self) -> float:
        mapping = {
            "fp16": 2.0,
            "bf16": 2.0,
            "int8": 1.0,
            "int4": 0.5,
            "nvfp4": 0.5625,  # 0.5 + ~12.5% scale metadata overhead
        }
        return mapping[self.value]


class KVCachePrecision(Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"

    @property
    def bytes_per_element(self) -> float:
        mapping = {
            "fp16": 2.0,
            "bf16": 2.0,
            "fp8": 1.0,
            "int8": 1.0,
        }
        return mapping[self.value]


class HardwareProfileName(Enum):
    H200_8GPU = "h200_8gpu"
    H200_4GPU = "h200_4gpu"
    RTX6000_8GPU = "rtx6000_8gpu"
    RTX6000_4GPU = "rtx6000_4gpu"


class DistributionType(Enum):
    FIXED = "fixed"
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    POISSON = "poisson"
    EMPIRICAL = "empirical"


class WorkloadType(Enum):
    FIXED_BATCH = "fixed_batch"
    MIXED_BATCH = "mixed_batch"
    AGENTIC = "agentic"
    HYBRID = "hybrid"
    HUMAN_IN_THE_LOOP = "human_in_the_loop"


# ─── Distribution Specification ──────────────────────────────────────────────

@dataclass
class DistributionSpec:
    """Specifies a probability distribution for Monte Carlo sampling."""
    type: DistributionType = DistributionType.FIXED
    value: float = 0.0            # For FIXED
    low: float = 0.0              # For UNIFORM
    high: float = 0.0             # For UNIFORM
    mean: float = 0.0             # For NORMAL
    std: float = 1.0              # For NORMAL
    mu: float = 0.0               # For LOGNORMAL (log-space mean)
    sigma: float = 1.0            # For LOGNORMAL (log-space std)
    lam: float = 1.0              # For POISSON (lambda)
    values: list[float] = field(default_factory=list)       # For EMPIRICAL
    probabilities: list[float] = field(default_factory=list) # For EMPIRICAL
    min_clip: Optional[float] = None  # Optional lower bound clipping
    max_clip: Optional[float] = None  # Optional upper bound clipping

    @staticmethod
    def fixed(value: float) -> DistributionSpec:
        return DistributionSpec(type=DistributionType.FIXED, value=value)

    @staticmethod
    def uniform(low: float, high: float) -> DistributionSpec:
        return DistributionSpec(type=DistributionType.UNIFORM, low=low, high=high)

    @staticmethod
    def normal(mean: float, std: float, min_clip: float = 1) -> DistributionSpec:
        return DistributionSpec(type=DistributionType.NORMAL, mean=mean, std=std, min_clip=min_clip)

    @staticmethod
    def lognormal(mu: float, sigma: float, min_clip: float = 1) -> DistributionSpec:
        return DistributionSpec(type=DistributionType.LOGNORMAL, mu=mu, sigma=sigma, min_clip=min_clip)

    @staticmethod
    def poisson(lam: float, min_clip: float = 0) -> DistributionSpec:
        return DistributionSpec(type=DistributionType.POISSON, lam=lam, min_clip=min_clip)


# ─── Hardware Profile ────────────────────────────────────────────────────────

@dataclass
class HardwareProfile:
    name: str
    num_gpus: int
    vram_per_gpu_gb: float
    memory_bandwidth_gbps: float
    interconnect: str  # "nvlink" or "pcie5"
    notes: str = ""
    catalog_id: Optional[str] = None
    vendor: Optional[str] = None
    gpu_id: Optional[str] = None
    topology_class: Optional[str] = None
    software_stack_ids: list[str] = field(default_factory=list)
    deployment_profile_ids: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)

    @property
    def total_vram_gb(self) -> float:
        return self.num_gpus * self.vram_per_gpu_gb


# ─── Model Architecture Config ───────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Qwen3-235B-A22B architecture parameters."""
    name: str = "Qwen3-235B-A22B"
    total_params_billions: float = 235.0
    active_params_billions: float = 22.0
    hidden_size: int = 4096
    head_dim: int = 128
    num_hidden_layers: int = 94
    num_attention_heads: int = 64     # Query heads
    num_key_value_heads: int = 4      # KV heads (GQA)
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1536
    intermediate_size: int = 12288    # Shared dense MLP
    vocab_size: int = 151936
    max_position_embeddings: int = 40960
    is_moe: bool = True


# ─── Workload Configs ─────────────────────────────────────────────────────────

@dataclass
class FixedBatchWorkload:
    """Fixed batch inference: all sequences have same token counts."""
    prompt_tokens: DistributionSpec = field(default_factory=lambda: DistributionSpec.fixed(1200))
    max_output_tokens: DistributionSpec = field(default_factory=lambda: DistributionSpec.fixed(500))
    num_parallel_jobs: int = 10


@dataclass
class MixedBatchClass:
    """One class within a mixed batch workload."""
    label: str = "medium"
    prompt_tokens: DistributionSpec = field(default_factory=lambda: DistributionSpec.fixed(1024))
    max_output_tokens: DistributionSpec = field(default_factory=lambda: DistributionSpec.fixed(512))
    count: int = 5


@dataclass
class MixedBatchWorkload:
    """Mixed batch: multiple classes of sequences."""
    classes: list[MixedBatchClass] = field(default_factory=list)

    @property
    def total_jobs(self) -> int:
        return sum(c.count for c in self.classes)


@dataclass
class AgenticWorkload:
    """Agentic inference workload with multi-turn tool-calling sequences."""
    initial_prompt_tokens: DistributionSpec = field(
        default_factory=lambda: DistributionSpec.lognormal(mu=7.0, sigma=0.3))
    max_turns_per_run: int = 10
    tool_call_probability: float = 0.7
    tool_result_tokens: DistributionSpec = field(
        default_factory=lambda: DistributionSpec.lognormal(mu=6.0, sigma=0.7))
    assistant_response_tokens: DistributionSpec = field(
        default_factory=lambda: DistributionSpec.lognormal(mu=5.5, sigma=0.5))
    summarization_probability: float = 0.0
    summarization_compression_ratio: float = 0.3
    stop_probability_per_turn: float = 0.15
    max_total_tokens_per_run: int = 32768
    num_parallel_agents: int = 10
    # Context management
    enable_context_reset: bool = False
    context_reset_threshold: int = 16384
    enable_max_context_cap: bool = True
    max_context_cap: int = 32768


@dataclass
class HumanInTheLoopWorkload:
    """Human-in-the-loop interactive coding session workload.

    Models long-lived sessions where context grows continuously:
    - Large static system prompt (rules, repo context, file excerpts)
    - Multi-round dialogue (user <-> assistant)
    - Tool outputs injected mid-conversation (builds, tests, file reads)
    - Optional context governance (summarization, hard caps, old-turn pruning)

    This is fundamentally different from batch or standard agentic workloads:
    - Sessions are long-lived and accumulate context over many rounds
    - The KV cache is the primary bottleneck (not throughput)
    - Each round adds: user_message + assistant_response + tool_outputs
    - Without governance, a session can easily reach 30,000+ tokens
    """
    # Session structure
    system_prompt_tokens: DistributionSpec = field(
        default_factory=lambda: DistributionSpec.lognormal(mu=7.6, sigma=0.4))  # ~2000 tokens median
    num_concurrent_sessions: int = 5

    # Round dynamics
    min_rounds: int = 5
    max_rounds: int = 30
    rounds_distribution: DistributionSpec = field(
        default_factory=lambda: DistributionSpec.lognormal(mu=2.5, sigma=0.5))  # ~12 rounds median

    # Per-round token components
    user_message_tokens: DistributionSpec = field(
        default_factory=lambda: DistributionSpec.lognormal(mu=5.5, sigma=0.8))  # ~245 tokens median
    assistant_response_tokens: DistributionSpec = field(
        default_factory=lambda: DistributionSpec.lognormal(mu=6.0, sigma=0.6))  # ~400 tokens median

    # Tool interactions (per round)
    tool_call_probability: float = 0.6
    num_tool_calls_per_round: DistributionSpec = field(
        default_factory=lambda: DistributionSpec.poisson(lam=2.0, min_clip=1))  # 1-5 tool calls
    tool_output_tokens: DistributionSpec = field(
        default_factory=lambda: DistributionSpec.lognormal(mu=6.5, sigma=0.7))  # ~665 tokens median

    # Context governance strategies
    enable_context_summarization: bool = False
    summarization_trigger_tokens: int = 16384
    summarization_compression_ratio: float = 0.25  # Keep 25% of old context

    enable_old_turn_pruning: bool = False
    pruning_keep_last_n_turns: int = 10
    pruning_estimated_tokens_per_turn: int = 1200  # Avg tokens per pruned turn

    enable_tool_output_truncation: bool = False
    tool_output_max_tokens: int = 2000

    enable_hard_context_cap: bool = True
    hard_context_cap: int = 32768

    # Session lifetime
    session_stop_probability_per_round: float = 0.05


@dataclass
class HybridWorkload:
    """Hybrid: fixed batch + agentic running concurrently."""
    fixed_batch: Optional[FixedBatchWorkload] = None
    agentic: Optional[AgenticWorkload] = None


# ─── Simulation Config ────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""
    scenario_name: str = "default"
    hardware_profile: HardwareProfileName = HardwareProfileName.H200_8GPU
    hardware_config: Optional[HardwareProfile] = None
    hardware_catalog_id: Optional[str] = None
    weight_precision: WeightPrecision = WeightPrecision.INT8
    kv_cache_precision: KVCachePrecision = KVCachePrecision.FP16
    tensor_parallel_degree: int = 8
    runtime_overhead_ratio: float = 0.10
    max_vram_utilization_target: float = 0.95
    safety_headroom_ratio: float = 0.05

    # Workload
    workload_type: WorkloadType = WorkloadType.FIXED_BATCH
    fixed_batch: Optional[FixedBatchWorkload] = None
    mixed_batch: Optional[MixedBatchWorkload] = None
    agentic: Optional[AgenticWorkload] = None
    hybrid: Optional[HybridWorkload] = None
    human_in_the_loop: Optional[HumanInTheLoopWorkload] = None

    # Monte Carlo
    monte_carlo_enabled: bool = False
    monte_carlo_iterations: int = 10000
    random_seed: Optional[int] = 42
    percentile_outputs: list[float] = field(default_factory=lambda: [50, 90, 95, 99])

    # Dynamic model / catalog selection
    model_source: str = "builtin"
    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    model_local_files_only: bool = False

    # Model override (optional)
    model_config: Optional[ModelConfig] = None

    # Software/runtime selection metadata
    software_stack_id: Optional[str] = None
    deployment_profile_id: Optional[str] = None
    selection_warnings: list[str] = field(default_factory=list)


# ─── Simulation Output ────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Output of a simulation run."""
    scenario_name: str = ""
    model_name: str = ""
    model_id: Optional[str] = None
    model_source: str = "builtin"
    hardware_profile: str = ""
    hardware_catalog_id: Optional[str] = None
    software_stack_id: Optional[str] = None
    deployment_profile_id: Optional[str] = None
    weight_precision: str = ""
    kv_cache_precision: str = ""
    tensor_parallel_degree: int = 0
    num_gpus: int = 0
    vram_per_gpu_gb: float = 0.0
    total_system_vram_gb: float = 0.0

    # Weight memory
    weight_memory_gb: float = 0.0

    # KV cache stats
    kv_cache_vram_mean_gb: float = 0.0
    kv_cache_vram_p50_gb: float = 0.0
    kv_cache_vram_p90_gb: float = 0.0
    kv_cache_vram_p95_gb: float = 0.0
    kv_cache_vram_p99_gb: float = 0.0
    kv_cache_vram_max_gb: float = 0.0

    # Total VRAM stats
    total_vram_mean_gb: float = 0.0
    total_vram_p50_gb: float = 0.0
    total_vram_p90_gb: float = 0.0
    total_vram_p95_gb: float = 0.0
    total_vram_p99_gb: float = 0.0
    total_vram_max_gb: float = 0.0

    # Per-GPU stats
    per_gpu_vram_mean_gb: float = 0.0
    per_gpu_vram_p95_gb: float = 0.0
    per_gpu_vram_p99_gb: float = 0.0

    # Runtime overhead
    runtime_overhead_mean_gb: float = 0.0

    # Risk metrics
    estimated_oom_probability: float = 0.0
    kv_cache_saturation_probability: float = 0.0
    vram_exceedance_probability: float = 0.0
    recommended_max_concurrency: Optional[int] = None
    qualitative_risk_level: str = "UNKNOWN"  # LOW / MODERATE / HIGH / CRITICAL

    # Metadata
    monte_carlo_iterations: int = 0
    total_concurrent_sequences: int = 0
    notes: list[str] = field(default_factory=list)
