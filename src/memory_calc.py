"""
Core VRAM calculation functions.

Mathematical Model:
    total_weight_memory = total_params × bytes_per_weight
    kv_per_token_per_layer = 2 × num_kv_heads × head_dim × bytes_per_kv_element
    kv_per_token = kv_per_token_per_layer × num_layers
    total_kv_cache = sum(seq_tokens_i) × kv_per_token
    runtime_overhead = (weights + kv_cache) × overhead_ratio
    total_vram = weights + kv_cache + overhead
    per_gpu_vram = total_vram / tensor_parallel_degree
"""

from __future__ import annotations
from .config import ModelConfig, WeightPrecision, KVCachePrecision, HardwareProfile


# ─── Constants ────────────────────────────────────────────────────────────────

BYTES_PER_GB = 1_073_741_824  # 2^30
BYTES_PER_GIB = BYTES_PER_GB  # We use GiB throughout for VRAM


# ─── Weight Memory ────────────────────────────────────────────────────────────

def calc_weight_memory_bytes(model: ModelConfig, precision: WeightPrecision) -> float:
    """
    Calculate total weight memory in bytes.

    For MoE models, ALL expert weights must be loaded into VRAM,
    not just the active experts. The total_params_billions figure
    includes all experts.

    Formula: total_params × bytes_per_param
    """
    total_params = model.total_params_billions * 1e9
    return total_params * precision.bytes_per_param


def calc_weight_memory_gb(model: ModelConfig, precision: WeightPrecision) -> float:
    """Calculate total weight memory in GB."""
    return calc_weight_memory_bytes(model, precision) / BYTES_PER_GB


# ─── KV Cache Memory ─────────────────────────────────────────────────────────

def calc_kv_bytes_per_token_per_layer(model: ModelConfig, kv_precision: KVCachePrecision) -> float:
    """
    Calculate KV cache bytes per token per layer.

    Formula: 2 × num_kv_heads × head_dim × bytes_per_element
    The factor of 2 accounts for both Key and Value tensors.

    With GQA (num_kv_heads=4, head_dim=128):
      FP16: 2 × 4 × 128 × 2 = 2048 bytes
      INT8: 2 × 4 × 128 × 1 = 1024 bytes
    """
    return (
        2  # K + V
        * model.num_key_value_heads
        * model.head_dim
        * kv_precision.bytes_per_element
    )


def calc_kv_bytes_per_token(model: ModelConfig, kv_precision: KVCachePrecision) -> float:
    """
    Calculate KV cache bytes per token across ALL layers.

    Formula: kv_per_token_per_layer × num_layers

    For Qwen3-235B (94 layers, 4 KV heads, head_dim=128):
      FP16: 2048 × 94 = 192,512 bytes ≈ 188 KB/token
      INT8: 1024 × 94 =  96,256 bytes ≈  94 KB/token
    """
    return calc_kv_bytes_per_token_per_layer(model, kv_precision) * model.num_hidden_layers


def calc_kv_bytes_per_token_gb(model: ModelConfig, kv_precision: KVCachePrecision) -> float:
    """KV cache memory per token in GB."""
    return calc_kv_bytes_per_token(model, kv_precision) / BYTES_PER_GB


def calc_total_kv_cache_bytes(
    model: ModelConfig,
    kv_precision: KVCachePrecision,
    total_active_tokens: int,
) -> float:
    """
    Calculate total KV cache memory for all active sequences.

    Formula: total_active_tokens × kv_bytes_per_token

    where total_active_tokens = sum of current sequence lengths
    across all concurrent requests/agents.
    """
    per_token = calc_kv_bytes_per_token(model, kv_precision)
    return total_active_tokens * per_token


def calc_total_kv_cache_gb(
    model: ModelConfig,
    kv_precision: KVCachePrecision,
    total_active_tokens: int,
) -> float:
    """Total KV cache memory in GB."""
    return calc_total_kv_cache_bytes(model, kv_precision, total_active_tokens) / BYTES_PER_GB


# ─── Runtime Overhead ─────────────────────────────────────────────────────────

def calc_runtime_overhead_gb(
    weight_memory_gb: float,
    kv_cache_gb: float,
    overhead_ratio: float = 0.10,
) -> float:
    """
    Calculate runtime overhead in GB.

    Includes: CUDA context (~0.5-1 GB), framework buffers, activation memory,
    scheduler state, memory allocator fragmentation.

    [ASSUMPTION] Modeled as proportional to (weights + kv_cache).
    This is a simplification; in practice CUDA context is fixed (~0.5-1 GB)
    and activations scale with batch size. The proportional model tends to be
    conservative for large models.

    Formula: (weight_memory + kv_cache) × overhead_ratio
    """
    return (weight_memory_gb + kv_cache_gb) * overhead_ratio


# ─── Total VRAM ──────────────────────────────────────────────────────────────

def calc_total_vram_gb(
    weight_memory_gb: float,
    kv_cache_gb: float,
    overhead_ratio: float = 0.10,
) -> float:
    """
    Calculate total VRAM required.

    Formula: weights + kv_cache + overhead
           = weights + kv_cache + (weights + kv_cache) × overhead_ratio
           = (weights + kv_cache) × (1 + overhead_ratio)
    """
    base = weight_memory_gb + kv_cache_gb
    return base * (1 + overhead_ratio)


def calc_per_gpu_vram_gb(total_vram_gb: float, tensor_parallel_degree: int) -> float:
    """
    Calculate per-GPU VRAM requirement.

    With tensor parallelism, model weights and KV cache are sharded
    evenly across GPUs.

    Formula: total_vram / TP_degree
    """
    return total_vram_gb / tensor_parallel_degree


# ─── OOM Check ────────────────────────────────────────────────────────────────

def check_oom(
    per_gpu_vram_gb: float,
    hardware: HardwareProfile,
    max_utilization: float = 0.95,
    safety_headroom: float = 0.05,
) -> tuple[bool, float]:
    """
    Check if the configuration would cause an Out-of-Memory condition.

    OOM condition: per_gpu_vram > gpu_memory × max_utilization × (1 - safety_headroom)

    Returns:
        (is_oom, effective_limit_gb)
    """
    effective_limit = hardware.vram_per_gpu_gb * max_utilization * (1 - safety_headroom)
    is_oom = per_gpu_vram_gb > effective_limit
    return is_oom, effective_limit


def calc_available_kv_cache_gb(
    hardware: HardwareProfile,
    model: ModelConfig,
    weight_precision: WeightPrecision,
    tensor_parallel_degree: int,
    overhead_ratio: float = 0.10,
    max_utilization: float = 0.95,
    safety_headroom: float = 0.05,
) -> float:
    """
    Calculate available VRAM budget for KV cache after weights and overhead.

    Formula:
        effective_limit = gpu_memory × max_util × (1 - headroom) × TP_degree
        weight_mem = total model weights
        weight_overhead = weight_mem × overhead_ratio
        available = effective_limit - weight_mem - weight_overhead

    This is the maximum total KV cache size before OOM.
    """
    effective_per_gpu = hardware.vram_per_gpu_gb * max_utilization * (1 - safety_headroom)
    effective_total = effective_per_gpu * tensor_parallel_degree

    weight_gb = calc_weight_memory_gb(model, weight_precision)
    # Overhead on weights alone (base overhead, before KV cache adds more)
    weight_overhead = weight_gb * overhead_ratio

    available = effective_total - weight_gb - weight_overhead
    return max(0.0, available)


def calc_max_tokens_from_budget(
    available_kv_gb: float,
    model: ModelConfig,
    kv_precision: KVCachePrecision,
    overhead_ratio: float = 0.10,
) -> int:
    """
    Calculate maximum total tokens that can fit in the KV cache budget.

    The KV cache itself also incurs overhead, so:
        available_kv_gb = kv_cache_gb × (1 + overhead_ratio)
        kv_cache_gb = available_kv_gb / (1 + overhead_ratio)
        max_tokens = kv_cache_gb / kv_per_token_gb
    """
    effective_kv_budget = available_kv_gb / (1 + overhead_ratio)
    kv_per_token_gb = calc_kv_bytes_per_token_gb(model, kv_precision)
    if kv_per_token_gb <= 0:
        return 0
    return int(effective_kv_budget / kv_per_token_gb)
