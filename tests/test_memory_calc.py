"""
Unit tests for memory calculations.
Validates formulas against hand-computed values.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import numpy as np
from src.config import (
    ModelConfig, WeightPrecision, KVCachePrecision,
    DistributionSpec, DistributionType,
)
from src.model_params import QWEN3_235B_A22B
from src.memory_calc import (
    calc_weight_memory_gb, calc_kv_bytes_per_token_per_layer,
    calc_kv_bytes_per_token, calc_total_kv_cache_gb,
    calc_total_vram_gb, calc_per_gpu_vram_gb, check_oom,
    calc_available_kv_cache_gb, calc_max_tokens_from_budget,
    BYTES_PER_GB,
)
from src.hardware import get_hardware_profile
from src.config import HardwareProfileName
from src.distributions import DistributionSampler


class TestWeightMemory(unittest.TestCase):
    """Test weight memory calculations."""

    def test_int8_weights(self):
        """235B params × 1 byte = 235 GB (in decimal GB, ~218.9 GiB)."""
        gb = calc_weight_memory_gb(QWEN3_235B_A22B, WeightPrecision.INT8)
        # 235e9 bytes / 2^30 ≈ 218.84 GiB
        self.assertAlmostEqual(gb, 235e9 / BYTES_PER_GB, places=1)
        print(f"  INT8 weights: {gb:.2f} GB")

    def test_fp16_weights(self):
        """235B params × 2 bytes = 470 GB."""
        gb = calc_weight_memory_gb(QWEN3_235B_A22B, WeightPrecision.FP16)
        self.assertAlmostEqual(gb, 470e9 / BYTES_PER_GB, places=1)
        print(f"  FP16 weights: {gb:.2f} GB")

    def test_int4_weights(self):
        """235B params × 0.5 bytes = 117.5 GB."""
        gb = calc_weight_memory_gb(QWEN3_235B_A22B, WeightPrecision.INT4)
        self.assertAlmostEqual(gb, 117.5e9 / BYTES_PER_GB, places=1)
        print(f"  INT4 weights: {gb:.2f} GB")

    def test_nvfp4_weights(self):
        """235B params × 0.5625 bytes (with scale overhead)."""
        gb = calc_weight_memory_gb(QWEN3_235B_A22B, WeightPrecision.NVFP4)
        expected = 235e9 * 0.5625 / BYTES_PER_GB
        self.assertAlmostEqual(gb, expected, places=1)
        print(f"  NVFP4 weights: {gb:.2f} GB")


class TestKVCache(unittest.TestCase):
    """Test KV cache calculations."""

    def test_kv_per_token_per_layer_fp16(self):
        """2 × 4 × 128 × 2 = 2048 bytes."""
        b = calc_kv_bytes_per_token_per_layer(QWEN3_235B_A22B, KVCachePrecision.FP16)
        self.assertEqual(b, 2048)
        print(f"  KV per token per layer (FP16): {b} bytes")

    def test_kv_per_token_per_layer_int8(self):
        """2 × 4 × 128 × 1 = 1024 bytes."""
        b = calc_kv_bytes_per_token_per_layer(QWEN3_235B_A22B, KVCachePrecision.INT8)
        self.assertEqual(b, 1024)
        print(f"  KV per token per layer (INT8): {b} bytes")

    def test_kv_per_token_all_layers_fp16(self):
        """2048 × 94 = 192,512 bytes ≈ 188 KB."""
        b = calc_kv_bytes_per_token(QWEN3_235B_A22B, KVCachePrecision.FP16)
        self.assertEqual(b, 192_512)
        print(f"  KV per token all layers (FP16): {b} bytes = {b/1024:.1f} KB")

    def test_kv_per_token_all_layers_int8(self):
        """1024 × 94 = 96,256 bytes ≈ 94 KB."""
        b = calc_kv_bytes_per_token(QWEN3_235B_A22B, KVCachePrecision.INT8)
        self.assertEqual(b, 96_256)
        print(f"  KV per token all layers (INT8): {b} bytes = {b/1024:.1f} KB")

    def test_total_kv_cache_10k_tokens_fp16(self):
        """10,000 tokens × 192,512 bytes ≈ 1.79 GB."""
        gb = calc_total_kv_cache_gb(QWEN3_235B_A22B, KVCachePrecision.FP16, 10_000)
        expected = 10_000 * 192_512 / BYTES_PER_GB
        self.assertAlmostEqual(gb, expected, places=3)
        print(f"  KV cache for 10K tokens (FP16): {gb:.3f} GB")

    def test_total_kv_cache_100k_tokens_fp16(self):
        """100,000 tokens × 192,512 bytes ≈ 17.93 GB."""
        gb = calc_total_kv_cache_gb(QWEN3_235B_A22B, KVCachePrecision.FP16, 100_000)
        print(f"  KV cache for 100K tokens (FP16): {gb:.3f} GB")
        self.assertGreater(gb, 15.0)
        self.assertLess(gb, 25.0)


class TestTotalVRAM(unittest.TestCase):
    """Test total VRAM calculations."""

    def test_basic_scenario(self):
        """INT8 weights + small KV cache + 10% overhead."""
        weight_gb = calc_weight_memory_gb(QWEN3_235B_A22B, WeightPrecision.INT8)
        kv_gb = calc_total_kv_cache_gb(QWEN3_235B_A22B, KVCachePrecision.FP16, 17_000)
        total_gb = calc_total_vram_gb(weight_gb, kv_gb, 0.10)

        expected = (weight_gb + kv_gb) * 1.10
        self.assertAlmostEqual(total_gb, expected, places=2)
        print(f"  Basic scenario: weights={weight_gb:.1f} + kv={kv_gb:.2f} + 10% overhead = {total_gb:.1f} GB")

    def test_per_gpu_tp8(self):
        """Per GPU should be total / 8."""
        total_gb = 300.0
        per_gpu = calc_per_gpu_vram_gb(total_gb, 8)
        self.assertAlmostEqual(per_gpu, 37.5, places=1)


class TestOOM(unittest.TestCase):
    """Test OOM condition checks."""

    def test_h200_8gpu_int8_safe(self):
        """8×H200 with INT8 should be safe for weights alone."""
        hw = get_hardware_profile(HardwareProfileName.H200_8GPU)
        weight_gb = calc_weight_memory_gb(QWEN3_235B_A22B, WeightPrecision.INT8)
        per_gpu = calc_per_gpu_vram_gb(weight_gb * 1.10, 8)  # + 10% overhead
        is_oom, limit = check_oom(per_gpu, hw)
        self.assertFalse(is_oom, f"per_gpu={per_gpu:.1f} GB should fit in limit={limit:.1f} GB")
        print(f"  H200 8×GPU INT8 weights per GPU: {per_gpu:.1f} GB, limit: {limit:.1f} GB — SAFE")

    def test_rtx6000_4gpu_fp16_oom(self):
        """4×RTX6000 with FP16 should OOM (470 GB weights > 384 GB total)."""
        hw = get_hardware_profile(HardwareProfileName.RTX6000_4GPU)
        weight_gb = calc_weight_memory_gb(QWEN3_235B_A22B, WeightPrecision.FP16)
        per_gpu = calc_per_gpu_vram_gb(weight_gb * 1.10, 4)
        is_oom, limit = check_oom(per_gpu, hw)
        self.assertTrue(is_oom, f"per_gpu={per_gpu:.1f} GB should NOT fit in limit={limit:.1f} GB")
        print(f"  RTX6000 4×GPU FP16 weights per GPU: {per_gpu:.1f} GB, limit: {limit:.1f} GB — OOM ✓")


class TestDistributions(unittest.TestCase):
    """Test distribution sampling."""

    def test_fixed(self):
        sampler = DistributionSampler(np.random.default_rng(42))
        spec = DistributionSpec.fixed(1200)
        vals = sampler.sample_int(spec, 10)
        self.assertTrue(np.all(vals == 1200))

    def test_lognormal_positive(self):
        sampler = DistributionSampler(np.random.default_rng(42))
        spec = DistributionSpec.lognormal(mu=7.0, sigma=0.3)
        vals = sampler.sample_int(spec, 1000)
        self.assertTrue(np.all(vals >= 1))
        # Median of lognormal(7.0, 0.3) = exp(7.0) ≈ 1097
        median = np.median(vals)
        self.assertGreater(median, 500)
        self.assertLess(median, 2000)

    def test_poisson(self):
        sampler = DistributionSampler(np.random.default_rng(42))
        spec = DistributionSpec.poisson(lam=4.0)
        vals = sampler.sample_int(spec, 10000)
        mean = np.mean(vals)
        self.assertAlmostEqual(mean, 4.0, delta=0.2)


class TestAvailableKVBudget(unittest.TestCase):
    """Test available KV cache budget calculation."""

    def test_h200_8gpu_int8(self):
        hw = get_hardware_profile(HardwareProfileName.H200_8GPU)
        avail = calc_available_kv_cache_gb(
            hw, QWEN3_235B_A22B, WeightPrecision.INT8, 8, 0.10, 0.95, 0.05
        )
        print(f"  H200 8×GPU INT8 available KV budget: {avail:.1f} GB")
        self.assertGreater(avail, 500)  # Should have >500 GB for KV cache

    def test_rtx6000_4gpu_int8(self):
        hw = get_hardware_profile(HardwareProfileName.RTX6000_4GPU)
        avail = calc_available_kv_cache_gb(
            hw, QWEN3_235B_A22B, WeightPrecision.INT8, 4, 0.10, 0.95, 0.05
        )
        print(f"  RTX6000 4×GPU INT8 available KV budget: {avail:.1f} GB")
        # 384 total - ~219 weights - ~22 overhead = ~143 GB for KV
        self.assertGreater(avail, 80)
        self.assertLess(avail, 200)

    def test_max_tokens_h200_8gpu(self):
        hw = get_hardware_profile(HardwareProfileName.H200_8GPU)
        avail = calc_available_kv_cache_gb(
            hw, QWEN3_235B_A22B, WeightPrecision.INT8, 8, 0.10, 0.95, 0.05
        )
        max_tok = calc_max_tokens_from_budget(
            avail, QWEN3_235B_A22B, KVCachePrecision.FP16, 0.10
        )
        print(f"  H200 8×GPU max KV tokens (FP16): {max_tok:,}")
        self.assertGreater(max_tok, 1_000_000)  # Should support >1M tokens


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Qwen3-235B-A22B Memory Calculation Tests")
    print("="*60 + "\n")
    unittest.main(verbosity=2)
