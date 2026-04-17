from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import TestCase

import numpy as np

from src.config import (
    AgenticWorkload,
    DistributionSpec,
    FixedBatchWorkload,
    HumanInTheLoopWorkload,
    HybridWorkload,
    MixedBatchClass,
    MixedBatchWorkload,
)
from src.distributions import DistributionSampler
from src.kv_cache_model import KVCacheModel
from src.reporting import result_to_dict
from src.simulator import run_scenario_file


class TestLegacyMemorySmokeScenarios(TestCase):
    def test_fixed_batch_legacy_scenario_smoke(self):
        result = run_scenario_file("scenarios/scenario_01_fixed_batch.yaml")
        payload = result_to_dict(result)
        self.assertEqual(payload["result_type"], "memory_simulation")
        self.assertEqual(payload["surface_status"], "deprecated_legacy_memory_engine")
        self.assertGreaterEqual(payload["recommended_max_concurrency"], 1)
        self.assertGreaterEqual(payload["estimated_oom_probability"], 0.0)
        self.assertLessEqual(payload["estimated_oom_probability"], 1.0)
        self.assertGreaterEqual(payload["kv_cache_saturation_probability"], 0.0)
        self.assertLessEqual(payload["kv_cache_saturation_probability"], 1.0)
        self.assertTrue(any("Legacy memory simulation surface" in note for note in payload["notes"]))

    def test_hitl_legacy_scenario_smoke(self):
        result = run_scenario_file("scenarios/scenario_09_hitl_light.yaml")
        payload = result_to_dict(result)
        self.assertEqual(payload["result_type"], "memory_simulation")
        self.assertGreater(payload["total_concurrent_sequences"], 0)
        self.assertGreaterEqual(payload["recommended_max_concurrency"], 1)
        self.assertLessEqual(payload["per_gpu_vram_p95_gb"], payload["vram_per_gpu_gb"] * 2)


class TestKVCacheModelDeterministicBehavior(TestCase):
    def setUp(self):
        self.sampler = DistributionSampler(np.random.default_rng(123))
        self.model = KVCacheModel(self.sampler)

    def test_fixed_batch_returns_expected_sequence_lengths(self):
        workload = FixedBatchWorkload(
            prompt_tokens=DistributionSpec.fixed(100),
            max_output_tokens=DistributionSpec.fixed(20),
            num_parallel_jobs=4,
        )
        peaks = self.model.simulate_fixed_batch(workload)
        self.assertEqual(peaks.tolist(), [120, 120, 120, 120])
        self.assertEqual(self.model.get_total_active_tokens(peaks), 480)
        self.assertEqual(self.model.get_max_sequence_length(peaks), 120)

    def test_mixed_batch_concatenates_all_classes(self):
        workload = MixedBatchWorkload(classes=[
            MixedBatchClass(label="small", prompt_tokens=DistributionSpec.fixed(50), max_output_tokens=DistributionSpec.fixed(10), count=2),
            MixedBatchClass(label="large", prompt_tokens=DistributionSpec.fixed(200), max_output_tokens=DistributionSpec.fixed(40), count=1),
        ])
        peaks = self.model.simulate_mixed_batch(workload)
        self.assertEqual(peaks.tolist(), [60, 60, 240])

    def test_agentic_peak_tracks_pre_cap_worst_moment(self):
        workload = AgenticWorkload(
            initial_prompt_tokens=DistributionSpec.fixed(1000),
            assistant_response_tokens=DistributionSpec.fixed(900),
            tool_result_tokens=DistributionSpec.fixed(700),
            tool_call_probability=1.0,
            stop_probability_per_turn=0.0,
            max_turns_per_run=10,
            enable_max_context_cap=True,
            max_context_cap=2500,
            max_total_tokens_per_run=12000,
            num_parallel_agents=1,
        )
        peak = self.model.simulate_agentic_single(workload)
        self.assertEqual(peak, 4100)

    def test_human_in_the_loop_peak_tracks_pre_governance_worst_moment(self):
        workload = HumanInTheLoopWorkload(
            system_prompt_tokens=DistributionSpec.fixed(500),
            rounds_distribution=DistributionSpec.fixed(6),
            min_rounds=6,
            max_rounds=6,
            user_message_tokens=DistributionSpec.fixed(300),
            assistant_response_tokens=DistributionSpec.fixed(400),
            tool_call_probability=1.0,
            num_tool_calls_per_round=DistributionSpec.fixed(2),
            tool_output_tokens=DistributionSpec.fixed(900),
            enable_tool_output_truncation=True,
            tool_output_max_tokens=250,
            enable_hard_context_cap=True,
            hard_context_cap=1600,
            enable_context_summarization=False,
            session_stop_probability_per_round=0.0,
        )
        peak = self.model.simulate_human_in_the_loop_single(workload)
        self.assertEqual(peak, 2800)

    def test_hybrid_combines_fixed_and_agentic_sequences(self):
        workload = HybridWorkload(
            fixed_batch=FixedBatchWorkload(
                prompt_tokens=DistributionSpec.fixed(64),
                max_output_tokens=DistributionSpec.fixed(32),
                num_parallel_jobs=2,
            ),
            agentic=AgenticWorkload(
                initial_prompt_tokens=DistributionSpec.fixed(200),
                assistant_response_tokens=DistributionSpec.fixed(50),
                tool_result_tokens=DistributionSpec.fixed(0),
                tool_call_probability=0.0,
                stop_probability_per_turn=1.0,
                max_turns_per_run=1,
                num_parallel_agents=1,
                enable_max_context_cap=False,
            ),
        )
        peaks = self.model.simulate_hybrid(workload)
        self.assertEqual(peaks.tolist(), [96, 96, 250])


class TestRunAllCatalogAuditCLI(TestCase):
    def test_catalog_audit_cli_prints_summary(self):
        repo_root = Path(__file__).resolve().parent.parent
        proc = subprocess.run(
            [sys.executable, "run_all.py", "--catalog-audit"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Catalog integrity audit", proc.stdout)
        self.assertIn("Errors:", proc.stdout)
        self.assertIn("Warnings:", proc.stdout)
