from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase

from src.calibration_rules import calibration_coverage_summary
from src.config import SimulationResult
from src.procurement import derive_workload_procurement_metrics, summarize_workload_results_for_procurement
from src.report_payload import build_customer_report_payload
from src.reporting import format_detailed_report, result_to_dict, results_to_json
from src.workload_simulation import WorkloadSimulationEngine, WorkloadSimulationResult


class TestProcurementDecisionBands(TestCase):
    def _make_result(self, **overrides):
        payload = {
            'throughput_per_sec_mean': 2.0,
            'throughput_per_sec_p50': 2.0,
            'configured_arrival_rate_per_sec': 1.0,
            'gpu_util_p95': 0.55,
            'cpu_util_p95': 0.2,
            'sla_hit_rate_mean': 0.995,
            'drop_rate_mean': 0.0,
            'qualitative_risk_level': 'LOW',
            'planning_profile': 'baseline',
            'reserved_capacity_fraction': 0.1,
        }
        payload.update(overrides)
        return SimpleNamespace(**payload)

    def test_procurement_band_unsuitable_when_safe_capacity_non_positive(self):
        metrics = derive_workload_procurement_metrics(self._make_result(throughput_per_sec_p50=0.0, throughput_per_sec_mean=0.0, sla_hit_rate_mean=0.0))
        self.assertEqual(metrics['procurement_band'], 'UNSUITABLE')

    def test_procurement_band_upsize_required_below_target_margin(self):
        metrics = derive_workload_procurement_metrics(self._make_result(configured_arrival_rate_per_sec=3.0))
        self.assertEqual(metrics['procurement_band'], 'UPSIZE_REQUIRED')

    def test_procurement_band_risky_for_high_risk_or_util(self):
        metrics = derive_workload_procurement_metrics(self._make_result(qualitative_risk_level='HIGH', configured_arrival_rate_per_sec=0.5))
        self.assertEqual(metrics['procurement_band'], 'RISKY')
        metrics = derive_workload_procurement_metrics(self._make_result(gpu_util_p95=0.91, configured_arrival_rate_per_sec=0.5))
        self.assertEqual(metrics['procurement_band'], 'RISKY')

    def test_procurement_band_safe_24x7_for_safe_profile_with_target_fit(self):
        metrics = derive_workload_procurement_metrics(self._make_result(planning_profile='safe_24x7', configured_arrival_rate_per_sec=1.0, throughput_per_sec_p50=3.0))
        self.assertEqual(metrics['procurement_band'], 'SAFE_24X7')

    def test_procurement_band_fit_when_target_is_met(self):
        metrics = derive_workload_procurement_metrics(self._make_result(configured_arrival_rate_per_sec=1.0, throughput_per_sec_p50=2.5))
        self.assertEqual(metrics['procurement_band'], 'FIT')
        self.assertTrue(metrics['target_fit'])

    def test_procurement_band_tune_or_scale_when_positive_capacity_but_no_target(self):
        metrics = derive_workload_procurement_metrics(self._make_result(configured_arrival_rate_per_sec=0.0, throughput_per_sec_p50=1.0))
        self.assertEqual(metrics['procurement_band'], 'TUNE_OR_SCALE')
        self.assertIsNone(metrics['target_fit'])

    def test_safe_capacity_is_never_above_steady_state(self):
        metrics = derive_workload_procurement_metrics(self._make_result(throughput_per_sec_p50=4.0, reserved_capacity_fraction=0.2))
        self.assertLessEqual(metrics['safe_capacity_24x7_per_sec'], metrics['steady_state_capacity_per_sec'])
        self.assertLessEqual(metrics['recommended_max_arrival_rate_per_sec'], metrics['safe_capacity_24x7_per_sec'])

    def test_procurement_summary_picks_expected_records(self):
        rows = [
            {'result_type': 'workload_simulation', 'scenario_name': 'a', 'throughput_per_sec_mean': 1.0, 'safe_capacity_24x7_per_sec': 0.8, 'latency_p95_ms': 150, 'qualitative_risk_level': 'LOW'},
            {'result_type': 'workload_simulation', 'scenario_name': 'b', 'throughput_per_sec_mean': 1.2, 'safe_capacity_24x7_per_sec': 0.7, 'latency_p95_ms': 90, 'qualitative_risk_level': 'HIGH'},
            {'result_type': 'workload_simulation', 'scenario_name': 'c', 'throughput_per_sec_mean': 0.9, 'safe_capacity_24x7_per_sec': 0.9, 'latency_p95_ms': 100, 'qualitative_risk_level': 'LOW'},
        ]
        summary = summarize_workload_results_for_procurement(rows)
        self.assertEqual(summary['best_throughput']['scenario_name'], 'b')
        self.assertEqual(summary['best_safe_24x7']['scenario_name'], 'c')
        self.assertEqual(summary['lowest_latency']['scenario_name'], 'b')
        self.assertEqual(summary['safest']['scenario_name'], 'c')


class TestReportingAndPayloadConsistency(TestCase):
    def _workload_result(self, **overrides) -> WorkloadSimulationResult:
        base = WorkloadSimulationResult(
            scenario_name='report_case',
            workload_id='lib_retrieval_rerank_generation',
            workload_class='retrieval_augmented',
            workload_family='retrieval_rerank_generation',
            runtime_family='vllm',
            runtime_mode='vllm_continuous_batching',
            execution_mode='container_on_bare_metal',
            hardware_catalog_id='nvidia_dgx_h200_8gpu',
            software_stack_id='nvidia_vllm_cuda',
            planning_profile='baseline',
            gpu_count=8,
            monte_carlo_trials=16,
            time_horizon_sec=300,
            configured_arrival_rate_per_sec=1.2,
            reserved_capacity_fraction=0.1,
            total_arrivals_mean=320.0,
            completed_mean=318.0,
            dropped_mean=2.0,
            throughput_per_sec_mean=1.15,
            throughput_per_sec_p50=1.1,
            throughput_per_sec_p95=1.3,
            latency_p50_ms=120.0,
            latency_p95_ms=220.0,
            latency_p99_ms=300.0,
            queue_wait_p95_ms=25.0,
            queue_wait_p99_ms=40.0,
            drop_rate_mean=0.005,
            drop_rate_p95=0.01,
            sla_hit_rate_mean=0.99,
            sla_hit_rate_p95=0.985,
            gpu_util_mean=0.6,
            gpu_util_p95=0.72,
            cpu_util_mean=0.1,
            cpu_util_p95=0.18,
            peak_inflight_p95=6.0,
            steady_state_capacity_per_sec=1.02,
            safe_capacity_24x7_per_sec=0.86,
            recommended_max_arrival_rate_per_sec=0.84,
            capacity_gap_per_sec=-0.34,
            capacity_headroom_ratio=0.28,
            procurement_band='UPSIZE_REQUIRED',
            recommended_action='Nagyobb hardver vagy több GPU szükséges a célterheléshez.',
            target_fit=False,
            bottleneck_stage='generate',
            bottleneck_breakdown={'generate': 1.0},
            stage_performance={'generate': {'pressure_score_mean': 15.0}},
            template_inputs={'request_rate_per_sec': 1.2, '_language_code': 'hu'},
            detailed_metrics={'family_display_name': 'Retrieval Rerank Generation', 'primary_metrics': [], 'secondary_metrics': [], 'decision_notes': ['Domináns bottleneck: generate.']},
            model_bindings={'llm_role': 'Qwen/Qwen2.5-32B-Instruct', 'embedding_role': 'nvidia/nv-embedqa-e5-v5'},
            requested_model_overrides={'llm_role': 'Qwen/Qwen2.5-32B-Instruct'},
            override_role_ids=['llm_role'],
            calibration_trace={'summary': calibration_coverage_summary({'generate': {'matched': False}}), 'stages': {'generate': {'matched': False}}, 'examples': {}},
            model_compatibility={'overall_status': 'warning', 'status_counts': {'ok': 1, 'warning': 1, 'error': 0}, 'blocking_roles': [], 'notes': ['KV cache headroom szűk.'], 'serving_notes': ['Ajánlott engine: vLLM.'], 'serving_plan': {'recommendation_lines': ['LLM: vLLM FP8 TP=4']}, 'checks': {'llm_role': {'model_id': 'Qwen/Qwen2.5-32B-Instruct', 'serving_recommendation': {'summary_line': 'vLLM · FP8 · TP=4 · ctx≈4096 · batch≈8', 'recommended_context_window_tokens': 4096, 'recommended_max_batch_size': 8, 'kv_cache_estimate': {'kv_cache_gb_total': 42.0}}}}},
            qualitative_risk_level='LOW',
            notes=['Calibration coverage: 0/1 stages matched to benchmark anchors.'],
        )
        for key, value in overrides.items():
            setattr(base, key, value)
        return base

    def test_result_to_dict_keeps_override_and_calibration_fields(self):
        payload = result_to_dict(self._workload_result())
        self.assertEqual(payload['override_role_ids'], ['llm_role'])
        self.assertIn('requested_model_overrides', payload)
        self.assertEqual(payload['calibration_trace']['summary']['coverage_status'], 'uncalibrated')

    def test_build_customer_report_payload_marks_uncalibrated_results_as_estimate_only(self):
        payload = build_customer_report_payload([self._workload_result()])
        self.assertTrue(payload['calibration_summary']['estimate_only'])
        self.assertEqual(payload['calibration_summary']['coverage_status'], 'uncalibrated')
        self.assertIn('Heurisztikus becslés', payload['executive_summary']['headline'])
        self.assertFalse(payload['comparison_views'][0]['override_active'] is False)
        self.assertEqual(payload['comparison_views'][0]['override_role_ids'], ['llm_role'])

    def test_format_detailed_report_prints_calibration_status(self):
        report = format_detailed_report(self._workload_result())
        self.assertIn('Coverage status', report)
        self.assertIn('heurisztikus', report.lower())

    def test_results_to_json_serializes_workload_results(self):
        blob = results_to_json([self._workload_result()])
        self.assertIn('"override_role_ids": [', blob)
        self.assertIn('"calibration_trace"', blob)

    def test_result_to_dict_memory_simulation_branch(self):
        result = SimulationResult(
            scenario_name='memory_case',
            model_name='Llama',
            model_id='meta-llama/Llama-3.1-8B',
            hardware_profile='DGX',
            weight_precision='fp8',
            kv_cache_precision='fp8',
            tensor_parallel_degree=4,
            num_gpus=4,
            vram_per_gpu_gb=141.0,
            total_system_vram_gb=564.0,
            weight_memory_gb=22.3,
            kv_cache_vram_mean_gb=10.0,
            kv_cache_vram_p95_gb=12.0,
            kv_cache_vram_p99_gb=13.0,
            kv_cache_vram_max_gb=14.0,
            total_vram_mean_gb=32.0,
            total_vram_p95_gb=35.0,
            total_vram_p99_gb=36.0,
            total_vram_max_gb=38.0,
            per_gpu_vram_mean_gb=8.0,
            per_gpu_vram_p95_gb=8.7,
            per_gpu_vram_p99_gb=9.0,
            runtime_overhead_mean_gb=2.0,
            estimated_oom_probability=0.01,
            kv_cache_saturation_probability=0.02,
            recommended_max_concurrency=16,
            qualitative_risk_level='LOW',
            monte_carlo_iterations=200,
            total_concurrent_sequences=12,
        )
        payload = result_to_dict(result)
        self.assertEqual(payload['result_type'], 'memory_simulation')
        self.assertEqual(payload['recommended_max_concurrency'], 16)


class TestWorkloadRiskAssessmentDecisionTable(TestCase):
    def setUp(self):
        self.engine = object.__new__(WorkloadSimulationEngine)

    def _risk(self, **overrides) -> str:
        payload = {
            "drop_rate_p95": 0.0,
            "drop_rate_mean": 0.0,
            "sla_hit_rate_mean": 0.995,
            "gpu_util_p95": 0.5,
        }
        payload.update(overrides)
        return self.engine._assess_risk(SimpleNamespace(**payload))

    def test_assess_risk_critical_triggers_have_precedence(self):
        cases = [
            {"drop_rate_p95": 0.051},
            {"sla_hit_rate_mean": 0.899},
            {"gpu_util_p95": 0.951},
            {"drop_rate_p95": 0.03, "sla_hit_rate_mean": 0.89},
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(self._risk(**case), "CRITICAL")

    def test_assess_risk_high_boundaries_are_strictly_above_threshold(self):
        self.assertEqual(self._risk(drop_rate_p95=0.02), "LOW")
        self.assertEqual(self._risk(sla_hit_rate_mean=0.95), "MEDIUM")
        self.assertEqual(self._risk(gpu_util_p95=0.88), "MEDIUM")
        self.assertEqual(self._risk(drop_rate_p95=0.021), "HIGH")
        self.assertEqual(self._risk(sla_hit_rate_mean=0.949), "HIGH")
        self.assertEqual(self._risk(gpu_util_p95=0.881), "HIGH")

    def test_assess_risk_medium_boundaries_are_strictly_above_threshold(self):
        self.assertEqual(self._risk(drop_rate_mean=0.005, gpu_util_p95=0.5, sla_hit_rate_mean=0.995), "LOW")
        self.assertEqual(self._risk(sla_hit_rate_mean=0.98, gpu_util_p95=0.5), "LOW")
        self.assertEqual(self._risk(drop_rate_mean=0.0051), "MEDIUM")
        self.assertEqual(self._risk(sla_hit_rate_mean=0.979), "MEDIUM")
        self.assertEqual(self._risk(gpu_util_p95=0.781), "MEDIUM")


class TestProcurementBoundaryBehavior(TestCase):
    def _make_result(self, **overrides):
        payload = {
            'throughput_per_sec_mean': 2.0,
            'throughput_per_sec_p50': 2.0,
            'configured_arrival_rate_per_sec': 1.0,
            'gpu_util_p95': 0.55,
            'cpu_util_p95': 0.2,
            'sla_hit_rate_mean': 1.0,
            'drop_rate_mean': 0.0,
            'qualitative_risk_level': 'LOW',
            'planning_profile': 'baseline',
            'reserved_capacity_fraction': 0.0,
        }
        payload.update(overrides)
        return SimpleNamespace(**payload)

    def test_exact_upsize_margin_does_not_trigger_upsize_required(self):
        result = self._make_result(throughput_per_sec_p50=0.9942, configured_arrival_rate_per_sec=1.0)
        metrics = derive_workload_procurement_metrics(result)
        self.assertEqual(metrics['safe_capacity_24x7_per_sec'], 0.85)
        self.assertNotEqual(metrics['procurement_band'], 'UPSIZE_REQUIRED')

    def test_gpu_util_exactly_ninety_percent_is_not_yet_risky(self):
        result = self._make_result(throughput_per_sec_p50=1.5, configured_arrival_rate_per_sec=0.5, gpu_util_p95=0.9)
        metrics = derive_workload_procurement_metrics(result)
        self.assertNotEqual(metrics['procurement_band'], 'RISKY')
        self.assertEqual(metrics['procurement_band'], 'FIT')

    def test_unknown_risk_defaults_to_tune_or_scale_when_no_target(self):
        result = self._make_result(configured_arrival_rate_per_sec=0.0, qualitative_risk_level='UNKNOWN', throughput_per_sec_p50=0.8)
        metrics = derive_workload_procurement_metrics(result)
        self.assertEqual(metrics['procurement_band'], 'TUNE_OR_SCALE')
        self.assertIsNone(metrics['target_fit'])
