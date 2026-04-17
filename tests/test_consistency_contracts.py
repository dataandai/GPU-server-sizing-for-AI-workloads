import sys
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blueprint_adapter import build_workload_from_blueprint
from src.catalog_loader import validate_catalog_integrity
from src.reporting import format_detailed_report, result_to_dict
from src.report_payload import build_customer_report_payload
from src.simulator import run_scenario_file
from src.template_adapter import build_workload_from_template
from src.workload_simulation import WorkloadSimulationEngine
from src.workload_validation import validate_workload_simulation_spec


class TestCatalogSemanticIntegrity(TestCase):
    def test_curated_catalog_cross_references_are_semantically_valid(self):
        errors, warnings = validate_catalog_integrity()
        self.assertEqual(errors, [], f"Catalog semantic errors: {errors}")
        # Warnings are allowed for intentionally generic/reference-like blueprint roles,
        # but the validator must at least return a stable list type.
        self.assertIsInstance(warnings, list)


class TestTemplateComposeRunContract(TestCase):
    def test_template_compose_run_report_keeps_override_and_language_contract(self):
        overrides = {
            "llm_role": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            "reranker_role": "BAAI/bge-reranker-v2-m3",
        }
        spec = build_workload_from_template(
            template_id="lib_retrieval_rerank_generation",
            scenario_name="contract_template_hu",
            hardware_catalog_id="nvidia_dgx_h200_8gpu",
            software_stack_id="nvidia_vllm_cuda",
            deployment_profile_id="bare_metal_container",
            language_code="hu",
            language_share=1.0,
            planning_profile="safe_24x7",
            monte_carlo_trials=4,
            time_horizon_sec=60,
            template_inputs={
                "request_rate_per_sec": 1.2,
                "avg_context_docs": 6,
                "avg_prompt_tokens": 1500,
                "avg_output_tokens": 220,
                "rerank_top_k": 16,
                "guardrail_enabled": False,
            },
            model_overrides=overrides,
        )
        errors, _warnings = validate_workload_simulation_spec(spec, strict_catalog=True)
        self.assertEqual(errors, [])
        self.assertEqual(spec["metadata"]["requested_model_overrides"], overrides)
        self.assertEqual(spec["metadata"]["override_role_ids"], ["llm_role", "reranker_role"])
        scaling = spec["metadata"]["language_scaling"]
        self.assertEqual(scaling["dominant_language"], "hu")
        self.assertGreater(scaling["generation_multiplier"], 1.0)

        result = WorkloadSimulationEngine(spec).run()
        self.assertEqual(result.requested_model_overrides, overrides)
        self.assertEqual(result.override_role_ids, ["llm_role", "reranker_role"])
        self.assertEqual(result.model_bindings["llm_role"], overrides["llm_role"])
        self.assertEqual(result.model_bindings["reranker_role"], overrides["reranker_role"])
        self.assertIn(result.model_compatibility.get("overall_status"), {"ok", "warning", "error"})

        payload = build_customer_report_payload([result])
        self.assertTrue(payload["comparison_views"][0]["override_active"])
        self.assertEqual(payload["comparison_views"][0]["override_role_ids"], ["llm_role", "reranker_role"])
        self.assertEqual(payload["comparison_views"][0]["language"], "hu")


class TestBlueprintComposeRunContract(TestCase):
    def test_blueprint_compose_run_report_keeps_override_and_calibration_contract(self):
        overrides = {"llm_role": "Qwen/Qwen2.5-32B-Instruct"}
        spec = build_workload_from_blueprint(
            blueprint_id="nvidia_rag_blueprint",
            template_id="bp_template_rag",
            scenario_name="contract_blueprint_hu",
            hardware_catalog_id="nvidia_dgx_h200_8gpu",
            software_stack_id="nvidia_vllm_cuda",
            deployment_profile_id="bare_metal_container",
            language_code="hu",
            language_share=0.9,
            planning_profile="baseline",
            monte_carlo_trials=4,
            time_horizon_sec=60,
            template_inputs={
                "query_rate_per_sec": 1.0,
                "avg_context_docs": 6,
                "query_prompt_tokens": 1600,
                "answer_tokens": 256,
            },
            model_overrides=overrides,
        )
        errors, _warnings = validate_workload_simulation_spec(spec, strict_catalog=True)
        self.assertEqual(errors, [])
        self.assertEqual(spec["metadata"]["requested_model_overrides"], overrides)
        self.assertEqual(spec["metadata"]["override_role_ids"], ["llm_role"])
        self.assertIn("model_compatibility", spec["metadata"])

        result = WorkloadSimulationEngine(spec).run()
        payload = result_to_dict(result)
        self.assertEqual(payload["requested_model_overrides"], overrides)
        self.assertEqual(payload["override_role_ids"], ["llm_role"])
        self.assertIn("coverage_status", payload["calibration_trace"]["summary"])
        self.assertIn(payload["calibration_trace"]["summary"]["coverage_status"], {"uncalibrated", "partial_low", "partial_medium", "anchored"})

        report_payload = build_customer_report_payload([result])
        self.assertEqual(report_payload["comparison_views"][0]["override_role_ids"], ["llm_role"])
        self.assertIn("coverage_status", report_payload["calibration_summary"])


class TestFamilySpecificDetailedMetrics(TestCase):
    def _run_blueprint(self, blueprint_id: str, template_id: str, template_inputs: dict):
        spec = build_workload_from_blueprint(
            blueprint_id=blueprint_id,
            template_id=template_id,
            scenario_name=f"metrics_{template_id}",
            hardware_catalog_id="nvidia_dgx_h200_8gpu",
            software_stack_id="nvidia_vllm_cuda",
            deployment_profile_id="bare_metal_container",
            language_code="en",
            language_share=1.0,
            monte_carlo_trials=4,
            time_horizon_sec=60,
            template_inputs=template_inputs,
        )
        return WorkloadSimulationEngine(spec).run()

    def test_realtime_voice_family_metrics_are_domain_specific(self):
        result = self._run_blueprint(
            "nvidia_realtime_voice_call_processing",
            "bp_template_realtime_voice_calls",
            {
                "concurrent_calls": 18,
                "audio_chunk_ms": 320,
                "diarization_enabled": True,
                "live_translation_enabled": True,
                "agent_assist_enabled": True,
                "tts_response_enabled": False,
            },
        )
        labels = {m["label"] for m in result.detailed_metrics["primary_metrics"]}
        self.assertIn("Cél párhuzamos hívások", labels)
        self.assertIn("24/7 safe párhuzamos hívások", labels)
        notes = " ".join(result.detailed_metrics["decision_notes"])
        self.assertIn("hívás", notes)

    def test_fraud_family_metrics_are_domain_specific(self):
        result = self._run_blueprint(
            "nvidia_financial_fraud_detection",
            "bp_template_financial_fraud_detection",
            {
                "transaction_events_per_sec": 120,
                "avg_entities_per_case": 5,
                "false_positive_sensitivity": 0.6,
                "graph_features_enabled": True,
                "realtime_blocking_enabled": True,
                "explainability_enabled": True,
            },
        )
        labels = {m["label"] for m in result.detailed_metrics["primary_metrics"]}
        self.assertIn("Cél tranzakció / mp", labels)
        self.assertIn("Safe entitás fan-out", labels)
        notes = " ".join(result.detailed_metrics["decision_notes"])
        self.assertIn("Realtime blocking", notes)

    def test_route_optimization_family_metrics_are_domain_specific(self):
        result = self._run_blueprint(
            "nvidia_route_optimization",
            "bp_template_route_optimization",
            {
                "solve_requests_per_minute": 9,
                "avg_stops_per_job": 220,
                "avg_vehicle_count": 54,
                "constraint_count": 45,
                "dynamic_replans_enabled": True,
                "eta_publish_enabled": False,
                "solve_sla_ms": 4500,
            },
        )
        labels = {m["label"] for m in result.detailed_metrics["primary_metrics"]}
        self.assertIn("Cél solve / perc", labels)
        self.assertIn("Safe stop volumen", labels)
        notes = " ".join(result.detailed_metrics["decision_notes"])
        self.assertIn("optimalizációs futások", notes)

    def test_predictive_forecasting_family_metrics_are_domain_specific(self):
        result = self._run_blueprint(
            "nvidia_predictive_maintenance_forecasting",
            "bp_template_predictive_maintenance_forecasting",
            {
                "monitored_assets": 280,
                "metrics_per_asset": 18,
                "refresh_interval_sec": 30,
                "lookback_window_min": 240,
                "forecast_horizon_min": 90,
                "anomaly_detection_enabled": True,
                "alert_publish_enabled": True,
            },
        )
        labels = {m["label"] for m in result.detailed_metrics["primary_metrics"]}
        self.assertIn("Cél asset lefedettség", labels)
        self.assertIn("Safe telemetry pont / mp", labels)
        notes = " ".join(result.detailed_metrics["decision_notes"])
        self.assertIn("asset", notes)

    def test_recommendation_ranking_family_metrics_are_domain_specific(self):
        result = self._run_blueprint(
            "nvidia_recommendation_ranking_reference",
            "bp_template_recommendation_ranking",
            {
                "recommendation_requests_per_sec": 10,
                "candidate_pool_size": 160,
                "catalog_filter_count": 7,
                "reranking_enabled": True,
                "personalization_enabled": True,
                "image_grounding_enabled": False,
            },
        )
        labels = {m["label"] for m in result.detailed_metrics["primary_metrics"]}
        self.assertIn("Cél kérések / mp", labels)
        self.assertIn("Safe candidate volume", labels)
        notes = " ".join(result.detailed_metrics["decision_notes"])
        self.assertIn("candidate volume", notes)


class TestLegacyMemorySurfacePolicy(TestCase):
    def test_legacy_memory_scenario_is_marked_as_deprecated_surface(self):
        result = run_scenario_file("scenarios/scenario_01_fixed_batch.yaml")
        payload = result_to_dict(result)
        self.assertEqual(payload["result_type"], "memory_simulation")
        self.assertEqual(payload["surface_status"], "deprecated_legacy_memory_engine")
        self.assertTrue(payload["estimate_only"])
        self.assertTrue(any("Legacy memory simulation surface" in note for note in payload["notes"]))
        report = format_detailed_report(result)
        self.assertIn("Deprecated legacy memory engine", report)
        self.assertIn("benchmark-calibrated workload simulator", report)
