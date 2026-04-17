import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.http_test_utils import wait_for_http_server

import json
import threading
import time
import urllib.request
from unittest import TestCase

from src.catalog_loader import (
    get_blueprint_bundle,
    list_blueprint_records,
    list_blueprint_template_records,
)
from src.blueprint_adapter import build_workload_from_blueprint
from ui_server import UIServerHandler, ThreadingHTTPServer


class TestBlueprintCatalog(TestCase):
    def test_blueprint_records_load_from_new_structure(self):
        records = list_blueprint_records()
        self.assertGreaterEqual(len(records), 5)
        self.assertTrue(any(r["blueprint_id"] == "nvidia_rag_blueprint" for r in records))

    def test_blueprint_bundle_contains_templates(self):
        bundle = get_blueprint_bundle("nvidia_vss_warehouse_2d")
        self.assertEqual(bundle["blueprint"]["blueprint_id"], "nvidia_vss_warehouse_2d")
        self.assertGreaterEqual(len(bundle["templates"]), 1)
        self.assertEqual(bundle["templates"][0]["blueprint_id"], "nvidia_vss_warehouse_2d")

    def test_blueprint_templates_load_from_new_structure(self):
        templates = list_blueprint_template_records()
        self.assertGreaterEqual(len(templates), 5)
        self.assertTrue(any(t["template_id"] == "bp_template_rag" for t in templates))


class TestUIServerCatalogEndpoints(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), UIServerHandler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        wait_for_http_server("127.0.0.1", cls.port)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)

    def _get_json(self, path: str) -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_catalog_summary_endpoint(self):
        payload = self._get_json("/api/catalog/summary")
        self.assertTrue(payload["ok"])
        self.assertGreaterEqual(payload["counts"]["blueprints"], 5)
        self.assertGreaterEqual(payload["counts"]["hardware"], 5)

    def test_blueprint_detail_endpoint(self):
        payload = self._get_json("/api/catalog/blueprints/nvidia_rag_blueprint")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["blueprint"]["blueprint_id"], "nvidia_rag_blueprint")
        self.assertGreaterEqual(len(payload["templates"]), 1)


class TestBlueprintAdapter(TestCase):
    def test_build_workload_from_blueprint_supports_role_overrides(self):
        spec = build_workload_from_blueprint(
            blueprint_id="nvidia_rag_blueprint",
            template_id="bp_template_rag",
            scenario_name="rag_hu",
            hardware_catalog_id="nvidia_dgx_h200_8gpu",
            software_stack_id="nvidia_vllm_cuda",
            deployment_profile_id="bare_metal_container",
            language_code="hu",
            language_share=0.9,
            model_overrides={"llm_role": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"},
            mean_arrival_rate_per_sec=4.5,
            planning_profile="safe_24x7",
        )
        self.assertEqual(spec["workload_definition"]["workload_class"], "retrieval_augmented")
        self.assertEqual(spec["runtime_profile"]["runtime_family"], "vllm")
        stages = spec["workload_definition"]["pipeline"]["stages"]
        llm_stage = next(s for s in stages if s["model_binding"]["role_id"] == "llm_role")
        self.assertEqual(llm_stage["model_binding"]["override_model_id"], "mistralai/Mistral-Small-3.1-24B-Instruct-2503")
        self.assertEqual(spec["workload_definition"]["input_profile"]["work_item"]["language_mix"][0]["language"], "hu")
        self.assertEqual(spec["deployment_profile"]["operational_policy"]["planning_profile"], "safe_24x7")
        self.assertTrue(spec["deployment_profile"]["operational_policy"]["n_plus_one_enabled"])
        self.assertIn("model_compatibility", spec["metadata"])
        self.assertIn(spec["metadata"]["model_compatibility"]["overall_status"], {"ok", "warning", "error"})
        self.assertIn("serving_plan", spec["metadata"]["model_compatibility"])


class TestUIServerBlueprintEndpoints(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), UIServerHandler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        wait_for_http_server("127.0.0.1", cls.port)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)

    def test_blueprint_roles_endpoint(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/catalog/blueprint-roles/nvidia_rag_blueprint") as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertTrue(any(r["role_id"] == "llm_role" for r in payload["records"]))

    def test_compose_blueprint_workload_endpoint(self):
        body = json.dumps({
            "blueprint_id": "nvidia_rag_blueprint",
            "template_id": "bp_template_rag",
            "scenario_name": "rag_blueprint_test",
            "hardware_catalog_id": "nvidia_dgx_h200_8gpu",
            "software_stack_id": "nvidia_vllm_cuda",
            "deployment_profile_id": "bare_metal_container",
            "language_code": "hu",
            "language_share": 0.8,
            "planning_profile": "high_throughput",
            "template_inputs": {"query_rate_per_sec": 1.2, "avg_context_docs": 8, "query_prompt_tokens": 1800, "answer_tokens": 320},
            "model_overrides": {"llm_role": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"}
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/compose-blueprint-workload",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertIn("workload_definition:", payload["yaml"])
        self.assertIn("mistralai/Mistral-Small-3.1-24B-Instruct-2503", payload["yaml"])
        self.assertIn("planning_profile: high_throughput", payload["yaml"])
        self.assertIn("model_compatibility", payload)
        self.assertIn(payload["model_compatibility"]["overall_status"], {"ok", "warning", "error"})
        self.assertIn("serving_plan", payload["model_compatibility"])

    def test_blueprint_model_compatibility_endpoint(self):
        body = json.dumps({
            "blueprint_id": "nvidia_rag_blueprint",
            "template_id": "bp_template_rag",
            "role_ids": ["embedding_role", "reranker_role", "llm_role"],
            "hardware_catalog_id": "nvidia_dgx_h200_8gpu",
            "software_stack_id": "nvidia_vllm_cuda",
            "deployment_profile_id": "bare_metal_container",
            "template_inputs": {"query_rate_per_sec": 1.0, "avg_context_docs": 6, "query_prompt_tokens": 1600, "answer_tokens": 256},
            "model_overrides": {"llm_role": "Qwen/Qwen2.5-32B-Instruct"}
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/blueprint-model-compatibility",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        self.assertTrue(payload["ok"])
        compat = payload["model_compatibility"]
        self.assertIn("llm_role", compat["checks"])
        self.assertIn("recommended_tensor_parallel_degree", compat["checks"]["llm_role"]["serving_recommendation"])

class TestBlueprintUIContext(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), UIServerHandler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        wait_for_http_server("127.0.0.1", cls.port)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)

    def _get_json(self, path: str) -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_blueprint_ui_context_filters_to_large_nvidia_hardware(self):
        payload = self._get_json(
            "/api/ui/blueprint-form?blueprint_id=nvidia_rag_blueprint&template_id=bp_template_rag"
        )
        self.assertTrue(payload["ok"])
        context = payload["context"]
        self.assertTrue(context["show_language_controls"])
        self.assertTrue(all(hid not in {"dell_r770_2gpu_h200_nvl", "dell_r770_2gpu_rtxpro6000"} for hid in context["allowed_hardware_ids"]))
        self.assertTrue(any(hid == "nvidia_dgx_h200_8gpu" for hid in context["allowed_hardware_ids"]))
        self.assertGreaterEqual(len(context["replaceable_roles"]), 1)
        llm_role = next(r for r in context["replaceable_roles"] if r["role_id"] == "llm_role")
        self.assertIn("candidate_models", llm_role)
        self.assertTrue(any(c["id"] == "Qwen/Qwen2.5-32B-Instruct" for c in llm_role["candidate_models"]))
        self.assertEqual(context["workload_family"], "enterprise_rag")

    def test_blueprint_ui_context_hides_mig_deployments_for_non_mig_stack(self):
        payload = self._get_json(
            "/api/ui/blueprint-form?blueprint_id=nvidia_rag_blueprint&template_id=bp_template_rag"
            "&hardware_catalog_id=nvidia_dgx_h200_8gpu&software_stack_id=nvidia_nim_llm"
        )
        self.assertTrue(payload["ok"])
        context = payload["context"]
        self.assertFalse(any(dep_id in {"vmware_vgpu_mig", "bare_metal_mig_partitioned"} for dep_id in context["allowed_deployment_profile_ids"]))


    def test_new_non_genai_blueprints_are_present(self):
        records = list_blueprint_records()
        ids = {r["blueprint_id"] for r in records}
        self.assertIn("nvidia_realtime_voice_call_processing", ids)
        self.assertIn("nvidia_route_optimization", ids)
        self.assertIn("nvidia_predictive_maintenance_forecasting", ids)
        self.assertIn("nvidia_recommendation_ranking_reference", ids)
        self.assertIn("nvidia_financial_fraud_detection", ids)

class TestNewBlueprintFamilies(TestCase):
    def test_voice_call_template_applies_template_inputs(self):
        spec = build_workload_from_blueprint(
            blueprint_id="nvidia_realtime_voice_call_processing",
            template_id="bp_template_realtime_voice_calls",
            scenario_name="voice_calls",
            hardware_catalog_id="dell_xe8640_4gpu_h100_sxm",
            software_stack_id="nvidia_riva_streaming",
            deployment_profile_id="bare_metal_container",
            template_inputs={
                "concurrent_calls": 30,
                "audio_chunk_ms": 250,
                "diarization_enabled": True,
                "live_translation_enabled": False,
                "agent_assist_enabled": True,
                "tts_response_enabled": False,
            },
        )
        self.assertEqual(spec["runtime_profile"]["runtime_family"], "speech_streaming")
        self.assertEqual(spec["workload_definition"]["workload_class"], "speech_processing")
        self.assertAlmostEqual(spec["workload_definition"]["input_profile"]["arrival_pattern"]["mean_arrival_rate_per_sec"], 120.0, places=2)
        stage_ids = [s["stage_id"] for s in spec["workload_definition"]["pipeline"]["stages"]]
        self.assertIn("speaker_diarization", stage_ids)
        self.assertIn("optional_agent_assist", stage_ids)
        self.assertNotIn("optional_live_translation", stage_ids)
        self.assertNotIn("optional_tts_response", stage_ids)

    def test_route_optimization_template_applies_template_inputs(self):
        spec = build_workload_from_blueprint(
            blueprint_id="nvidia_route_optimization",
            template_id="bp_template_route_optimization",
            scenario_name="route_opt",
            hardware_catalog_id="dell_r770_2gpu_h200_nvl",
            software_stack_id="nvidia_cuopt_solver",
            deployment_profile_id="bare_metal_container",
            template_inputs={
                "solve_requests_per_minute": 12,
                "avg_stops_per_job": 240,
                "avg_vehicle_count": 60,
                "constraint_count": 55,
                "dynamic_replans_enabled": True,
                "eta_publish_enabled": False,
                "solve_sla_ms": 5000,
            },
        )
        self.assertEqual(spec["runtime_profile"]["runtime_family"], "solver_service")
        self.assertEqual(spec["workload_definition"]["workload_class"], "batch_processing")
        self.assertAlmostEqual(spec["workload_definition"]["input_profile"]["arrival_pattern"]["mean_arrival_rate_per_sec"], 0.2, places=3)
        self.assertEqual(spec["workload_definition"]["sla_policy"]["latency_target_ms_p95"], 5000.0)
        stage_ids = [s["stage_id"] for s in spec["workload_definition"]["pipeline"]["stages"]]
        self.assertNotIn("optional_publish_dispatch", stage_ids)

    def test_predictive_maintenance_template_applies_template_inputs(self):
        spec = build_workload_from_blueprint(
            blueprint_id="nvidia_predictive_maintenance_forecasting",
            template_id="bp_template_predictive_maintenance_forecasting",
            scenario_name="predictive_ops",
            hardware_catalog_id="dell_r770_2gpu_h200_nvl",
            software_stack_id="nvidia_triton_inference",
            deployment_profile_id="bare_metal_container",
            template_inputs={
                "monitored_assets": 300,
                "metrics_per_asset": 20,
                "refresh_interval_sec": 30,
                "lookback_window_min": 240,
                "forecast_horizon_min": 90,
                "anomaly_detection_enabled": False,
                "alert_publish_enabled": True,
            },
        )
        self.assertEqual(spec["runtime_profile"]["runtime_family"], "triton")
        self.assertEqual(spec["workload_definition"]["workload_class"], "stream_processing")
        self.assertAlmostEqual(spec["workload_definition"]["input_profile"]["arrival_pattern"]["mean_arrival_rate_per_sec"], 10.0, places=2)
        stage_ids = [s["stage_id"] for s in spec["workload_definition"]["pipeline"]["stages"]]
        self.assertNotIn("optional_detect_anomalies", stage_ids)
        self.assertIn("optional_publish_alerts", stage_ids)

    def test_recommendation_ranking_template_applies_template_inputs(self):
        spec = build_workload_from_blueprint(
            blueprint_id="nvidia_recommendation_ranking_reference",
            template_id="bp_template_recommendation_ranking",
            scenario_name="ranking",
            hardware_catalog_id="dell_r770_2gpu_h200_nvl",
            software_stack_id="nvidia_triton_inference",
            deployment_profile_id="bare_metal_container",
            template_inputs={
                "recommendation_requests_per_sec": 12,
                "candidate_pool_size": 200,
                "catalog_filter_count": 8,
                "reranking_enabled": True,
                "personalization_enabled": False,
                "image_grounding_enabled": False,
            },
        )
        self.assertEqual(spec["runtime_profile"]["runtime_family"], "triton")
        self.assertEqual(spec["workload_definition"]["workload_class"], "retrieval_augmented")
        stage_ids = [s["stage_id"] for s in spec["workload_definition"]["pipeline"]["stages"]]
        self.assertIn("rerank_candidates", stage_ids)
        self.assertNotIn("optional_personalization_features", stage_ids)
        self.assertNotIn("optional_image_grounding", stage_ids)

    def test_financial_fraud_template_applies_template_inputs(self):
        spec = build_workload_from_blueprint(
            blueprint_id="nvidia_financial_fraud_detection",
            template_id="bp_template_financial_fraud_detection",
            scenario_name="fraud",
            hardware_catalog_id="dell_r770_2gpu_h200_nvl",
            software_stack_id="nvidia_triton_inference",
            deployment_profile_id="bare_metal_container",
            template_inputs={
                "transaction_events_per_sec": 220,
                "avg_entities_per_case": 10,
                "false_positive_sensitivity": 0.9,
                "graph_features_enabled": False,
                "realtime_blocking_enabled": True,
                "explainability_enabled": False,
            },
        )
        self.assertEqual(spec["runtime_profile"]["runtime_family"], "triton")
        self.assertEqual(spec["workload_definition"]["sla_policy"]["latency_target_ms_p95"], 250.0)
        stage_ids = [s["stage_id"] for s in spec["workload_definition"]["pipeline"]["stages"]]
        self.assertNotIn("optional_graph_risk", stage_ids)
        self.assertNotIn("optional_case_explainability", stage_ids)

class TestTemplateInputsInContext(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), UIServerHandler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        wait_for_http_server("127.0.0.1", cls.port)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)

    def _get_json(self, path: str) -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_context_exposes_template_fields_for_voice_calls(self):
        payload = self._get_json(
            "/api/ui/blueprint-form?blueprint_id=nvidia_realtime_voice_call_processing&template_id=bp_template_realtime_voice_calls"
        )
        self.assertTrue(payload["ok"])
        context = payload["context"]
        self.assertTrue(context["field_visibility"]["bp_template_inputs"])
        names = {f["name"] for f in context["template_fields"]}
        self.assertIn("concurrent_calls", names)
        self.assertIn("agent_assist_enabled", names)


class TestAdditionalTemplateInputsInContext(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), UIServerHandler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        wait_for_http_server("127.0.0.1", cls.port)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)

    def _get_json(self, path: str) -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_context_exposes_template_fields_for_fraud_detection(self):
        payload = self._get_json(
            "/api/ui/blueprint-form?blueprint_id=nvidia_financial_fraud_detection&template_id=bp_template_financial_fraud_detection"
        )
        self.assertTrue(payload["ok"])
        context = payload["context"]
        self.assertTrue(context["field_visibility"]["bp_template_inputs"])
        self.assertIn("nvidia_triton_inference", context["allowed_software_stack_ids"])
        names = {f["name"] for f in context["template_fields"]}
        self.assertIn("transaction_events_per_sec", names)
        self.assertIn("graph_features_enabled", names)
