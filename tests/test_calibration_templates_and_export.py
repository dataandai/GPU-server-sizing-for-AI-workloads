import json
import threading
import time
import urllib.request
from pathlib import Path
from unittest import TestCase

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.http_test_utils import wait_for_http_server

from src.blueprint_adapter import build_workload_from_blueprint
from src.template_adapter import build_workload_from_template, list_workload_template_records
from src.workload_simulation import WorkloadSimulationEngine
from src.report_export import export_customer_report_html
from src.pdf_export import export_customer_report_pdf
from src.reporting import result_to_dict
from ui_server import ThreadingHTTPServer, UIServerHandler


class TestCalibrationAnchors(TestCase):
    def test_rag_blueprint_matches_calibration_anchor(self):
        spec = build_workload_from_blueprint(
            blueprint_id="nvidia_rag_blueprint",
            template_id="bp_template_rag",
            scenario_name="rag_calibrated",
            hardware_catalog_id="nvidia_dgx_h200_8gpu",
            software_stack_id="nvidia_vllm_cuda",
            deployment_profile_id="bare_metal_container",
            monte_carlo_trials=4,
            time_horizon_sec=90,
            template_inputs={
                "query_rate_per_sec": 1.2,
                "avg_context_docs": 8,
                "reranking_enabled": True,
                "document_parsing_enabled": False,
                "multimodal_query_enabled": False,
                "query_prompt_tokens": 1800,
                "answer_tokens": 320,
            },
        )
        result = WorkloadSimulationEngine(spec).run()
        summary = result.calibration_trace["summary"]
        self.assertGreaterEqual(summary["matched_stage_count"], 3)
        self.assertIn("rag_generate_answer_default", summary["matched_anchor_ids"])


class TestAdvancedTemplateAdapter(TestCase):
    def test_library_template_builds_valid_workload(self):
        spec = build_workload_from_template(
            template_id="lib_multi_stage_agentic_workflow",
            scenario_name="agentic_library",
            hardware_catalog_id="nvidia_dgx_h200_8gpu",
            software_stack_id="nvidia_tensorrt_llm",
            deployment_profile_id="bare_metal_container",
            template_inputs={
                "concurrent_runs": 5,
                "avg_steps_per_run": 7,
                "avg_tool_calls_per_run": 4,
                "avg_docs_per_step": 5,
                "report_length_tokens": 2600,
                "human_gate_enabled": True,
            },
            monte_carlo_trials=4,
            time_horizon_sec=120,
        )
        self.assertEqual(spec["metadata"]["template_source"], "workload_library")
        self.assertEqual(spec["metadata"]["source_template_library_id"], "lib_multi_stage_agentic_workflow")
        self.assertEqual(spec["runtime_profile"]["runtime_family"], "tensorrt_llm")
        stage_ids = [s["stage_id"] for s in spec["workload_definition"]["pipeline"]["stages"]]
        self.assertIn("optional_review", stage_ids)
        result = WorkloadSimulationEngine(spec).run()
        self.assertEqual(result.source_template_id, "lib_multi_stage_agentic_workflow")
        self.assertGreaterEqual(result.throughput_per_sec_mean, 0.0)


    def test_library_template_supports_model_overrides_for_all_roles(self):
        spec = build_workload_from_template(
            template_id="lib_multi_stage_agentic_workflow",
            scenario_name="agentic_override_case",
            hardware_catalog_id="nvidia_dgx_h200_8gpu",
            software_stack_id="nvidia_tensorrt_llm",
            deployment_profile_id="bare_metal_container",
            model_overrides={
                "planner_role": "Qwen/Qwen2.5-32B-Instruct",
                "reasoning_role": "meta-llama/Llama-3.1-70B-Instruct",
                "writer_role": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                "review_role": "meta-llama/Llama-Guard-3-8B",
            },
            template_inputs={"human_gate_enabled": True},
        )
        bindings = {
            stage["model_binding"]["role_id"]: stage["model_binding"]["override_model_id"]
            for stage in spec["workload_definition"]["pipeline"]["stages"]
        }
        self.assertEqual(bindings["planner_role"], "Qwen/Qwen2.5-32B-Instruct")
        self.assertEqual(bindings["reasoning_role"], "meta-llama/Llama-3.1-70B-Instruct")
        self.assertEqual(bindings["writer_role"], "mistralai/Mistral-Small-3.1-24B-Instruct-2503")
        self.assertEqual(bindings["review_role"], "meta-llama/Llama-Guard-3-8B")

    def test_template_catalog_contains_expected_families(self):
        ids = {record["template_id"] for record in list_workload_template_records()}
        self.assertIn("lib_retrieval_rerank_generation", ids)
        self.assertIn("lib_batch_document_pipeline", ids)
        self.assertIn("lib_hybrid_multimodal_pipeline", ids)


class TestReportExport(TestCase):
    def test_html_and_pdf_export(self):
        spec = build_workload_from_blueprint(
            blueprint_id="nvidia_rag_blueprint",
            template_id="bp_template_rag",
            scenario_name="report_export_case",
            hardware_catalog_id="nvidia_dgx_h200_8gpu",
            software_stack_id="nvidia_vllm_cuda",
            deployment_profile_id="bare_metal_container",
            monte_carlo_trials=4,
            time_horizon_sec=90,
        )
        result = WorkloadSimulationEngine(spec).run()
        base = Path("output/results/report_export_case")
        html_path = export_customer_report_html([result], base.with_suffix(".html"))
        pdf_path = export_customer_report_pdf([result], base.with_suffix(".pdf"))
        self.assertTrue(html_path.exists())
        self.assertTrue(pdf_path.exists())
        self.assertIn("report_export_case", html_path.read_text(encoding="utf-8"))
        self.assertGreater(pdf_path.stat().st_size, 1000)


class TestTemplateAndReportEndpoints(TestCase):
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

    def test_workload_template_catalog_endpoint(self):
        payload = self._get_json("/api/catalog/workload-templates")
        self.assertTrue(payload["ok"])
        ids = {r["template_id"] for r in payload["records"]}
        self.assertIn("lib_streaming_queue_heavy_inference", ids)


    def test_template_ui_context_exposes_on_prem_roles_and_non_virtualized_deployments(self):
        payload = self._get_json(
            "/api/ui/template-form?template_id=lib_multi_stage_agentic_workflow"
            "&hardware_catalog_id=nvidia_dgx_h200_8gpu&software_stack_id=nvidia_tensorrt_llm"
        )
        self.assertTrue(payload["ok"])
        context = payload["context"]
        role_ids = {r["role_id"] for r in context["replaceable_roles"]}
        self.assertIn("planner_role", role_ids)
        self.assertIn("writer_role", role_ids)
        self.assertTrue(all(dep_id not in {"vmware_vgpu_time_sliced", "vmware_vgpu_mig", "vm_passthrough_gpu"} for dep_id in context["allowed_deployment_profile_ids"]))

    def test_template_model_compatibility_endpoint_returns_role_checks(self):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/template-model-compatibility",
            data=json.dumps({
                "role_ids": ["reasoning_role", "writer_role"],
                "hardware_catalog_id": "dell_r770_2gpu_rtxpro6000",
                "software_stack_id": "nvidia_tensorrt_llm",
                "deployment_profile_id": "bare_metal_container",
                "model_overrides": {
                    "reasoning_role": "Qwen/Qwen2.5-72B-Instruct",
                    "writer_role": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
                },
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        self.assertTrue(payload["ok"])
        compat = payload["model_compatibility"]
        self.assertIn(compat["overall_status"], {"ok", "warning", "error"})
        self.assertIn("reasoning_role", compat["checks"])
        self.assertIn("writer_role", compat["checks"])
        self.assertIn("serving_plan", compat)
        self.assertIn("recommended_tensor_parallel_degree", compat["checks"]["reasoning_role"]["serving_recommendation"])

    def test_compose_template_workload_and_export_report(self):
        compose_body = json.dumps({
            "template_id": "lib_retrieval_rerank_generation",
            "scenario_name": "ui_template_case",
            "hardware_catalog_id": "nvidia_dgx_h200_8gpu",
            "software_stack_id": "nvidia_vllm_cuda",
            "deployment_profile_id": "bare_metal_container",
            "template_inputs": {
                "request_rate_per_sec": 1.5,
                "avg_context_docs": 6,
                "avg_prompt_tokens": 1600,
                "avg_output_tokens": 280,
                "rerank_top_k": 16,
                "guardrail_enabled": False,
            },
            "model_overrides": {
                "llm_role": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                "reranker_role": "BAAI/bge-reranker-v2-m3"
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/compose-template-workload",
            data=compose_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        self.assertTrue(payload["ok"])
        yaml_text = payload["yaml"]
        self.assertIn("mistralai/Mistral-Small-3.1-24B-Instruct-2503", yaml_text)
        self.assertIn("BAAI/bge-reranker-v2-m3", yaml_text)
        self.assertIn("model_compatibility", payload)
        self.assertIn(payload["model_compatibility"]["overall_status"], {"ok", "warning", "error"})
        self.assertIn("serving_plan", payload["model_compatibility"])

        run_req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/run",
            data=json.dumps({"yaml": yaml_text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(run_req) as resp:
            run_payload = json.loads(resp.read().decode("utf-8"))
        self.assertTrue(run_payload["ok"])
        self.assertEqual(run_payload["results"][0]["scenario_name"], "ui_template_case")

        export_req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/export-report",
            data=json.dumps({"results": run_payload["results"], "format": "html", "report_name": "ui_template_case"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(export_req) as resp:
            export_payload = json.loads(resp.read().decode("utf-8"))
        self.assertTrue(export_payload["ok"])
        report_path = Path(export_payload["report_file"])
        self.assertTrue(report_path.exists())
        report_html = report_path.read_text(encoding="utf-8")
        self.assertIn("Model kompatibilitás", report_html)
        self.assertIn("Serving javaslat", report_html)
