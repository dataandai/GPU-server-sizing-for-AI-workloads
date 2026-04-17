import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest import TestCase

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.http_test_utils import wait_for_http_server

from src.blueprint_adapter import build_workload_from_blueprint
from src.workload_simulation import WorkloadSimulationEngine
from src.workload_validation import canonical_reserved_capacity_fraction, validate_workload_simulation_spec
from ui_server import ThreadingHTTPServer, UIServerHandler


class TestAuditGuards(TestCase):
    def test_reserved_capacity_is_canonicalized(self):
        spec = {
            "workload_definition": {"pipeline": {"stages": []}},
            "runtime_profile": {"runtime_family": "vllm", "scheduler_model": {"mode": "vllm_continuous_batching"}},
            "deployment_profile": {
                "execution_mode": "bare_metal",
                "hardware_binding": {"hardware_catalog_id": "nvidia_dgx_h200_8gpu", "software_stack_id": "nvidia_vllm_cuda"},
                "availability": {"reserved_capacity_percent": 20},
            },
            "simulation_profile": {},
        }
        self.assertAlmostEqual(canonical_reserved_capacity_fraction(spec["deployment_profile"]), 0.2)

    def test_irrelevant_partitioning_fields_are_rejected(self):
        spec = {
            "workload_definition": {
                "pipeline": {"stages": []},
            },
            "runtime_profile": {"runtime_family": "vllm", "scheduler_model": {"mode": "vllm_continuous_batching"}},
            "deployment_profile": {
                "execution_mode": "bare_metal",
                "hardware_binding": {"hardware_catalog_id": "nvidia_dgx_h200_8gpu", "software_stack_id": "nvidia_vllm_cuda"},
                "partitioning": {"mode": "none", "mig_profile": "3g.71gb"},
                "availability": {"reserved_capacity_fraction": 0.1},
                "operational_policy": {"reserved_capacity_fraction": 0.1},
            },
            "simulation_profile": {},
        }
        errors, _ = validate_workload_simulation_spec(spec, strict_catalog=False)
        self.assertTrue(any("MIG/vGPU" in e for e in errors))

    def test_selected_model_binding_is_used_consistently(self):
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
            mean_arrival_rate_per_sec=2.0,
        )
        engine = WorkloadSimulationEngine(spec)
        llm_stage = next(s for s in engine.workload["pipeline"]["stages"] if s["model_binding"]["role_id"] == "llm_role")
        self.assertEqual(llm_stage["model_binding"]["selected_model_id"], "mistralai/Mistral-Small-3.1-24B-Instruct-2503")
        self.assertEqual(engine._resolve_model_bindings()["llm_role"], "mistralai/Mistral-Small-3.1-24B-Instruct-2503")


class TestUIServerValidationResponses(TestCase):
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

    def test_compose_blueprint_invalid_combo_returns_400(self):
        body = json.dumps({
            "blueprint_id": "nvidia_rag_blueprint",
            "template_id": "bp_template_rag",
            "scenario_name": "invalid_combo",
            "hardware_catalog_id": "nvidia_dgx_h200_8gpu",
            "software_stack_id": "amd_vllm_rocm",
            "deployment_profile_id": "bare_metal_container",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/compose-blueprint-workload",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)
        payload = json.loads(ctx.exception.read().decode("utf-8"))
        self.assertFalse(payload["ok"])
        self.assertIn("vendor fókusza", payload["error"])
