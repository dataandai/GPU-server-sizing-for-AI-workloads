import json
import numpy as np
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from unittest import TestCase

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.http_test_utils import wait_for_http_server

from src.blueprint_adapter import build_workload_from_blueprint
from src.reporting import result_to_dict
from src.simulator import run_scenario_file
from src.workload_simulation import WorkloadSimulationEngine, WorkloadSimulationResult, is_workload_simulation_file
from ui_server import UIServerHandler, ThreadingHTTPServer


WORKLOAD_YAML = {
    "schema_version": "1.0",
    "name": "Test workload sim",
    "workload_definition": {
        "workload_id": "test_pipeline",
        "workload_class": "retrieval_augmented",
        "input_profile": {
            "arrival_pattern": {"mode": "poisson", "mean_arrival_rate_per_sec": 2.0},
            "work_item": {
                "unit_type": "request",
                "size_distribution": {"distribution": "lognormal", "p50": 800, "p95": 2400},
                "complexity_distribution": {"distribution": "normal", "mean": 1.0, "stddev": 0.2},
                "language_mix": [{"language": "hu", "share": 0.8}],
            },
        },
        "pipeline": {
            "topology": "linear",
            "stages": [
                {
                    "stage_id": "rerank",
                    "stage_type": "rerank",
                    "language_sensitive": True,
                    "model_binding": {
                        "role_id": "reranker_role",
                        "default_model_id": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
                        "override_model_id": "BAAI/bge-reranker-v2-m3",
                        "model_source": "huggingface",
                    },
                    "resource_profile": {
                        "primary_device": "gpu",
                        "gpu_memory_gb_per_work_item": 0.25,
                        "service_time_model": {"mode": "linear", "base_ms": 6, "ms_per_input_unit": 0.004},
                    },
                    "batching_profile": {
                        "batching_mode": "dynamic",
                        "max_batch_size": 16,
                        "batch_timeout_ms": 8,
                        "allow_mixed_lengths": True,
                    },
                },
                {
                    "stage_id": "generate",
                    "stage_type": "generator",
                    "language_sensitive": True,
                    "model_binding": {
                        "role_id": "llm_role",
                        "default_model_id": "nvidia/llama-3.3-nemotron-super-49b-v1",
                        "override_model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                        "model_source": "huggingface",
                    },
                    "resource_profile": {
                        "primary_device": "gpu",
                        "gpu_memory_gb_per_work_item": 1.0,
                        "service_time_model": {"mode": "reference_scaled", "reference_profile_id": "rag_blueprint_llm_stage"},
                    },
                    "batching_profile": {
                        "batching_mode": "continuous",
                        "max_batch_size": 32,
                        "batch_timeout_ms": 10,
                        "max_batched_tokens": 16384,
                        "allow_mixed_lengths": True,
                    },
                    "failure_policy": {"drop_on_overload": True},
                },
            ],
        },
        "sla_policy": {"latency_target_ms_p95": 1800, "throughput_target_per_sec": 1.5, "max_queue_wait_ms": 500},
    },
    "runtime_profile": {
        "runtime_id": "vllm_latency_balanced",
        "runtime_family": "vllm",
        "scheduler_model": {
            "mode": "vllm_continuous_batching",
            "chunked_prefill": True,
            "decode_priority": True,
            "max_num_batched_tokens": 16384,
            "prefix_cache_enabled": True,
            "prefix_cache_hit_rate": 0.25,
        },
        "runtime_penalties": {
            "small_batch_efficiency_penalty": 1.05,
            "mixed_length_batch_penalty": 1.03,
        },
    },
    "deployment_profile": {
        "deployment_id": "bare_metal_h200_4gpu",
        "execution_mode": "bare_metal",
        "hardware_binding": {
            "hardware_catalog_id": "nvidia_dgx_h200_8gpu",
            "gpu_count": 4,
            "tp_size": 4,
            "software_stack_id": "nvidia_vllm_cuda",
        },
        "partitioning": {"mode": "none", "isolation_level": "hard"},
        "overhead_model": {"throughput_penalty_factor": 1.0, "latency_jitter_stddev_ms": 2},
        "availability": {"reserved_capacity_percent": 20, "max_safe_gpu_utilization": 0.8},
    },
    "simulation_profile": {
        "monte_carlo_trials": 20,
        "warmup_sec": 5,
        "time_horizon_sec": 120,
        "time_step_ms": 10,
        "random_variables": [
            {
                "name": "arrival_burst",
                "applies_to": "input",
                "distribution": "lognormal",
                "parameters": {"mean": 0.0, "sigma": 0.15},
                "target_path": "workload_definition.input_profile.arrival_pattern.mean_arrival_rate_per_sec",
            }
        ],
    },
}


class TestWorkloadSimulationEngine(TestCase):
    def test_detection_and_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "workload.yaml"
            path.write_text(yaml.safe_dump(WORKLOAD_YAML, sort_keys=False), encoding="utf-8")
            self.assertTrue(is_workload_simulation_file(path))
            result = run_scenario_file(str(path))

        self.assertIsInstance(result, WorkloadSimulationResult)
        self.assertEqual(result.result_type, "workload_simulation")
        self.assertEqual(result.workload_id, "test_pipeline")
        self.assertIn("llm_role", result.model_bindings)
        self.assertEqual(result.model_bindings["llm_role"], "mistralai/Mistral-Small-3.1-24B-Instruct-2503")
        self.assertGreaterEqual(result.throughput_per_sec_mean, 0.0)
        self.assertGreaterEqual(result.latency_p95_ms, result.latency_p50_ms)
        self.assertIn(result.procurement_band, {"FIT", "SAFE_24X7", "TUNE_OR_SCALE", "UPSIZE_REQUIRED", "RISKY", "UNSUITABLE"})
        self.assertGreaterEqual(result.safe_capacity_24x7_per_sec, 0.0)
        payload = result_to_dict(result)
        self.assertEqual(payload["result_type"], "workload_simulation")
        self.assertIn("procurement_band", payload)
        self.assertIn("safe_capacity_24x7_per_sec", payload)
        self.assertIn("stage_performance", payload)
        self.assertIn("detailed_metrics", payload)
        self.assertTrue(isinstance(payload["stage_performance"], dict))


    def test_hungarian_language_mix_increases_generated_token_load(self):
        hu_spec = json.loads(json.dumps(WORKLOAD_YAML))
        en_spec = json.loads(json.dumps(WORKLOAD_YAML))
        hu_spec["workload_definition"]["input_profile"]["work_item"]["language_mix"] = [{"language": "hu", "share": 1.0}]
        en_spec["workload_definition"]["input_profile"]["work_item"]["language_mix"] = [{"language": "en", "share": 1.0}]
        rng_hu = np.random.default_rng(7)
        rng_en = np.random.default_rng(7)
        engine_hu = WorkloadSimulationEngine(hu_spec)
        engine_en = WorkloadSimulationEngine(en_spec)
        items_hu = engine_hu._generate_work_items(hu_spec["workload_definition"], 120000, rng_hu)
        items_en = engine_en._generate_work_items(en_spec["workload_definition"], 120000, rng_en)
        self.assertEqual(len(items_hu), len(items_en))
        hu_avg_output = sum(item["output_units"] for item in items_hu) / max(1, len(items_hu))
        en_avg_output = sum(item["output_units"] for item in items_en) / max(1, len(items_en))
        self.assertGreater(hu_avg_output, en_avg_output * 1.5)

    def test_cli_writes_workload_json(self):
        repo_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "workload.yaml"
            out_json = Path(tmpdir) / "result.json"
            path.write_text(yaml.safe_dump(WORKLOAD_YAML, sort_keys=False), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "run_all.py", str(path), "--json-out", str(out_json)],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("JSON results written to:", proc.stdout)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 1)
            self.assertEqual(payload[0]["result_type"], "workload_simulation")


    def test_voice_workload_result_contains_detailed_metrics(self):
        spec = build_workload_from_blueprint(
            blueprint_id="nvidia_realtime_voice_call_processing",
            template_id="bp_template_realtime_voice_calls",
            scenario_name="voice_detail",
            hardware_catalog_id="dell_xe8640_4gpu_h100_sxm",
            software_stack_id="nvidia_riva_streaming",
            deployment_profile_id="bare_metal_container",
            monte_carlo_trials=4,
            time_horizon_sec=120,
            template_inputs={
                "concurrent_calls": 16,
                "audio_chunk_ms": 320,
                "diarization_enabled": True,
                "live_translation_enabled": True,
                "agent_assist_enabled": True,
                "tts_response_enabled": False,
            },
        )
        result = WorkloadSimulationEngine(spec).run()
        self.assertEqual(result.workload_family, "realtime_voice_call_processing")
        self.assertIn("family_display_name", result.detailed_metrics)
        self.assertGreaterEqual(len(result.detailed_metrics.get("primary_metrics", [])), 2)
        self.assertIn("stream_asr", result.stage_performance)

    def test_fraud_workload_result_contains_family_specific_cards(self):
        spec = build_workload_from_blueprint(
            blueprint_id="nvidia_financial_fraud_detection",
            template_id="bp_template_financial_fraud_detection",
            scenario_name="fraud_detail",
            hardware_catalog_id="dell_r770_2gpu_h200_nvl",
            software_stack_id="nvidia_triton_inference",
            deployment_profile_id="bare_metal_container",
            monte_carlo_trials=4,
            time_horizon_sec=120,
            template_inputs={
                "transaction_events_per_sec": 120,
                "avg_entities_per_case": 8,
                "false_positive_sensitivity": 0.8,
                "graph_features_enabled": True,
                "realtime_blocking_enabled": True,
                "explainability_enabled": True,
            },
        )
        result = WorkloadSimulationEngine(spec).run()
        labels = {m["label"] for m in result.detailed_metrics.get("primary_metrics", [])}
        self.assertIn("Cél tranzakció / mp", labels)
        self.assertIn("classify_fraud", result.stage_performance)


class TestWorkloadUIServer(TestCase):
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

    def test_workload_schema_endpoint(self):
        payload = self._get_json("/api/catalog/workload-schema")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["schema"]["title"], "Workload Simulation Schema")

    def test_workload_run_endpoint(self):
        body = json.dumps({"yaml": yaml.safe_dump(WORKLOAD_YAML, sort_keys=False)}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/run",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["results"][0]["result_type"], "workload_simulation")
        self.assertIn("result_download_url", payload)
        self.assertIn("procurement_band", payload["summary"])
