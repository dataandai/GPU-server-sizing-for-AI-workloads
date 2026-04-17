import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import TestCase


class TestRunAllJsonOutput(TestCase):
    def test_run_all_writes_results_json(self):
        repo_root = Path(__file__).resolve().parent.parent
        scenario_yaml = """
scenario_name: "cli_json_write"
hardware_catalog_id: "nvidia_dgx_b200_8gpu"
software_stack_id: "nvidia_vllm_cuda"
deployment_profile_id: "bare_metal_container"
weight_precision: "int8"
kv_cache_precision: "fp8"
workload_type: "fixed_batch"
workload:
  prompt_tokens: 1024
  max_output_tokens: 256
  num_parallel_jobs: 8
model_config:
  name: "Llama-70B-Instruct"
  total_params_billions: 70
  active_params_billions: 70
  hidden_size: 8192
  num_hidden_layers: 80
  num_attention_heads: 64
  num_key_value_heads: 8
  intermediate_size: 28672
  vocab_size: 128256
  max_position_embeddings: 131072
monte_carlo_enabled: false
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            scenario = tmp / "scenario.yaml"
            out_json = tmp / "results.json"
            scenario.write_text(scenario_yaml, encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, "run_all.py", str(scenario), "--json-out", str(out_json)],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )

            assert "JSON results written to:" in proc.stdout
            assert out_json.exists()
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            assert len(payload) == 1
            assert payload[0]["scenario_name"] == "cli_json_write"
