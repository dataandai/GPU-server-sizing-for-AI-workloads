import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.catalog_loader import get_gpu_record, resolve_hardware_profile
import src.model_loader as model_loader
from src.model_loader import (
    _download_json_file,
    _download_text_file,
    load_model_config_from_huggingface,
)
from src.workloads import load_scenario


class TestCatalogLoader(TestCase):
    def test_resolve_hardware_profile_from_catalog(self):
        profile = resolve_hardware_profile("nvidia_dgx_b200_8gpu")
        self.assertEqual(profile.catalog_id, "nvidia_dgx_b200_8gpu")
        self.assertEqual(profile.num_gpus, 8)
        self.assertEqual(profile.vram_per_gpu_gb, 180.0)
        self.assertEqual(profile.interconnect, "nvlink")
        self.assertEqual(profile.gpu_id, "nvidia_b200_sxm_180gb")


    def test_nvfp4_only_exposed_for_blackwell_gpu_catalog_entries(self):
        h200 = get_gpu_record("nvidia_h200_sxm5_141gb")
        b200 = get_gpu_record("nvidia_b200_sxm_180gb")
        self.assertNotIn("nvfp4", [str(x).lower() for x in h200['default_precision_targets']])
        self.assertIn("nvfp4", [str(x).lower() for x in b200['default_precision_targets']])

    def test_load_scenario_with_catalog_hardware_and_inline_model(self):
        scenario_yaml = """
scenario_name: "catalog_inline_model"
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
            scenario_path = Path(tmpdir) / "scenario.yaml"
            scenario_path.write_text(scenario_yaml, encoding="utf-8")
            config = load_scenario(str(scenario_path))

        self.assertEqual(config.hardware_catalog_id, "nvidia_dgx_b200_8gpu")
        self.assertIsNotNone(config.hardware_config)
        self.assertEqual(config.tensor_parallel_degree, 8)
        self.assertEqual(config.software_stack_id, "nvidia_vllm_cuda")
        self.assertEqual(config.deployment_profile_id, "bare_metal_container")
        self.assertEqual(config.model_config.name, "Llama-70B-Instruct")
        self.assertEqual(config.model_config.total_params_billions, 70.0)


class TestHuggingFaceModelLoader(TestCase):
    def setUp(self):
        load_model_config_from_huggingface.cache_clear()
        _download_text_file.cache_clear()
        _download_json_file.cache_clear()

    def test_load_model_config_from_hf_mocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config.json").write_text(
                """
{
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 14336,
  "vocab_size": 151936,
  "max_position_embeddings": 32768,
  "tie_word_embeddings": false
}
""".strip(),
                encoding="utf-8",
            )
            (root / "README.md").write_text(
                "Model card for Example-14B. 14B parameters, 14B active parameters.",
                encoding="utf-8",
            )

            class FakeApi:
                def __init__(self, *args, **kwargs):
                    pass

                def model_info(self, repo_id, revision=None, files_metadata=False):
                    return SimpleNamespace(id=repo_id, cardData={"params": "14B"})

            def fake_download(repo_id, filename, revision=None, local_files_only=False, **kwargs):
                return str(root / filename)

            with mock.patch("src.model_loader.HfApi", FakeApi), mock.patch(
                "src.model_loader.hf_hub_download", side_effect=fake_download
            ):
                cfg = load_model_config_from_huggingface("acme/Example-14B")

        self.assertEqual(cfg.name, "acme/Example-14B")
        self.assertEqual(cfg.total_params_billions, 14.0)
        self.assertEqual(cfg.active_params_billions, 14.0)
        self.assertEqual(cfg.head_dim, 128)
        self.assertFalse(cfg.is_moe)


    def test_load_model_config_from_hf_local_files_only_skips_model_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config.json").write_text(
                "{\"hidden_size\":4096,\"num_hidden_layers\":32,\"num_attention_heads\":32,\"num_key_value_heads\":8,\"intermediate_size\":14336,\"vocab_size\":151936,\"max_position_embeddings\":32768}",
                encoding="utf-8",
            )
            (root / "README.md").write_text(
                "Example-14B cached model. 14B parameters.",
                encoding="utf-8",
            )

            def fake_download(repo_id, filename, revision=None, local_files_only=False, **kwargs):
                return str(root / filename)

            with mock.patch("src.model_loader.HfApi", side_effect=AssertionError("HfApi should not be called")), mock.patch(
                "src.model_loader.hf_hub_download", side_effect=fake_download
            ):
                cfg = load_model_config_from_huggingface("https://huggingface.co/acme/Example-14B", local_files_only=True)

        self.assertEqual(cfg.name, "acme/Example-14B")
        self.assertEqual(cfg.total_params_billions, 14.0)

    def test_model_config_from_mapping_accepts_mixtral_aliases(self):
        cfg = model_loader.model_config_from_mapping({
            "name": "mixtral-test",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "model_type": "mixtral",
            "total_params_billions": 46.7,
            "active_params_billions": 12.9,
        })
        self.assertTrue(cfg.is_moe)
        self.assertEqual(cfg.num_experts, 8)
        self.assertEqual(cfg.num_experts_per_tok, 2)

    def test_model_config_from_mapping_accepts_deepseek_aliases(self):
        cfg = model_loader.model_config_from_mapping({
            "name": "deepseek-test",
            "hidden_size": 7168,
            "num_attention_heads": 128,
            "num_hidden_layers": 61,
            "num_key_value_heads": 128,
            "intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "model_type": "deepseek_v3",
            "total_params_billions": 671.0,
            "active_params_billions": 37.0,
        })
        self.assertTrue(cfg.is_moe)
        self.assertEqual(cfg.num_experts, 256)
        self.assertEqual(cfg.num_experts_per_tok, 8)
        self.assertEqual(cfg.moe_intermediate_size, 2048)

