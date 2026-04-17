"""
Qwen3-235B-A22B architecture constants.
All values sourced from official HuggingFace config.json unless marked as [ASSUMPTION].
"""

from .config import ModelConfig


# Official config from https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json
QWEN3_235B_A22B = ModelConfig(
    name="Qwen3-235B-A22B",
    total_params_billions=235.0,       # Official
    active_params_billions=22.0,       # Official
    hidden_size=4096,                  # config.json ✓
    head_dim=128,                      # config.json ✓
    num_hidden_layers=94,              # config.json ✓
    num_attention_heads=64,            # config.json ✓ (query heads)
    num_key_value_heads=4,             # config.json ✓ (GQA)
    num_experts=128,                   # config.json ✓
    num_experts_per_tok=8,             # config.json ✓
    moe_intermediate_size=1536,        # config.json ✓
    intermediate_size=12288,           # config.json ✓ (shared dense MLP)
    vocab_size=151936,                 # config.json ✓
    max_position_embeddings=40960,     # config.json ✓ (native, extendable via YaRN)
    is_moe=True,
)


def get_default_model() -> ModelConfig:
    """Return the default Qwen3-235B-A22B model configuration."""
    return QWEN3_235B_A22B
