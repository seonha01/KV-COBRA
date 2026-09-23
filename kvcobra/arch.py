"""Model geometry helpers.

Everything KV-COBRA needs to know about a HuggingFace causal LM is the
number of layers, the number of *KV* heads (GQA models have fewer KV heads
than query heads) and the per-head dimension ``d``. ``get_model_arch`` reads
these from the HF config without loading weights; ``locate_layers`` returns
the ``nn.ModuleList`` of decoder blocks so we can attach forward hooks to
``self_attn.k_proj`` / ``self_attn.v_proj``.

Tested with Llama-3.1-8B, Mistral-7B-v0.3 and Qwen2.5-7B-Instruct (the three
models of the paper).
"""
from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoConfig


@dataclass(frozen=True)
class ModelArch:
    """Static geometry of a decoder-only transformer."""
    model_id: str
    n_layers: int
    n_heads: int        # query heads
    n_kv_heads: int     # key/value heads (== n_heads without GQA)
    head_dim: int
    hidden_size: int

    @property
    def gqa_group(self) -> int:
        """Number of query heads that share one KV head."""
        return max(1, self.n_heads // self.n_kv_heads)


def get_model_arch(model_id: str) -> ModelArch:
    """Read (layers, heads, kv_heads, head_dim) from a HuggingFace config.

    Multimodal configs that nest the LM under ``text_config`` are handled
    transparently.
    """
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", cfg)

    n_layers = text_cfg.num_hidden_layers
    n_heads = text_cfg.num_attention_heads
    n_kv_heads = getattr(text_cfg, "num_key_value_heads", n_heads)
    hidden = text_cfg.hidden_size
    head_dim = getattr(text_cfg, "head_dim", None) or (hidden // n_heads)

    return ModelArch(
        model_id=model_id,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
    )


def locate_layers(model):
    """Return the decoder-block ``ModuleList`` regardless of model wrapping.

    Supports ``model.model.layers`` (Llama / Mistral / Qwen) and the
    ``language_model`` wrappers used by multimodal checkpoints.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
    raise RuntimeError(
        f"Cannot locate transformer layers in {type(model).__name__}")
