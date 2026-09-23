"""Hook-based pre-RoPE KV compressor (simulated quantization).

``KVCompressor`` registers forward hooks on ``self_attn.k_proj`` and/or
``self_attn.v_proj`` of every layer and replaces the projection output, head
by head, with its compressed-then-reconstructed version. Attention therefore
runs on exactly the tensor a real compressed cache would decode to, while
the rest of the model is untouched. This is the "fake-quant" path used for
all accuracy numbers in the paper.

Per-head configuration (one dict per ``(layer, kv_head)``)::

    {"method": "svdq",
     "V_r": <fp32 tensor d x r>,        # top-r SVD basis (possibly KL-reordered)
     "bits": b,                         # uniform bit-width for the r latents
     "hadamard_signs": <fp32 tensor>}   # ±1 vector of length next_pow2(r)

The compression of one head with ``T`` tokens, ``x ∈ R^{T x d}``::

    z      = x @ V_r                              # project        [T, r]
    z_rot  = RHT(pad(z))                          # rotate         [T, r']
    z_hat  = uniform_quant(z_rot, b)              # per-channel min/max over T
    x_hat  = unpad(RHT^-1(z_hat)) @ V_r^T         # back-project   [T, d]

Note the quantizer is *per channel over the tokens seen in the hook call*,
so results depend on the evaluation protocol (window size for PPL, prompt
for generation) — the runners in ``scripts/`` fix these to the paper's.
"""
from __future__ import annotations

from typing import Any

import torch

from .arch import locate_layers
from .hadamard import (
    inverse_random_hadamard_transform,
    next_power_of_two,
    random_hadamard_transform,
)

Key = tuple[int, int]


def uniform_quant(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-channel asymmetric uniform quant-dequant (min/max over dim 0)."""
    n_levels = 2 ** bits
    mn = x.min(dim=0).values
    mx = x.max(dim=0).values
    scale = ((mx - mn) / (n_levels - 1)).clamp(min=1e-8)
    q = ((x - mn) / scale).round().clamp(0, n_levels - 1)
    return q * scale + mn


def svdq_compress(
    x: torch.Tensor,
    V_r: torch.Tensor,
    bits: int,
    hadamard_signs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project → (RHT) → uniform quant → (RHT⁻¹) → back-project. See module doc."""
    z = x @ V_r
    if hadamard_signs is not None:
        r_orig = z.shape[-1]
        padded = next_power_of_two(r_orig)
        z_p = torch.nn.functional.pad(z, (0, padded - r_orig)) if padded != r_orig else z
        z_rot = random_hadamard_transform(z_p, hadamard_signs, normalize=True)
        z_rot_q = uniform_quant(z_rot, bits)
        z_hat_p = inverse_random_hadamard_transform(z_rot_q, hadamard_signs, normalize=True)
        z_hat = z_hat_p[..., :r_orig]
    else:
        z_hat = uniform_quant(z, bits)
    return z_hat @ V_r.T


class KVCompressor:
    """Install/remove forward hooks that compress K and/or V per head.

    Args:
        k_configs: ``{(layer, head): cfg}`` for keys; empty dict = keys untouched.
        v_configs: same for values.
        n_layers, n_kv_heads, head_dim: model geometry.

    Use as a context manager or call :meth:`install_hooks` /
    :meth:`remove_hooks` explicitly.
    """

    def __init__(
        self,
        k_configs: dict[Key, dict[str, Any]],
        v_configs: dict[Key, dict[str, Any]],
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        self.k_configs = k_configs
        self.v_configs = v_configs
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self._hooks: list[Any] = []
        self._model = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def install_hooks(self, model) -> None:
        layers = locate_layers(model)
        for li in range(self.n_layers):
            attn = layers[li].self_attn
            if self.k_configs:
                self._hooks.append(attn.k_proj.register_forward_hook(
                    self._make_hook(li, self.k_configs)))
            if self.v_configs:
                self._hooks.append(attn.v_proj.register_forward_hook(
                    self._make_hook(li, self.v_configs)))
        self._model = model

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._model = None

    def attach(self, model):
        """``with comp.attach(model): ...`` — hooks removed on exit."""
        self._pending_model = model
        return self

    def __enter__(self):
        self.install_hooks(self._pending_model)
        return self

    def __exit__(self, *exc):
        self.remove_hooks()
        return False

    # ── summary helpers ──────────────────────────────────────────────────

    def allocation_table(self, side: str = "k") -> list[dict]:
        """Rows of ``{layer, head, rank, bits, budget}`` for inspection/export."""
        cfgs = self.k_configs if side == "k" else self.v_configs
        rows = []
        for (li, hi), cfg in sorted(cfgs.items()):
            r = int(cfg["V_r"].shape[-1])
            rows.append({"layer": li, "head": hi, "rank": r,
                         "bits": int(cfg["bits"]), "budget": r * int(cfg["bits"])})
        return rows

    # ── internals ────────────────────────────────────────────────────────

    @staticmethod
    def _compress_head(x: torch.Tensor, cfg: dict) -> torch.Tensor:
        m = cfg["method"]
        if m == "none":
            return x
        if m == "svdq":
            signs = cfg.get("hadamard_signs")
            if signs is not None and signs.device != x.device:
                signs = signs.to(x.device)
            if cfg["V_r"].device != x.device:
                cfg["V_r"] = cfg["V_r"].to(x.device)
            return svdq_compress(x, cfg["V_r"], cfg["bits"], hadamard_signs=signs)
        raise ValueError(f"Unknown compression method: {m}")

    def _make_hook(self, layer_idx: int, configs: dict):
        nH, d = self.n_kv_heads, self.head_dim

        def hook_fn(module, args, output: torch.Tensor) -> torch.Tensor:
            bsz, seq_len, _ = output.shape
            reshaped = output.reshape(bsz, seq_len, nH, d)
            for h in range(nH):
                cfg = configs.get((layer_idx, h))
                if cfg is None or cfg["method"] == "none":
                    continue
                x = reshaped[0, :, h, :].float()
                x_hat = self._compress_head(x, cfg)
                reshaped[0, :, h, :] = x_hat.to(reshaped.dtype)
            return reshaped.reshape(bsz, seq_len, -1)

        return hook_fn
