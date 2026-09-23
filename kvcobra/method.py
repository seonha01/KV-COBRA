"""Assemble a KV-COBRA compressor from a calibration (paper Sec. 3).

Two variants share everything except the per-direction weights used by the
allocator:

* ``"mse"``  – C1/C2 on the key eigenvalues (reconstruction-MSE objective).
* ``"kl"``   – C1/C2 on ``w_i = sigma_Q,i^2 * sigma_K,i^2`` with the SVD basis
  re-sorted by ``w`` (attention-KL objective). **This is the main method.**

Both then quantize the retained latents with a random Hadamard rotation and
a single uniform bit-width per head (``KVCompressor``).

Sides:
    ``k_only``        compress keys only (all main-paper numbers)
    ``v_only``        compress values only
    ``kv_symmetric``  same bpd on K and V
    ``kv_asymmetric`` total ``2*bpd`` split as ``k_fraction`` / ``1-k_fraction``

The V side never uses C2 or KL reordering (its spectrum is nearly flat, so
equalizing marginal distortion just starves heads); it gets a uniform
per-head budget and plain C1, as in the paper's K+V appendix.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .allocation import (
    optimal_rank_bits,
    query_variance,
    reorder_by_attention_kl,
    uniform_budgets,
    water_filling_budgets,
)
from .arch import ModelArch
from .calibration import Calibration
from .compressor import KVCompressor
from .hadamard import generate_signs, next_power_of_two

Key = tuple[int, int]

VARIANT_NAMES = {"mse": "KV-COBRA-MSE", "kl": "KV-COBRA-KL"}


@dataclass(frozen=True)
class KVCobraSpec:
    """One experiment cell."""
    variant: str = "kl"                # "kl" (main) or "mse"
    bits_per_dim: float = 1.0          # average bits per key dimension
    kv_side: str = "k_only"
    k_fraction: float = 0.5            # only for kv_asymmetric
    hadamard_seed: int = 42            # the only random element
    c1_rounding: str = "floor"         # "floor" (released) | "round" (legacy, see allocation.py)

    @property
    def name(self) -> str:
        return VARIANT_NAMES[self.variant]

    def __post_init__(self):
        if self.variant not in VARIANT_NAMES:
            raise ValueError(f"variant must be one of {list(VARIANT_NAMES)}")
        if self.kv_side not in ("k_only", "v_only", "kv_symmetric", "kv_asymmetric"):
            raise ValueError(f"unknown kv_side {self.kv_side!r}")
        if self.c1_rounding not in ("floor", "round"):
            raise ValueError("c1_rounding must be 'floor' or 'round'")


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────


def _rank_bit_configs(eig, basis, nL: int, nH: int, d: int,
                      budgets: dict[Key, int], min_rank: int = 2,
                      rounding: str = "floor") -> dict[Key, dict]:
    """Run C1 on every head at its budget and emit ``svdq`` configs."""
    cfgs: dict[Key, dict] = {}
    for li in range(nL):
        for hi in range(nH):
            B = budgets[(li, hi)]
            ev = eig[(li, hi)]
            V_full, _ = basis[(li, hi)]
            r, b = optimal_rank_bits(ev, B=B, d=d, min_rank=min_rank, min_bits=2,
                                     rounding=rounding)
            b = max(2, min(8, b))
            r = max(min_rank, min(d, B // b))
            cfgs[(li, hi)] = {
                "method": "svdq",
                "V_r": V_full[:, :r].float(),
                "bits": b,
            }
    return cfgs


def _attach_hadamard(cfgs: dict[Key, dict], seed: int,
                     cache: dict[tuple[int, int], torch.Tensor]) -> dict[Key, dict]:
    """Give every head a ±1 sign vector sized to its padded rank (cached per size)."""
    for cfg in cfgs.values():
        if cfg.get("method") != "svdq":
            continue
        r = cfg["V_r"].shape[-1]
        padded = next_power_of_two(r)
        key = (padded, seed)
        if key not in cache:
            cache[key] = generate_signs(padded, seed=seed, device="cpu").float()
        cfg["hadamard_signs"] = cache[key]
    return cfgs


# ─────────────────────────────────────────────────────────────────────────────
# Public builder
# ─────────────────────────────────────────────────────────────────────────────


def build_kv_cobra(
    arch: ModelArch,
    calib: Calibration,
    spec: KVCobraSpec | None = None,
    **spec_kwargs,
) -> KVCompressor:
    """Build the compressor for one cell.

    Either pass a :class:`KVCobraSpec` or its fields as keyword arguments::

        comp = build_kv_cobra(arch, calib, variant="kl", bits_per_dim=1.0,
                              hadamard_seed=43)

    The per-head ``(rank, bits)`` table is available afterwards through
    ``comp.allocation_table()``.
    """
    spec = spec or KVCobraSpec(**spec_kwargs)
    d = arch.head_dim
    nL, nH = arch.n_layers, arch.n_kv_heads
    sign_cache: dict[tuple[int, int], torch.Tensor] = {}

    if spec.variant == "kl":
        q_var = (query_variance(calib.raw_q, calib.k_basis, nL, nH)
                 if calib.raw_q is not None else None)
    else:
        q_var = None

    rnd = spec.c1_rounding

    def k_side(bpd: float) -> dict[Key, dict]:
        B_avg = int(bpd * d)
        if spec.variant == "kl":
            w_eig, reord_basis = reorder_by_attention_kl(
                calib.k_eig, calib.k_basis, q_var, nL, nH)
            budgets = water_filling_budgets(w_eig, B_avg, nL, nH, d, rounding=rnd)
            cfgs = _rank_bit_configs(w_eig, reord_basis, nL, nH, d, budgets, rounding=rnd)
        else:
            budgets = water_filling_budgets(calib.k_eig, B_avg, nL, nH, d, rounding=rnd)
            cfgs = _rank_bit_configs(calib.k_eig, calib.k_basis, nL, nH, d, budgets,
                                     rounding=rnd)
        return _attach_hadamard(cfgs, spec.hadamard_seed, sign_cache)

    def v_side(bpd: float) -> dict[Key, dict]:
        B_avg = int(bpd * d)
        budgets = uniform_budgets(B_avg, nL, nH)
        cfgs = _rank_bit_configs(calib.v_eig, calib.v_basis, nL, nH, d, budgets,
                                 rounding=rnd)
        return _attach_hadamard(cfgs, spec.hadamard_seed, sign_cache)

    if spec.kv_side == "k_only":
        return KVCompressor(k_side(spec.bits_per_dim), {}, nL, nH, d)
    if spec.kv_side == "v_only":
        return KVCompressor({}, v_side(spec.bits_per_dim), nL, nH, d)
    if spec.kv_side == "kv_symmetric":
        return KVCompressor(k_side(spec.bits_per_dim), v_side(spec.bits_per_dim),
                            nL, nH, d)
    total = 2 * spec.bits_per_dim
    return KVCompressor(k_side(total * spec.k_fraction),
                        v_side(total * (1 - spec.k_fraction)), nL, nH, d)
