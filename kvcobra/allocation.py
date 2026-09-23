"""The two allocation levels of KV-COBRA (paper Sec. 3.1–3.2) and the KL reordering.

Notation (one KV head, head dimension ``d``):

* ``lam[i]``  – ``i``-th eigenvalue of the head's key covariance (descending).
* ``B``       – the head's bit budget, ``B = r * b`` (``bpd * d`` on average).
* ``D(r, b)`` – distortion of keeping the top-``r`` directions and quantizing
  them with ``b`` bits each,

      D(r, b) = sum_{i>r} lam[i]  +  q(b) * sum_{i<=r} lam[i],
      q(b)    = 2^(-2b) / 12                      (Bennett, uniform quantizer)

C1 (``optimal_rank_bits``)
    Given ``B``, pick ``(r*, b*)`` minimizing ``D`` subject to ``r*b <= B``
    over the even rank grid — an ``O(d/2)`` enumeration.

C2 (``water_filling_budgets``)
    Given the *average* budget, redistribute budget across all ``L x H``
    heads so that their marginal distortions equalize (damped proportional
    update, 5 rounds). This is the shipped solver of the paper.

KL variant (``query_variance`` + ``reorder_by_attention_kl``)
    Replace ``lam[i]`` by ``w[i] = sigma_Q,i^2 * sigma_K,i^2`` (query variance
    in the K-SVD basis times key eigenvalue) and re-sort the basis by ``w``.
    C1/C2 then minimize the attention-KL surrogate instead of key MSE
    (paper Sec. 3.4).
"""
from __future__ import annotations

import numpy as np
import torch

Key = tuple[int, int]          # (layer, kv_head)

MIN_BITS = 2
MAX_BITS = 8


def bennett_distortion(bits: int) -> float:
    """Per-dimension distortion coefficient ``q(b)`` of a uniform quantizer."""
    return (1.0 / 12.0) * (2.0 ** (-2 * bits))


# ─────────────────────────────────────────────────────────────────────────────
# C1 — per-head rank / bit-width
# ─────────────────────────────────────────────────────────────────────────────


ROUNDING_MODES = ("floor", "round")


def optimal_rank_bits(
    eigenvalues: np.ndarray,
    B: int,
    d: int = 128,
    min_rank: int = 2,
    min_bits: int = MIN_BITS,
    max_bits: int = MAX_BITS,
    rounding: str = "floor",
) -> tuple[int, int]:
    """C1: ``(r*, b*) = argmin D(r, b)`` over the even rank grid.

    Candidate ranks are ``2, 4, ..., min(d, B // min_bits)``; for each rank
    the bit-width is derived from ``B / r`` and clipped to
    ``[min_bits, max_bits]``.

    ``rounding`` selects how ``b`` is derived from ``B / r``:

    * ``"floor"`` – ``b = B // r``. Every candidate satisfies ``r * b <= B``.
      This is the released code and the default.
    * ``"round"`` – ``b = int(round(B / r))``. Legacy behaviour of the code
      that produced the paper's bit-sweep points at 1.5–4.0 bits/dim for
      seeds 43–45 (see ``docs/REPRODUCTION.md``). It can pick pairs with
      ``r * b`` slightly above ``B`` (e.g. ``B=256, r=34 → b=8``), which at
      2–3 bits/dim buys noticeably lower perplexity than the feasible grid.

    Args:
        eigenvalues: descending spectrum (length ``d``); may also be the
            KL weights ``w`` of the KL variant.
        B: integer bit budget for this head.
    Returns:
        ``(r, b)``.
    """
    if rounding not in ROUNDING_MODES:
        raise ValueError(f"rounding must be one of {ROUNDING_MODES}")
    lam = np.asarray(eigenvalues, dtype=np.float64)
    if lam.sum() < 1e-30:
        r = max(min_rank, B // max(min_bits, 1))
        return r, max(min_bits, B // max(r, 1))

    dist = {b: bennett_distortion(b) for b in range(2, 9)}

    r_max = min(d, max(min_rank, B // max(min_bits, 1)))
    rs = list(range(max(min_rank, 2), r_max + 1, 2))
    if not rs:
        rs = [min_rank]

    best_r, best_b, best_D = rs[0], min_bits, float("inf")
    for r in rs:
        b_int = B // r if rounding == "floor" else int(round(B / r))
        if b_int < min_bits:
            continue
        b_int = min(b_int, max_bits)
        proj_loss = lam[r:].sum() if r < len(lam) else 0.0
        quant_loss = dist.get(b_int, dist[min_bits]) * lam[:r].sum()
        D = proj_loss + quant_loss
        if D < best_D:
            best_D, best_r, best_b = D, r, b_int
    return best_r, best_b


def head_distortion(eigenvalues: np.ndarray, r: int, b: int) -> float:
    """``D(r, b)`` for a given spectrum (used by C2)."""
    ev = eigenvalues
    proj_loss = ev[r:].sum() if r < len(ev) else 0.0
    return proj_loss + bennett_distortion(b) * ev[:r].sum()


# ─────────────────────────────────────────────────────────────────────────────
# C2 — cross-head budget redistribution (damped water-filling)
# ─────────────────────────────────────────────────────────────────────────────


def uniform_budgets(avg_B: int, n_layers: int, n_heads: int) -> dict[Key, int]:
    """Every head gets the average budget (no C2)."""
    return {(li, hi): avg_B for li in range(n_layers) for hi in range(n_heads)}


def water_filling_budgets(
    eigvals: dict[Key, np.ndarray],
    avg_B: int,
    n_layers: int,
    n_heads: int,
    d: int = 128,
    iterations: int = 5,
    damping: float = 0.3,
    rounding: str = "floor",
) -> dict[Key, int]:
    """C2: equalize per-head distortion by moving budget between heads.

    Each round: evaluate every head at its current budget with C1, compare
    its distortion ``D_k`` to the mean, and move ``avg_B * (sqrt(D_k/mean) - 1)
    * damping`` bits toward heads whose distortion is above average. The
    total is then renormalized to ``avg_B * n_layers * n_heads``. Budgets
    never drop below ``B_min = max(4, avg_B // 4)``.

    ``rounding`` is forwarded to C1 (see :func:`optimal_rank_bits`).

    Returns:
        ``{(layer, head): integer budget}``.
    """
    budgets = uniform_budgets(avg_B, n_layers, n_heads)
    total_B = avg_B * n_layers * n_heads
    B_min = max(4, avg_B // 4)

    for _ in range(iterations):
        distortions: dict[Key, float] = {}
        for key, B in budgets.items():
            ev = eigvals[key]
            r, b = optimal_rank_bits(ev, B=B, d=d, min_rank=2, min_bits=2,
                                     rounding=rounding)
            b = max(2, min(8, b))
            r = max(2, min(d, B // b))
            distortions[key] = head_distortion(ev, r, b)

        mean_D = max(float(np.mean(list(distortions.values()))), 1e-30)
        for key in budgets:
            ratio = distortions[key] / mean_D
            adj = int(avg_B * (np.sqrt(ratio) - 1) * damping)
            budgets[key] = max(B_min, budgets[key] + adj)

        current = sum(budgets.values())
        if current > 0:
            for key in budgets:
                budgets[key] = max(B_min, int(budgets[key] * total_B / current))

    return budgets


# ─────────────────────────────────────────────────────────────────────────────
# KL variant — query-aware reordering of the SVD basis
# ─────────────────────────────────────────────────────────────────────────────


def query_variance(
    raw_q: dict[Key, torch.Tensor],
    k_basis: dict[Key, tuple[torch.Tensor, torch.Tensor]],
    n_layers: int,
    n_heads: int,
) -> dict[Key, np.ndarray]:
    """``sigma_Q,i^2``: variance of the (centered) queries along each K-SVD direction.

    Computed in fp32 on CPU from the raw calibration queries.
    """
    q_var: dict[Key, np.ndarray] = {}
    for li in range(n_layers):
        for hi in range(n_heads):
            Q_raw = raw_q.get((li, hi))
            V_full, _ = k_basis.get((li, hi), (None, None))
            if Q_raw is None or V_full is None:
                continue
            Q_f = Q_raw.float().cpu()
            V_f = V_full.float().cpu()
            Q_rot = (Q_f - Q_f.mean(0, keepdim=True)) @ V_f
            q_var[(li, hi)] = (Q_rot * Q_rot).mean(0).cpu().numpy()
    return q_var


def reorder_by_attention_kl(
    eig: dict[Key, np.ndarray],
    basis: dict[Key, tuple[torch.Tensor, torch.Tensor]],
    q_var: dict[Key, np.ndarray] | None,
    n_layers: int,
    n_heads: int,
):
    """Sort every head's directions by ``w_i = sigma_Q,i^2 * sigma_K,i^2``.

    Returns:
        ``(w_eig, reordered_basis)`` — the sorted weights (fed to C1/C2 in
        place of the eigenvalues) and the basis with columns permuted to
        match. If ``q_var`` is ``None`` the weights reduce to the eigenvalues.
    """
    w_eig: dict[Key, np.ndarray] = {}
    reord_basis: dict[Key, tuple[torch.Tensor, np.ndarray]] = {}
    for li in range(n_layers):
        for hi in range(n_heads):
            if (li, hi) not in eig or (li, hi) not in basis:
                continue
            ev = np.asarray(eig[(li, hi)], dtype=np.float64)
            qv = np.asarray(
                q_var.get((li, hi), np.ones_like(ev)) if q_var else np.ones_like(ev),
                dtype=np.float64)
            w = np.maximum(qv, 1e-30) * np.maximum(ev, 1e-30)
            order = np.argsort(w)[::-1]
            w_eig[(li, hi)] = w[order]
            V_full, _ = basis[(li, hi)]
            idx = torch.from_numpy(order.copy()).long()
            reord_basis[(li, hi)] = (V_full[:, idx], w[order])
    return w_eig, reord_basis
