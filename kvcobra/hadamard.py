"""Randomized Hadamard transform (RHT) on the rank-``r`` latent.

After projecting a key onto its top-``r`` SVD directions the latent
coordinates have very unequal variances (they are the eigenvalues).
A uniform scalar quantizer would then waste bits on the small channels.
KV-COBRA fixes this with a fixed random sign flip followed by a Walsh–
Hadamard transform: every rotated channel has (approximately) the same
variance and is pushed toward a Gaussian marginal, so a *single* uniform
quantizer per head is enough (paper Sec. 3.3, Appendix A.3).

The rotation is orthogonal and involutory, so the inverse is the same
transform applied in reverse order.  The sign vector is generated from a
seed (``generate_signs``) and is the only random element of the whole
method — the five seeds of the paper are five sign draws.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def fast_walsh_hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """In-place-free butterfly WHT along the last dim (must be a power of 2).

    Returns ``H_d x`` (scaled by ``1/sqrt(d)`` when ``normalize``), in
    ``O(d log d)``.
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"Last dim must be power of 2, got {d}"

    result = x.clone()
    h = 1
    while h < d:
        result = result.reshape(*result.shape[:-1], -1, 2, h)
        a = result[..., 0, :].clone()   # clone: a, b are views of result
        b = result[..., 1, :].clone()
        result[..., 0, :] = a + b
        result[..., 1, :] = a - b
        result = result.reshape(*x.shape)
        h *= 2

    if normalize:
        result = result / math.sqrt(d)
    return result


def random_hadamard_transform(
    x: torch.Tensor,
    signs: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """RHT: ``H_d (D_s x)`` — sign flip, then WHT."""
    return fast_walsh_hadamard_transform(x * signs, normalize=normalize)


def inverse_random_hadamard_transform(
    x: torch.Tensor,
    signs: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Inverse RHT: ``D_s (H_d x)`` — WHT, then the same sign flip.

    Valid because the normalized WHT is its own inverse and ``D_s^2 = I``.
    """
    return fast_walsh_hadamard_transform(x, normalize=normalize) * signs


def generate_signs(d: int, seed: int = 42,
                   device: Optional[torch.device] = None) -> torch.Tensor:
    """Deterministic ±1 sign vector of length ``d`` from ``seed`` (float32)."""
    dev = device or torch.device("cpu")
    gen = torch.Generator(device=dev).manual_seed(seed)
    signs = torch.randint(0, 2, (d,), generator=gen, device=dev) * 2 - 1
    return signs.float()


def next_power_of_two(n: int) -> int:
    """Smallest power of two ``>= n`` (``n`` itself if already a power of two)."""
    return 1 << (n - 1).bit_length() if (n & (n - 1)) else n
