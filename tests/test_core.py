"""Fast CPU unit tests (no model download): run ``pytest tests/``."""
import numpy as np
import torch

from kvcobra.allocation import (optimal_rank_bits, water_filling_budgets,
                                uniform_budgets, reorder_by_attention_kl)
from kvcobra.compressor import KVCompressor, svdq_compress, uniform_quant
from kvcobra.hadamard import (fast_walsh_hadamard_transform, generate_signs,
                              inverse_random_hadamard_transform,
                              next_power_of_two, random_hadamard_transform)


def _spectrum(d=128, decay=0.05, seed=0):
    rng = np.random.default_rng(seed)
    lam = np.exp(-decay * np.arange(d)) * (1 + 0.1 * rng.random(d))
    return np.sort(lam)[::-1]


def test_hadamard_is_orthonormal_and_involutory():
    x = torch.randn(64, 128, dtype=torch.float64)
    signs = generate_signs(128, seed=7).double()
    y = random_hadamard_transform(x, signs)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1))          # orthonormal
    assert torch.allclose(inverse_random_hadamard_transform(y, signs), x)
    h = fast_walsh_hadamard_transform(torch.eye(8, dtype=torch.float64))
    assert torch.allclose(h @ h.T, torch.eye(8, dtype=torch.float64))


def test_signs_are_deterministic():
    assert torch.equal(generate_signs(32, seed=43), generate_signs(32, seed=43))
    assert not torch.equal(generate_signs(32, seed=43), generate_signs(32, seed=44))
    assert next_power_of_two(48) == 64 and next_power_of_two(64) == 64


def test_c1_is_feasible_and_prefers_rank_when_spectrum_is_flat():
    lam = _spectrum()
    for B in (16, 32, 64, 128, 256, 512):
        r, b = optimal_rank_bits(lam, B=B, d=128)
        assert 2 <= r <= 128 and 2 <= b <= 8 and r * b <= B
    r_flat, _ = optimal_rank_bits(np.ones(128), B=256, d=128)
    r_steep, _ = optimal_rank_bits(_spectrum(decay=0.5), B=256, d=128)
    assert r_flat > r_steep


def test_c2_conserves_total_budget_and_respects_floor():
    nL, nH, d, avg_B = 4, 2, 128, 128
    eig = {(l, h): _spectrum(decay=0.02 * (1 + l + h), seed=l * 10 + h)
           for l in range(nL) for h in range(nH)}
    budgets = water_filling_budgets(eig, avg_B, nL, nH, d)
    assert set(budgets) == set(eig)
    assert min(budgets.values()) >= max(4, avg_B // 4)
    assert abs(sum(budgets.values()) - avg_B * nL * nH) <= 0.02 * avg_B * nL * nH
    assert len(set(budgets.values())) > 1          # C2 actually redistributes
    assert uniform_budgets(avg_B, nL, nH) == {k: avg_B for k in eig}


def test_kl_reorder_sorts_by_weight_and_permutes_basis():
    d = 16
    rng = np.random.default_rng(3)
    ev = np.sort(rng.random(d) + 0.5)[::-1]
    V = torch.eye(d)
    qv = rng.random(d) + 0.5               # random query weights → non-trivial permutation
    w, basis = reorder_by_attention_kl({(0, 0): ev}, {(0, 0): (V, None)},
                                       {(0, 0): qv}, 1, 1)
    assert np.all(np.diff(w[(0, 0)]) <= 0)
    V_reord, _ = basis[(0, 0)]
    assert torch.equal(V_reord[:, 0], V[:, np.argmax(ev * qv)])


def test_svdq_reconstruction_error_decreases_with_bits_and_rank():
    torch.manual_seed(0)
    T, d = 256, 32
    V = torch.linalg.qr(torch.randn(d, d))[0]
    x = torch.randn(T, d) @ torch.diag(torch.linspace(4, 0.1, d)) @ V.T
    err = {}
    for r in (8, 32):
        for b in (2, 4, 8):
            signs = generate_signs(next_power_of_two(r), seed=1)
            x_hat = svdq_compress(x, V[:, :r], b, hadamard_signs=signs)
            err[(r, b)] = (x - x_hat).pow(2).mean().item()
    assert err[(32, 8)] < err[(32, 4)] < err[(32, 2)]
    assert err[(32, 4)] < err[(8, 4)]
    assert err[(32, 8)] < 1e-3


def test_uniform_quant_hits_extremes():
    x = torch.tensor([[0.0, -1.0], [1.0, 1.0], [0.5, 0.0]])
    q = uniform_quant(x, bits=2)
    assert torch.allclose(q.min(0).values, x.min(0).values)
    assert torch.allclose(q.max(0).values, x.max(0).values)


def test_compressor_hooks_change_only_configured_heads():
    torch.manual_seed(0)
    nL, nH, d, T = 2, 2, 8, 5

    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.k_proj = torch.nn.Linear(d, nH * d, bias=False)
            self.v_proj = torch.nn.Linear(d, nH * d, bias=False)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer() for _ in range(nL)])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    m = Model()
    x = torch.randn(1, T, d)
    ref = m.model.layers[0].self_attn.k_proj(x).reshape(1, T, nH, d)
    V = torch.eye(d)
    cfg = {(0, 0): {"method": "svdq", "V_r": V[:, :4], "bits": 2,
                    "hadamard_signs": generate_signs(4, seed=1)}}
    comp = KVCompressor(cfg, {}, nL, nH, d)
    with comp.attach(m):
        out = m.model.layers[0].self_attn.k_proj(x).reshape(1, T, nH, d)
    assert torch.equal(out[0, :, 1], ref[0, :, 1])        # untouched head
    assert not torch.equal(out[0, :, 0], ref[0, :, 0])    # compressed head
    assert comp.allocation_table() == [{"layer": 0, "head": 0, "rank": 4, "bits": 2, "budget": 8}]
    after = m.model.layers[0].self_attn.k_proj(x).reshape(1, T, nH, d)
    assert torch.equal(after, ref)                        # hooks removed
