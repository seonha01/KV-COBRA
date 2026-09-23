"""One-shot calibration: collect pre-RoPE K/V (and Q) activations, eigendecompose.

KV-COBRA only needs *second-order statistics* of each KV head:

* ``k_eig[(layer, head)]``   – eigenvalues of the uncentered key covariance
  ``K^T K / n`` (sorted descending),
* ``k_basis[(layer, head)]`` – the matching eigenvectors (the per-head SVD
  basis that the compressor projects onto),
* the same for values, and
* ``raw_q[(layer, head)]``   – raw query activations, used only by the KL
  variant to weight SVD directions by the query variance.

Everything is collected with forward hooks on ``k_proj`` / ``v_proj`` /
``q_proj`` **before** rotary embeddings are applied, which is the space in
which KV-COBRA compresses keys (see paper, Sec. 3 and Appendix G).

The default calibration set matches the paper: the first 32 WikiText-2
*train* paragraphs with at least 200 characters, each truncated to 1024
tokens.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .arch import ModelArch, locate_layers


# ─────────────────────────────────────────────────────────────────────────────
# Calibration texts
# ─────────────────────────────────────────────────────────────────────────────


def load_calibration_texts(
    n_samples: int = 32,
    min_length: int = 200,
    dataset_name: str = "Salesforce/wikitext",
    subset: str = "wikitext-2-raw-v1",
    split: str = "train",
) -> list[str]:
    """Return the first ``n_samples`` texts of ``split`` with ``len >= min_length``.

    Iteration order is the dataset order, so the selection is deterministic
    (this is what makes the paper numbers reproducible: same texts → same
    covariance → same allocation).
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, subset, split=split)
    texts: list[str] = []
    for item in dataset:
        text = item.get("text", "")
        if len(text) >= min_length:
            texts.append(text)
        if len(texts) >= n_samples:
            break
    return texts


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Calibration:
    """Per-(layer, kv_head) second-order statistics of one model.

    Attributes:
        k_eig / v_eig:     ``{(layer, head): eigenvalues[d]}`` sorted descending.
        k_basis / v_basis: ``{(layer, head): (V_fp16[d, d], sqrt_eig_fp16[d])}``.
            Column ``i`` of ``V`` is the eigenvector of eigenvalue ``i``.
        n_cal_tokens:      number of calibration tokens per head.
        raw_q:             ``{(layer, head): Q[n_tokens, d]}`` on CPU; queries of
            one GQA group are averaged so there is exactly one entry per KV
            head. ``None`` if queries were not collected.
    """
    k_eig: dict[tuple[int, int], np.ndarray]
    v_eig: dict[tuple[int, int], np.ndarray]
    k_basis: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]
    v_basis: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]
    n_cal_tokens: int
    raw_q: dict[tuple[int, int], torch.Tensor] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Activation collection
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _collect_activations(
    model,
    tokenizer,
    arch: ModelArch,
    texts: Sequence[str],
    max_seq_len: int,
    collect_queries: bool,
):
    """Run ``texts`` through the model, capturing per-head K, V (and Q) on CPU."""
    layers = locate_layers(model)
    dev = next(model.parameters()).device
    nH, d = arch.n_kv_heads, arch.head_dim
    nQ = arch.n_heads
    group = max(1, nQ // nH)

    k_data: dict[tuple[int, int], list[torch.Tensor]] = {}
    v_data: dict[tuple[int, int], list[torch.Tensor]] = {}
    q_data: dict[tuple[int, int], list[torch.Tensor]] | None = (
        {} if collect_queries else None)
    hooks = []

    def make_kv_hook(li: int, target: dict):
        def fn(_mod, _args, out):
            reshaped = out.reshape(out.shape[0], out.shape[1], nH, d)
            for h in range(nH):
                target.setdefault((li, h), []).append(
                    reshaped[0, :, h, :].detach().cpu())
        return fn

    def make_q_hook(li: int):
        def fn(_mod, _args, out):
            reshaped = out.reshape(out.shape[0], out.shape[1], nQ, d)
            # Average the query heads of each GQA group: that is the object
            # the softmax actually compares against this KV head's keys.
            for kh in range(nH):
                start = kh * group
                q_group = reshaped[0, :, start:start + group, :].mean(dim=1)
                q_data.setdefault((li, kh), []).append(q_group.detach().cpu())
        return fn

    for li in range(arch.n_layers):
        attn = layers[li].self_attn
        hooks.append(attn.k_proj.register_forward_hook(make_kv_hook(li, k_data)))
        hooks.append(attn.v_proj.register_forward_hook(make_kv_hook(li, v_data)))
        if collect_queries:
            hooks.append(attn.q_proj.register_forward_hook(make_q_hook(li)))

    model.eval()
    for text in texts:
        inp = tokenizer(text, return_tensors="pt",
                        max_length=max_seq_len, truncation=True).to(dev)
        if inp.input_ids.shape[1] < 10:
            continue
        model(**inp, use_cache=False)

    for h in hooks:
        h.remove()

    k_cat = {key: torch.cat(v, dim=0) for key, v in k_data.items()}
    v_cat = {key: torch.cat(v, dim=0) for key, v in v_data.items()}
    q_cat = ({key: torch.cat(v, dim=0) for key, v in q_data.items()}
             if q_data is not None else None)
    return k_cat, v_cat, q_cat


def _eigendecompose(data: dict[tuple[int, int], torch.Tensor], device):
    """Sorted-descending eigendecomposition of ``X^T X / n`` for every head.

    Computed in fp32 on ``device``; eigenvalues are returned as float64-free
    numpy arrays (fp32 precision) and the basis is stored in fp16, exactly as
    in the paper's code (the compressor later up-casts ``V_r`` to fp32).
    """
    eigs: dict[tuple[int, int], np.ndarray] = {}
    bases: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for key, X_cpu in data.items():
        X = X_cpu.to(device).float()
        n = X.shape[0]
        cov = (X.T @ X) / max(n, 1)
        evals, evecs = torch.linalg.eigh(cov)      # ascending
        evals = evals.flip(0).clamp(min=0)          # → descending, no negatives
        evecs = evecs.flip(1)
        eigs[key] = evals.cpu().numpy()
        bases[key] = (evecs.half(), evals.sqrt().half())
        del X, cov
    return eigs, bases


def collect_calibration(
    model,
    tokenizer,
    arch: ModelArch,
    texts: Sequence[str],
    max_seq_len: int = 1024,
    collect_queries: bool = True,
) -> Calibration:
    """Collect activations and eigendecompose; the single entry point.

    Args:
        model: HuggingFace causal LM (fp16) already on GPU.
        tokenizer: matching tokenizer.
        arch: output of :func:`kvcobra.get_model_arch`.
        texts: calibration texts (see :func:`load_calibration_texts`).
        max_seq_len: truncation length per text (paper: 1024).
        collect_queries: keep raw queries (required by the KL variant).

    Returns:
        :class:`Calibration`.
    """
    dev = next(model.parameters()).device
    k_data, v_data, q_data = _collect_activations(
        model, tokenizer, arch, texts, max_seq_len, collect_queries)
    n_tokens = next(iter(k_data.values())).shape[0] if k_data else 0

    k_eig, k_basis = _eigendecompose(k_data, dev)
    v_eig, v_basis = _eigendecompose(v_data, dev)

    del k_data, v_data
    gc.collect()
    torch.cuda.empty_cache()

    return Calibration(
        k_eig=k_eig, v_eig=v_eig,
        k_basis=k_basis, v_basis=v_basis,
        n_cal_tokens=n_tokens,
        raw_q=q_data,
    )
