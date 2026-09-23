"""Sliding-window perplexity (paper protocol).

Datasets
    ``wikitext2``  Salesforce/wikitext · wikitext-2-raw-v1 · test
    ``ptb``        Penn Treebank test split (Mikolov preprocessing), 3 761 sentences
    ``c4``         allenai/c4 · en · validation, streamed and concatenated
                   until 2,000,000 characters

Protocol
    The first ``max_eval_tokens`` (32 768) tokens of the concatenated text
    are scored with a window of 2048 tokens and stride 512: each window is
    a fresh forward pass (``use_cache=False``) and only the last 512 tokens
    of every window after the first contribute to the loss, so every token
    is predicted with ≥1536 tokens of context.

The KV compressor hooks see one full window per call, so the per-channel
min/max of the quantizer is taken over 2048 tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PPLConfig:
    max_eval_tokens: int = 32768
    window: int = 2048
    stride: int = 512
    c4_max_chars: int = 2_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders (return one long string)
# ─────────────────────────────────────────────────────────────────────────────


def _load_wikitext2_text() -> str:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(ds["text"])


PTB_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"
PTB_CACHE = Path(__file__).resolve().parents[2] / "data" / "ptb" / "ptb.test.txt"


def _load_ptb_text() -> str:
    """PTB test split (Mikolov's preprocessed version, 3 761 sentences).

    The HuggingFace ``ptb_text_only`` dataset is a loading script that recent
    ``datasets`` releases refuse to run. Its script simply read the file
    below from GitHub and stripped each line, so we do the same (verified
    byte-identical to the HF dataset text used for the paper). The file is
    cached under ``data/ptb/``.
    """
    import urllib.request
    if PTB_CACHE.exists():
        raw = PTB_CACHE.read_text(encoding="utf-8")
    else:
        raw = urllib.request.urlopen(PTB_URL, timeout=60).read().decode("utf-8")
        PTB_CACHE.parent.mkdir(parents=True, exist_ok=True)
        PTB_CACHE.write_text(raw, encoding="utf-8")
    return "\n".join(line.strip() for line in raw.splitlines())


def _load_c4_text(max_chars: int) -> str:
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    buf: list[str] = []
    total = 0
    for item in ds:
        t = item.get("text", "")
        if not t:
            continue
        buf.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return "\n\n".join(buf)


def load_eval_ids(tokenizer, dataset: str, cfg: PPLConfig) -> torch.Tensor:
    """Load, tokenize and crop ``dataset`` ∈ {wikitext2, ptb, c4} → ids ``[1, T]``."""
    if dataset == "wikitext2":
        text = _load_wikitext2_text()
    elif dataset == "ptb":
        text = _load_ptb_text()
    elif dataset == "c4":
        text = _load_c4_text(cfg.c4_max_chars)
    else:
        raise ValueError(f"unknown dataset {dataset!r}; options: wikitext2, ptb, c4")
    ids = tokenizer(text, return_tensors="pt").input_ids
    if cfg.max_eval_tokens and cfg.max_eval_tokens > 0:
        ids = ids[:, : cfg.max_eval_tokens]
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def eval_ppl(model, ids: torch.Tensor, cfg: PPLConfig | None = None,
             compressor=None) -> float:
    """``exp(mean NLL)`` over the stride regions of a sliding window.

    Args:
        model: HF causal LM on GPU.
        ids: token ids ``[1, T]`` on the model's device.
        cfg: :class:`PPLConfig` (defaults = paper).
        compressor: optional :class:`kvcobra.KVCompressor`.
    """
    cfg = cfg or PPLConfig()
    if compressor is not None:
        compressor.install_hooks(model)
    try:
        seq_len = ids.shape[1]
        nll_sum = 0.0
        n_tok = 0
        model.eval()
        for begin in range(0, seq_len - 1, cfg.stride):
            end = min(begin + cfg.window, seq_len)
            chunk = ids[:, begin:end]
            out = model(chunk, use_cache=False)
            logits = out.logits[:, :-1, :].contiguous()
            labels = chunk[:, 1:].contiguous()
            if begin > 0:
                overlap = cfg.window - cfg.stride
                logits = logits[:, overlap:, :]
                labels = labels[:, overlap:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            )
            nll_sum += loss.item()
            n_tok += labels.numel()
            if end >= seq_len:
                break
        return float(np.exp(nll_sum / max(n_tok, 1)))
    finally:
        if compressor is not None:
            compressor.remove_hooks()
