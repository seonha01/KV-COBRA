"""KV-COBRA: KV Cache Compression via Co-Optimized Bit-Rank Allocation.

Public API (everything you need for the paper's main experiments):

    from kvcobra import (
        get_model_arch, collect_calibration, load_calibration_texts,
        build_kv_cobra, KVCompressor,
    )

Typical flow
------------
1. ``arch = get_model_arch(model_id)``                       – read layer/head geometry
2. ``calib = collect_calibration(model, tok, arch, texts)``  – one-shot calibration
3. ``comp = build_kv_cobra(arch, calib, variant="kl", bits_per_dim=1.0, seed=43)``
4. ``comp.install_hooks(model)`` … evaluate … ``comp.remove_hooks()``

The evaluation helpers live in :mod:`kvcobra.eval` (perplexity, zero-shot,
LongBench) and the command-line runners in ``scripts/``.
"""
from .arch import ModelArch, get_model_arch, locate_layers
from .calibration import (
    Calibration,
    collect_calibration,
    load_calibration_texts,
)
from .compressor import KVCompressor
from .method import KVCobraSpec, build_kv_cobra

__version__ = "1.0.0"

__all__ = [
    "ModelArch",
    "get_model_arch",
    "locate_layers",
    "Calibration",
    "collect_calibration",
    "load_calibration_texts",
    "KVCompressor",
    "KVCobraSpec",
    "build_kv_cobra",
]
