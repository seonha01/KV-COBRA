"""Shared plumbing for the command-line runners in ``scripts/``.

* the three paper models and their short names,
* the paper's bit-per-dimension grid,
* model loading + calibration with the paper's seeds,
* an incremental, resumable CSV/JSON result table.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .arch import ModelArch, get_model_arch
from .calibration import Calibration, collect_calibration, load_calibration_texts
from .method import KVCobraSpec

# Short name → HuggingFace id. Any other HF id can be passed directly.
MODELS = {
    "llama31_8b": "meta-llama/Llama-3.1-8B",
    "mistral7b":  "mistralai/Mistral-7B-v0.3",
    "qwen25_7b":  "Qwen/Qwen2.5-7B-Instruct",
}
PAPER_BITS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
PAPER_SEEDS = (43, 44, 45, 46, 47)
VARIANTS = ("mse", "kl")

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def resolve_model(name: str) -> tuple[str, str]:
    """``"llama31_8b"`` → ``("llama31_8b", "meta-llama/Llama-3.1-8B")``;
    an unknown name is treated as a HF id and given a filesystem-safe short name."""
    if name in MODELS:
        return name, MODELS[name]
    short = name.split("/")[-1].lower().replace("-", "_").replace(".", "")
    return short, name


def set_seed(seed: int) -> None:
    """Seed torch/numpy. Only the Hadamard sign draw actually depends on it."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_model_and_tokenizer(model_id: str, dtype=torch.float16):
    """fp16 model with ``device_map="auto"`` (restrict GPUs with CUDA_VISIBLE_DEVICES)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:      # transformers >= 5 renamed torch_dtype → dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto", trust_remote_code=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def calibrate(model, tokenizer, arch: ModelArch, n_cal: int = 32,
              cal_seq_len: int = 1024, log=print) -> Calibration:
    """Paper calibration: 32 WikiText-2 train paragraphs × ≤1024 tokens."""
    t0 = time.time()
    texts = load_calibration_texts(n_samples=n_cal, min_length=200)
    calib = collect_calibration(model, tokenizer, arch, texts,
                                max_seq_len=cal_seq_len, collect_queries=True)
    log(f"  calibration: {len(texts)} texts, {calib.n_cal_tokens} tokens/head, "
        f"{time.time() - t0:.1f}s")
    return calib


def make_specs(variants, bits, kv_side: str, seed: int,
               c1_rounding: str = "floor") -> list[KVCobraSpec]:
    """Cartesian product of variants × bits (variant-major, like the paper runs)."""
    return [KVCobraSpec(variant=v, bits_per_dim=float(b), kv_side=kv_side,
                        hadamard_seed=seed, c1_rounding=c1_rounding)
            for v in variants for b in bits]


class ResultTable:
    """Incremental CSV (+JSON) with resume support.

    ``key_cols`` identify a finished cell; :meth:`has` lets a runner skip it.
    """

    def __init__(self, path: Path, key_cols: tuple[str, ...]):
        self.path = Path(path)
        self.key_cols = key_cols
        self.rows: list[dict] = []
        if self.path.exists():
            df = pd.read_csv(self.path)
            if "c1_rounding" not in df.columns:      # files written before the option existed
                df["c1_rounding"] = "floor"
            self.rows = df.to_dict("records")

    def _key(self, row: dict) -> tuple:
        out = []
        for c in self.key_cols:
            v = row.get(c)
            if c == "c1_rounding":
                if row.get("method") == "FP16":
                    v = "-"                                    # FP16 has no allocation
                elif v is None or v != v:                      # missing / NaN → floor
                    v = "floor"
            out.append(float(v) if c == "bits_per_dim" else v)
        return tuple(out)

    def has(self, **key) -> bool:
        k = self._key(key)
        return any(self._key(r) == k for r in self.rows)

    def add(self, row: dict) -> None:
        self.rows.append(row)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
        self.path.with_suffix(".json").write_text(json.dumps(self.rows, indent=2))

    def __len__(self) -> int:
        return len(self.rows)
