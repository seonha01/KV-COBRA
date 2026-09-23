"""LongBench (THUDM) QA tasks with greedy generation and token-F1.

Tasks (first ``max_samples`` = 50 examples of each file, in file order)
    narrativeqa · qasper · multifieldqa_en · hotpotqa · musique

Data
    The official ``data.zip`` of the ``THUDM/LongBench`` dataset repo,
    unpacked to ``<data_dir>/<task>.jsonl`` (``scripts/download_longbench.py``).

Protocol
    prompt  = "Context: <context[:3*4096 chars]>\\n\\nQuestion: <input>\\n\\nAnswer:"
    tokens  = first 4096 tokens of the prompt (truncation on the right)
    output  = 64 greedy tokens; score = max over references of word-level F1
    (lower-cased, whitespace-split, set overlap), averaged over samples ×100.

The KV compressor sees the whole prompt in the prefill hook call and then
one token per decode step (each decode step is quantized on its own).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DEFAULT_TASKS = ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "musique")
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "longbench"


@dataclass(frozen=True)
class LongBenchConfig:
    tasks: tuple[str, ...] = DEFAULT_TASKS
    max_samples: int = 50
    max_input_len: int = 4096
    max_gen_len: int = 64
    data_dir: Path = DEFAULT_DATA_DIR


def _normalize(text: str) -> list[str]:
    return text.lower().split()


def f1_score(prediction: str, reference: str) -> float:
    """Word-set F1 between prediction and one reference."""
    pred = set(_normalize(prediction))
    ref = set(_normalize(reference))
    if not pred or not ref:
        return 0.0
    common = pred & ref
    if not common:
        return 0.0
    prec = len(common) / len(pred)
    rec = len(common) / len(ref)
    return 2 * prec * rec / (prec + rec)


def load_task(task: str, data_dir: Path) -> list[dict]:
    path = Path(data_dir) / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"LongBench file not found: {path}\n"
            f"Run `python scripts/download_longbench.py` first.")
    with open(path) as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def eval_longbench(model, tokenizer, cfg: LongBenchConfig | None = None,
                   compressor=None) -> dict[str, float]:
    """Return ``{task: F1×100, ..., "mean": ...}``."""
    cfg = cfg or LongBenchConfig()
    device = next(model.parameters()).device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if compressor is not None:
        compressor.install_hooks(model)

    results: dict[str, float] = {}
    try:
        model.eval()
        for task in cfg.tasks:
            samples = load_task(task, cfg.data_dir)[: cfg.max_samples]
            scores: list[float] = []
            for sample in samples:
                context = sample.get("context", "")
                question = sample.get("input", "")
                answers = sample.get("answers", [])
                if isinstance(answers, str):
                    answers = [answers]
                if not answers:
                    continue
                prompt = (f"Context: {context[: cfg.max_input_len * 3]}\n\n"
                          f"Question: {question}\n\nAnswer:")
                inputs = tokenizer(prompt, return_tensors="pt",
                                   max_length=cfg.max_input_len, truncation=True).to(device)
                out = model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_gen_len,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
                pred = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                        skip_special_tokens=True)
                scores.append(max(f1_score(pred, a) for a in answers))
            results[task] = float(np.mean(scores) * 100) if scores else 0.0
    finally:
        if compressor is not None:
            compressor.remove_hooks()

    valid = [v for v in results.values() if not np.isnan(v)]
    results["mean"] = float(np.mean(valid)) if valid else float("nan")
    return results
