#!/usr/bin/env python
"""LongBench sweep for one model and one seed (paper Table 1, "LongBench" columns).

Example::

    python scripts/download_longbench.py                  # once
    CUDA_VISIBLE_DEVICES=0 python scripts/run_longbench.py --model llama31_8b --seed 43

Writes ``results/longbench/<model>_seed<seed>.csv`` (per-task F1 ×100 and
mean). This is by far the slowest evaluation (generation); resumable.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvcobra import build_kv_cobra, get_model_arch                        # noqa: E402
from kvcobra.eval import LongBenchConfig, eval_longbench                  # noqa: E402
from kvcobra.eval.longbench import DEFAULT_DATA_DIR, DEFAULT_TASKS        # noqa: E402
from kvcobra.experiment import (PAPER_BITS, RESULTS_DIR, VARIANTS,        # noqa: E402
                                ResultTable, calibrate,
                                load_model_and_tokenizer, make_specs,
                                resolve_model, set_seed)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--bits", nargs="+", type=float, default=list(PAPER_BITS))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--kv-side", default="k_only")
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--n-cal", type=int, default=32)
    p.add_argument("--cal-seq-len", type=int, default=1024)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "longbench")
    a = p.parse_args()

    short, model_id = resolve_model(a.model)
    set_seed(a.seed)
    t0 = time.time()
    log = lambda s: print(f"[{short}/longbench/seed{a.seed}] {s}", flush=True)  # noqa: E731

    arch = get_model_arch(model_id)
    model, tokenizer = load_model_and_tokenizer(model_id)
    calib = calibrate(model, tokenizer, arch, a.n_cal, a.cal_seq_len, log)
    cfg = LongBenchConfig(tasks=tuple(a.tasks), max_samples=a.max_samples,
                          data_dir=a.data_dir)

    table = ResultTable(a.out_dir / f"{short}_seed{a.seed}.csv",
                        key_cols=("method", "kv_side", "bits_per_dim"))
    if len(table):
        log(f"resuming from {table.path} ({len(table)} rows)")

    if not table.has(method="FP16", kv_side="fp16", bits_per_dim=16.0):
        tc = time.time()
        scores = eval_longbench(model, tokenizer, cfg)
        table.add({"model": short, "seed": a.seed, "method": "FP16", "kv_side": "fp16",
                   "bits_per_dim": 16.0, **scores, "wall_s": time.time() - tc})
        log(f"  FP16 mean F1={scores['mean']:.2f}  [{time.time() - tc:.0f}s]")

    specs = make_specs(a.variants, a.bits, a.kv_side, a.seed)
    for i, spec in enumerate(specs, 1):
        if table.has(method=spec.name, kv_side=spec.kv_side, bits_per_dim=spec.bits_per_dim):
            log(f"[{i:2d}/{len(specs)}] {spec.name:<12} {spec.bits_per_dim:.1f}b  (done)")
            continue
        tc = time.time()
        comp = build_kv_cobra(arch, calib, spec)
        scores = eval_longbench(model, tokenizer, cfg, compressor=comp)
        table.add({"model": short, "seed": a.seed, "method": spec.name,
                   "kv_side": spec.kv_side, "bits_per_dim": spec.bits_per_dim,
                   **scores, "wall_s": time.time() - tc})
        log(f"[{i:2d}/{len(specs)}] {spec.name:<12} {spec.bits_per_dim:.1f}b "
            f"mean F1={scores['mean']:.2f}  [{time.time() - tc:.0f}s]")
        del comp
        torch.cuda.empty_cache()
        gc.collect()

    log(f"done in {(time.time() - t0) / 60:.1f} min → {table.path}")


if __name__ == "__main__":
    main()
