#!/usr/bin/env python
"""Perplexity sweep for one model and one seed (paper Fig. 4 / Table 1, PPL rows).

Example (paper setting)::

    CUDA_VISIBLE_DEVICES=0 python scripts/run_ppl.py --model llama31_8b --seed 43

Writes ``results/ppl/<model>_seed<seed>.csv`` with one row per
(method, bits_per_dim, dataset) plus the FP16 reference rows. Re-running
skips cells that are already in the CSV.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvcobra import build_kv_cobra, get_model_arch                    # noqa: E402
from kvcobra.eval import PPLConfig, eval_ppl, load_eval_ids           # noqa: E402
from kvcobra.experiment import (PAPER_BITS, RESULTS_DIR, VARIANTS,    # noqa: E402
                                ResultTable, calibrate,
                                load_model_and_tokenizer, make_specs,
                                resolve_model, set_seed)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="short name or HF id")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--bits", nargs="+", type=float, default=list(PAPER_BITS))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--kv-side", default="k_only")
    p.add_argument("--datasets", nargs="+", default=["wikitext2", "ptb", "c4"])
    p.add_argument("--n-eval-tokens", type=int, default=32768)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--n-cal", type=int, default=32)
    p.add_argument("--cal-seq-len", type=int, default=1024)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "ppl")
    a = p.parse_args()

    short, model_id = resolve_model(a.model)
    set_seed(a.seed)
    t0 = time.time()
    log = lambda s: print(f"[{short}/ppl/seed{a.seed}] {s}", flush=True)  # noqa: E731

    arch = get_model_arch(model_id)
    log(f"{model_id}: layers={arch.n_layers} kv_heads={arch.n_kv_heads} "
        f"head_dim={arch.head_dim}")
    model, tokenizer = load_model_and_tokenizer(model_id)
    calib = calibrate(model, tokenizer, arch, a.n_cal, a.cal_seq_len, log)

    cfg = PPLConfig(max_eval_tokens=a.n_eval_tokens, window=a.window, stride=a.stride)
    device = next(model.parameters()).device
    ids_by_ds = {}
    for ds in a.datasets:
        ids_by_ds[ds] = load_eval_ids(tokenizer, ds, cfg).to(device)
        log(f"  {ds}: {ids_by_ds[ds].shape[1]} tokens")

    table = ResultTable(a.out_dir / f"{short}_seed{a.seed}.csv",
                        key_cols=("method", "kv_side", "bits_per_dim", "dataset"))
    if len(table):
        log(f"resuming from {table.path} ({len(table)} rows)")

    # ── FP16 reference ────────────────────────────────────────────────────
    fp16 = {}
    for ds, ids in ids_by_ds.items():
        existing = [r for r in table.rows if r["method"] == "FP16" and r["dataset"] == ds]
        if existing:
            fp16[ds] = float(existing[0]["ppl"])
            continue
        ppl = eval_ppl(model, ids, cfg)
        fp16[ds] = ppl
        table.add({"model": short, "seed": a.seed, "method": "FP16", "kv_side": "fp16",
                   "bits_per_dim": 16.0, "dataset": ds, "ppl": ppl, "delta": 0.0,
                   "wall_s": 0.0})
        log(f"  FP16 {ds:<9} PPL={ppl:.4f}")

    # ── KV-COBRA cells ────────────────────────────────────────────────────
    specs = make_specs(a.variants, a.bits, a.kv_side, a.seed)
    for i, spec in enumerate(specs, 1):
        todo = [ds for ds in ids_by_ds if not table.has(
            method=spec.name, kv_side=spec.kv_side, bits_per_dim=spec.bits_per_dim, dataset=ds)]
        if not todo:
            log(f"[{i:2d}/{len(specs)}] {spec.name:<12} {spec.bits_per_dim:.1f}b  (done)")
            continue
        comp = build_kv_cobra(arch, calib, spec)
        for ds in todo:
            tc = time.time()
            ppl = eval_ppl(model, ids_by_ds[ds], cfg, compressor=comp)
            table.add({"model": short, "seed": a.seed, "method": spec.name,
                       "kv_side": spec.kv_side, "bits_per_dim": spec.bits_per_dim,
                       "dataset": ds, "ppl": ppl, "delta": ppl - fp16[ds],
                       "wall_s": time.time() - tc})
            log(f"[{i:2d}/{len(specs)}] {spec.name:<12} {spec.bits_per_dim:.1f}b "
                f"{ds:<9} PPL={ppl:9.3f}  Δ={ppl - fp16[ds]:+9.3f}  [{time.time() - tc:.0f}s]")
        del comp
        torch.cuda.empty_cache()
        gc.collect()

    log(f"done in {(time.time() - t0) / 60:.1f} min → {table.path}")


if __name__ == "__main__":
    main()
