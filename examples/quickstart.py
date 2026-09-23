#!/usr/bin/env python
"""Minimal end-to-end example: compress the keys of a model at 1 bit/dim and
measure WikiText-2 perplexity with and without KV-COBRA.

    python examples/quickstart.py --model meta-llama/Llama-3.1-8B --bits 1.0
    python examples/quickstart.py --model Qwen/Qwen2.5-0.5B --bits 2.0   # small & quick
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                            # noqa: E402
from kvcobra import (build_kv_cobra, collect_calibration, get_model_arch,   # noqa: E402
                     load_calibration_texts)
from kvcobra.eval import PPLConfig, eval_ppl, load_eval_ids             # noqa: E402
from kvcobra.experiment import load_model_and_tokenizer                 # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--bits", type=float, default=1.0)
    p.add_argument("--variant", default="kl", choices=["kl", "mse"])
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--eval-tokens", type=int, default=8192)
    a = p.parse_args()

    # 1. model + geometry
    arch = get_model_arch(a.model)
    model, tok = load_model_and_tokenizer(a.model)
    print(f"{a.model}: {arch.n_layers} layers × {arch.n_kv_heads} KV heads × d={arch.head_dim}")

    # 2. one-shot calibration (32 WikiText-2 paragraphs, ≤1024 tokens each)
    calib = collect_calibration(model, tok, arch, load_calibration_texts(), max_seq_len=1024)

    # 3. allocate (C1 + C2 [+ KL reordering]) and build the hook-based compressor
    comp = build_kv_cobra(arch, calib, variant=a.variant, bits_per_dim=a.bits,
                          hadamard_seed=a.seed)
    table = comp.allocation_table()
    ranks = sorted({r["rank"] for r in table}); bits = sorted({r["bits"] for r in table})
    print(f"allocation: {len(table)} heads, ranks {ranks[0]}–{ranks[-1]}, bit-widths {bits}")
    avg = sum(r["budget"] for r in table) / len(table) / arch.head_dim
    print(f"average budget = {avg:.3f} bits/dim (target {a.bits})")

    # 4. evaluate
    cfg = PPLConfig(max_eval_tokens=a.eval_tokens)
    ids = load_eval_ids(tok, "wikitext2", cfg).to(next(model.parameters()).device)
    fp16 = eval_ppl(model, ids, cfg)
    comp_ppl = eval_ppl(model, ids, cfg, compressor=comp)
    print(f"WikiText-2 PPL   FP16 = {fp16:.3f}   KV-COBRA-{a.variant.upper()} @ {a.bits} bpd = {comp_ppl:.3f}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
