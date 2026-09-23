#!/usr/bin/env python
"""Aggregate ``results/`` (or ``reference/paper``) over seeds into paper-style tables.

    python scripts/summarize_results.py                       # reproduced results
    python scripts/summarize_results.py --source reference    # the paper's own runs
    python scripts/summarize_results.py --bits 1.0 --markdown # Table 1 (1 bpd)

For every (model, method, bits) it reports mean ± std over the available
seeds: PPL averaged over WikiText-2/PTB/C4, zero-shot mean accuracy and
LongBench mean F1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ["llama31_8b", "mistral7b", "qwen25_7b"]
METHOD_ORDER = ["FP16", "KV-COBRA-MSE", "KV-COBRA-KL"]


def _load(kind: str, base: Path) -> pd.DataFrame:
    files = sorted((base / kind).glob("*_seed*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def summarize(base: Path, bits: list[float] | None):
    out = []
    ppl = _load("ppl", base)
    if not ppl.empty:
        # per (model, seed, method, bpd): mean over datasets → then over seeds
        if "c1_rounding" not in ppl.columns:
            ppl["c1_rounding"] = "floor"
        ppl["c1_rounding"] = ppl["c1_rounding"].fillna("floor")
        g = (ppl.groupby(["model", "seed", "method", "bits_per_dim", "c1_rounding"])["ppl"].mean()
             .reset_index())
        g["metric"] = "ppl"
        out.append(g.rename(columns={"ppl": "value"}))
    for kind in ("zeroshot", "longbench"):
        d = _load(kind, base)
        if d.empty:
            continue
        if "c1_rounding" not in d.columns:
            d["c1_rounding"] = "floor"
        d["c1_rounding"] = d["c1_rounding"].fillna("floor")
        g = d[["model", "seed", "method", "bits_per_dim", "c1_rounding", "mean"]].copy()
        g["metric"] = kind
        out.append(g.rename(columns={"mean": "value"}))
    if not out:
        print(f"no results under {base}")
        return None
    df = pd.concat(out, ignore_index=True)
    if bits:
        df = df[df.bits_per_dim.isin(bits) | (df.method == "FP16")]
    df.loc[df.method == "FP16", "c1_rounding"] = "-"
    agg = (df.groupby(["metric", "model", "method", "bits_per_dim", "c1_rounding"])["value"]
           .agg(["mean", "std", "count"]).reset_index())
    agg["std"] = agg["std"].fillna(0.0)
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["results", "reference"], default="results")
    p.add_argument("--bits", nargs="+", type=float, default=None)
    p.add_argument("--markdown", action="store_true")
    a = p.parse_args()
    base = ROOT / "results" if a.source == "results" else ROOT / "reference" / "paper"
    agg = summarize(base, a.bits)
    if agg is None:
        return
    for metric in ("ppl", "zeroshot", "longbench"):
        sub = agg[agg.metric == metric]
        if sub.empty:
            continue
        label = {"ppl": "PPL (mean of WikiText-2/PTB/C4, ↓)",
                 "zeroshot": "Zero-shot accuracy % (5-task mean, ↑)",
                 "longbench": "LongBench F1 (5-task mean, ↑)"}[metric]
        print(f"\n## {label}")
        if a.markdown:
            print("| model | method | bpd | C1 rounding | mean ± std | seeds |")
            print("|---|---|---|---|---|---|")
        for model in MODEL_ORDER:
            for method in METHOD_ORDER:
                rows = sub[(sub.model == model) & (sub.method == method)].sort_values("bits_per_dim")
                for _, r in rows.iterrows():
                    bpd = "fp16" if r.method == "FP16" else f"{r.bits_per_dim:.1f}"
                    if a.markdown:
                        print(f"| {model} | {method} | {bpd} | {r['c1_rounding']} | {r['mean']:.2f} ± {r['std']:.2f} | {int(r['count'])} |")
                    else:
                        print(f"  {model:<12} {method:<13} {bpd:>5} [{r['c1_rounding']:<5}] {r['mean']:8.2f} ± {r['std']:5.2f}  (n={int(r['count'])})")


if __name__ == "__main__":
    main()
