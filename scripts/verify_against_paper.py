#!/usr/bin/env python
"""Compare reproduced results (``results/``) with the paper runs (``reference/paper``).

Every cell present in *both* is compared on its score column(s):

    ppl        → ppl               (per dataset)
    zeroshot   → mean + 5 tasks
    longbench  → mean + 5 tasks

The paper's CSVs mix two C1 rounding modes (see docs/REPRODUCTION.md):
rows at 0.5 / 1.0 bpd and every row of seeds 46–47 come from the released
``floor`` code, rows at 1.5–4.0 bpd of seeds 43–45 from the legacy
``round`` code. Each reference row is therefore compared with the
reproduced row of the *matching* mode (``expected_mode``).

Prints a per-kind table of matched cells, max |Δ|, and lists any cell whose
difference exceeds ``--atol``. Exit status 1 if any such cell exists.

    python scripts/verify_against_paper.py                 # all kinds
    python scripts/verify_against_paper.py --kind ppl --atol 1e-4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = {
    "ppl":       ["model", "seed", "method", "kv_side", "bits_per_dim", "dataset"],
    "zeroshot":  ["model", "seed", "method", "kv_side", "bits_per_dim"],
    "longbench": ["model", "seed", "method", "kv_side", "bits_per_dim"],
}
SCORE_COLS = {
    "ppl":       ["ppl"],
    "zeroshot":  ["mean", "arc_c", "hellaswag", "piqa", "winogrande", "mmlu"],
    "longbench": ["mean", "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "musique"],
}


def expected_mode(seed: int, bits: float, method: str) -> str:
    """Which C1 rounding produced a given reference row."""
    if method == "FP16":
        return "-"
    if bits <= 1.0 or int(seed) in (46, 47):
        return "floor"
    return "round"


def _load_dir(d: Path) -> pd.DataFrame:
    files = sorted(d.glob("*_seed*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def compare(kind: str, atol: float, results_dir: Path, reference_dir: Path):
    ref = _load_dir(reference_dir / kind)
    new = _load_dir(results_dir / kind)
    keys, cols = KEYS[kind], [c for c in SCORE_COLS[kind]]
    if ref.empty or new.empty:
        print(f"[{kind}] reference rows={len(ref)} reproduced rows={len(new)} — nothing to compare")
        return 0, 0, []
    cols = [c for c in cols if c in ref.columns and c in new.columns]
    ref = ref.copy()
    ref["c1_rounding"] = [expected_mode(s, b, mth) for s, b, mth
                          in zip(ref.seed, ref.bits_per_dim, ref.method)]
    if "c1_rounding" not in new.columns:
        new = new.copy(); new["c1_rounding"] = "floor"
    new = new.copy()
    new.loc[new.method == "FP16", "c1_rounding"] = "-"
    new["c1_rounding"] = new["c1_rounding"].fillna("floor")
    m = ref.merge(new, on=keys + ["c1_rounding"], suffixes=("_paper", "_repro"))
    bad = []
    max_abs = 0.0
    for _, r in m.iterrows():
        for c in cols:
            diff = abs(float(r[f"{c}_paper"]) - float(r[f"{c}_repro"]))
            max_abs = max(max_abs, diff)
            if diff > atol:
                bad.append((tuple(r[k] for k in keys), c, r[f"{c}_paper"], r[f"{c}_repro"], diff))
    n_cells = len(m)
    print(f"[{kind}] reference cells={len(ref)}  reproduced cells={len(new)}  "
          f"compared={n_cells}  max|Δ|={max_abs:.3e}  "
          f"{'ALL WITHIN TOL' if not bad else f'{len(bad)} DIFFERENCES > {atol:g}'}")
    if n_cells:
        # per-model / per-method summary of max |Δ| on the primary score
        prim = cols[0]
        m["absdiff"] = (m[f"{prim}_paper"] - m[f"{prim}_repro"]).abs()
        summ = m.groupby(["model", "method", "c1_rounding"])["absdiff"].agg(["count", "max"])
        for (model, method, mode), row in summ.iterrows():
            print(f"    {model:<12} {method:<13} [{mode:<5}] n={int(row['count']):3d}  "
                  f"max|Δ{prim}|={row['max']:.3e}")
    for key, c, a, b, d in bad[:40]:
        print(f"    MISMATCH {key} {c}: paper={a:.6f} repro={b:.6f} |Δ|={d:.3e}")
    if len(bad) > 40:
        print(f"    … {len(bad) - 40} more")
    return n_cells, len(bad), bad


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", nargs="+", default=["ppl", "zeroshot", "longbench"])
    p.add_argument("--atol", type=float, default=1e-6,
                   help="absolute tolerance on the score columns")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--reference-dir", type=Path, default=ROOT / "reference" / "paper")
    a = p.parse_args()
    total_bad = 0
    for kind in a.kind:
        _, n_bad, _ = compare(kind, a.atol, a.results_dir, a.reference_dir)
        total_bad += n_bad
    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()
