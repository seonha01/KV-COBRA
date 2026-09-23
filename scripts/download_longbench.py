#!/usr/bin/env python
"""Fetch the official LongBench ``data.zip`` and unpack the QA tasks.

The paper evaluates the first 50 examples of five tasks in the *file order*
of the official release, so we use the exact ``data.zip`` from the
``THUDM/LongBench`` dataset repository rather than a re-hosted copy.

    python scripts/download_longbench.py            # → data/longbench/<task>.jsonl
    python scripts/download_longbench.py --all      # unpack every task
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvcobra.eval.longbench import DEFAULT_DATA_DIR, DEFAULT_TASKS   # noqa: E402

# md5 of the five files inside the official data.zip (as used for the paper)
EXPECTED_MD5 = {
    "narrativeqa": "af9f571b33d83a864306f89542ba86a6",
    "qasper": "bf52f8d44f72ae69fc19484722b51ef2",
    "multifieldqa_en": "8a39cbf71c9b0d418b383e3e3871b0e7",
    "hotpotqa": "578c8c30ffda6ffee6cd0d25f20a54fa",
    "musique": "b5e71ba4f0b1101dfbea3de27bd0bb9d",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--all", action="store_true", help="unpack all 21+ tasks")
    a = p.parse_args()

    from huggingface_hub import hf_hub_download
    print("downloading THUDM/LongBench data.zip …", flush=True)
    zip_path = hf_hub_download(repo_id="THUDM/LongBench", filename="data.zip",
                               repo_type="dataset")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".jsonl")]
        wanted = names if a.all else [n for n in names
                                      if Path(n).stem in DEFAULT_TASKS]
        for n in wanted:
            data = z.read(n)
            out = a.out_dir / Path(n).name
            out.write_bytes(data)
            md5 = hashlib.md5(data).hexdigest()
            exp = EXPECTED_MD5.get(Path(n).stem)
            flag = "" if exp is None else ("  ✓ matches paper" if exp == md5 else "  ✗ DIFFERS from paper copy")
            print(f"  {out.name:<28} {len(data) / 1e6:6.1f} MB  md5={md5}{flag}")
    print(f"done → {a.out_dir}")


if __name__ == "__main__":
    main()
