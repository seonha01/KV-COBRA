#!/usr/bin/env python
"""Run the whole paper matrix as a job queue over several GPUs.

Each job is one ``(kind, model, seed, bits)`` invocation of a runner script.
Jobs are ordered so that the cheap/most-important cells finish first:

    1. PPL, every model × seed, full bit sweep                (~1.5 h / model)
    2. zero-shot + LongBench at 0.5 and 1.0 bpd, all 5 seeds    (paper Table 1)
    3. zero-shot + LongBench at the remaining bits, seeds 43–45 (paper Fig. 4)

Workers (one per GPU) pull the next job; every runner is resumable, so the
queue can be stopped and restarted at any time.

    python scripts/run_queue.py --gpus 0 1 2
    python scripts/run_queue.py --gpus 0 --models llama31_8b --seeds 43 --kinds ppl
    python scripts/run_queue.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kvcobra.experiment import MODELS, PAPER_BITS, PAPER_SEEDS   # noqa: E402

RUNNER = {"ppl": "run_ppl.py", "zeroshot": "run_zeroshot.py", "longbench": "run_longbench.py"}
HEADLINE_BITS = (0.5, 1.0)
SWEEP_SEEDS = (43, 44, 45)


def build_jobs(models, seeds, kinds, full_sweep_all_seeds: bool):
    jobs = []
    if "ppl" in kinds:
        for m in models:
            for s in seeds:
                jobs.append(("ppl", m, s, PAPER_BITS))
    for kind in ("zeroshot", "longbench"):
        if kind not in kinds:
            continue
        for m in models:
            for s in seeds:
                jobs.append((kind, m, s, HEADLINE_BITS))
    rest = tuple(b for b in PAPER_BITS if b not in HEADLINE_BITS)
    for kind in ("zeroshot", "longbench"):
        if kind not in kinds:
            continue
        for m in models:
            for s in seeds:
                if full_sweep_all_seeds or s in SWEEP_SEEDS:
                    jobs.append((kind, m, s, rest))
    return jobs


def worker(gpu: str, q: Queue, log_dir: Path, dry: bool):
    while True:
        try:
            kind, model, seed, bits = q.get_nowait()
        except Exception:
            return
        cmd = [sys.executable, str(ROOT / "scripts" / RUNNER[kind]),
               "--model", model, "--seed", str(seed), "--bits", *[str(b) for b in bits]]
        tag = f"{kind}_{model}_seed{seed}_{'-'.join(str(b) for b in bits)}"
        print(f"[gpu{gpu}] START {tag}  {time.strftime('%H:%M:%S')}", flush=True)
        if dry:
            q.task_done()
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f"{tag}.log", "a") as f:
            t0 = time.time()
            rc = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"[gpu{gpu}] {'DONE' if rc == 0 else f'FAIL(rc={rc})'} {tag}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        q.task_done()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", nargs="+", default=["0"])
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    p.add_argument("--kinds", nargs="+", default=["ppl", "zeroshot", "longbench"])
    p.add_argument("--full-sweep-all-seeds", action="store_true",
                   help="run the full bit sweep for every seed (paper: seeds 43-45 only)")
    p.add_argument("--log-dir", type=Path, default=ROOT / "results" / "logs")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    jobs = build_jobs(a.models, a.seeds, a.kinds, a.full_sweep_all_seeds)
    print(f"{len(jobs)} jobs on GPUs {a.gpus}")
    q: Queue = Queue()
    for j in jobs:
        q.put(j)
    threads = [threading.Thread(target=worker, args=(g, q, a.log_dir, a.dry_run), daemon=True)
               for g in a.gpus]
    for t in threads:
        t.start()
        time.sleep(2)   # stagger model loading
    for t in threads:
        t.join()
    print("queue finished")


if __name__ == "__main__":
    main()
