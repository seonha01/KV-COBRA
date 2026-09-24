#!/usr/bin/env python
"""Run the whole paper matrix as a job queue over several GPUs.

Each job is one ``(kind, model, seed, bits, c1_rounding)`` invocation of a
runner script. Jobs are ordered so that the cheap / most important cells
finish first:

    1. PPL, floor rounding, every model × seed, full bit sweep
    2. PPL, round (legacy) rounding, seeds 43–45, full bit sweep
    3. zero-shot + LongBench, floor, 0.5 and 1.0 bpd, all 5 seeds   (paper Table 1)
    4. zero-shot + LongBench, round,  1.5–4.0 bpd, seeds 43–45      (paper Fig. 4 points)
    5. zero-shot + LongBench, floor,  1.5–4.0 bpd, seeds 43–45      (released code)

See docs/REPRODUCTION.md for why two C1 rounding modes exist.

Workers (one per GPU) pull the next job; every runner is resumable, so the
queue can be stopped and restarted at any time. Jobs are claimed through
lock files in the log directory, so several queue instances (e.g. started
at different times on different GPUs) can share the same job list safely.

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


def build_jobs(models, seeds, kinds, full_sweep_all_seeds: bool, skip_released_sweep: bool):
    jobs = []
    rest = tuple(b for b in PAPER_BITS if b not in HEADLINE_BITS)
    sweep_seeds = [s for s in seeds if full_sweep_all_seeds or s in SWEEP_SEEDS]
    if "ppl" in kinds:
        for m in models:
            for s in seeds:
                jobs.append(("ppl", m, s, PAPER_BITS, "floor"))
        for m in models:
            for s in sweep_seeds:
                jobs.append(("ppl", m, s, PAPER_BITS, "round"))
    gen_kinds = [k for k in ("zeroshot", "longbench") if k in kinds]
    for kind in gen_kinds:
        for m in models:
            for s in seeds:
                jobs.append((kind, m, s, HEADLINE_BITS, "floor"))
    for kind in gen_kinds:
        for m in models:
            for s in sweep_seeds:
                jobs.append((kind, m, s, rest, "round"))
    if not skip_released_sweep:
        for kind in gen_kinds:
            for m in models:
                for s in sweep_seeds:
                    jobs.append((kind, m, s, rest, "floor"))
    return jobs


def job_tag(kind, model, seed, bits, rounding) -> str:
    return f"{kind}_{model}_seed{seed}_{rounding}_{'-'.join(str(b) for b in bits)}"


def job_is_complete(kind, model, seed, bits, rounding) -> bool:
    """True if the result CSV already holds every cell of this job (skip without loading a model)."""
    import pandas as pd
    csv = ROOT / "results" / kind / f"{model}_seed{seed}.csv"
    if not csv.exists():
        return False
    df = pd.read_csv(csv)
    if "c1_rounding" not in df.columns:
        df["c1_rounding"] = "floor"
    df["c1_rounding"] = df["c1_rounding"].fillna("floor")
    if not (df.method == "FP16").any():
        return False
    n_ds = 3 if kind == "ppl" else 1
    for name in ("KV-COBRA-MSE", "KV-COBRA-KL"):
        for b in bits:
            sel = df[(df.method == name) & (df.bits_per_dim.astype(float) == float(b))
                     & (df.c1_rounding == rounding)]
            if len(sel) < n_ds:
                return False
    return True


def _claim(tag: str, log_dir: Path) -> bool:
    """Atomically claim a job. Returns False if it is done or held by a live process."""
    if (log_dir / f"{tag}.done").exists():
        return False
    lock = log_dir / f"{tag}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            pid = int(lock.read_text().strip() or 0)
        except ValueError:
            pid = 0
        if pid and Path(f"/proc/{pid}").exists():
            return False                      # another live queue instance owns it
        lock.write_text(str(os.getpid()))     # stale lock (dead owner): take over
        return True


def worker(gpu: str, q: Queue, log_dir: Path, dry: bool):
    log_dir.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            kind, model, seed, bits, rounding = q.get_nowait()
        except Exception:
            return
        cmd = [sys.executable, str(ROOT / "scripts" / RUNNER[kind]),
               "--model", model, "--seed", str(seed), "--c1-rounding", rounding,
               "--bits", *[str(b) for b in bits]]
        tag = job_tag(kind, model, seed, bits, rounding)
        if dry:
            print(f"[gpu{gpu}] START {tag}  {time.strftime('%H:%M:%S')}", flush=True)
            q.task_done()
            continue
        if not _claim(tag, log_dir):
            q.task_done()
            continue
        print(f"[gpu{gpu}] START {tag}  {time.strftime('%H:%M:%S')}", flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        with open(log_dir / f"{tag}.log", "a") as f:
            t0 = time.time()
            rc = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"[gpu{gpu}] {'DONE' if rc == 0 else f'FAIL(rc={rc})'} {tag}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if rc == 0:
            (log_dir / f"{tag}.done").touch()
        (log_dir / f"{tag}.lock").unlink(missing_ok=True)
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
    p.add_argument("--skip-released-sweep", action="store_true",
                   help="omit priority-5 jobs (floor rounding at 1.5-4.0 bpd for zero-shot/LongBench)")
    p.add_argument("--log-dir", type=Path, default=ROOT / "results" / "logs")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    jobs = build_jobs(a.models, a.seeds, a.kinds, a.full_sweep_all_seeds, a.skip_released_sweep)
    if not a.dry_run:
        a.log_dir.mkdir(parents=True, exist_ok=True)
        n_done = 0
        for j in jobs:
            if job_is_complete(*j):
                (a.log_dir / f"{job_tag(*j)}.done").touch()
                n_done += 1
        print(f"{len(jobs)} jobs on GPUs {a.gpus} ({n_done} already complete in results/)")
    else:
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
