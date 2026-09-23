"""Evaluation protocols of the paper's main results.

* :mod:`kvcobra.eval.ppl`       – sliding-window perplexity (WikiText-2, PTB, C4)
* :mod:`kvcobra.eval.zeroshot`  – 5 multiple-choice benchmarks, log-prob ranking
* :mod:`kvcobra.eval.longbench` – 5 LongBench QA tasks, greedy generation, F1

Every ``eval_*`` function takes an optional ``compressor``; when given, its
hooks are installed for the duration of the evaluation and removed after.
"""
from .ppl import PPLConfig, eval_ppl, load_eval_ids
from .zeroshot import ZeroShotConfig, eval_zeroshot
from .longbench import LongBenchConfig, eval_longbench

__all__ = [
    "PPLConfig", "eval_ppl", "load_eval_ids",
    "ZeroShotConfig", "eval_zeroshot",
    "LongBenchConfig", "eval_longbench",
]
