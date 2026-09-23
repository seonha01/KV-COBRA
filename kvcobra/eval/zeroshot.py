"""Zero-shot multiple-choice accuracy by log-probability ranking.

Benchmarks (first ``max_samples`` = 200 examples of each split, in dataset order)
    ``arc_c``       allenai/ai2_arc · ARC-Challenge · test
    ``hellaswag``   Rowan/hellaswag · validation
    ``piqa``        baber/piqa (parquet mirror of ybisk/piqa) · validation
    ``winogrande``  allenai/winogrande · winogrande_xl · validation
    ``mmlu``        cais/mmlu · all · test

Scoring
    For every choice we score the prompt ``"<question> <choice>"`` (truncated
    to 1024 tokens) with one forward pass and sum the log-probabilities of the
    choice tokens; the argmax choice is compared with the gold answer.
    The KV compressor sees one prompt per hook call.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_BENCHMARKS = ("arc_c", "hellaswag", "piqa", "winogrande", "mmlu")


@dataclass(frozen=True)
class ZeroShotConfig:
    benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS
    max_samples: int = 200
    max_prompt_len: int = 1024


# ─────────────────────────────────────────────────────────────────────────────
# Loaders → list of {question, choices, answer}
# ─────────────────────────────────────────────────────────────────────────────


def _load_arc(subset: str, n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", subset, split="test")
    out = []
    for ex in ds:
        choices = ex["choices"]["text"]
        ak = ex["answerKey"]
        ai = ord(ak.upper()) - ord("A") if ak.isalpha() else int(ak) - 1
        out.append({"question": ex["question"], "choices": choices, "answer": ai})
        if len(out) >= n:
            break
    return out


def _load_hellaswag(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    out = []
    for ex in ds:
        out.append({"question": ex["ctx"], "choices": ex["endings"],
                    "answer": int(ex["label"])})
        if len(out) >= n:
            break
    return out


def _load_piqa(n: int) -> list[dict]:
    from datasets import load_dataset
    last_err = None
    for repo in ("baber/piqa", "ybisk/piqa"):
        try:
            ds = load_dataset(repo, split="validation")
        except TypeError:
            ds = load_dataset(repo, split="validation", trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
        out = []
        for ex in ds:
            out.append({
                "question": ex.get("goal", ex.get("question", "")),
                "choices": [ex["sol1"], ex["sol2"]],
                "answer": int(ex.get("label", ex.get("answer", 0))),
            })
            if len(out) >= n:
                break
        return out
    raise RuntimeError(f"failed to load piqa: {last_err}")


def _load_winogrande(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    out = []
    for ex in ds:
        s = ex["sentence"]
        out.append({
            "question": "",
            "choices": [s.replace("_", ex["option1"]), s.replace("_", ex["option2"])],
            "answer": int(ex["answer"]) - 1,
        })
        if len(out) >= n:
            break
    return out


def _load_mmlu(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    out = []
    for ex in ds:
        out.append({"question": ex["question"], "choices": ex["choices"],
                    "answer": int(ex["answer"])})
        if len(out) >= n:
            break
    return out


LOADERS = {
    "arc_c":      lambda n: _load_arc("ARC-Challenge", n),
    "arc_e":      lambda n: _load_arc("ARC-Easy", n),
    "hellaswag":  _load_hellaswag,
    "piqa":       _load_piqa,
    "winogrande": _load_winogrande,
    "mmlu":       _load_mmlu,
}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _mc_accuracy(model, tokenizer, items: list[dict], max_prompt_len: int) -> float:
    device = next(model.parameters()).device
    correct = total = 0
    for item in items:
        log_probs: list[float] = []
        for ch in item["choices"]:
            prompt = (item["question"] + " " + ch).strip()
            inp = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=max_prompt_len).to(device)
            out = model(**inp, use_cache=False)
            ch_ids = tokenizer(f" {ch}", return_tensors="pt").input_ids[0, 1:]
            nc = len(ch_ids)
            if nc == 0:
                log_probs.append(-1e10)
                continue
            sl = inp.input_ids.shape[1]
            st = max(0, sl - nc)
            lp = torch.nn.functional.log_softmax(out.logits[0, st - 1:sl - 1], dim=-1)
            tgt = ch_ids[: lp.shape[0]].to(device)
            log_probs.append(lp.gather(1, tgt.unsqueeze(1)).squeeze(1).sum().item())
        if int(np.argmax(log_probs)) == int(item["answer"]):
            correct += 1
        total += 1
    return correct / max(total, 1) * 100


def eval_zeroshot(model, tokenizer, cfg: ZeroShotConfig | None = None,
                  compressor=None) -> dict[str, float]:
    """Return ``{benchmark: accuracy%, ..., "mean": ...}``."""
    cfg = cfg or ZeroShotConfig()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if compressor is not None:
        compressor.install_hooks(model)

    results: dict[str, float] = {}
    try:
        model.eval()
        for bench in cfg.benchmarks:
            if bench not in LOADERS:
                print(f"  [zeroshot] unknown benchmark: {bench}")
                results[bench] = float("nan")
                continue
            try:
                items = LOADERS[bench](cfg.max_samples)
                results[bench] = _mc_accuracy(model, tokenizer, items, cfg.max_prompt_len)
            except Exception as e:  # noqa: BLE001
                print(f"  [zeroshot] {bench} FAILED: {type(e).__name__}: {str(e)[:160]}")
                results[bench] = float("nan")
    finally:
        if compressor is not None:
            compressor.remove_hooks()

    valid = [v for v in results.values() if not np.isnan(v)]
    results["mean"] = float(np.mean(valid)) if valid else float("nan")
    return results
