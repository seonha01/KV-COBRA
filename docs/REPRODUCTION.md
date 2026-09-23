# Reproducing the paper's main results

This document describes exactly what was run for the paper, how this repo
re-runs it, and how to check the numbers.

## 1. What "main results" means here

| Metric | Datasets | Protocol | Runner |
|---|---|---|---|
| Perplexity | WikiText-2 test, PTB test, C4 validation | first 32 768 tokens, sliding window 2048 / stride 512 | `scripts/run_ppl.py` |
| Zero-shot accuracy | ARC-Challenge, HellaSwag, PIQA, WinoGrande, MMLU | first 200 examples each, log-prob ranking of choices, prompt ≤ 1024 tokens | `scripts/run_zeroshot.py` |
| LongBench F1 | NarrativeQA, Qasper, MultiFieldQA-en, HotpotQA, MuSiQue | first 50 examples each, prompt ≤ 4096 tokens, 64 greedy tokens, word-F1 | `scripts/run_longbench.py` |

Models: `meta-llama/Llama-3.1-8B`, `mistralai/Mistral-7B-v0.3`,
`Qwen/Qwen2.5-7B-Instruct`, all fp16, keys compressed pre-RoPE
(`kv_side = k_only`). Methods: **KV-COBRA-KL** (main), **KV-COBRA-MSE**
(ablation) and the uncompressed **FP16** reference.

Bit budgets: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 bits per key dimension.

## 2. Seeds — what they control

The whole pipeline is deterministic except for the **random sign vector of
the Hadamard rotation**. Calibration texts are the first 32 WikiText-2
train paragraphs (in dataset order), the covariance/eigendecomposition is
deterministic, and C1/C2 are deterministic integer programs. `--seed N`
therefore only changes the sign draw (`kvcobra.hadamard.generate_signs`).

Paper seeds: **43, 44, 45, 46, 47**. Coverage in the paper's own runs
(and therefore in `reference/paper/`):

| kind | seeds 43–45 | seeds 46–47 |
|---|---|---|
| PPL | all 7 budgets | 0.5, 1.0, 2.0 |
| zero-shot | all 7 budgets | 0.5, 1.0 |
| LongBench | all 7 budgets (Mistral seed 43: 0.5, 1.0 only) | 0.5, 1.0 |

`scripts/run_queue.py` reproduces the union of this (it also fills the few
cells the paper did not run).

## 2b. Two C1 rounding modes — read this before comparing numbers

While re-running the paper's experiments we found that its result CSVs were
produced by **two versions of the C1 solver** (`optimal_rank_bits`):

| rows in `reference/paper` | C1 bit-width rule | in this repo |
|---|---|---|
| 0.5 and 1.0 bpd, all seeds; every row of seeds 46–47 | `b = B // r` (feasible, `r·b ≤ B`) | `--c1-rounding floor` (**default**, identical to the released code) |
| 1.5–4.0 bpd, seeds 43–45 | `b = int(round(B / r))` (legacy; may exceed the budget slightly) | `--c1-rounding round` |

The original sweep (April) used the legacy rule; the rule was then fixed to
floor division, and only the 0.5/1.0 bpd cells (plus the extra seeds 46–47)
were re-run with the fixed code. The two rules give the *same* numbers to all
printed digits wherever they coincide, but at 1.5–3.0 bpd they differ a lot,
e.g. Llama-3.1-8B, seed 43, WikiText-2 PPL:

| bpd | KV-COBRA-MSE floor | KV-COBRA-MSE round (paper) | KV-COBRA-KL floor | KV-COBRA-KL round (paper) |
|---|---|---|---|---|
| 1.5 | 17.55 | 13.03 | 10.83 | 7.36 |
| 2.0 | 14.23 | 6.84 | 7.19 | 6.68 |
| 2.5 | 6.14 | 6.11 | 6.17 | 6.04 |
| 4.0 | 5.79 | 5.80 | 5.78 | 5.78 |

`scripts/verify_against_paper.py` knows this provenance (`expected_mode`) and
compares every reference row with the reproduced row of the matching mode;
`scripts/run_queue.py` runs both modes where needed. Both are bit-exact
against the paper CSVs (see the verification report in the README).

## 3. Running

```bash
python scripts/download_longbench.py          # once; prints md5 ✓ against the paper copy
python scripts/run_queue.py --gpus 0 1 2      # everything (both rounding modes), resumable
python scripts/run_queue.py --gpus 0 1 2 --skip-released-sweep   # omit floor @1.5-4.0 for zs/LB
scripts/status.sh                             # progress
python scripts/verify_against_paper.py        # compare with reference/paper
python scripts/summarize_results.py --bits 1.0 --markdown   # Table-1 style summary
```

Approximate cost on one RTX 6000 Ada (48 GB) per model and seed:

| kind | per cell | cells (full sweep) | time |
|---|---|---|---|
| PPL (3 datasets) | ~30 s × 3 | 14 | ~25 min |
| zero-shot | 10–20 min | 14 | 2.5–4.5 h |
| LongBench | 45–95 min | 14 | 10–22 h |

## 4. Verifying bit-exactness

`reference/paper/<kind>/<model>_seed<seed>.csv` are the CSVs written by the
original experiment scripts (restricted to FP16 / KV-COBRA rows).
`scripts/verify_against_paper.py` joins them with `results/` on
(model, seed, method, kv_side, bits_per_dim[, dataset]) and reports the
maximum absolute difference of every score column. With the pinned
environment the perplexities agree to all printed digits (`|Δ| = 0`);
generation-based metrics (LongBench) are also expected to match exactly
because decoding is greedy and every forward pass is bit-identical, but
minor GPU-kernel nondeterminism across driver versions can in principle
flip a token — the script makes any such cell visible.

## 5. Data provenance

* **Calibration / WikiText-2**: `Salesforce/wikitext`, `wikitext-2-raw-v1`
  (identical content to the legacy `wikitext` id used originally).
* **PTB**: the HF `ptb_text_only` loading script no longer runs on current
  `datasets`; it merely downloaded
  `https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt`
  and stripped each line. `kvcobra.eval.ppl` does exactly that (verified
  byte-identical to the cached HF text used for the paper) and caches the
  file under `data/ptb/`.
* **C4**: `allenai/c4`, `en`, validation split, streamed in order until
  2 000 000 characters.
* **LongBench**: official `data.zip` of the `THUDM/LongBench` dataset repo;
  the downloader prints the md5 of each task file and marks it ✓ when it
  equals the copy used for the paper.
* **Zero-shot**: `allenai/ai2_arc`, `Rowan/hellaswag`, `baber/piqa`
  (parquet mirror of `ybisk/piqa`), `allenai/winogrande`, `cais/mmlu`.

## 6. Environment used for the paper and for this reproduction

Python 3.11.15 · torch 2.5.1+cu121 · transformers 5.5.3 · datasets 4.8.4 ·
numpy 1.26.4 · accelerate 1.13.0 · NVIDIA RTX 6000 Ada (driver for CUDA 12.x).
