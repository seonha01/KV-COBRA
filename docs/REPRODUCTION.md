# Reproduction notes

## Protocols

| Metric | Data | Protocol |
|---|---|---|
| Perplexity | WikiText-2 test, PTB test, C4 validation | first 32,768 tokens, window 2048, stride 512 |
| Zero-shot | ARC-C, HellaSwag, PIQA, WinoGrande, MMLU | first 200 examples, log-prob ranking, prompt ≤ 1024 tokens |
| LongBench | NarrativeQA, Qasper, MultiFieldQA-en, HotpotQA, MuSiQue | first 50 examples, prompt ≤ 4096 tokens, 64 greedy tokens, word F1 |

Models in fp16, keys compressed pre-RoPE (`k_only`). Calibration: first 32 WikiText-2 train paragraphs (≥200 chars), ≤1024 tokens each. Budgets: 0.5, 1, 1.5, 2, 2.5, 3, 4 bpd.

## Seeds

The pipeline is deterministic except for the Hadamard sign draw, so `--seed` only changes that. Paper seeds: 43–47. The paper's runs cover all budgets for seeds 43–45 and 0.5/1.0 bpd (PPL also 2.0) for seeds 46–47; `run_queue.py` reproduces that set.

## C1 rounding

The paper's CSVs come from two versions of the C1 solver:

| Reference rows | Rule | Flag |
|---|---|---|
| 0.5/1.0 bpd (all seeds), all rows of seeds 46–47 | `b = B // r` | `--c1-rounding floor` (default) |
| 1.5–4.0 bpd, seeds 43–45 | `b = int(round(B / r))` | `--c1-rounding round` |

The rule was changed to floor division after the original sweep, and only the 0.5/1.0 bpd cells were re-run. The two rules agree where they coincide but differ at 1.5–3 bpd (Llama-3.1-8B, seed 43, WikiText-2, MSE/KL: 2 bpd floor 14.23/7.19 vs round 6.84/6.68; 4 bpd 5.79/5.78 vs 5.80/5.78). `verify_against_paper.py` compares each reference row with the reproduced row of the matching rule.

## Running

```bash
python scripts/download_longbench.py
python scripts/run_queue.py --gpus 0 1 2 --skip-released-sweep
scripts/status.sh
python scripts/verify_against_paper.py
python scripts/summarize_results.py --bits 1.0
```

Per model and seed on one RTX 6000 Ada: PPL ~25 min, zero-shot 2.5–4.5 h, LongBench 10–22 h (full sweep).

## Data

- WikiText-2: `Salesforce/wikitext`, `wikitext-2-raw-v1`.
- PTB: `ptb.test.txt` from `wojzaremba/lstm` on GitHub, lines stripped (identical to the HF `ptb_text_only` text; the HF loading script no longer runs).
- C4: `allenai/c4` en validation, streamed to 2,000,000 characters.
- LongBench: `data.zip` from `THUDM/LongBench`; the downloader checks md5 against the paper copy.
- Zero-shot: `allenai/ai2_arc`, `Rowan/hellaswag`, `baber/piqa`, `allenai/winogrande`, `cais/mmlu`.

## Environment

Python 3.11.15, torch 2.5.1+cu121, transformers 5.5.3, datasets 4.8.4, numpy 1.26.4, accelerate 1.13.0.
