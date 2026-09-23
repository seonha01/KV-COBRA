# KV-COBRA: KV Cache Compression via Co-Optimized Bit-Rank Allocation

Reference implementation of **KV-COBRA** ([arXiv:2609.24298](https://arxiv.org/abs/2609.24298)),
reduced to what is needed to reproduce the paper's **main results**:
perplexity, zero-shot accuracy and LongBench F1 for KV-COBRA-KL /
KV-COBRA-MSE against the FP16 model on Llama-3.1-8B, Mistral-7B-v0.3 and
Qwen2.5-7B-Instruct. No third-party baselines, no RULER, no CUDA kernels —
just the method and the three evaluation protocols, written to be read and
reused.

> **Bit-exact.** The code paths that produce numbers were ported operation
> by operation from the experiment code behind the paper. The paper's own
> result CSVs ship in `reference/paper/` and `scripts/verify_against_paper.py`
> checks a re-run against them (perplexities agree to every printed digit).

---

## 1. The method in one page

Decoding an LLM is bounded by reading the KV cache. KV-COBRA stores every
**key** of every attention head in a compressed form: project onto the
head's top-`r` SVD directions, rotate, quantize each latent with `b` bits.
What is new is *how `(r, b)` are chosen*: **per head**, and **jointly**.

For one KV head with key-covariance eigenvalues `λ_1 ≥ … ≥ λ_d`
(head dimension `d`, typically 128), keeping `r` directions at `b` bits costs
`B = r·b` bits per token and incurs

```
D(r, b) = Σ_{i>r} λ_i  +  q(b) · Σ_{i≤r} λ_i ,       q(b) = 2^(-2b) / 12
          └─ truncation ─┘   └──── quantization ────┘
```

| Level | What it decides | Code |
|---|---|---|
| **C1** — per-head rank/bit-width | for a given head budget `B`, `(r*, b*) = argmin D(r,b)` s.t. `r·b ≤ B` (enumerate even `r`, `b = ⌊B/r⌋ ∈ [2, 8]`) | `kvcobra/allocation.py::optimal_rank_bits` |
| **C2** — cross-head budget | redistribute the global budget `bpd·d` per head so that all heads sit at (approximately) the same marginal distortion: damped water-filling, 5 rounds | `kvcobra/allocation.py::water_filling_budgets` |
| **KL reordering** (main method) | replace `λ_i` by `w_i = σ²_{Q,i}·σ²_{K,i}` — the query variance along direction `i` times the key eigenvalue — and re-sort the basis by `w`. C1/C2 then minimize an attention-KL surrogate instead of key MSE | `kvcobra/allocation.py::query_variance`, `reorder_by_attention_kl` |
| **Rotate-and-quantize** | random Hadamard transform on the `r` latents (flattens the per-channel variance, Gaussianizes) followed by one uniform `b`-bit quantizer per head | `kvcobra/hadamard.py`, `kvcobra/compressor.py` |

Everything runs once at calibration time on ~6 k tokens of WikiText-2
(seconds), produces a static per-head table, and adds nothing per token.
`KV-COBRA-MSE` = C1 + C2 + rotation on the eigenvalues; `KV-COBRA-KL` =
the same on the KL-reordered weights.

The compressor is applied through **forward hooks** on `k_proj`
(pre-RoPE): the projection output is replaced head-by-head by its
compressed-then-reconstructed value, so the model runs on exactly the
tensor a real compressed cache would decode to. This "simulated
quantization" path is what all accuracy numbers in the paper use.

## 2. Installation

```bash
git clone <this repo> && cd KV-COBRA
conda create -n kvcobra python=3.11 -y && conda activate kvcobra
pip install -r requirements.txt        # pinned versions used for the paper
pip install -e .
huggingface-cli login                  # Llama-3.1 is a gated checkpoint
python -m pytest tests/                # 8 fast CPU tests
```

## 3. Quick start

```python
from kvcobra import get_model_arch, load_calibration_texts, collect_calibration, build_kv_cobra
from kvcobra.eval import PPLConfig, load_eval_ids, eval_ppl
from kvcobra.experiment import load_model_and_tokenizer

model_id = "meta-llama/Llama-3.1-8B"
arch = get_model_arch(model_id)                       # layers / KV heads / head_dim
model, tok = load_model_and_tokenizer(model_id)       # fp16, device_map="auto"

calib = collect_calibration(model, tok, arch, load_calibration_texts())   # ~10 s
comp  = build_kv_cobra(arch, calib, variant="kl", bits_per_dim=1.0, hadamard_seed=43)

print(comp.allocation_table()[:3])    # per-head rows: {'layer', 'head', 'rank', 'bits', 'budget' (= rank*bits)}

cfg = PPLConfig()                                      # 32k tokens, window 2048, stride 512
ids = load_eval_ids(tok, "wikitext2", cfg).to(model.device)
print("FP16    :", eval_ppl(model, ids, cfg))
print("KV-COBRA:", eval_ppl(model, ids, cfg, compressor=comp))   # hooks installed for the call only
```

or simply `python examples/quickstart.py --model Qwen/Qwen2.5-0.5B --bits 2.0`.

`build_kv_cobra` also accepts `kv_side="v_only" | "kv_symmetric" | "kv_asymmetric"`
(with `k_fraction`) for the joint K+V setting of the paper's appendix; the
main results use the default `k_only`.

## 4. Reproducing the paper

```bash
python scripts/download_longbench.py            # official data.zip, md5-checked (once)

# one model / one seed / one metric (each is resumable)
CUDA_VISIBLE_DEVICES=0 python scripts/run_ppl.py       --model llama31_8b --seed 43
CUDA_VISIBLE_DEVICES=0 python scripts/run_zeroshot.py  --model llama31_8b --seed 43
CUDA_VISIBLE_DEVICES=0 python scripts/run_longbench.py --model llama31_8b --seed 43

# the whole matrix (3 models × seeds 43–47 × 3 metrics) as a GPU job queue
python scripts/run_queue.py --gpus 0 1 2
scripts/status.sh

# compare with the paper's CSVs, then aggregate over seeds
python scripts/verify_against_paper.py
python scripts/summarize_results.py --bits 1.0 --markdown
```

Outputs land in `results/<ppl|zeroshot|longbench>/<model>_seed<seed>.csv`
(+ `.json`). Runtime, seed coverage, data provenance and the environment
are documented in [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

### Paper numbers (1 bit / dim, keys only; mean over the five seeds)

| Model | PPL ↓ (WT2/PTB/C4 mean) MSE / KL / FP16 | 0-shot ↑ MSE / KL / FP16 | LongBench F1 ↑ MSE / KL / FP16 |
|---|---|---|---|
| Llama-3.1-8B | 24.8 / **20.3** / 7.7 | 51.5 / **51.6** / 55.4 | 9.3 / **11.4** / 15.8 |
| Mistral-7B-v0.3 | 27.2 / **15.9** / 13.2 | 54.7 / **55.1** / 56.3 | 9.5 / **10.7** / 11.5 |
| Qwen2.5-7B-Instruct | **13.0** / 13.2 / 9.5 | 51.7 / **52.5** / 54.1 | 9.1 / **9.5** / 10.6 |

(`python scripts/summarize_results.py --source reference --bits 1.0` prints
this table with standard deviations from the shipped reference CSVs.)

## 5. Repository layout

```
kvcobra/
  arch.py           model geometry (layers, KV heads, head_dim) + layer lookup
  calibration.py    calibration texts, K/V/Q hooks, per-head eigendecomposition
  allocation.py     C1 (optimal_rank_bits), C2 (water_filling_budgets), KL reordering
  hadamard.py       Walsh–Hadamard transform, random signs
  compressor.py     KVCompressor: hook-based project → rotate → quantize → reconstruct
  method.py         build_kv_cobra(): glues calibration + allocation into a compressor
  experiment.py     model table, seeds, result tables used by the scripts
  eval/             ppl.py · zeroshot.py · longbench.py (paper protocols)
scripts/            run_ppl.py · run_zeroshot.py · run_longbench.py · run_queue.py
                    download_longbench.py · verify_against_paper.py · summarize_results.py
reference/paper/    the paper's own result CSVs (FP16 + KV-COBRA rows, seeds 43–47)
examples/           quickstart.py
tests/              CPU unit tests
docs/               REPRODUCTION.md
```

## 6. Reproducibility notes

* **Determinism.** Given a seed the pipeline is deterministic; the seed
  only draws the Hadamard sign vector. Calibration uses the first 32
  WikiText-2 train paragraphs ≥ 200 characters, truncated to 1024 tokens.
* **Quantizer statistics.** The uniform quantizer takes per-channel min/max
  over the tokens seen in one hook call (a 2048-token window for PPL, one
  prompt for zero-shot, the prefill and then single tokens for generation).
  Changing window/stride or batching would change the numbers.
* **Pinned datasets.** PTB is read from the original Mikolov file (the HF
  loading script is dead on current `datasets`), LongBench from the
  official `data.zip`; both are checked against the copies used for the
  paper.

## 7. Citation

```bibtex
@article{kvcobra2026,
  title   = {KV-COBRA: KV Cache Compression via Co-Optimized Bit-Rank Allocation},
  journal = {arXiv preprint arXiv:2609.24298},
  year    = {2026}
}
```
