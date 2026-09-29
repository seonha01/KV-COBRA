# KV-COBRA

Reference implementation of *KV-COBRA: KV Cache Compression via Co-Optimized Bit-Rank Allocation* (NeurIPS 2026, [arXiv:2609.24298](https://arxiv.org/abs/2609.24298)).

[Paper](https://arxiv.org/abs/2609.24298) · [Project page](https://seonha01.github.io/KV-COBRA/)

The repo contains the method (per-head rank/bit allocation, C1 + C2, KL reordering, Hadamard rotate-and-quantize) and the three evaluation protocols of the paper's main results (perplexity, zero-shot, LongBench) for KV-COBRA-MSE, KV-COBRA-KL and FP16 on Llama-3.1-8B, Mistral-7B-v0.3 and Qwen2.5-7B-Instruct. Baselines, RULER and CUDA kernels are not included.

## Install

```bash
conda create -n kvcobra python=3.11 -y && conda activate kvcobra
pip install -r requirements.txt && pip install -e .
huggingface-cli login        # Llama-3.1 is gated
pytest tests/
```

## Usage

```python
from kvcobra import get_model_arch, load_calibration_texts, collect_calibration, build_kv_cobra
from kvcobra.eval import PPLConfig, load_eval_ids, eval_ppl
from kvcobra.experiment import load_model_and_tokenizer

model_id = "meta-llama/Llama-3.1-8B"
arch = get_model_arch(model_id)
model, tok = load_model_and_tokenizer(model_id)
calib = collect_calibration(model, tok, arch, load_calibration_texts())
comp = build_kv_cobra(arch, calib, variant="kl", bits_per_dim=1.0, hadamard_seed=43)

cfg = PPLConfig()
ids = load_eval_ids(tok, "wikitext2", cfg).to(model.device)
print(eval_ppl(model, ids, cfg), eval_ppl(model, ids, cfg, compressor=comp))
```

`comp.allocation_table()` lists the per-head (rank, bits). `kv_side` accepts `k_only` (default), `v_only`, `kv_symmetric`, `kv_asymmetric`.

## Reproducing the paper

```bash
python scripts/download_longbench.py
python scripts/run_ppl.py       --model llama31_8b --seed 43
python scripts/run_zeroshot.py  --model llama31_8b --seed 43
python scripts/run_longbench.py --model llama31_8b --seed 43
python scripts/run_queue.py --gpus 0 1 2 --skip-released-sweep   # whole matrix
python scripts/verify_against_paper.py
python scripts/summarize_results.py --bits 1.0
```

Results go to `results/<kind>/<model>_seed<seed>.csv`. The paper's own CSVs are in `reference/paper/`; `verify_against_paper.py` compares cell by cell. Details, seed coverage and runtime: [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

Note: the paper's sweep points at 1.5–4.0 bpd (seeds 43–45) were produced with an older C1 rounding rule (`b = round(B/r)`). It is kept as `--c1-rounding round`; the default `floor` is the released rule. See docs/REPRODUCTION.md.

## Layout

```
kvcobra/        arch, calibration, allocation (C1/C2/KL), hadamard, compressor, method, eval/
scripts/        runners, queue, downloader, verification, summary
reference/      paper result CSVs
docs/           REPRODUCTION.md, VERIFICATION.md
```

## Citation

```bibtex
@inproceedings{ha2026kvcobra,
  title     = {KV-COBRA: KV Cache Compression via Co-Optimized Bit-Rank Allocation},
  author    = {Ha, Sihyeon and Lee, Jaeho and Jeon, Yo-Seb},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
