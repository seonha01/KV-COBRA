# KV-COBRA

**KV-COBRA: KV Cache Compression via Co-Optimized Bit-Rank Allocation**  
*NeurIPS 2026*

This branch hosts the **project webpage** for the KV-COBRA paper.

👉 **Project page:**  
https://seonha01.github.io/KV-COBRA/

---

## 🔗 Code

The reference implementation and the reproduction scripts for the paper's main results live on the `main` branch:

👉 **Code repository:**  
https://github.com/seonha01/KV-COBRA

---

## 📌 About

KV-COBRA compresses the KV cache of large language models by choosing, for every attention head, how many SVD directions to keep and how many bits to spend on each, and by redistributing the total bit budget across heads. It adds no per-token overhead and stays accurate down to 0.5 bits per dimension.

Please refer to the **project webpage** for:
- Overview and motivation
- Method illustration
- Experimental results
- Citation information

---

## 📄 Citation

```bibtex
@inproceedings{ha2026kvcobra,
  title     = {KV-COBRA: KV Cache Compression via Co-Optimized Bit-Rank Allocation},
  author    = {Ha, Sihyeon and Lee, Jaeho and Jeon, Yo-Seb},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
