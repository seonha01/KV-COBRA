# Verification report — reproduced vs. paper CSVs

Generated 2026-09-25 by `python scripts/verify_against_paper.py`.
Cells are compared with the reproduced run of the matching C1 rounding mode (see docs/REPRODUCTION.md).

```
[ppl] reference cells=519  reproduced cells=1053  compared=519  max|Δ|=3.553e-15  ALL WITHIN TOL
    llama31_8b   FP16          [-    ] n= 15  max|Δppl|=0.000e+00
    llama31_8b   KV-COBRA-KL   [floor] n= 36  max|Δppl|=0.000e+00
    llama31_8b   KV-COBRA-KL   [round] n= 45  max|Δppl|=0.000e+00
    llama31_8b   KV-COBRA-MSE  [floor] n= 36  max|Δppl|=0.000e+00
    llama31_8b   KV-COBRA-MSE  [round] n= 45  max|Δppl|=3.553e-15
    mistral7b    FP16          [-    ] n= 15  max|Δppl|=0.000e+00
    mistral7b    KV-COBRA-KL   [floor] n= 36  max|Δppl|=0.000e+00
    mistral7b    KV-COBRA-KL   [round] n= 39  max|Δppl|=3.553e-15
    mistral7b    KV-COBRA-MSE  [floor] n= 36  max|Δppl|=0.000e+00
    mistral7b    KV-COBRA-MSE  [round] n= 39  max|Δppl|=3.553e-15
    qwen25_7b    FP16          [-    ] n= 15  max|Δppl|=0.000e+00
    qwen25_7b    KV-COBRA-KL   [floor] n= 36  max|Δppl|=0.000e+00
    qwen25_7b    KV-COBRA-KL   [round] n= 45  max|Δppl|=0.000e+00
    qwen25_7b    KV-COBRA-MSE  [floor] n= 36  max|Δppl|=0.000e+00
    qwen25_7b    KV-COBRA-MSE  [round] n= 45  max|Δppl|=0.000e+00
[zeroshot] reference cells=165  reproduced cells=153  compared=153  max|Δ|=0.000e+00  ALL WITHIN TOL
    llama31_8b   FP16          [-    ] n=  5  max|Δmean|=0.000e+00
    llama31_8b   KV-COBRA-KL   [floor] n= 10  max|Δmean|=0.000e+00
    llama31_8b   KV-COBRA-KL   [round] n= 15  max|Δmean|=0.000e+00
    llama31_8b   KV-COBRA-MSE  [floor] n= 10  max|Δmean|=0.000e+00
    llama31_8b   KV-COBRA-MSE  [round] n= 15  max|Δmean|=0.000e+00
    mistral7b    FP16          [-    ] n=  5  max|Δmean|=0.000e+00
    mistral7b    KV-COBRA-KL   [floor] n= 10  max|Δmean|=0.000e+00
    mistral7b    KV-COBRA-KL   [round] n= 12  max|Δmean|=0.000e+00
    mistral7b    KV-COBRA-MSE  [floor] n= 10  max|Δmean|=0.000e+00
    mistral7b    KV-COBRA-MSE  [round] n= 15  max|Δmean|=0.000e+00
    qwen25_7b    FP16          [-    ] n=  5  max|Δmean|=0.000e+00
    qwen25_7b    KV-COBRA-KL   [floor] n= 10  max|Δmean|=0.000e+00
    qwen25_7b    KV-COBRA-KL   [round] n= 10  max|Δmean|=0.000e+00
    qwen25_7b    KV-COBRA-MSE  [floor] n= 10  max|Δmean|=0.000e+00
    qwen25_7b    KV-COBRA-MSE  [round] n= 11  max|Δmean|=0.000e+00
[longbench] reference cells=155  reproduced cells=75  compared=75  max|Δ|=3.553e-15  ALL WITHIN TOL
    llama31_8b   FP16          [-    ] n=  5  max|Δmean|=0.000e+00
    llama31_8b   KV-COBRA-KL   [floor] n= 10  max|Δmean|=0.000e+00
    llama31_8b   KV-COBRA-MSE  [floor] n= 10  max|Δmean|=0.000e+00
    mistral7b    FP16          [-    ] n=  5  max|Δmean|=0.000e+00
    mistral7b    KV-COBRA-KL   [floor] n= 10  max|Δmean|=0.000e+00
    mistral7b    KV-COBRA-MSE  [floor] n= 10  max|Δmean|=0.000e+00
    qwen25_7b    FP16          [-    ] n=  5  max|Δmean|=0.000e+00
    qwen25_7b    KV-COBRA-KL   [floor] n= 10  max|Δmean|=0.000e+00
    qwen25_7b    KV-COBRA-MSE  [floor] n= 10  max|Δmean|=0.000e+00
```
