# C-JEPA — Raw Extraction Notes

Sources:
- https://arxiv.org/abs/2410.19560
- https://ar5iv.labs.arxiv.org/html/2410.19560

## Problem statement
- I-JEPA EMA does not robustly prevent representation collapse.
- I-JEPA prediction does not accurately learn the mean of patch representations.

## Solution: combine I-JEPA with VICReg

### Total loss
L = L_JEPA + beta_vicreg * L_VICReg
beta_vicreg = 0.001

### VICReg decomposition
L_VICReg = beta_sim * L_sim + beta_std * L_std + beta_cov * L_cov
beta_sim = 25, beta_std = 25, beta_cov = 1

- L_std (variance) and L_cov (covariance) prevent collapse.
- L_sim (invariance) regularizes mean of augmented views.

## Training (ViT-B/16, ImageNet-1K, 600 epochs)
- Batch size 2048.
- LR: linear warmup 1e-4 -> 1e-3 over 15 epochs, cosine decay to 1e-6.
- Weight decay: 0.04 -> 0.4.
- EMA: 0.996 -> 1.0 linear.

## Headline numbers (ViT-B/16, 600 epochs)
- Linear probe: I-JEPA 72.9 -> C-JEPA 73.7 (+0.8).
- Fine-tune: I-JEPA 83.5 -> C-JEPA 84.5 (+1.0).

## Ablation (100 epochs, ViT-B/16, linear probe)
- I-JEPA baseline: 63.7
- + variance/covariance only: 68.3
- Full C-JEPA (sim + var + cov): 69.5

## Takeaway
- C-JEPA also converges faster than I-JEPA (the VICReg terms accelerate the
  representation quality curve, not just the asymptote).
