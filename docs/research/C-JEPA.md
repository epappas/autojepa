# C-JEPA: Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning

**Authors:** Shentong Mo, Shengbang Tong
**Venue / Year:** arXiv preprint, 2024
**arXiv:** 2410.19560 — https://arxiv.org/abs/2410.19560
**Status:** Distilled 2026-05-15

## 1. One-line thesis

Adding VICReg's variance + covariance + invariance regularizer to I-JEPA's
loss patches I-JEPA's two known weaknesses — partial EMA-only collapse and
inaccurate mean-of-patch prediction — and yields measurable gains on linear
probe and fine-tune at the same compute budget.

## 2. Method

- Keep the full I-JEPA architecture: context encoder, EMA target encoder,
  predictor, multi-block masking.
- Add a VICReg head over augmented views (the EMA-target patch embeddings vs
  online-encoder patch embeddings):
  ```
  L = L_JEPA + beta_vicreg * L_VICReg
  L_VICReg = 25 * L_sim + 25 * L_std + 1 * L_cov
  beta_vicreg = 0.001
  ```
- L_std (variance) keeps each embedding dimension's std above a hinge.
- L_cov (covariance) decorrelates dimensions across the batch.
- L_sim (invariance) regularizes the mean of two augmented views.
- Training schedule unchanged from I-JEPA (batch 2048, LR 1e-3 with cosine,
  EMA 0.996 -> 1.0).

## 3. Results

ViT-B/16, ImageNet-1K, 600 epochs:

| Metric | I-JEPA | C-JEPA | Delta |
| --- | --- | --- | --- |
| Linear probe top-1 | 72.9 | 73.7 | +0.8 |
| Fine-tune top-1 | 83.5 | 84.5 | +1.0 |

100-epoch ablation (ViT-B/16, linear probe):
- I-JEPA baseline: 63.7
- + variance/covariance only: 68.3
- Full C-JEPA: 69.5

C-JEPA also shows faster convergence — early-epoch linear-probe scores
dominate I-JEPA, not just the asymptote.

## 4. Why it matters for AutoJEPA

This is the AutoJEPA default loss configuration.

- Per the writeup, the AutoJEPA hybrid policy treats VICReg-aware loss as the
  default, with weights (`beta_vicreg`, `beta_sim`, `beta_std`, `beta_cov`)
  exposed as first-class search axes in `autojepa.models.losses`.
- The variance and covariance terms map directly onto the `variance_check()`
  required call enforced by the program.md validator. C-JEPA gives us a
  principled regularizer that *trains the model to satisfy the
  collapse-detection invariants*, not just monitor them.
- The fast-convergence property is load-bearing for the forecaster
  recalibration: with C-JEPA, the early plateau region used by the
  parameter-exploration phase (~2000 steps) is more informative because the
  representation quality signal moves earlier.
- C-JEPA's hyperparameters (beta scaling 25/25/1, beta_vicreg 0.001) seed the
  Phase 2 baseline; the LLM is then allowed to mutate these via diffs.

## 5. Caveats / known limitations

- Only validated on ImageNet-1K with ViT-B/16. No video, audio, or CNN data.
- VICReg adds a per-step compute overhead (variance + covariance over batch
  embeddings). At very large batches the covariance term dominates memory.
- The hyperparameter beta_vicreg = 0.001 is small enough that mis-tuning is
  silent — bad weights look like a no-op rather than a regression. The LLM
  diff loop must be told this is a sensitive axis.
- "Partial collapse" is a stronger failure mode than the paper measures
  against; C-JEPA prevents full collapse but does not eliminate dimensional
  collapse. Pair with RankMe / LiDAR detectors.

## 6. References to other corpus entries

- [I-JEPA](I-JEPA.md) — the baseline C-JEPA fixes.
- [V-JEPA 2](V-JEPA-2.md) — same encoder/predictor/EMA template at video scale.
- [CNN-JEPA](CNN-JEPA.md) — orthogonal architectural fix; can be combined with
  C-JEPA's loss in principle.
