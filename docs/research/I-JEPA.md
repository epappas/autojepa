# I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture

**Authors:** Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas
**Venue / Year:** CVPR 2023
**arXiv:** 2301.08243 — https://arxiv.org/abs/2301.08243
**Status:** Distilled 2026-05-15

## 1. One-line thesis

Predicting the embeddings of multiple semantic-scale target blocks from a
single spatially-distributed context block, in latent space and without
hand-crafted augmentations, yields a scalable self-supervised image learner.

## 2. Method

- Three networks: context encoder f_theta (trained), target encoder f_theta_bar
  (EMA of f_theta, no gradient), predictor g_phi (small ViT, trained).
- Per image: sample one context block (scale 0.85-1.0, aspect 1.0) and M=4
  target blocks (scale 0.15-0.20, aspect 0.75-1.5). Remove target overlaps from
  context.
- Encode context tokens with f_theta. For each target block, run g_phi on
  context tokens plus learnable mask tokens carrying the target's positional
  embeddings.
- Loss: average L2 distance between predicted embeddings and EMA-target
  embeddings of the same target patches.
- EMA momentum: 0.996 -> 1.0 linear schedule.
- Predictor: narrow ViT, 384 dim, depth 6 (B/16), 12 (L/16, H/16, H/14), 16 (G/16).
- No image augmentations beyond random horizontal flip and resized crop.

## 3. Results

- ViT-H/14 trained on 16 A100s in under 72 hours.
- ImageNet-1K linear probe, ViT-H/14 @224, 300 epochs: 79.3 top-1.
- ImageNet-1% low-shot: ViT-H/14 = 73.3, ViT-H/16 @448 = 77.3.
- Strong on local-structure tasks: object counting (Clevr/Count) and depth
  prediction (Clevr/Dist), where contrastive methods underperform.
- Masking ablation (ViT-B/16, 1% ImageNet): multi-block 54.2 vs single block
  20.2, random patch 17.6, rasterized quadrant 15.5 — the masking choice is
  the dominant design lever.

## 4. Why it matters for AutoJEPA

This is the foundational primitive of the entire codebase.

- I-JEPA's multi-block sampler is the prior implementation for
  `autojepa.masking.MultiBlockInfillMask`.
- The context-encoder / EMA-target-encoder / predictor triple is the canonical
  module layout in `autojepa.models.{encoders,predictors}`, with the
  `target_encoder.update_ema()` required call enforced by the program.md
  validator.
- The L2 latent loss is the default in `autojepa.models.losses` (alternatives
  in C-JEPA, see [C-JEPA](C-JEPA.md)).
- The known masking-strategy sensitivity (~3x downstream gap between schemes)
  is the empirical justification for putting `masking_strategy` in the default
  hybrid search dimensions of AutoJEPA's policy (writeup section on hybrid
  thresholds).
- I-JEPA's hyperparameters (batch 2048, EMA 0.996->1.0, LR cosine to 1e-6) are
  the seed values for the Phase 2 CIFAR-10 validation gate.

## 5. Caveats / known limitations

- ViT-only. Convolutional backbones need a different masking treatment — see
  [CNN-JEPA](CNN-JEPA.md).
- EMA alone is not provably collapse-proof. Empirically rare on ImageNet-scale
  training but a real risk on small datasets and short schedules — addressed
  by [C-JEPA](C-JEPA.md).
- Training loss is not a useful checkpoint-selection signal: it can decrease
  while representations collapse. AutoJEPA must use downstream probes
  (`autojepa.eval`), not loss, as the campaign objective.
- Image-only. Audio masking transfers do NOT work — see
  [JEPA Audio Design Choices](JEPA-Audio-Design-Choices.md).

## 6. References to other corpus entries

- [V-JEPA 2](V-JEPA-2.md) — temporal extension and predictor scaling.
- [C-JEPA](C-JEPA.md) — addresses the EMA-collapse and mean-prediction gaps.
- [CNN-JEPA](CNN-JEPA.md) — adapts the recipe to convolutional encoders.
- [A-JEPA](A-JEPA.md) — direct audio port of the I-JEPA recipe.
- [JEPA Audio Design Choices](JEPA-Audio-Design-Choices.md) — counter-evidence
  on which I-JEPA priors transfer to audio.
