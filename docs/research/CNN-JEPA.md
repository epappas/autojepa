# CNN-JEPA: Self-Supervised Pretraining Convolutional Neural Networks Using JEPA

**Authors:** Andras Kalapos, Balint Gyires-Toth
**Venue / Year:** ICMLA 2024
**arXiv:** 2408.07514 — https://arxiv.org/abs/2408.07514
**Status:** Distilled 2026-05-15

## 1. One-line thesis

The JEPA recipe ports to ResNet-style CNNs by zeroing convolutional
activations at masked spatial locations on every layer (sparse-conv encoder)
and using a tiny depthwise-separable conv predictor — and beats both ViT
I-JEPA and BYOL/SimCLR/VICReg on ImageNet-100 with 17-35% less wall-clock.

## 2. Method

- **Sparse CNN encoder:** standard CNN forward pass, but after each conv layer
  the activations at masked spatial locations are zeroed. The mask is upscaled
  to each layer's resolution. This avoids re-implementing sparse conv kernels
  while preventing masked-region leakage through the receptive field.
- **Predictor:** 3 layers of depthwise-separable conv (3x3) with BN + ReLU.
  ~90% fewer params than dense-conv predictor and slightly better.
- **Masking:** multi-block, patch size 32x32 px (1x1 in ResNet-50's final
  feature map). The union of all masked blocks is treated as a single
  prediction region (simplifies bookkeeping, no quality cost).
- **No projector network**, **no augmentation stack** beyond standard
  resized-crop + flip.
- Standard EMA target update, L2 latent loss.

## 3. Results

- ImageNet-100 linear top-1 (ResNet-50): **73.3** — beats ViT-Small/Base
  I-JEPA on the same dataset.
- ImageNet-1K linear top-1 (ResNet-50, 100 epochs): 54.23.
- Wall-clock: 17-35% faster per-epoch than BYOL / SimCLR on the same hardware.
- Batch size 512 (128/device * 4 GPUs); 200 epochs ImageNet-100 in ~13 hours;
  100 epochs ImageNet-1K in ~70 hours.

## 4. Why it matters for AutoJEPA

- CNN-JEPA is the entry point for *non-ViT backbones* in AutoJEPA. The
  `autojepa.models.encoders` module exposes both `ViTEncoder` and
  `SparseCNNEncoder`, where the latter implements the
  zero-after-each-conv-layer trick from this paper.
- The depthwise-separable predictor is the reference for
  `autojepa.models.predictors.ConvPredictor`. It is the cheap-compute baseline
  for hybrid-policy search when the LLM is allowed to swap predictor types.
- Demonstrates that "minimal augmentations + no projector" is viable. This
  matches AutoJEPA's principle of keeping the prepare.py / train.py contract
  small: fewer external dependencies, fewer brittle interactions for the LLM
  to mutate.
- The 17-35% wall-clock advantage is a genuine campaign-level benefit:
  AutoJEPA Phase 2 validation on CIFAR-10 should default to the CNN-JEPA
  recipe to maximize iterations per GPU-hour during the framework-validation
  gate.

## 5. Caveats / known limitations

- Headline numbers are ImageNet-100, not -1K. The 100-epoch ImageNet-1K
  number (54.23) is well below ViT I-JEPA at the same compute. Treat
  CNN-JEPA as a CNN-domain baseline, not a state-of-the-art claim.
- The zero-after-each-conv masking trick is correct only if you are willing
  to accept BatchNorm statistics being computed with some inputs masked.
  Empirically fine in this paper but a fragile invariant for any LLM diff
  that touches the encoder.
- No collapse-prevention beyond EMA. Pairing with [C-JEPA](C-JEPA.md)'s
  VICReg head is untested but plausible.
- Predictor is unconditional on patch position (it's a small conv stack);
  positional information comes only through the spatial layout of the
  context activations. Different from the I-JEPA predictor, where mask tokens
  carry explicit positional embeddings.

## 6. References to other corpus entries

- [I-JEPA](I-JEPA.md) — the ViT-based baseline this paper adapts.
- [C-JEPA](C-JEPA.md) — orthogonal loss-side improvement.
- [V-JEPA 2](V-JEPA-2.md) — opposite scaling regime; useful contrast.
