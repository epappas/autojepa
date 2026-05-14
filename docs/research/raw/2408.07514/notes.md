# CNN-JEPA — Raw Extraction Notes

Sources:
- https://arxiv.org/abs/2408.07514
- https://ar5iv.labs.arxiv.org/html/2408.07514

## Why this paper exists
- I-JEPA is ViT-only. CNNs don't have explicit "patch tokens", so masking is
  non-trivial: pixel-level masks leak information through receptive fields.

## Sparse CNN encoder
- Treats masking as zeroing convolutional layer outputs at masked locations.
- Mask is upscaled to each layer's activation grid and applied after each conv.
- Stays compatible with standard GPU-optimized dense conv kernels.
- EMA teacher update unchanged.

## Predictor
- 3 layers of depthwise separable convolution, 3x3, BN, ReLU.
- ~90% parameter reduction vs standard conv predictor; better accuracy.

## Masking
- Patch size 32x32 px (corresponds to 1x1 in final feature map of ResNet-50).
- Multi-block masking adapted from I-JEPA.
- All masked blocks treated as one prediction region (simplifies, no loss in quality).

## Training
- Batch size 512 (128/device * 4 GPUs).
- ImageNet-100: 200 epochs, ~13 hours total.
- ImageNet-1K: 100 epochs, ~70 hours.

## Headline results
- ImageNet-100 ResNet-50 linear top-1: 73.3.
- ImageNet-1K ResNet-50 linear top-1 @ 100 epochs: 54.23.
- 17-35% less training time per epoch vs BYOL / SimCLR.
- Outperforms I-JEPA (ViT-Small/Base) on ImageNet-100.

## Notable simplifications
- No separate projector network.
- Minimal augmentations.
