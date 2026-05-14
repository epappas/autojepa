# I-JEPA — Raw Extraction Notes

Sources:
- https://arxiv.org/abs/2301.08243
- https://ar5iv.labs.arxiv.org/html/2301.08243

## Core architecture
- Image-based Joint-Embedding Predictive Architecture.
- Three networks: context encoder f_theta, target encoder f_theta_bar (EMA), predictor g_phi.
- Predict representations of target blocks from a single context block, in latent space.
- No pixel reconstruction; no hand-crafted augmentations.

## Masking
- M = 4 target blocks per image.
- Target block scale range: (0.15, 0.20). Aspect ratio: (0.75, 1.5).
- Single context block scale range: (0.85, 1.0). Unit aspect ratio.
- Overlap with target blocks is removed from context, leaving spatially distributed context.

## Predictor
- Narrow ViT, embedding dim 384.
- Depth: 6 for ViT-B/16; 12 for ViT-L/16, ViT-H/16, ViT-H/14; 16 for ViT-G/16.
- Inputs: context tokens plus learnable mask tokens with positional embeddings of target.

## Loss / EMA
- L2 distance between predicted and target patch-level representations (averaged).
- EMA momentum: 0.996 -> 1.0 linear over training.

## Training (ViT-H/14, ImageNet, 300 epochs)
- Batch size 2048.
- LR: linear warmup 1e-4 -> 1e-3 over 15 epochs, cosine decay to 1e-6.
- Weight decay: 0.04 -> 0.4 linear schedule.
- 16 A100 GPUs, < 72 hours.

## Headline results
- ImageNet linear probe ViT-H/14 @224: 79.3 top-1 (300 epochs).
- ImageNet 1% low-shot, ViT-H/14: 73.3 top-1.
- ImageNet 1% low-shot, ViT-H/16 @448: 77.3 top-1.

## Masking ablations (ViT-B/16, 1% ImageNet)
- Multi-block (proposed): 54.2
- Rasterized quadrant: 15.5
- Single block: 20.2
- Random patch: 17.6

## Notable downstream
- Strong on object counting (Clevr/Count) and depth prediction (Clevr/Dist) - implies
  representations capture local quantitative structure, not just global class labels.
