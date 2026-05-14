# V-JEPA 2 — Raw Extraction Notes

Sources:
- https://arxiv.org/abs/2506.09985
- https://ar5iv.labs.arxiv.org/html/2506.09985

## Two-stage recipe
1. Action-free V-JEPA 2 pretraining on web video + image (VideoMix22M).
2. V-JEPA 2-AC: action-conditioned post-training on < 62 hours of Droid robot video.

## Encoder
- ViT-L (300M) up to ViT-g (~1B params).
- Frozen during V-JEPA 2-AC post-training.
- Progressive-resolution training; 8.4x GPU time reduction vs full-res training.

## Predictor
- ~300M params transformer.
- 24 layers, 16 heads, 1024 hidden dim, GELU.
- Block-causal attention pattern (NOT vanilla cross-attention): each patch attends to
  action, end-effector state, other patches at same timestep, plus prior timesteps.

## Pretraining data
- VideoMix22M = 22M samples across Something-Something v2, Kinetics, HowTo100M,
  YT-Temporal-1B, ImageNet.
- > 1M hours of internet video.

## EMA / collapse prevention
- Target encoder is EMA of online encoder.
- Stop-gradient on target.

## V-JEPA 2-AC training
- Teacher-forcing loss + two-step rollout loss.
- Loss in representation space, not pixel.
- < 62 hours Droid robot video.

## Headline numbers
- Something-Something v2 attentive probe: 77.3 top-1.
- Epic-Kitchens-100 action anticipation: 39.7 recall@5 (+44% rel. vs prior SOTA).
- PerceptionTest: 84.0; TempCompass: 76.9 (with LLM alignment, 8B scale).

## Robotic deployment (Franka, zero-shot, two labs)
- Reach: end-effector within < 4 cm of goal.
- Grasp: ~65% average success.
- Pick-and-place: 75% cup, 65% box.
- No environment-specific training, no reward engineering.
