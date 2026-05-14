# JEPA Audio Design Choices — Raw Extraction Notes

Sources:
- https://arxiv.org/abs/2405.08679
- https://ar5iv.labs.arxiv.org/html/2405.08679

## Goal
- Systematic ablation of JEPA design choices for audio (mel-spectrogram input).
- Find which image-domain decisions transfer and which fail.

## Architecture
- Encoder: ViT-Base, 12 layers, 768 dim, 12 heads, Flash-Attention.
- Predictor: narrow ViT, 8 layers, 512 dim, 16 heads.
- Patch size 16x16.
- Distance: smoothed L1 (slightly beats normalized L2).
- EMA: linear interpolation tau_0 -> tau_T over 300 epochs.

## Masking strategies tested
1. Unstructured (random patches).
2. Multi-block (I-JEPA-style).
3. Time-only (mask whole time slices, keep all frequency bins).

## Key cross-modal findings
- Multi-block (best for images, +36% there) is *worse* than unstructured for audio.
- Unstructured masking is the winner across all 8 downstream tasks.
- Reason: audio events span wide frequency ranges; local connectivity matters less.
- Latent-domain masking (mask context but full input to target) - which I-JEPA
  uses - DEGRADES audio representation quality. Audio needs *both* context and
  target inputs masked.

## Downstream linear-probe scores (best column)
| Task | Acc / Score |
| --- | --- |
| ESC-50 (env. sounds) | 90.0 |
| UrbanSound8K | 87.7 |
| Speech Commands V2 | 95.4 |
| VoxCeleb1 (speaker ID) | 73.1 |
| CREMA-D (emotion) | 72.6 |
| GTZAN (music genre) | 86.9 |
| NSynth (instruments) | 76.8 |
| Surge (pitch) | 42.8 |

## Punchline
- "Optimal design choices differ between audio and image domains."
- Don't assume I-JEPA defaults transfer; sweep masking + target-input policy first.
