# A-JEPA: Joint-Embedding Predictive Architecture Can Listen

**Authors:** Zhengcong Fei, Mingyuan Fan, Junshi Huang
**Venue / Year:** arXiv preprint, 2023 (cs.SD)
**arXiv:** 2311.15830 — https://arxiv.org/abs/2311.15830
**Status:** Distilled 2026-05-15

## 1. One-line thesis

I-JEPA generalizes to mel-spectrogram audio if the masking schedule
*curriculum-anneals* from random block masking (image-style) to
time-frequency-aware masking, accounting for audio's strong local correlation
in both axes — yielding state-of-the-art audio-only SSL on AudioSet, ESC-50,
and Speech Commands.

## 2. Method

- Input: 128-band mel-spectrogram at 16 kHz. Patch embedded with a 16x16 conv,
  non-overlapping.
- Architecture: ViT-B (12 layers, 86M) context encoder, EMA target encoder
  with identical structure, 16-layer Transformer decoder/predictor.
- **Curriculum masking:**
  - Phase A (early): random block masking, M=4 target blocks, scale (0.15, 0.20).
  - Phase B (late): time-frequency aware masking, 3 target blocks, scale
    (0.05, 0.075).
  - Schedule:
    `f(s) = min(1, sqrt(s * (1 - c0^2) / S) + c0^2)` with c0 = 0.01.
    s = current step, S = total. f(s) is the probability of using the
    audio-aware mask at step s.
- Loss: standard JEPA L2 in latent space.
- Pretraining: batch 512, LR 2e-4, 24 epochs on AudioSet-2M.
- Fine-tuning: 10% regularized patch masking on the target dataset; masked
  tokens cannot independently form attention weights (instead of input
  dropout / zero-fill).

## 3. Results

| Benchmark | A-JEPA |
| --- | --- |
| AudioSet-2M mAP | 48.6 (+1.3 vs AudioMAE among audio-only SSL) |
| AudioSet-20K mAP | 38.4 |
| ESC-50 acc | 96.3 |
| Speech Commands V2 acc | 98.5 |
| Speech Commands V1 acc | 97.7 |
| VoxCeleb (SID) acc | 95.8 |

## 4. Why it matters for AutoJEPA

- A-JEPA is the audio-domain validation target for AutoJEPA. The masking
  curriculum is the prior implementation for
  `autojepa.masking.CurriculumMask`, which composes a generic block mask in
  the early phase and an audio-aware (time-frequency) mask in the late phase.
- The curriculum schedule (square-root-with-floor) generalizes:
  `autojepa.masking.schedules.SquareRootCurriculum` is the parameterized form
  exposed to the LLM diff loop, with c0 and S as searchable axes.
- The fine-tuning trick (regularized attention masking instead of token
  dropout) is the reference for
  `autojepa.masking.RegularizedAttentionMask`, used during the eval-phase
  probe runs to harden the linear / k-NN probes against the train-eval
  masking-policy mismatch.
- Confirms that the I-JEPA encoder/predictor/EMA contract transfers to a new
  modality with zero architectural change — only the masking and the
  curriculum need to be re-tuned. This is exactly the kind of high-leverage
  axis the hybrid policy is designed to discover.

## 5. Caveats / known limitations

- Reports image-style multi-block masking as part of the curriculum; the
  follow-up [JEPA Audio Design Choices](JEPA-Audio-Design-Choices.md)
  paper finds *unstructured* masking actually beats multi-block on audio,
  so A-JEPA's masking ablation is incomplete and partially superseded.
- Uses standard ViT despite spectrogram axes being non-isotropic (time vs
  frequency have different statistics). No architectural anisotropy — only
  the masking is anisotropic.
- 24 pretraining epochs on AudioSet-2M is short; the absolute numbers are not
  necessarily ceilings.
- No collapse-prevention beyond EMA. Pair with [C-JEPA](C-JEPA.md)'s VICReg
  loss for AutoJEPA's audio campaigns.

## 6. References to other corpus entries

- [I-JEPA](I-JEPA.md) — direct architectural ancestor.
- [JEPA Audio Design Choices](JEPA-Audio-Design-Choices.md) — independent
  systematic ablation that contradicts A-JEPA's masking conclusions.
- [C-JEPA](C-JEPA.md) — orthogonal loss-side improvement applicable here.
