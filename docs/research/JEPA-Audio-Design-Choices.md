# Investigating Design Choices in Joint-Embedding Predictive Architectures for General Audio Representation Learning

**Authors:** Alain Riou, Stefan Lattner, Gaetan Hadjeres, Geoffroy Peeters
**Venue / Year:** ICASSP 2024 — Self-supervision in Audio, Speech and Beyond workshop
**arXiv:** 2405.08679 — https://arxiv.org/abs/2405.08679
**Status:** Distilled 2026-05-15

## 1. One-line thesis

Several JEPA design choices that are best-practice in the image domain
actively *hurt* audio representation quality — most importantly, multi-block
masking and latent-domain (target-only-unmasked) masking both lose to
unstructured masking with both branches masked, across 8 diverse audio tasks.

## 2. Method

- Input: mel-spectrogram. Encoder ViT-B (12 layers, 768 dim, 12 heads) with
  Flash-Attention. Predictor narrow ViT (8 layers, 512 dim, 16 heads).
  Patch 16x16. Distance: smoothed L1 (slightly beats normalized L2).
  EMA: linear interpolation tau_0 -> tau_T over 300 epochs.
- Systematic ablation over three masking strategies:
  1. **Unstructured** (random patches).
  2. **Multi-block** (I-JEPA-style).
  3. **Time-only** (mask whole time slices, keep all frequency bins).
- Systematic ablation over masking *symmetry*:
  - Latent-domain masking: target encoder sees full input (I-JEPA default).
  - Both-branches masking: both context and target encoders see masked input.
- Evaluation by linear probe across 8 tasks (environmental sound, speech,
  music, pitch, emotion, speaker ID, instruments).

## 3. Results

Best linear-probe scores reported:

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

Headline qualitative findings:
- Unstructured masking beats multi-block on every one of the 8 tasks.
- Latent-domain masking (I-JEPA's choice) *degrades* audio quality vs masking
  both branches.
- Smoothed L1 slightly beats normalized L2.

## 4. Why it matters for AutoJEPA

This paper is the strongest argument for *why AutoJEPA needs an
LLM-driven diff loop in the first place* — image-domain JEPA defaults are not
even close to optimal once you cross modalities, and the wrong defaults are
not visible from training loss alone.

- Concretely, this paper invalidates the assumption that
  `autojepa.masking.MultiBlockInfillMask` is the right default for audio
  campaigns. The hybrid policy must include `mask_strategy` ∈
  {multi_block, unstructured, time_only} as a categorical search axis, with
  unstructured as the audio prior.
- The latent-domain-masking finding maps to a `target_branch_masking` boolean
  in the AutoJEPA training contract. Default true for image, false for audio.
- The smoothed-L1 result motivates exposing the loss-distance metric
  (L1 / smoothed-L1 / L2 / normalized-L2) as a search axis in
  `autojepa.models.losses`, not just the loss-function family.
- Provides a clean second audio benchmark suite (ESC-50, UrbanSound8K, SPCv2,
  VoxCeleb, CREMA-D, GTZAN, NSynth, Surge) suitable for the
  `autojepa.eval` linear-probe panel; the eight-task average is a far less
  noisy campaign objective than any single-task probe.

## 5. Caveats / known limitations

- Workshop paper; no ImageNet-scale or AudioSet-2M result. Absolute numbers
  are below [A-JEPA](A-JEPA.md), which used a much larger pretraining set.
  The paper's value is in the *relative* design-choice findings, not
  state-of-the-art numbers.
- Does not test curriculum masking ([A-JEPA](A-JEPA.md)'s contribution), so
  the question "does unstructured-only beat A-JEPA's curriculum?" is open.
  AutoJEPA's hybrid policy should test both.
- Compute envelope is small; conclusions might soften at scale (the multi-block
  prior was developed precisely because it scaled).
- Single backbone (ViT-B). Findings may not transfer to CNN encoders.

## 6. References to other corpus entries

- [A-JEPA](A-JEPA.md) — the audio JEPA paper this work partially supersedes.
- [I-JEPA](I-JEPA.md) — source of the image-domain priors that fail here.
- [C-JEPA](C-JEPA.md) — orthogonal loss-side improvement; untested in this
  paper.
