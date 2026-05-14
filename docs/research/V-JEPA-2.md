# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

**Authors:** Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas
**Venue / Year:** arXiv preprint, 2025 (FAIR)
**arXiv:** 2506.09985 — https://arxiv.org/abs/2506.09985
**Status:** Distilled 2026-05-15

## 1. One-line thesis

A two-stage recipe — an action-free JEPA pretrained on >1M hours of internet
video, then post-trained on <62 hours of robot video with a latent
action-conditioned predictor — yields a single model that understands,
anticipates, and plans in the physical world.

## 2. Method

- **Stage 1 (V-JEPA 2):** action-free joint-embedding predictive pretraining
  on VideoMix22M (Something-Something v2, Kinetics, HowTo100M, YT-Temporal-1B,
  ImageNet). Encoder ViT-L (300M) up to ViT-g (~1B params). Progressive-
  resolution training delivers an 8.4x reduction in GPU time vs full-res.
- **Predictor:** ~300M parameter transformer, 24 layers, 16 heads, 1024 hidden,
  GELU. Block-causal attention pattern: each patch attends to action token,
  end-effector state, other patches at the same timestep, and patches from
  prior timesteps.
- **Stage 2 (V-JEPA 2-AC):** post-train predictor (encoder frozen) on Droid
  robot video, conditioning on latent action tokens. Loss combines
  teacher-forcing prediction with two-step rollout; targets are EMA encoder
  embeddings, not pixels.
- **Collapse prevention:** EMA target + stop-gradient on the target branch.
- For LLM alignment / VQA, the encoder is plugged into an 8B LLM.

## 3. Results

- Something-Something v2 attentive probe: **77.3 top-1**.
- Epic-Kitchens-100 action anticipation: **39.7 recall@5** (44% relative over
  prior SOTA).
- VQA at 8B scale (with LLM alignment): PerceptionTest 84.0, TempCompass 76.9.
- **Zero-shot Franka manipulation in two different labs**, no environment
  fine-tuning, no reward shaping: reach within 4 cm of goal; ~65% grasp
  success; 75% pick-and-place on cup, 65% on box.

## 4. Why it matters for AutoJEPA

- V-JEPA 2's predictor scaling (~300M parameters, 24 layers) is the empirical
  ceiling justifying the predictor-depth axis being a core hybrid-policy
  search dimension in `autojepa.models.predictors`. Image-only I-JEPA gets
  away with depth-12; video / world-model regimes do not.
- The block-causal attention pattern (action + state + cross-time) motivates
  the cross-attention / causal-attention variant in
  `autojepa.models.predictors`. The default predictor remains the I-JEPA
  bidirectional ViT; the V-JEPA-2-style predictor is the alternative
  registered as `predictors.BlockCausalPredictor`.
- The two-stage scheme (frozen encoder + action-conditioned predictor
  post-training) is a template for downstream specialization campaigns where
  the AutoJEPA loop should freeze the encoder and only let the LLM mutate
  predictor / loss / rollout-length parameters.
- The progressive-resolution training trick (8.4x speedup) is a candidate
  parameter-axis for the framework's parallel iteration scheduler, since it
  changes wall-clock per iteration without changing the underlying objective.

## 5. Caveats / known limitations

- Compute envelope is far outside AutoJEPA Phase 2 budgets. Use V-JEPA 2 only
  as architectural reference; do not attempt to reproduce numbers.
- Video-domain primitives (temporal patches, action conditioning) are
  out-of-scope for AutoJEPA v1, which targets image and audio domains.
- The action-conditioning protocol assumes Droid-style trajectories; transfer
  to other embodiments is not characterized in the paper.

## 6. References to other corpus entries

- [I-JEPA](I-JEPA.md) — image-domain ancestor of the encoder/predictor design.
- [C-JEPA](C-JEPA.md) — orthogonal collapse-prevention strategy applicable to
  the V-JEPA-2 predictor as well.
- [A-JEPA](A-JEPA.md) — temporal-modality counterpart in audio.
