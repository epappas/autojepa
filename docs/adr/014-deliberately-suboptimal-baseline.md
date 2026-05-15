# ADR-014: Phase-2 baseline is deliberately suboptimal

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** Architecture writeup §12.1 (Phase 2 risk + mitigation); Phase-2 implementation of `examples/ijepa-cifar10/train.py`

## Context

Writeup §12.1 names the central risk of the entire framework approach:

> The framework's value proposition rests on the LLM proposing useful
> code diffs. If `llm_diff` produces only rejected or neutral diffs on
> a known-good I-JEPA recipe, the LLM either (a) doesn't have enough
> room to improve a strong baseline, or (b) the program.md is
> under-guided. Mitigation: instrument Phase 2 to test on a
> *deliberately suboptimal* baseline (smaller predictor, no VICReg)
> so the LLM has room to find genuine improvements.

A strong I-JEPA-on-CIFAR baseline (proper anti-collapse loss, deeper
predictor, multi-block masking with M=4 targets, cosine LR schedule)
hits ~0.55-0.60 probe_auroc at 4k steps. There is little room above
that floor for an LLM to find improvements without architectural work
that exceeds the per-iteration budget. A campaign against such a
baseline could legitimately produce zero useful diffs even if the
framework were working perfectly — a Type-II framework error.

## Decision

`examples/ijepa-cifar10/train.py` ships a **deliberately weak**
default configuration:

| Lever                  | Strong baseline | AutoJEPA Phase-2 default | Headroom for LLM |
|------------------------|-----------------|--------------------------|------------------|
| `predictor_depth`      | 6               | 2                        | Up to encoder depth (12) |
| `predictor_embed_dim`  | 384             | 128                      | Up to encoder dim (192) |
| `num_targets`          | 4               | 2                        | 1-8 |
| Anti-collapse loss     | VICReg + L2     | plain L2                 | Add VICReg / Barlow / DINO |
| LR schedule            | Cosine to 0     | Constant                 | Add cosine / warmup |
| Masking diversity      | Multi-block + future | Multi-block only       | CompositeMask + others |

The baseline alone is expected to hit ~0.30 probe_auroc at 4k steps.
The Phase-2 falsifier gate (`phase2_falsifier`, `probe_auroc > 0.40`)
requires the LLM to contribute measurable lift — the gap from 0.30
baseline to 0.40 gate is the LLM's contribution.

## Consequences

- **Positive:** The Phase-2 result is interpretable. A pass means the
  LLM contributed measurable lift; a fail means either the LLM is too
  weak for the design space or `program.md` is under-guided. Both are
  actionable diagnoses.
- **Positive:** Diff suggestions from the LLM map to obvious wins
  (add VICReg, deepen predictor, cosine LR), so even a small LLM has
  a reasonable chance of contributing.
- **Negative:** A reader landing on `train.py` may assume the choices
  reflect best practice and replicate them in their own work. The
  module docstring + `program.md` + this ADR call out the suboptimality
  explicitly. The README §"Headroom for the LLM" reinforces it.
- **Negative:** A "production" I-JEPA-CIFAR run would not use these
  defaults. AutoJEPA does not ship a production CIFAR baseline; the
  campaign-discovered hyperparameters in
  `artifacts/ijepa-cifar10/versions/v####/` ARE the production
  baseline once the campaign closes.

## How to apply

- Phase-3 `examples/trace-jepa/` will follow the same pattern: ship
  an intentionally weakened baseline so the campaign has headroom.
  The specific weakening choices will be documented in
  `examples/trace-jepa/program.md` and a sibling ADR.
- A future "production" example (post-Phase-5 public release) ships
  the campaign-discovered best config rather than the suboptimal
  baseline. That example carries a different name (e.g.,
  `examples/ijepa-cifar10-best`) so the falsifier and the production
  recipe do not collide.
- A future Phase-2 rerun (after a meaningful framework change) uses
  the SAME baseline so the Phase-2 outcome is comparable across
  framework versions. This ADR pins the baseline.
