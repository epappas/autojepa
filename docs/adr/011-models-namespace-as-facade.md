# ADR-011: `autojepa.models` namespace is a facade over stable-pretraining

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** Phase-1 implementation of `models/ema.py` and `models/losses.py`. Extends ADR-003.

## Context

ADR-003 established that probes and label-aware collapse signals are
wrapped from `stable_pretraining` rather than reimplemented. While
implementing Phase-1 we discovered that `stable_pretraining` also
provides:

- `stable_pretraining.TeacherStudentWrapper` with the exact
  `base_ema_coefficient=0.996, final_ema_coefficient=1.0` defaults
  cited by I-JEPA.
- `stable_pretraining.callbacks.TeacherStudentCallback` that drives
  the EMA update on each Lightning step.
- `stable_pretraining.losses.{VICRegLoss, BarlowTwinsLoss, BYOLLoss,
  DINOv1Loss, NTXEntLoss, NegativeCosineSimilarity}` — every SSL loss
  the writeup §6.4 program.md template lists as a high-value diff
  target.

Reimplementing any of these inside `autojepa.models` would be:
- Pure duplication (the math is the same).
- A maintenance liability (we would have to track upstream bugfixes by
  hand).
- A divergence risk (subtle differences in normalization or detach
  conventions are how SSL collapse defenses fail silently).

## Decision

`autojepa.models` is a **facade namespace**. It re-exports
`stable_pretraining` primitives under AutoJEPA-flavored names and adds
only:

1. Closed-form helpers that don't exist upstream (`l1_loss`,
   `l2_loss` — used by the I-JEPA-shaped predictor objective).
2. AutoJEPA-specific config dataclasses (`EMAConfig`) that surface the
   parameters the hybrid policy will search over (writeup §6.3).
3. AutoJEPA-specific invariant assertions (`assert_no_grad_on_target`)
   that the AST validator's `required_calls` list (writeup §6.4
   program.md) calls out as hard-fail gates.
4. A flat `LOSS_REGISTRY: dict[str, Callable]` mapping string keys to
   loss factories, so the LLM diff policy can swap losses by editing
   a config field instead of rewriting import statements.

The facade pattern means an example `train.py` only needs:

```python
from autojepa.models import losses, ema
encoder = build_some_encoder(...)
target = ema.build_target_encoder(encoder)
loss_fn = losses.build_loss("vicreg", sim_coeff=25, var_coeff=25, cov_coeff=1)
```

instead of importing from `stable_pretraining` directly.

## Consequences

- **Positive:** Single import surface for `train.py` authors and for the
  LLM diff policy's prompt context (which sees `autojepa.models.*`
  namespaces).
- **Positive:** `LOSS_REGISTRY` keys become a stable contract — the
  policy's param search can propose `loss_type: "barlow_twins"` and
  the loss is resolved at trial start without code-diffing.
- **Positive:** When stable-pretraining renames something, only the
  facade module breaks — `train.py` files in user campaigns are
  insulated.
- **Negative:** A user familiar with `stable_pretraining` directly may
  be confused by the AutoJEPA-flavored re-exports. Mitigation: the
  `losses.py` and `ema.py` module docstrings cite the upstream
  symbols by name.
- **Negative:** Some upstream features (e.g., bespoke kwargs on
  `VICRegLoss`) leak through the registry's `**kwargs` dispatch and
  are coupled to the upstream signature. Acceptable: the kwargs
  contract is documented at the upstream site.

## How to apply

- New SSL losses go into `LOSS_REGISTRY` under their canonical paper
  name (snake_case). If upstream adds them, just register; if upstream
  doesn't, implement once in `losses.py` next to the registry.
- New EMA-related primitives go into `models/ema.py`. AutoJEPA-specific
  invariants (e.g., `assert_no_grad_on_target`) live alongside the
  factory.
- New encoder / predictor architectures go into `models/encoders.py`
  / `models/predictors.py` and follow the same facade-over-`timm`
  convention.
