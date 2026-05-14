# ADR-012: `objective.metric` is the forecast target — no separate field

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** Phase-1 implementation discovery while wiring the recalibrated forecaster (writeup §6.2). Refines ADR-008.

## Context

ADR-008 said the forecaster recalibration should "add a new `forecast_target`
field" alongside the SSL plateau-aware `min_steps` / `poll_interval_s`
defaults. Implementation revealed that the inherited
`autoresearch-rl` plumbing already wires `ObjectiveConfig.metric`
through to the `IntraIterationGuard` constructor at three call sites:

- `src/autojepa/controller/parallel_engine.py:592` —
  `IntraIterationGuard(metric=objective.metric, ...)`
- `src/autojepa/controller/engine.py:333` — same
- `src/autojepa/controller/continuous.py:38, 58` —
  `metric=objective.metric if objective else "val_bpb"`

Inside the guard
(`src/autojepa/controller/intra_iteration.py:157`), the cumulative
metric series is built by extracting `r.metrics[self._metric]` from
each progress report. The metric name is whatever string is in
`objective.metric`.

Adding a separate `forecast_target` field would create two parallel
sources of truth for "which scalar drives the forecaster" and force
the engine wiring to choose between them. That is the kind of
configuration drift ADR-008 was trying to prevent.

## Decision

`ObjectiveConfig.metric` IS the forecast target. AutoJEPA does not
add a separate `forecast_target` field to
`IntraIterationCancelConfig` or anywhere else. The recalibration in
ADR-008 is delivered through:

1. New default `ObjectiveConfig.metric: str = "probe_auroc"` (was `"val_bpb"`).
2. New default `ObjectiveConfig.direction: Literal["min", "max"] = "max"` (was `"min"`).
3. New default `IntraIterationCancelConfig.min_steps: int = 2000` (was 5).
4. New default `IntraIterationCancelConfig.poll_interval_s: float = 30.0` (was 5.0).
5. New default `IntraIterationCancelConfig.min_reports_before_decide: int = 10` (was 5).
6. The mirror `GuardConfig` dataclass in `controller/intra_iteration.py`
   gets the same defaults so unit tests of the guard match runtime
   reality.

The `continuous.py` fallback `metric="val_bpb"` is left in place as a
defensive default for legacy callers that pass `objective=None`.
Production AutoJEPA campaigns always set `objective.metric` and never
hit the fallback.

## Consequences

- **Positive:** No two-clock confusion. Per-campaign override of the
  forecast target is the same edit as per-campaign override of the
  keep/discard objective.
- **Positive:** ADR-008's intent is met without expanding the config
  surface.
- **Positive:** Inherited `autoresearch-rl` tests pass unchanged
  (552/561 with the same 9 example-fixture failures from ADR-006).
- **Negative:** Anyone reading ADR-008 would expect a `forecast_target`
  field to exist. This ADR is the cross-reference; ADR-008's
  status line stays Accepted with a "see ADR-012 for the
  implementation detail" pointer.

## How to apply

- Future code that needs to know "which metric is the forecaster
  watching" reads `ObjectiveConfig.metric`. There is no other source.
- Future code that proposes a per-iteration override of the forecast
  target overrides `objective.metric` for that iteration only — the
  hybrid policy already supports per-iteration param overrides.
- ADR-008's reference table is amended-by-reference: the row
  `forecast_target | (n/a) | probe_auroc` is satisfied by
  `objective.metric` defaulting to `probe_auroc` instead of being a
  separate field.
