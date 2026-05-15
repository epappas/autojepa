# ADR-013: SSL plateau-then-rise is a known forecaster limitation

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** Phase-1 batch-6 implementation discovery while writing `tests/test_forecaster_ssl.py`. Refines ADR-008 / ADR-012.

## Context

ADR-008 set out a forecaster recalibration for SSL workloads: bump
`min_steps`, `poll_interval_s`, and `min_reports_before_decide`
defaults so the forecaster does not over-cancel JEPA trials in their
plateau phase. ADR-012 then refined the implementation to use
`objective.metric` directly (no separate `forecast_target` field).

While writing the SSL plateau regression test
(`tests/test_forecaster_ssl.py`) we directly probed the inherited
power-law fit `y = a*x^b + c` against a synthetic SSL trajectory:

```
SERIES = [0.50, 0.51, 0.52, 0.55, 0.55, 0.55, 0.55, 0.58, 0.62, 0.65, 0.68]
forecast_value(negated, len) = -0.544
actual final               = -0.680
```

The fit underestimates the final value by 0.136. Because of this, the
forecaster reports `should_early_stop=True` against a target of -0.6
even though the actual trajectory ends at -0.68. The same fit at
2*len still predicts -0.544 — the power law has saturated on the
early plateau and refuses to extrapolate the steep late drop.

## Why this happens

A power-law `a*x^b + c` is monotone-decreasing-with-slowing-rate. SSL
plateau-then-rise is plateau-then-fast-drop (in the negated form the
forecaster sees). The two shapes are not in the same family, so a
least-squares power-law fit picks the dominant early-plateau pattern
and extrapolates flat. Adding more data points at the end pulls the
fit incrementally but cannot recover the actual asymptote on a
small series.

This is the limitation writeup §12.2 anticipated:
> "The `min_steps`, `min_reports_before_decide`, and `forecast_target`
> defaults are educated guesses. Expect a 1-2 wk recalibration burn-in
> where the forecaster either over-cancels or under-cancels. Track
> false-positive and false-negative cancellation rates explicitly."

## Decision

AutoJEPA does not change the `forecasting.py` algorithm in Phase 1.
We accept the over-cancellation as a documented limitation and rely
on three framework-level mitigations:

1. **`min_steps=2000` default (ADR-008).** The IntraIterationGuard
   does not invoke the forecaster until at least 2000 trial steps
   have elapsed. Assuming a checkpoint cadence of one report per
   ~40 steps, that is ~50 reports — well past the typical SSL early
   plateau where the fit is broken.
2. **`min_reports_before_decide=10` default (ADR-008).** The guard
   refuses to call the forecaster on fewer than 10 reports.
3. **Multi-seed scoring (ADR-009).** The campaign-level objective is
   `mean(probe_auroc)` across 3 seeds, which smooths plateau-noise
   and reduces the impact of any single trial's mis-cancellation.

## Consequences

- **Positive:** No algorithmic surgery in Phase 1. The inherited
  forecaster code path is unchanged, which keeps cherry-pick
  compatibility with upstream `autoresearch-rl`.
- **Positive:** The over-cancellation regime is documented, tested,
  and covered by an explicit test
  (`tests/test_forecaster_ssl.py::TestPlateauThenRiseLimitation`).
  Anyone modifying `forecasting.py` will see the regression test
  and the failure-mode comment together.
- **Negative:** Trials that genuinely break out of a long plateau
  earlier than step 2000 (rare with the recommended training
  schedules, but possible with aggressive learning-rate warmups) may
  still be cancelled before their breakout is visible. Mitigation:
  per-campaign `IntraIterationCancelConfig.min_steps` override in
  the example config.
- **Negative:** Under-cancellation in the opposite direction
  (forecaster optimistically extrapolates flat past the actual
  asymptote) is also possible. This is bounded by the existing
  `should_early_stop` math which only reports early-stop when the
  forecast strictly exceeds the target.

## How to apply

- Phase-2 `examples/ijepa-cifar10/config.yaml` declares
  `intra_iteration_cancel.enabled: true` and either accepts the
  recalibrated defaults or overrides `min_steps` upward if the chosen
  training schedule has a longer plateau.
- Phase-4 hardening tasks may include a forecaster-algorithm refresh
  (sigmoid-shaped or piecewise-linear fits would handle plateau-then-rise
  better). Whenever that work lands, the regression test in
  `tests/test_forecaster_ssl.py::TestPlateauThenRiseLimitation` will
  flip from documenting the limitation to documenting the fix.
- Per writeup §12.2, the first 50 iterations of any new AutoJEPA
  campaign should reserve a no-forecaster control group to measure
  false-positive and false-negative cancellation rates.

## Refinement of prior ADRs

ADR-008 said the recalibration should "address" plateau over-cancellation.
ADR-013 narrows the claim: the recalibration *moves* the over-cancellation
window to a regime where it does not bite typical SSL training
schedules — it does not eliminate the underlying algorithmic limitation.
