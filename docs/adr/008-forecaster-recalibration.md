# ADR-008: Recalibrate forecaster for SSL plateau curves

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** Architecture writeup §6.2

## Context

The inherited `autojepa/forecasting.py` fits a power-law `y = a*x^b + c`
to the early progress series and decides whether to cancel a trial
based on extrapolated final value. Its defaults
(`min_steps=5`, `poll_interval_s=5.0`, `min_reports_before_decide=5`)
are calibrated for `autoresearch-rl`'s primary curves — RL reward and
supervised-fine-tuning loss — which:

1. Move within the first 50 steps.
2. Have a clear monotone trend amenable to power-law extrapolation.

JEPA learning curves have a different shape:

1. Fast initial drop in `L_predict` (first 100-500 steps).
2. **Long plateau** during which `L_predict` barely moves but
   `probe_auroc` continues to climb.
3. Slow further descent in `L_predict` over many thousands of steps.

Applying the upstream defaults to a JEPA trial cancels it during the
plateau phase, before the probe-quality signal has had time to develop.
Equivalent to over-pruning.

## Decision

Recalibrate forecaster defaults for SSL workloads:

| Field | autoresearch-rl default | AutoJEPA default | Reason |
|---|---|---|---|
| `min_steps` | 5 | 2000 | Plateau extends past step 5 |
| `poll_interval_s` | 5.0 | 30.0 | Probe AUROC reported at checkpoint cadence, not per step |
| `min_reports_before_decide` | 5 | 10 | More samples needed to distinguish plateau from genuine flat |
| `forecast_target` | (n/a) | `probe_auroc` | Extrapolate downstream score, not training loss |

The new `forecast_target` field (Phase-1 forecaster adaptation) tells
the forecaster which key in the `metrics={...}` dict to extrapolate.
Default for AutoJEPA is `probe_auroc` (ADR-004).

## Consequences

- **Positive:** Plateau-phase trials are no longer wrongly cancelled.
- **Positive:** Forecaster signal is on the same scalar that drives
  keep/discard, eliminating two-clock surprise.
- **Negative:** Cancellation decisions are slower (~30 s vs 5 s poll
  cadence). Acceptable on Basilica where the per-iteration cost is
  GPU-minutes, not GPU-seconds.
- **Negative:** The new defaults are educated guesses. Per writeup
  §12.2, expect a 1-2 wk recalibration burn-in. Reserve a
  no-forecaster control group for the first 50 iterations to measure
  false-positive and false-negative cancellation rates.

## How to apply

- `src/autojepa/config.py::ControllerConfig.intra_iteration_cancel`
  defaults reflect the table above.
- `src/autojepa/forecasting.py` accepts a `forecast_target: str`
  parameter and reads that key from each `emit_progress` call instead
  of the legacy `val_bpb`-shaped scalar.
- A new test in `tests/test_forecasting.py` covers the
  monotone-then-plateau curve shape: the forecaster must NOT cancel
  during the plateau when the target metric is still climbing.
