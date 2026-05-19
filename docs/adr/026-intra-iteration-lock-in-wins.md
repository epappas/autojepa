# ADR-026: IntraIterationGuard locks in wins — never cancel an iter whose series already beats best

- **Status:** Accepted
- **Date:** 2026-05-19
- **Deciders:** epappas
- **Source:** `intra_iteration_cancel` forecaster bug tracked across
  v21/v23/v24/v25 in `docs/phase-2-fix-diary.md`; finally diagnosed
  and fixed in Phase-4 hardening 2026-05-19.
- **Refines:** ADR-008 (forecaster recalibration), ADR-013 (plateau
  limitation).

## Context

The `IntraIterationGuard` watches the live progress series of an
in-flight trial and uses a power-law forecaster to decide whether
the trial can still beat the run-wide best. When the forecast says
"won't beat best", the guard writes a cancel control file and the
trial exits early.

This is correct for trials that have NEVER reached the bar (their
series sits well below best and the forecaster correctly predicts
they'll continue to do so). It is INCORRECT for trials whose
observed series has already crossed the bar but whose forecast
predicts a final dip below it. Live evidence across four campaigns:

| Campaign | Iter | Peak probe | Best | Outcome |
|---|---|---|---|---|
| v21 | 1 | 0.295 | 0.273 | cancelled |
| v23 | 1 | 0.295 | 0.250 | cancelled |
| v24 | 1 | 0.264 | 0.252 | cancelled |
| v25 | 1 | 0.264 | 0.254 | cancelled |

In every case the trial had ALREADY produced a probe value above
the run-wide best — it was already a keep-worthy outcome. The
forecaster fit late-stage SSL noise (typical pattern: rising probe
with a small dip near the final reporting step) into a decaying
power-law tail and concluded "predicted final value < best." Every
one of these iters would have raised the run-wide best if allowed
to complete. Estimated cost across the four campaigns:
~$10-15 in unrecovered GPU time + significant analytic damage
(falsely suggesting "param search has plateaued" earlier than it
actually had).

ADR-013 acknowledged "SSL plateau-then-rise is a known forecaster
limitation" and accepted lost wins as a tradeoff. The cases above
are NOT plateau-then-rise — they're rise-then-dip on a CURRENT
value that already beats best. That's not a limitation of the
forecaster; that's a misuse of its output.

## Decision

In `IntraIterationGuard.evaluate`, short-circuit to `"continue"`
when the observed series already crosses the bar:

```python
if self._direction == "max":
    best_f = float(best_value)
    if max(series) >= best_f:
        return ("continue", "current_already_beats_best")
    # ... forecast as before
else:  # min
    best_f = float(best_value)
    if min(series) <= best_f:
        return ("continue", "current_already_beats_best")
    # ... forecast as before
```

The new reason string `current_already_beats_best` is distinct from
the existing `forecast_above_best` / `forecast_below_best` reasons,
so post-mortem analysis can identify which guards fired.

Order matters: the lock-in check runs AFTER the
`min_reports_before_decide` guard and `no_best_yet` guard, but
BEFORE the forecaster invocation. We don't want a single lucky
early report (before the minimum reports threshold) to lock in
prematurely.

## Consequences

- **Positive:** v21/v23/v24/v25 iter=1-style "iter has already won
  but forecaster cancels it" failure mode is eliminated.
- **Positive:** No worsening of the doomed-series case. Tests
  `test_max_direction_cancels_when_series_never_reaches_best` and
  `test_min_direction_cancels_doomed_series` lock in the unchanged
  behaviour for trials that never crossed best.
- **Positive:** Fixes a real `lost wins` bug without touching the
  forecaster itself (which still works correctly within its design
  domain). Minimal blast radius.
- **Negative:** Iters that beat best but then completely collapse
  late-stage now spend more compute before completion. We accept
  this because (a) the final reported value is still kept-worthy
  if it beats best, and (b) a true collapse triggers the canary
  guard in `train.py` independently.
- **Negative:** Adds a tiny per-poll computation (`max(series)` /
  `min(series)`). Series length is bounded by
  `max_reports_per_iter ~ MAX_STEPS / report_interval`, typically
  a few thousand entries — O(N) is fine at 30s poll cadence.

## How to apply

- Any future cancellation policy must distinguish "trial can never
  win" from "trial has already won and might decay." The latter is
  ALWAYS a continue.
- When ADR-008's forecaster is replaced (e.g. with a Bayesian
  changepoint detector), the lock-in guard MUST survive. Add a
  regression test asserting `max(series) >= best -> continue`
  before swapping the underlying forecaster.

## Related work / follow-ups

- The "wall hit during post-training upload window" failure (v26
  iter=1 elapsed=5920s > wall=5400s, mis-blamed on the forecaster)
  is a SEPARATE bug not addressed here. Tracked for Phase-4
  follow-up. The fix is likely on the basilica adapter side
  (timeout_s should cover the model-download window, not just the
  training window).
