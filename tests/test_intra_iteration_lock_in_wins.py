"""Regression tests for IntraIterationGuard "lock in wins" guard (ADR-026).

Before this fix, the power-law forecaster could cancel iters whose
CURRENT series already contained values at-or-above best, on the basis
of a forecasted final dip. Live evidence:

- v21 iter=1: peak probe=0.295 vs best=0.273 -> cancelled
- v23 iter=1: peak probe=0.295 vs best=0.250 -> cancelled
- v24 iter=1: peak probe=0.264 vs best=0.252 -> cancelled
- v25 iter=1: peak probe=0.264 vs best=0.254 -> cancelled

In every case the trial had already produced a probe value above the
run-wide best; the iter would have been kept if allowed to complete.
The forecaster fit late-stage SSL noise (rising probe with a small dip
near the end of training) into a decaying tail and concluded "won't
beat best." See docs/phase-2-fix-diary.md 2026-05-19.

Fix: in `evaluate()`, short-circuit to "continue" when the observed
series already crosses best (for max direction) or sinks below best
(for min direction). Iters that have ALREADY won cannot be doomed.
"""
from __future__ import annotations

from pathlib import Path

from autojepa.controller.intra_iteration import GuardConfig, IntraIterationGuard
from autojepa.target.progress_reader import ProgressReader


def _make_guard(
    tmp_path: Path, *, direction: str, best_value: float | None,
    min_reports: int = 5,
) -> IntraIterationGuard:
    progress = tmp_path / "p.jsonl"
    progress.touch()
    reader = ProgressReader(str(progress))
    return IntraIterationGuard(
        reader=reader,
        control_path=str(tmp_path / "c.json"),
        metric="probe_auroc",
        direction=direction,
        best_value=best_value,
        config=GuardConfig(min_reports_before_decide=min_reports),
    )


# --- max direction: lock in wins ---


def test_max_direction_continues_when_series_already_beats_best(
    tmp_path: Path,
) -> None:
    """v25 iter=1 pattern: series peaks ABOVE best then dips slightly.
    The forecaster used to fit the dip as a decay and cancel. With
    the lock-in guard, we MUST continue because the iter has already
    produced a probe value above best — it is already a keep-worthy
    outcome and the engine should record the final value, not throw
    it away."""
    guard = _make_guard(tmp_path, direction="max", best_value=0.254)
    # Rising probe that already exceeded best=0.254, with a small dip.
    series = [0.20, 0.23, 0.26, 0.264, 0.262, 0.256]
    decision, reason = guard.evaluate(series)
    assert decision == "continue", (
        f"iter with peak {max(series)} > best 0.254 must not cancel "
        f"on a forecasted dip; got reason={reason}"
    )
    assert reason == "current_already_beats_best"


def test_max_direction_cancels_when_series_never_reaches_best(
    tmp_path: Path,
) -> None:
    """Sanity: doomed series (stuck well below best) must still cancel.
    The lock-in guard is precisely scoped to series that have ALREADY
    won; series that never crossed best are still the forecaster's
    business."""
    guard = _make_guard(tmp_path, direction="max", best_value=0.95)
    decision, reason = guard.evaluate([0.45, 0.48, 0.50, 0.51, 0.50, 0.50])
    assert decision == "cancel"
    assert reason == "forecast_below_best"


def test_max_direction_continues_at_exact_tie_with_best(tmp_path: Path) -> None:
    """A tie counts as already-won. >= is the correct comparison —
    keeping a value identical to best is harmless (loop just keeps
    the existing best), but throwing it away on a noise-forecast is
    a real wins-lost case (see v24 iter=1)."""
    guard = _make_guard(tmp_path, direction="max", best_value=0.252)
    series = [0.20, 0.22, 0.24, 0.252, 0.25, 0.24]
    decision, _ = guard.evaluate(series)
    assert decision == "continue"


# --- min direction: same guard, lower-is-better ---


def test_min_direction_continues_when_series_already_below_best(
    tmp_path: Path,
) -> None:
    """Mirror case for min direction: iter has already gone below best
    (= already a keep) and the forecaster predicts a bump back up.
    Lock in the win."""
    guard = _make_guard(tmp_path, direction="min", best_value=0.40)
    # Series dropped below 0.40 (current best is 0.40), then jitters up.
    series = [1.0, 0.8, 0.6, 0.40, 0.39, 0.41, 0.43]
    decision, reason = guard.evaluate(series)
    assert decision == "continue", (
        f"iter that already crossed best=0.40 must not be cancelled; "
        f"got reason={reason}"
    )
    assert reason == "current_already_beats_best"


def test_min_direction_cancels_doomed_series(tmp_path: Path) -> None:
    """Sanity: stuck-high series (never reached best) still cancels."""
    guard = _make_guard(tmp_path, direction="min", best_value=0.40)
    decision, reason = guard.evaluate([0.95, 0.93, 0.92, 0.91, 0.90, 0.90])
    assert decision == "cancel"
    assert reason == "forecast_above_best"


# --- pre-existing semantics preserved ---


def test_insufficient_reports_still_short_circuits(tmp_path: Path) -> None:
    """The min_reports_before_decide guard MUST still fire before the
    lock-in check; otherwise a single lucky early report could let an
    iter never be cancellable. Defensive ordering."""
    guard = _make_guard(
        tmp_path, direction="max", best_value=0.5, min_reports=5,
    )
    decision, reason = guard.evaluate([0.6, 0.7])  # only 2 reports, > best
    assert decision == "continue"
    assert reason == "insufficient_reports"


def test_no_best_yet_short_circuits(tmp_path: Path) -> None:
    """No best to compare against → continue. Unchanged behaviour."""
    guard = _make_guard(tmp_path, direction="max", best_value=None)
    decision, reason = guard.evaluate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert decision == "continue"
    assert reason == "no_best_yet"
