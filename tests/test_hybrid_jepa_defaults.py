"""Regression tests for the AutoJEPA HybridPolicy default widening.

Per writeup §6.3 / ADR-008, AutoJEPA stays in param-exploration mode
longer than upstream autoresearch-rl. These constants are the contract
campaign configs may rely on by omission.
"""

from __future__ import annotations

import inspect

from autojepa.policy.hybrid import HybridPolicy


def _defaults() -> dict[str, int]:
    sig = inspect.signature(HybridPolicy.__init__)
    return {
        name: p.default
        for name, p in sig.parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def test_param_explore_iters_default_widened() -> None:
    assert _defaults()["param_explore_iters"] == 25


def test_stall_threshold_default_widened() -> None:
    assert _defaults()["stall_threshold"] == 5


def test_diff_failure_limit_widened() -> None:
    # writeup §12.5: ~30-40% diff iter failure rate is normal; floor of 5
    assert _defaults()["diff_failure_limit"] == 5
