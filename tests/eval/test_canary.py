"""Unit tests for autojepa.eval.canary.

The canary protocol from writeup §7.4 is implemented as a special-case
`gates.Gate` that fires at iteration 0 against `canary_loss`. Tests
here validate the config boundaries and the Gate construction; the
end-to-end behavior (canary loss series -> gate evaluation -> abort)
is covered by the existing tests in tests/test_gates.py.
"""

from __future__ import annotations

import pytest

from autojepa.eval.canary import CanaryConfig, build_canary_gate
from autojepa.gates import Gate, GateEngine


class TestCanaryConfig:
    def test_defaults_match_writeup(self) -> None:
        cfg = CanaryConfig()
        assert cfg.n_samples == 1000
        assert cfg.max_steps == 200
        assert cfg.loss_metric == "canary_loss"
        assert cfg.loss_threshold == 0.05
        assert cfg.on_fail_action == "abort_campaign"

    def test_rejects_zero_samples(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            CanaryConfig(n_samples=0)

    def test_rejects_zero_max_steps(self) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            CanaryConfig(max_steps=0)

    def test_rejects_non_positive_threshold(self) -> None:
        with pytest.raises(ValueError, match="loss_threshold"):
            CanaryConfig(loss_threshold=0.0)

    def test_rejects_unknown_action(self) -> None:
        with pytest.raises(ValueError, match="on_fail_action"):
            CanaryConfig(on_fail_action="ignore")


class TestBuildCanaryGate:
    def test_default_gate_shape(self) -> None:
        gate = build_canary_gate()
        assert isinstance(gate, Gate)
        assert gate.name == "sanity_overfit_canary"
        assert gate.after_iters == 0
        assert gate.on_fail == "abort_campaign"
        assert len(gate.require) == 1
        req = gate.require[0]
        assert req.metric == "canary_loss"
        assert req.op == "<"
        assert req.threshold == 0.05

    def test_custom_threshold(self) -> None:
        gate = build_canary_gate(CanaryConfig(loss_threshold=0.01))
        assert gate.require[0].threshold == 0.01

    def test_warn_action_propagates(self) -> None:
        gate = build_canary_gate(CanaryConfig(on_fail_action="warn"))
        assert gate.on_fail == "warn"

    def test_engine_aborts_on_failed_canary(self) -> None:
        engine = GateEngine([build_canary_gate()])
        # iter=0: canary loss above threshold -> should abort
        should, reason = engine.should_abort(0, {"canary_loss": 0.5})
        assert should is True
        assert "sanity_overfit_canary" in reason
        assert "canary_loss" in reason

    def test_engine_passes_when_canary_satisfied(self) -> None:
        engine = GateEngine([build_canary_gate()])
        should, _ = engine.should_abort(0, {"canary_loss": 0.01})
        assert should is False

    def test_engine_with_warn_does_not_abort(self) -> None:
        engine = GateEngine([build_canary_gate(CanaryConfig(on_fail_action="warn"))])
        should, _ = engine.should_abort(0, {"canary_loss": 0.5})
        assert should is False
