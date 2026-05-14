"""Unit tests for autojepa.gates.

Validates the decision-gate engine introduced in writeup §7.6:
- Per-requirement comparison ops on a metrics dict
- Activation cutoff via `after_iters`
- abort_campaign vs warn semantics
- Pydantic <-> dataclass round-trip for YAML-loaded configs
"""

from __future__ import annotations

import pytest

from autojepa.gates import (
    Gate,
    GateConfig,
    GateEngine,
    GateRequirement,
    GateRequirementConfig,
    build_engine,
)


class TestGateRequirementEval:
    def test_greater_than_passes(self) -> None:
        r = GateRequirement(metric="probe_auroc", op=">", threshold=0.7)
        assert r.evaluate({"probe_auroc": 0.71})
        assert not r.evaluate({"probe_auroc": 0.7})
        assert not r.evaluate({"probe_auroc": 0.5})

    def test_greater_or_equal(self) -> None:
        r = GateRequirement(metric="probe_auroc", op=">=", threshold=0.7)
        assert r.evaluate({"probe_auroc": 0.7})
        assert r.evaluate({"probe_auroc": 0.71})
        assert not r.evaluate({"probe_auroc": 0.69})

    def test_less_than(self) -> None:
        r = GateRequirement(metric="probe_fpr", op="<", threshold=0.05)
        assert r.evaluate({"probe_fpr": 0.04})
        assert not r.evaluate({"probe_fpr": 0.05})

    def test_equality(self) -> None:
        r = GateRequirement(metric="x", op="==", threshold=1.0)
        assert r.evaluate({"x": 1.0})
        assert not r.evaluate({"x": 1.0001})

    def test_missing_metric_returns_false(self) -> None:
        r = GateRequirement(metric="missing", op=">", threshold=0.0)
        assert not r.evaluate({"other": 1.0})

    def test_describe_includes_actual_value(self) -> None:
        r = GateRequirement(metric="probe_auroc", op=">", threshold=0.7)
        d = r.describe({"probe_auroc": 0.55})
        assert "probe_auroc" in d and ">" in d and "0.7" in d and "0.55" in d

    def test_describe_marks_missing(self) -> None:
        r = GateRequirement(metric="probe_auroc", op=">", threshold=0.7)
        d = r.describe({})
        assert "missing" in d


class TestGateValidation:
    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Gate(name="", after_iters=0, require=(GateRequirement("x", ">", 0.0),))

    def test_rejects_negative_after_iters(self) -> None:
        with pytest.raises(ValueError, match="after_iters"):
            Gate(name="g", after_iters=-1, require=(GateRequirement("x", ">", 0.0),))

    def test_rejects_no_requirements(self) -> None:
        with pytest.raises(ValueError, match="at least one requirement"):
            Gate(name="g", after_iters=0, require=())


class TestGateEngineEvaluate:
    def _engine(self) -> GateEngine:
        return GateEngine(
            [
                Gate(
                    name="mvp",
                    after_iters=20,
                    require=(
                        GateRequirement("probe_auroc", ">", 0.7),
                        GateRequirement("probe_fpr", "<", 0.05),
                    ),
                    on_fail="abort_campaign",
                ),
                Gate(
                    name="warning",
                    after_iters=5,
                    require=(GateRequirement("probe_auroc", ">", 0.4),),
                    on_fail="warn",
                ),
            ]
        )

    def test_no_gates_active_before_after_iters(self) -> None:
        results = self._engine().evaluate(0, {"probe_auroc": 0.0, "probe_fpr": 1.0})
        assert results == []

    def test_warn_gate_activates_first(self) -> None:
        results = self._engine().evaluate(5, {"probe_auroc": 0.5})
        assert [r.name for r in results] == ["warning"]
        assert results[0].passed is True

    def test_warn_gate_fails_when_below_threshold(self) -> None:
        results = self._engine().evaluate(5, {"probe_auroc": 0.3})
        assert [r.name for r in results] == ["warning"]
        assert results[0].passed is False
        assert "probe_auroc" in results[0].failures[0]

    def test_mvp_gate_fires_at_after_iters(self) -> None:
        results = self._engine().evaluate(20, {"probe_auroc": 0.8, "probe_fpr": 0.04})
        names = [r.name for r in results]
        assert "mvp" in names
        mvp = next(r for r in results if r.name == "mvp")
        assert mvp.passed is True

    def test_mvp_gate_fails_when_one_requirement_misses(self) -> None:
        results = self._engine().evaluate(20, {"probe_auroc": 0.8, "probe_fpr": 0.10})
        mvp = next(r for r in results if r.name == "mvp")
        assert mvp.passed is False
        assert any("probe_fpr" in f for f in mvp.failures)
        assert all("probe_auroc" not in f for f in mvp.failures)


class TestGateEngineShouldAbort:
    def test_does_not_abort_when_no_gate_failed(self) -> None:
        eng = GateEngine([
            Gate(
                name="mvp",
                after_iters=10,
                require=(GateRequirement("probe_auroc", ">", 0.5),),
                on_fail="abort_campaign",
            )
        ])
        should, reason = eng.should_abort(20, {"probe_auroc": 0.7})
        assert not should
        assert reason == ""

    def test_aborts_with_descriptive_reason(self) -> None:
        eng = GateEngine([
            Gate(
                name="mvp",
                after_iters=10,
                require=(GateRequirement("probe_auroc", ">", 0.7),),
                on_fail="abort_campaign",
            )
        ])
        should, reason = eng.should_abort(20, {"probe_auroc": 0.5})
        assert should
        assert "mvp" in reason and "iter 20" in reason and "probe_auroc" in reason

    def test_warn_failure_does_not_abort(self) -> None:
        eng = GateEngine([
            Gate(
                name="warning",
                after_iters=10,
                require=(GateRequirement("probe_auroc", ">", 0.7),),
                on_fail="warn",
            )
        ])
        should, _ = eng.should_abort(20, {"probe_auroc": 0.5})
        assert not should


class TestGateRequirementConfig:
    def test_parses_greater_than(self) -> None:
        req = GateRequirementConfig(metric="probe_auroc", expr=">0.7").to_requirement()
        assert req == GateRequirement("probe_auroc", ">", 0.7)

    def test_parses_less_or_equal(self) -> None:
        req = GateRequirementConfig(metric="loss", expr="<=2.5").to_requirement()
        assert req == GateRequirement("loss", "<=", 2.5)

    def test_parses_scientific_notation(self) -> None:
        req = GateRequirementConfig(metric="x", expr=">1e-3").to_requirement()
        assert req == GateRequirement("x", ">", 1e-3)

    def test_rejects_malformed_expression(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            GateRequirementConfig(metric="x", expr="garbage")
        with pytest.raises(ValueError, match="must match"):
            GateRequirementConfig(metric="x", expr=">")
        with pytest.raises(ValueError, match="must match"):
            GateRequirementConfig(metric="x", expr="0.5")


class TestGateConfigRoundTrip:
    def test_yaml_shape_to_engine(self) -> None:
        cfgs = [
            GateConfig(
                name="mvp",
                after_iters=20,
                require={"probe_auroc": ">0.7", "probe_fpr": "<0.05"},
                on_fail="abort_campaign",
            )
        ]
        engine = build_engine(cfgs)
        should, reason = engine.should_abort(20, {"probe_auroc": 0.5, "probe_fpr": 0.1})
        assert should
        assert "mvp" in reason

    def test_empty_require_rejected(self) -> None:
        cfg = GateConfig(name="g", after_iters=0, require={})
        with pytest.raises(ValueError, match="at least one requirement"):
            cfg.to_gate()
