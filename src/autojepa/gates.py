"""Decision-gate engine for AutoJEPA campaigns.

Net-new module introduced in writeup §7.6. Per-campaign explicit kill
criteria make "is AutoJEPA working on this problem?" mechanically
falsifiable.

Example YAML:

    gates:
      - name: mvp_validation
        after_iters: 20
        require:
          probe_auroc: ">0.7"
          probe_fpr: "<0.05"
        on_fail: abort_campaign

`GateEngine.should_abort(iter_count, latest_metrics)` returns
`(should_abort, reason)`. The continuous loop calls it after each
iteration commit and aborts the campaign when any `on_fail:
abort_campaign` gate fails.

The mechanism mirrors the explicit comparison criteria in MLE-bench
(writeup §7.6 reference). Threshold strings are parsed into typed
`GateRequirement` objects so the engine has no string handling on the
hot path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator

GateOp = Literal["<", "<=", ">", ">=", "==", "!="]
OnFail = Literal["abort_campaign", "warn"]

_OP_PATTERN = re.compile(r"^\s*(<=|>=|==|!=|<|>)\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$")
_VALID_OPS: tuple[GateOp, ...] = ("<", "<=", ">", ">=", "==", "!=")


@dataclass(frozen=True)
class GateRequirement:
    """A single requirement: `<metric> <op> <threshold>`."""

    metric: str
    op: GateOp
    threshold: float

    def evaluate(self, metrics: dict[str, float]) -> bool:
        if self.metric not in metrics:
            return False
        v = float(metrics[self.metric])
        if self.op == "<":
            return v < self.threshold
        if self.op == "<=":
            return v <= self.threshold
        if self.op == ">":
            return v > self.threshold
        if self.op == ">=":
            return v >= self.threshold
        if self.op == "==":
            return v == self.threshold
        return v != self.threshold

    def describe(self, metrics: dict[str, float]) -> str:
        if self.metric not in metrics:
            return f"{self.metric} {self.op} {self.threshold} [missing]"
        actual = metrics[self.metric]
        return f"{self.metric} {self.op} {self.threshold} [actual={actual}]"


@dataclass(frozen=True)
class Gate:
    """One named gate that activates after `after_iters` iterations."""

    name: str
    after_iters: int
    require: tuple[GateRequirement, ...]
    on_fail: OnFail = "abort_campaign"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("gate name must be non-empty")
        if self.after_iters < 0:
            raise ValueError(f"after_iters must be non-negative; got {self.after_iters}")
        if not self.require:
            raise ValueError(f"gate '{self.name}' must declare at least one requirement")


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate evaluation."""

    name: str
    passed: bool
    on_fail: OnFail
    failures: tuple[str, ...]


class GateEngine:
    """Evaluates a list of gates against the latest metrics dict."""

    def __init__(self, gates: list[Gate]):
        self._gates = list(gates)

    def evaluate(self, iter_count: int, latest_metrics: dict[str, float]) -> list[GateResult]:
        out: list[GateResult] = []
        for g in self._gates:
            if iter_count < g.after_iters:
                continue
            failures = tuple(r.describe(latest_metrics) for r in g.require if not r.evaluate(latest_metrics))
            out.append(
                GateResult(
                    name=g.name,
                    passed=not failures,
                    on_fail=g.on_fail,
                    failures=failures,
                )
            )
        return out

    def should_abort(
        self, iter_count: int, latest_metrics: dict[str, float]
    ) -> tuple[bool, str]:
        """Return (True, reason) when any abort_campaign gate fails."""
        results = self.evaluate(iter_count, latest_metrics)
        for r in results:
            if not r.passed and r.on_fail == "abort_campaign":
                joined = "; ".join(r.failures)
                return True, f"gate '{r.name}' failed at iter {iter_count}: {joined}"
        return False, ""


class GateRequirementConfig(BaseModel):
    """Pydantic-friendly raw form: `{ "<metric>": "<op><threshold>" }`."""

    metric: str
    expr: str

    @field_validator("expr")
    @classmethod
    def _check_expr(cls, v: str) -> str:
        if not _OP_PATTERN.match(v):
            raise ValueError(
                f"expression {v!r} must match '<op><value>' where op in {list(_VALID_OPS)}"
            )
        return v

    def to_requirement(self) -> GateRequirement:
        m = _OP_PATTERN.match(self.expr)
        if m is None:
            raise ValueError(f"unparseable expression: {self.expr!r}")
        op_raw, threshold_raw = m.group(1), m.group(2)
        return GateRequirement(metric=self.metric, op=op_raw, threshold=float(threshold_raw))


class GateConfig(BaseModel):
    """Pydantic form of one gate as it appears in a campaign config."""

    name: str
    after_iters: int = 0
    require: dict[str, str] = Field(default_factory=dict)
    on_fail: OnFail = "abort_campaign"

    def to_gate(self) -> Gate:
        if not self.require:
            raise ValueError(f"gate '{self.name}' must declare at least one requirement")
        reqs = [
            GateRequirementConfig(metric=metric, expr=expr).to_requirement()
            for metric, expr in self.require.items()
        ]
        return Gate(
            name=self.name,
            after_iters=self.after_iters,
            require=tuple(reqs),
            on_fail=self.on_fail,
        )


def build_engine(configs: list[GateConfig]) -> GateEngine:
    """Materialize a `GateEngine` from a list of YAML-loaded `GateConfig`s."""
    return GateEngine([c.to_gate() for c in configs])
