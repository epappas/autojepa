from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Proposal:
    """Base proposal type shared by all policy families."""

    rationale: str = ""
    # Engine-injected env overrides forwarded to the executor (e.g.
    # AR_MODEL_DIR). ParamProposal already smuggles these through its
    # `params` dict, but DiffProposal carries no params, so the engine
    # had no way to reach the target adapter for diff iters. v29 iter=3
    # surfaced this: training succeeded with the LLM-authored scheduler
    # diff (probe_auroc=0.2402, outcome.json written) but the controller
    # marked the iter as failed because outcome.json went to the
    # /app/artifacts/ fallback path instead of $AR_MODEL_DIR/outcome.json.
    # See ADR-022 + docs/phase-2-fix-diary.md 2026-05-18.
    env_overrides: dict[str, str] = field(default_factory=dict)


@dataclass
class ParamProposal(Proposal):
    """Proposal carrying hyperparameter overrides (continuous loop)."""

    params: dict[str, object] = field(default_factory=dict)


@dataclass
class DiffProposal(Proposal):
    """Proposal carrying a unified diff string (legacy loop)."""

    diff: str = ""


class Policy(Protocol):
    """Unified policy protocol for both param and diff proposals."""

    def propose(self, state: dict) -> Proposal: ...


def propose_batch(policy: Policy, state: dict, k: int) -> list[Proposal]:
    """Default batch shim: ask the policy k times.

    Policies that can do better (LLMParamPolicy with one prompt asking for
    a diverse k-array; RandomPolicy with seeded-random k draws; GridPolicy
    advancing through k consecutive cells) should override by exposing
    their own propose_batch(state, k) method, which this helper will call
    in preference to the loop.
    """
    if k <= 0:
        return []
    native = getattr(policy, "propose_batch", None)
    if callable(native):
        # Avoid infinite recursion if the impl is itself this helper.
        if getattr(native, "__module__", None) != __name__:
            return list(native(state, k))
    return [policy.propose(state) for _ in range(k)]


@runtime_checkable
class Learnable(Protocol):
    """Policy that can receive reward feedback after each proposal."""

    def record_reward(self, reward: float) -> None: ...
