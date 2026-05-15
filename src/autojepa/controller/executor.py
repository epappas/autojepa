from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from autojepa.config import ObjectiveConfig
from autojepa.policy.interface import ParamProposal, Proposal
from autojepa.target.interface import TargetAdapter


@dataclass
class Outcome:
    status: str
    metrics: dict[str, float]
    stdout: str
    stderr: str
    elapsed_s: float
    run_dir: str
    judge_signals: dict | None = None


class Executor(Protocol):
    def execute(self, proposal: Proposal, run_dir: str) -> Outcome: ...


class Evaluator(Protocol):
    def score(self, outcome: Outcome, objective: ObjectiveConfig) -> float | None: ...


class TargetExecutor:
    """Wraps a TargetAdapter for param-based proposals.

    This is the live executor used by the continuous loop. The legacy
    SandboxExecutor (which patched diffs into a git worktree and ran
    them through sandbox/runner.py with stdout heuristics) was removed
    in batch 7 along with the rest of the legacy controller/loop.py
    path. See docs/research/AutoresearchRL-Inheritance-Map.md §2-§9.
    """

    def __init__(self, target: TargetAdapter) -> None:
        self._target = target

    def execute(self, proposal: Proposal, run_dir: str) -> Outcome:
        assert isinstance(proposal, ParamProposal)
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        try:
            train_out = self._target.run(run_dir=run_dir, params=proposal.params)
            if train_out.status != "ok":
                outcome = train_out
            else:
                outcome = self._target.eval(run_dir=run_dir, params=proposal.params)
        except Exception as exc:
            return Outcome(
                status="failed", metrics={}, stdout="",
                stderr=str(exc), elapsed_s=0.0, run_dir=run_dir,
            )
        return Outcome(
            status=outcome.status,
            metrics=outcome.metrics,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            elapsed_s=outcome.elapsed_s,
            run_dir=outcome.run_dir,
        )


class MetricEvaluator:
    """Extracts objective metric and normalizes direction.

    The live evaluator. The legacy JudgeEvaluator (which scored from
    sandbox-side judge_next_state heuristics) was removed in batch 7;
    its `judge_signals` field on Outcome is preserved as Optional for
    diff-executor wire-compatibility but is unused by the live path.
    """

    def score(self, outcome: Outcome, objective: ObjectiveConfig) -> float | None:
        if objective.metric not in outcome.metrics:
            return None
        value = float(outcome.metrics[objective.metric])
        return value if objective.direction == "min" else -value
