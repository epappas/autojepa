"""Sanity-overfit canary protocol.

Writeup §7.4: before any campaign starts, AutoJEPA runs a 1k-sample
overfit test. If the model cannot drive `L_predict` below threshold on
a tiny dataset, the tokenizer or data pipeline is broken — kill the
campaign immediately. Cost: 1 iteration; saves catastrophic
miscalibration of multi-iteration runs.

The canary is implemented as a special-case `gates.Gate` that fires at
iteration 0 (after_iters=0) and gates `canary_loss` against a
configured ceiling. The trial subprocess opts in by emitting
`canary_loss` in its first checkpoint:

    emit_progress(step=0, step_target=N, metrics={
        "canary_loss": current_overfit_loss,
        ...
    })

This module provides the typed config and the `gates.Gate` factory;
it does NOT run training itself — the canary protocol runs inside the
campaign's `train.py` like any other iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

from autojepa.gates import Gate, GateRequirement


@dataclass(frozen=True)
class CanaryConfig:
    """Hyperparameters of the sanity-overfit canary.

    Defaults follow the writeup §7.4 sketch: 1k samples, ~200 steps,
    L_predict driven below 0.05. These are starting points; an example
    that uses a different model size or loss formulation should override.
    """

    n_samples: int = 1000
    max_steps: int = 200
    loss_metric: str = "canary_loss"
    loss_threshold: float = 0.05
    on_fail_action: str = "abort_campaign"

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive; got {self.n_samples}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {self.max_steps}")
        if self.loss_threshold <= 0:
            raise ValueError(f"loss_threshold must be positive; got {self.loss_threshold}")
        if self.on_fail_action not in ("abort_campaign", "warn"):
            raise ValueError(
                f"on_fail_action must be 'abort_campaign' or 'warn'; got {self.on_fail_action!r}"
            )


def build_canary_gate(config: CanaryConfig | None = None) -> Gate:
    """Construct the iter-0 canary gate from a config.

    The returned gate fires immediately (`after_iters=0`) and asserts
    `canary_loss < loss_threshold`. Wire it into the campaign's
    `GateEngine` alongside any campaign-specific gates.
    """
    cfg = config or CanaryConfig()
    return Gate(
        name="sanity_overfit_canary",
        after_iters=0,
        require=(
            GateRequirement(
                metric=cfg.loss_metric,
                op="<",
                threshold=cfg.loss_threshold,
            ),
        ),
        on_fail=cfg.on_fail_action,  # type: ignore[arg-type]
    )
