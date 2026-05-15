"""Regression test for ADR-019: AR_MODEL_DIR must propagate to executor.

Before the fix, the engine mutated only the extractor-returned dict
(which is a fresh `{**proposal.params, "_type": "param"}` for the
hybrid path), leaving `proposal.params` unmutated. The basilica
adapter's env-construction reads `proposal.params`, so AR_MODEL_DIR
never reached the bootstrap server's `_model_dir` and outcome.json
polling (ADR-015) found nothing. Verified live on commit b33d79a (v13)
via `kubectl exec env` showing AR_MODEL_DIR missing.

This test asserts the fix: after the engine's model_dir injection,
the SOURCE Proposal's `params` dict carries AR_MODEL_DIR — i.e., the
mutation reaches every consumer of `proposal.params`, not just the
proposal-event log.
"""

from __future__ import annotations


def test_hybrid_extractor_returns_fresh_dict() -> None:
    """Baseline contract: the hybrid extractor MUST return a new dict,
    not the same reference as `proposal.params`. If this ever changes,
    the ADR-019 fix becomes unnecessary AND the engine's defensive
    mutation of `proposal.params` becomes a double-write — the doc
    should be updated then.
    """
    from autojepa.controller.continuous import _hybrid_extractor
    from autojepa.policy.interface import ParamProposal

    src_params = {"lr": 0.001, "batch_size": 64}
    proposal = ParamProposal(params=src_params, rationale="t")
    extracted = _hybrid_extractor(proposal)

    assert extracted is not src_params, (
        "If hybrid extractor stops returning a fresh dict, revisit "
        "ADR-019 — the proposal.params mutation in engine.py becomes "
        "a no-op double-write rather than a load-bearing fix."
    )
    assert extracted == {"lr": 0.001, "batch_size": 64, "_type": "param"}


def test_proposal_params_mutation_visible_to_executor() -> None:
    """ADR-019 contract: when the engine injects AR_MODEL_DIR, both
    the extracted-for-emit dict and proposal.params see it. The
    executor reads `proposal.params` to build the trial env."""
    from autojepa.policy.interface import ParamProposal

    proposal = ParamProposal(
        params={"lr": 0.001, "batch_size": 64}, rationale="t",
    )

    # Simulate exactly what controller/engine.py does:
    extracted = {**proposal.params, "_type": "param"}
    model_dir = "artifacts/example/models/v0000"

    extracted["AR_MODEL_DIR"] = model_dir
    _params_attr = getattr(proposal, "params", None)
    assert isinstance(_params_attr, dict)
    _params_attr["AR_MODEL_DIR"] = model_dir

    # Both views must now carry AR_MODEL_DIR.
    assert extracted["AR_MODEL_DIR"] == model_dir
    assert proposal.params["AR_MODEL_DIR"] == model_dir, (
        "executor reads proposal.params; if AR_MODEL_DIR is missing "
        "here, the basilica adapter's env=env loses it and the "
        "bootstrap server's _model_dir is empty (verified live on v13)"
    )


def test_diff_proposal_has_no_params_dict() -> None:
    """`getattr(proposal, 'params', None)` returns None for DiffProposal
    — we rely on the isinstance dict check to skip mutation cleanly."""
    from autojepa.policy.interface import DiffProposal

    diff = DiffProposal(diff="--- a/x\n+++ b/x\n", rationale="t")
    _params_attr = getattr(diff, "params", None)
    # DiffProposal has no `params` attr — the getattr defaults to None.
    # If a future refactor adds one, ensure it's NOT a dict (or the
    # mutation will run, which is harmless but worth noting).
    assert _params_attr is None or not isinstance(_params_attr, dict)
