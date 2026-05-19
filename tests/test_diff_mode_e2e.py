"""End-to-end smoke for the diff-mode pipeline.

This test wires `run_experiment` -> `HybridExecutor` -> `DiffExecutor`
-> a fake target that *actually executes* the LLM-modified source via
a Python subprocess, then asserts the engine emits an `iteration`
event with `probe_auroc` populated.

The point is NOT to train a real model. The point is to exercise the
full plumbing chain end-to-end in <60s on CPU so the next ADR-022 style
bug ("AR_MODEL_DIR didn't propagate", "diff was truncated to 200
chars", "executor asserted ParamProposal", "patch rejected the hunk
count") is caught in seconds instead of campaigns.

The v30 iter=4 CosineAnnealingLR diff (the first real Phase-2 LLM-diff
ratchet, see docs/phase-2-fix-diary.md 2026-05-18 end-of-day) is used
as the exact-shape diff input. We rewrite it to apply against a
2-statement skeleton train.py rather than the full 400-line ijepa-
cifar10/train.py — the structural shape is what matters here, not the
ML semantics.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from autojepa.config import (
    ComparabilityConfig,
    ControllerConfig,
    ObjectiveConfig,
    TelemetryConfig,
)
from autojepa.controller.diff_executor import DiffExecutor, HybridExecutor
from autojepa.controller.engine import run_experiment
from autojepa.controller.executor import MetricEvaluator, TargetExecutor
from autojepa.policy.interface import DiffProposal, Proposal
from autojepa.target.interface import RunOutcome


# Minimal trainable target whose shape mirrors examples/ijepa-cifar10/
# train.py: an optimizer construction block + a step loop. The v30 diff
# adds a scheduler between them. After applying, the patched script
# prints a HIGHER probe_auroc than baseline.
_BASELINE_TRAIN = """\
#!/usr/bin/env python3
import os
import sys

LEARNING_RATE = 1e-3
MAX_STEPS = 4

def main() -> int:
    optimizer = "AdamW"
    # placeholder used by diff-anchor; mimics the optimizer-construction
    # block in examples/ijepa-cifar10/train.py around line 241-245
    _opts = (optimizer, LEARNING_RATE)

    probe = 0.50
    for step in range(MAX_STEPS):
        probe += 0.01

    print(f"probe_auroc={probe:.4f}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""


# Same structural shape as v30 iter=4: insert a scheduler construction
# after the optimizer block; add a scheduler.step() inside the step
# loop. Hand-crafted to apply on _BASELINE_TRAIN above. The post-diff
# script raises probe_auroc by an extra +0.05 per step so the engine
# can see the diff actually changed runtime behaviour.
_V30_LIKE_DIFF = """\
--- a/train.py
+++ b/train.py
@@ -10,9 +10,14 @@ def main() -> int:
     optimizer = "AdamW"
     # placeholder used by diff-anchor; mimics the optimizer-construction
     # block in examples/ijepa-cifar10/train.py around line 241-245
     _opts = (optimizer, LEARNING_RATE)
+    scheduler = {
+        "kind": "CosineAnnealingLR",
+        "T_max": MAX_STEPS,
+        "eta_min": LEARNING_RATE * 0.01,
+    }

     probe = 0.50
     for step in range(MAX_STEPS):
-        probe += 0.01
+        probe += 0.01 + 0.05 * (scheduler["T_max"] > 0)

     print(f"probe_auroc={probe:.4f}", flush=True)
     return 0
"""


class _SubprocessDiffTarget:
    """TargetAdapter that runs AR_MODIFIED_SOURCE via Python subprocess.

    Stand-in for BasilicaTarget but local + CPU. Decodes the base64-
    encoded modified source the same way the basilica bootstrap server
    does, writes it to a temp file, runs it via the current python,
    parses `probe_auroc=...` out of stdout. Records every params dict
    it received so the test can assert AR_MODIFIED_SOURCE/AR_MODEL_DIR
    propagation.
    """

    def __init__(self) -> None:
        self.run_calls: list[dict[str, object]] = []
        self.eval_calls: list[dict[str, object]] = []

    def _run_source(self, run_dir: str, params: dict[str, object]) -> RunOutcome:
        import subprocess
        import time

        encoded = params.get("AR_MODIFIED_SOURCE")
        assert isinstance(encoded, str) and encoded, (
            "DiffExecutor must pass AR_MODIFIED_SOURCE as a base64 string"
        )
        modified = base64.b64decode(encoded.encode("ascii")).decode("utf-8")

        # AR_MODEL_DIR must reach the target (ADR-019/022). DiffExecutor
        # merges env_overrides into params; the engine populates
        # env_overrides because telemetry.model_output_dir is set.
        model_dir = params.get("AR_MODEL_DIR")
        assert isinstance(model_dir, str) and model_dir, (
            "AR_MODEL_DIR must propagate to the target adapter; see ADR-022"
        )

        Path(run_dir).mkdir(parents=True, exist_ok=True)
        script_path = Path(run_dir) / "patched_train.py"
        script_path.write_text(modified, encoding="utf-8")

        start = time.monotonic()
        cp = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=run_dir, capture_output=True, text=True, timeout=20,
        )
        elapsed = time.monotonic() - start

        metrics: dict[str, float] = {}
        for line in (cp.stdout or "").splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                try:
                    metrics[k.strip()] = float(v.strip())
                except ValueError:
                    continue

        status = "ok" if cp.returncode == 0 else "failed"
        return RunOutcome(
            status=status, metrics=metrics, stdout=cp.stdout or "",
            stderr=cp.stderr or "", elapsed_s=elapsed, run_dir=run_dir,
        )

    def run(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome:
        self.run_calls.append(dict(params))
        return self._run_source(run_dir, params)

    def eval(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome:
        self.eval_calls.append(dict(params))
        # eval phase reuses the same script — train.py wrote final
        # metrics to stdout; just re-emit them by reading the run.
        return self._run_source(run_dir, params)


class _OneShotDiffPolicy:
    """Returns the v30-shaped DiffProposal once, then stops the loop.

    The engine's run_experiment stops at max_iterations=1, so this
    policy only needs to fire once.
    """

    def __init__(self, diff: str) -> None:
        self._diff = diff
        self.calls = 0

    def propose(self, state: dict) -> Proposal:  # noqa: ARG002
        self.calls += 1
        return DiffProposal(diff=self._diff, rationale="llm-diff")


def _read_events(trace_path: Path) -> list[dict]:
    if not trace_path.exists():
        return []
    out: list[dict] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _diff_hybrid_extractor(proposal: Proposal) -> dict:
    """Mirror controller/continuous.py:_hybrid_extractor for diff path."""
    assert isinstance(proposal, DiffProposal)
    return {"diff": proposal.diff, "diff_len": len(proposal.diff), "_type": "diff"}


def _state_builder(mutable_file: str):
    def builder(history: list[dict], program: str) -> dict:
        return {
            "history": history, "program": program,
            "source": Path(mutable_file).read_text(encoding="utf-8"),
            "mutable_file": mutable_file,
        }
    return builder


def test_diff_mode_emits_probe_auroc_iteration_event(tmp_path: Path) -> None:
    """Full chain: policy -> HybridExecutor -> DiffExecutor -> target
    subprocess -> outcome -> engine -> iteration event with probe_auroc.

    Would have caught ADR-022 (AR_MODEL_DIR not in env), the
    `[:200]` truncation, the `assert isinstance(ParamProposal)` regression,
    and the patch-rejects-wrong-hunk-count bug in a single ~5s test
    instead of 13 Basilica campaigns.
    """
    # 1. Stage a real train.py in a sandbox.
    mutable = tmp_path / "train.py"
    mutable.write_text(_BASELINE_TRAIN, encoding="utf-8")

    # 2. Construct the real production executors.
    target = _SubprocessDiffTarget()
    diff_exec = DiffExecutor(target, str(mutable))
    target_exec = TargetExecutor(target)
    executor = HybridExecutor(target_exec, diff_exec)

    # 3. Real engine. trace_path must exist so we can read events.
    trace_path = tmp_path / "events.jsonl"
    artifacts = tmp_path / "artifacts"
    versions = tmp_path / "versions"
    model_output = tmp_path / "models"

    result = run_experiment(
        executor=executor,
        evaluator=MetricEvaluator(),
        policy=_OneShotDiffPolicy(_V30_LIKE_DIFF),
        objective=ObjectiveConfig(metric="probe_auroc", direction="max"),
        controller=ControllerConfig(max_iterations=1),
        telemetry=TelemetryConfig(
            trace_path=str(trace_path),
            ledger_path=str(artifacts / "results.tsv"),
            artifacts_dir=str(artifacts),
            versions_dir=str(versions),
            model_output_dir=str(model_output),
        ),
        comparability_cfg=ComparabilityConfig(strict=False),
        proposal_state_builder=_state_builder(str(mutable)),
        proposal_params_extractor=_diff_hybrid_extractor,
        enable_run_manifest=False,
        enable_versions=False,
        enable_tracker=False,
        enable_forecasting=False,
    )

    # 4. Iteration event with probe_auroc populated -> chain works.
    events = _read_events(trace_path)
    iter_events = [e for e in events if e.get("type") == "iteration"]
    assert iter_events, (
        f"engine emitted no iteration event; chain broken upstream. "
        f"All events: {[e.get('type') for e in events]}"
    )
    iter_event = iter_events[0]
    assert "probe_auroc" in iter_event.get("metrics", {}), (
        f"iteration event missing probe_auroc — the LLM-modified "
        f"train.py either never ran or never returned the metric. "
        f"Event: {iter_event}"
    )
    probe = iter_event["metrics"]["probe_auroc"]
    assert isinstance(probe, (int, float)) and probe > 0.0

    # 5. The result must reflect a real best_value (loop didn't no-op).
    assert result.iterations == 1
    assert result.best_value is not None and result.best_value > 0.0

    # 6. Target must have received AR_MODIFIED_SOURCE AND AR_MODEL_DIR.
    assert len(target.run_calls) == 1
    params = target.run_calls[0]
    assert "AR_MODIFIED_SOURCE" in params
    assert "AR_MODIFIED_TARGET" in params
    assert params["AR_MODIFIED_TARGET"] == "train.py"
    assert "AR_MODEL_DIR" in params, "ADR-022 regression: AR_MODEL_DIR lost"

    # 7. The base64-encoded source MUST contain the scheduler addition
    # (i.e., the diff was actually applied before reaching the target).
    decoded = base64.b64decode(str(params["AR_MODIFIED_SOURCE"])).decode("utf-8")
    assert "CosineAnnealingLR" in decoded, (
        "AR_MODIFIED_SOURCE doesn't contain the scheduler diff — "
        "DiffExecutor accepted the proposal but didn't apply the diff"
    )

    # 8. Diff was kept: with model_output_dir set, the proposal's
    # env_overrides should have been populated with AR_MODEL_DIR.
    # The on-disk source must have been restored to original after
    # execute (the on_keep callback wasn't passed here, so the diff
    # must NOT have been persisted to mutable file).
    restored = mutable.read_text(encoding="utf-8")
    assert restored == _BASELINE_TRAIN, (
        "DiffExecutor.finally must restore the mutable file; "
        "without on_keep callback, no persistence is expected"
    )


def test_diff_mode_proposal_event_includes_rationale(tmp_path: Path) -> None:
    """ADR-020: rationale must reach the proposal event.

    Independent assertion that the v30-shape diff doesn't lose its
    `rationale="llm-diff"` between policy and trace. If this regresses,
    the FALLBACK monitor (which keys on rationale string) goes blind.
    """
    mutable = tmp_path / "train.py"
    mutable.write_text(_BASELINE_TRAIN, encoding="utf-8")
    target = _SubprocessDiffTarget()
    diff_exec = DiffExecutor(target, str(mutable))
    executor = HybridExecutor(TargetExecutor(target), diff_exec)

    trace_path = tmp_path / "events.jsonl"
    run_experiment(
        executor=executor,
        evaluator=MetricEvaluator(),
        policy=_OneShotDiffPolicy(_V30_LIKE_DIFF),
        objective=ObjectiveConfig(metric="probe_auroc", direction="max"),
        controller=ControllerConfig(max_iterations=1),
        telemetry=TelemetryConfig(
            trace_path=str(trace_path),
            ledger_path=str(tmp_path / "results.tsv"),
            artifacts_dir=str(tmp_path / "artifacts"),
            versions_dir=str(tmp_path / "versions"),
            model_output_dir=str(tmp_path / "models"),
        ),
        comparability_cfg=ComparabilityConfig(strict=False),
        proposal_state_builder=_state_builder(str(mutable)),
        proposal_params_extractor=_diff_hybrid_extractor,
        enable_run_manifest=False,
        enable_versions=False,
        enable_tracker=False,
        enable_forecasting=False,
    )

    events = _read_events(trace_path)
    proposal_events = [e for e in events if e.get("type") == "proposal"]
    assert proposal_events, "no proposal event emitted"
    assert proposal_events[0].get("rationale") == "llm-diff"
