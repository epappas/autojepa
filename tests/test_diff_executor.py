from __future__ import annotations

from autojepa.controller.contract import ContractConfig
from autojepa.controller.diff_executor import (
    DiffExecutor,
    HybridExecutor,
    _apply_diff_in_memory,
    _persist_diff,
)
from autojepa.controller.executor import TargetExecutor
from autojepa.policy.interface import DiffProposal, ParamProposal
from autojepa.target.interface import RunOutcome


ORIGINAL_SOURCE = """\
import torch
LEARNING_RATE = 0.0026
EPOCHS = 10
"""

VALID_DIFF = """\
--- a/train.py
+++ b/train.py
@@ -1,3 +1,3 @@
 import torch
-LEARNING_RATE = 0.0026
+LEARNING_RATE = 0.0020
 EPOCHS = 10
"""


class FakeTarget:
    """Minimal target adapter for testing."""

    def __init__(self, metrics: dict[str, float] | None = None) -> None:
        self._metrics = metrics or {"val_bpb": 1.23}
        self.last_params: dict[str, object] = {}

    def run(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome:
        self.last_params = params
        return RunOutcome(
            status="ok", metrics=self._metrics, stdout="val_bpb=1.23",
            stderr="", elapsed_s=1.0, run_dir=run_dir,
        )

    def eval(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome:
        return RunOutcome(
            status="ok", metrics=self._metrics, stdout="val_bpb=1.23",
            stderr="", elapsed_s=0.5, run_dir=run_dir,
        )


# --- _apply_diff_in_memory ---


def test_apply_diff_in_memory_success():
    modified = _apply_diff_in_memory(ORIGINAL_SOURCE, VALID_DIFF, "train.py")
    assert modified is not None
    assert "LEARNING_RATE = 0.0020" in modified
    assert "LEARNING_RATE = 0.0026" not in modified


def test_apply_diff_in_memory_bad_diff():
    bad_diff = """\
--- a/train.py
+++ b/train.py
@@ -100 +100 @@
-nonexistent line
+replaced
"""
    result = _apply_diff_in_memory(ORIGINAL_SOURCE, bad_diff, "train.py")
    assert result is None


def test_apply_diff_in_memory_preserves_other_lines():
    modified = _apply_diff_in_memory(ORIGINAL_SOURCE, VALID_DIFF, "train.py")
    assert modified is not None
    assert "import torch" in modified
    assert "EPOCHS = 10" in modified


def test_apply_diff_in_memory_tolerates_wrong_hunk_count():
    """LLM-generated diffs routinely have wrong line counts in @@ headers
    (e.g. header claims `+12` but body has 11 +/context lines). GNU
    patch rejects this as "malformed patch" regardless of --fuzz; the
    recount pass must rewrite headers to match body before patch sees
    the diff. Live evidence: v26 iter=5 (Claude-authored
    CosineAnnealingLR diff) had `@@ -176,6 +176,12 @@` for a hunk
    whose body actually contained 5 added + 6 context = 11 after-side
    lines, not 12.
    """
    # Same edit content as VALID_DIFF, but header lies: claims -1,3 +1,3
    # when body actually has 3 context+removed and 3 context+added.
    # (Hand-tested: VALID_DIFF's header is correct; here we make a
    # version with a wrong larger count to test recount.)
    wrong_count_diff = """\
--- a/train.py
+++ b/train.py
@@ -1,7 +1,7 @@
 import torch
-LEARNING_RATE = 0.0026
+LEARNING_RATE = 0.0020
 EPOCHS = 10
"""
    modified = _apply_diff_in_memory(ORIGINAL_SOURCE, wrong_count_diff, "train.py")
    assert modified is not None, (
        "recount should fix the wrong -1,7 +1,7 header so patch accepts the diff"
    )
    assert "LEARNING_RATE = 0.0020" in modified
    assert "LEARNING_RATE = 0.0026" not in modified


def test_apply_diff_in_memory_tolerates_offset_hunk_header():
    """LLM-generated diffs routinely get hunk line numbers wrong while
    keeping the context lines accurate. The patch backend with fuzz
    must absorb the offset rather than reject the whole diff (as
    `git apply` did in v25 — see docs/phase-2-fix-diary.md 2026-05-17
    "the diff path was never wired" entry).
    """
    # Same edit as VALID_DIFF, but the hunk header claims to start at
    # line 50 even though the file is only 3 lines long. git apply
    # would reject this as "corrupt patch"; patch --fuzz=5 finds the
    # context and applies the edit at the correct location.
    offset_diff = """\
--- a/train.py
+++ b/train.py
@@ -50,3 +50,3 @@
 import torch
-LEARNING_RATE = 0.0026
+LEARNING_RATE = 0.0020
 EPOCHS = 10
"""
    modified = _apply_diff_in_memory(ORIGINAL_SOURCE, offset_diff, "train.py")
    assert modified is not None, "patch --fuzz should tolerate wrong line numbers"
    assert "LEARNING_RATE = 0.0020" in modified
    assert "LEARNING_RATE = 0.0026" not in modified


# --- _persist_diff ---


def test_persist_diff_writes_modified_source(tmp_path):
    src = tmp_path / "train.py"
    src.write_text(ORIGINAL_SOURCE, encoding="utf-8")

    ok = _persist_diff(str(src), VALID_DIFF)
    assert ok is True

    content = src.read_text(encoding="utf-8")
    assert "LEARNING_RATE = 0.0020" in content
    assert "LEARNING_RATE = 0.0026" not in content


def test_persist_diff_returns_false_on_bad_diff(tmp_path):
    src = tmp_path / "train.py"
    src.write_text(ORIGINAL_SOURCE, encoding="utf-8")

    bad_diff = "--- a/train.py\n+++ b/train.py\n@@ -100 +100 @@\n-x\n+y\n"
    ok = _persist_diff(str(src), bad_diff)
    assert ok is False

    # Original unchanged
    content = src.read_text(encoding="utf-8")
    assert "LEARNING_RATE = 0.0026" in content


# --- DiffExecutor ---


def test_diff_executor_success(tmp_path):
    src = tmp_path / "train.py"
    src.write_text(ORIGINAL_SOURCE, encoding="utf-8")

    target = FakeTarget()
    executor = DiffExecutor(target, str(src))
    run_dir = str(tmp_path / "run-0000")

    proposal = DiffProposal(diff=VALID_DIFF, rationale="test")
    outcome = executor.execute(proposal, run_dir)

    assert outcome.status == "ok"
    assert "val_bpb" in outcome.metrics
    assert "AR_MODIFIED_SOURCE" in target.last_params
    assert "AR_MODIFIED_TARGET" in target.last_params


def test_diff_executor_rejects_empty_diff(tmp_path):
    src = tmp_path / "train.py"
    src.write_text(ORIGINAL_SOURCE, encoding="utf-8")

    target = FakeTarget()
    executor = DiffExecutor(target, str(src))
    run_dir = str(tmp_path / "run-0000")

    proposal = DiffProposal(diff="", rationale="test")
    outcome = executor.execute(proposal, run_dir)

    assert outcome.status == "rejected"
    assert "empty" in outcome.stderr


def test_diff_executor_rejects_forbidden_token(tmp_path):
    src = tmp_path / "train.py"
    src.write_text(ORIGINAL_SOURCE, encoding="utf-8")

    forbidden_diff = """\
--- a/train.py
+++ b/train.py
@@ -1 +1 @@
-import torch
+import socket
"""
    target = FakeTarget()
    executor = DiffExecutor(target, str(src))
    run_dir = str(tmp_path / "run-0000")

    proposal = DiffProposal(diff=forbidden_diff, rationale="test")
    outcome = executor.execute(proposal, run_dir)

    assert outcome.status == "rejected"
    assert "forbidden" in outcome.stderr


def test_diff_executor_contract_rejection(tmp_path):
    src = tmp_path / "train.py"
    src.write_text(ORIGINAL_SOURCE, encoding="utf-8")

    # Diff touches frozen_file
    frozen_diff = """\
--- a/model.py
+++ b/model.py
@@ -1 +1 @@
-x = 1
+x = 2
"""
    contract = ContractConfig(
        frozen_file="model.py",
        mutable_file="train.py",
        program_file="program.txt",
        strict=True,
    )
    target = FakeTarget()
    executor = DiffExecutor(target, str(src), contract)
    run_dir = str(tmp_path / "run-0000")

    proposal = DiffProposal(diff=frozen_diff, rationale="test")
    outcome = executor.execute(proposal, run_dir)

    assert outcome.status == "rejected"
    assert "frozen" in outcome.stderr


def test_diff_executor_bad_apply(tmp_path):
    src = tmp_path / "train.py"
    src.write_text(ORIGINAL_SOURCE, encoding="utf-8")

    bad_diff = """\
--- a/train.py
+++ b/train.py
@@ -100 +100 @@
-nonexistent
+replaced
"""
    target = FakeTarget()
    executor = DiffExecutor(target, str(src))
    run_dir = str(tmp_path / "run-0000")

    proposal = DiffProposal(diff=bad_diff, rationale="test")
    outcome = executor.execute(proposal, run_dir)

    assert outcome.status == "rejected"
    assert "apply failed" in outcome.stderr


def test_diff_executor_passes_base64_to_target(tmp_path):
    import base64

    src = tmp_path / "train.py"
    src.write_text(ORIGINAL_SOURCE, encoding="utf-8")

    target = FakeTarget()
    executor = DiffExecutor(target, str(src))
    run_dir = str(tmp_path / "run-0000")

    proposal = DiffProposal(diff=VALID_DIFF, rationale="test")
    executor.execute(proposal, run_dir)

    encoded = target.last_params["AR_MODIFIED_SOURCE"]
    decoded = base64.b64decode(str(encoded)).decode("utf-8")
    assert "LEARNING_RATE = 0.0020" in decoded


# --- HybridExecutor ---


def test_hybrid_executor_dispatches_diff(tmp_path):
    src = tmp_path / "train.py"
    src.write_text(ORIGINAL_SOURCE, encoding="utf-8")

    target = FakeTarget()
    target_exec = TargetExecutor(target)
    diff_exec = DiffExecutor(target, str(src))
    hybrid = HybridExecutor(target_exec, diff_exec)

    run_dir = str(tmp_path / "run-0000")
    proposal = DiffProposal(diff=VALID_DIFF, rationale="test")
    outcome = hybrid.execute(proposal, run_dir)

    assert outcome.status == "ok"
    assert "AR_MODIFIED_SOURCE" in target.last_params


def test_hybrid_executor_dispatches_param(tmp_path):
    target = FakeTarget()
    target_exec = TargetExecutor(target)
    diff_exec = DiffExecutor(target, "/tmp/nonexistent.py")
    hybrid = HybridExecutor(target_exec, diff_exec)

    run_dir = str(tmp_path / "run-0000")
    proposal = ParamProposal(params={"lr": 0.001}, rationale="test")
    outcome = hybrid.execute(proposal, run_dir)

    assert outcome.status == "ok"
    assert target.last_params == {"lr": 0.001}
