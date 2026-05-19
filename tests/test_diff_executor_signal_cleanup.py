"""Regression tests for DiffExecutor SIGTERM/SIGKILL cleanup (ADR-024).

Before this fix, the `try/finally` block in `DiffExecutor.execute`
restored the mutable file only on normal completion or Python
exception. A SIGTERM mid-`target.run()` left the patched diff on
disk; subsequent iterations then read this dirty state as `source`
and stacked diffs on top of leftover modifications. Live evidence:
v30 ended with iter=12 (in-flight, fallback no-op) leftover on top
of iter=4's kept CosineAnnealingLR scheduler. Verified manually
after the kill — `git status` showed train.py dirty with both
modifications stacked. See docs/phase-2-fix-diary.md 2026-05-19.
"""
from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from autojepa.controller.diff_executor import (
    _restore_on_signal,
    recover_restore_marker,
)
from autojepa.policy.interface import DiffProposal
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


# --- _restore_on_signal: signal handler restores the file ---


def test_restore_on_signal_restores_on_sigterm(tmp_path: Path) -> None:
    """Sending SIGTERM into the with-block must restore the file via
    the installed handler, BEFORE the previous handler (which here
    just records the call) chains. The original SIGTERM disposition
    is restored on exit."""
    mutable = tmp_path / "train.py"
    mutable.write_text(ORIGINAL_SOURCE, encoding="utf-8")

    delivered = {"count": 0}

    def _prev(signum: int, frame: object) -> None:
        delivered["count"] += 1

    prev = signal.signal(signal.SIGTERM, _prev)
    try:
        with _restore_on_signal(str(mutable), ORIGINAL_SOURCE):
            # Mutate the file (as DiffExecutor.execute would).
            mutable.write_text("DIRTY: leftover diff\n", encoding="utf-8")
            assert "DIRTY" in mutable.read_text(encoding="utf-8")
            # Now self-signal: our handler restores then chains to _prev.
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.05)
    finally:
        signal.signal(signal.SIGTERM, prev)

    # File must be restored.
    restored = mutable.read_text(encoding="utf-8")
    assert restored == ORIGINAL_SOURCE, (
        f"SIGTERM handler did not restore the file. Content was: {restored!r}"
    )
    # Previous handler must have been called (chain semantics preserved).
    assert delivered["count"] >= 1, "previous SIGTERM handler was not chained"


def test_restore_on_signal_clears_marker_on_normal_exit(tmp_path: Path) -> None:
    """The restore-marker sidecar must be cleaned up on normal exit
    so we don't leave breadcrumbs that the next boot would replay."""
    mutable = tmp_path / "train.py"
    mutable.write_text(ORIGINAL_SOURCE, encoding="utf-8")

    with _restore_on_signal(str(mutable), ORIGINAL_SOURCE) as marker:
        assert marker is not None and marker.exists(), (
            "marker must be written at entry so SIGKILL recovery works"
        )
        mutable.write_text("CHANGED", encoding="utf-8")

    assert marker is not None and not marker.exists(), (
        "marker must be cleared on normal exit; otherwise next boot "
        "would falsely recover and overwrite intentional edits"
    )


def test_restore_on_signal_restores_disposition_on_exit(tmp_path: Path) -> None:
    """After the with-block exits cleanly, signal handlers must point
    back to what they were before — not lingering with our handler."""
    mutable = tmp_path / "train.py"
    mutable.write_text(ORIGINAL_SOURCE, encoding="utf-8")

    sentinel = object()

    def _custom(signum: int, frame: object) -> None:
        pass

    prev_term = signal.signal(signal.SIGTERM, _custom)
    try:
        with _restore_on_signal(str(mutable), ORIGINAL_SOURCE):
            current = signal.getsignal(signal.SIGTERM)
            assert current is not _custom, (
                "our handler must displace the previous one while active"
            )
        after = signal.getsignal(signal.SIGTERM)
        assert after is _custom, (
            "previous SIGTERM handler must be restored on context exit"
        )
        _ = sentinel  # avoid unused warning
    finally:
        signal.signal(signal.SIGTERM, prev_term)


# --- recover_restore_marker: SIGKILL breadcrumb recovery ---


def test_recover_restore_marker_restores_from_sidecar(tmp_path: Path) -> None:
    """SIGKILL is uncatchable — no signal handler runs. The sidecar
    marker is the only recovery path: next engine boot reads it and
    restores the file before any diff iteration sees the dirty state.
    """
    mutable = tmp_path / "train.py"
    mutable.write_text("DIRTY: leftover diff after SIGKILL\n", encoding="utf-8")
    marker = mutable.with_suffix(mutable.suffix + ".autojepa-restore")
    marker.write_text(ORIGINAL_SOURCE, encoding="utf-8")

    recovered = recover_restore_marker(str(mutable))
    assert recovered is True
    assert mutable.read_text(encoding="utf-8") == ORIGINAL_SOURCE
    assert not marker.exists(), "marker must be removed after recovery"


def test_recover_restore_marker_no_marker_returns_false(tmp_path: Path) -> None:
    """No marker, no recovery — must be a clean no-op."""
    mutable = tmp_path / "train.py"
    mutable.write_text(ORIGINAL_SOURCE, encoding="utf-8")
    assert recover_restore_marker(str(mutable)) is False
    assert mutable.read_text(encoding="utf-8") == ORIGINAL_SOURCE


# --- DiffExecutor end-to-end: child process kill mid-execute ---


def _execute_and_block(mutable_path: str, diff: str, ready_path: str) -> None:
    """Run DiffExecutor.execute with a target that signals "ready" then
    blocks forever, simulating a hung target.run().

    Invoked as a separate process so the parent can SIGTERM it.
    """
    from autojepa.controller.diff_executor import DiffExecutor as _DE

    class _BlockingTarget:
        def run(self, *, run_dir: str, params: dict) -> RunOutcome:  # noqa: ARG002
            Path(ready_path).write_text("ready", encoding="utf-8")
            time.sleep(60)  # parent will SIGTERM us before this returns
            return RunOutcome(
                status="ok", metrics={}, stdout="", stderr="",
                elapsed_s=60.0, run_dir=run_dir,
            )

        def eval(self, *, run_dir: str, params: dict) -> RunOutcome:  # noqa: ARG002
            return RunOutcome(
                status="ok", metrics={}, stdout="", stderr="",
                elapsed_s=0.0, run_dir=run_dir,
            )

    executor = _DE(_BlockingTarget(), mutable_path)
    proposal = DiffProposal(diff=diff, rationale="test")
    try:
        executor.execute(proposal, str(Path(mutable_path).parent / "run"))
    except SystemExit:
        pass


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="SIGTERM semantics differ on Windows; this test exercises POSIX behaviour",
)
def test_diff_executor_restores_on_sigterm_mid_run(tmp_path: Path) -> None:
    """End-to-end: kill the executor mid-`target.run()`, assert the
    mutable file ends up restored to its original content. Without
    the signal handler, the file would stay dirty (the bug we hit
    in v30 after the kill).
    """
    mutable = tmp_path / "train.py"
    mutable.write_text(ORIGINAL_SOURCE, encoding="utf-8")
    ready = tmp_path / "ready"

    ctx = multiprocessing.get_context("fork")
    proc = ctx.Process(
        target=_execute_and_block,
        args=(str(mutable), VALID_DIFF, str(ready)),
    )
    proc.start()
    try:
        # Wait for target.run() to start (file is patched at that point).
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not ready.exists():
            time.sleep(0.05)
        assert ready.exists(), "child never reached the blocking target.run()"
        # While blocked, the file MUST be dirty — DiffExecutor wrote it.
        dirty = mutable.read_text(encoding="utf-8")
        assert "0.0020" in dirty, (
            "DiffExecutor should have written the patched source before "
            "calling target.run(); content was: " + repr(dirty)
        )
        # Now SIGTERM the child.
        os.kill(proc.pid, signal.SIGTERM)
        proc.join(timeout=10.0)
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join()

    # File must be restored after the kill.
    restored = mutable.read_text(encoding="utf-8")
    assert restored == ORIGINAL_SOURCE, (
        f"DiffExecutor must restore mutable_file on SIGTERM. Content was:\n"
        f"{restored}"
    )
    # And the marker must be cleared (so next boot doesn't double-restore).
    marker = mutable.with_suffix(mutable.suffix + ".autojepa-restore")
    assert not marker.exists(), "marker must be unlinked by the signal handler"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="SIGKILL semantics differ on Windows",
)
def test_diff_executor_sidecar_survives_sigkill(tmp_path: Path) -> None:
    """SIGKILL is uncatchable — even the signal handler won't run.
    But the sidecar marker MUST survive on disk, and a subsequent
    `recover_restore_marker` call MUST restore the file. This is the
    cross-process recovery path for the "kill -9 the controller"
    scenario.
    """
    mutable = tmp_path / "train.py"
    mutable.write_text(ORIGINAL_SOURCE, encoding="utf-8")
    ready = tmp_path / "ready"

    ctx = multiprocessing.get_context("fork")
    proc = ctx.Process(
        target=_execute_and_block,
        args=(str(mutable), VALID_DIFF, str(ready)),
    )
    proc.start()
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not ready.exists():
            time.sleep(0.05)
        assert ready.exists()
        # SIGKILL the child — no handler runs, file stays dirty.
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=10.0)
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join()

    # After SIGKILL, file is dirty and the sidecar marker survives.
    assert "0.0020" in mutable.read_text(encoding="utf-8"), (
        "SIGKILL pre-condition: file should be dirty post-kill (no handler ran)"
    )
    marker = mutable.with_suffix(mutable.suffix + ".autojepa-restore")
    assert marker.exists(), (
        "sidecar marker must survive across processes so next-boot recovery works"
    )

    # Simulate next engine boot: recovery restores the original.
    assert recover_restore_marker(str(mutable)) is True
    assert mutable.read_text(encoding="utf-8") == ORIGINAL_SOURCE
    assert not marker.exists()
