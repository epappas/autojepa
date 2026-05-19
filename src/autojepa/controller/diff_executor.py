"""Diff-based executor for code modification proposals.

Validates diffs (safety + contract), applies them in-memory via a temp
git repo, and passes the modified source as a base64 env var to the
target adapter.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import signal
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Iterator, Union

from autojepa.controller.contract import ContractConfig, validate_diff_against_contract
from autojepa.controller.executor import Outcome, TargetExecutor
from autojepa.policy.interface import DiffProposal, ParamProposal, Proposal
from autojepa.sandbox.validator import validate_diff, validate_required_calls
from autojepa.target.interface import TargetAdapter

logger = logging.getLogger(__name__)

SignalHandler = Union[Callable[[int, FrameType | None], Any], int, None]

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "autojepa",
    "GIT_AUTHOR_EMAIL": "aj@local",
    "GIT_COMMITTER_NAME": "autojepa",
    "GIT_COMMITTER_EMAIL": "aj@local",
}


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$")


def _recount_hunks(diff: str) -> str:
    """Rewrite each @@ hunk header so the counts match the actual body.

    LLM-authored diffs routinely have wrong line counts in @@ headers
    even when the body content is correct (e.g. header claims +12
    lines but body has 11). GNU `patch` rejects this as "malformed
    patch" regardless of --fuzz, because fuzz handles position drift,
    not internal hunk-count mismatch. This recount pass walks each
    hunk's body, counts context/added/removed lines, and rewrites the
    header. Position numbers are preserved (let patch's --fuzz
    relocate).

    Live evidence: v26 iter=5 (Claude-authored CosineAnnealingLR diff,
    712 chars, semantically correct) had `@@ -176,6 +176,12 @@` for a
    hunk whose body actually contained 5 added + 6 context lines (i.e.
    "+11", not "+12"). patch rejected it; recount + patch accepts.
    """
    lines = diff.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip("\n")
        m = _HUNK_HEADER_RE.match(stripped)
        if not m:
            out.append(line)
            i += 1
            continue

        old_start = int(m.group(1))
        new_start = int(m.group(2))
        tail = m.group(3)

        # Find end of this hunk: next @@ header, next file header, or EOF.
        body_start = i + 1
        body_end = body_start
        while body_end < len(lines):
            bl = lines[body_end].rstrip("\n")
            if (
                bl.startswith("@@")
                or bl.startswith("--- ")
                or bl.startswith("+++ ")
                or bl.startswith("diff --git")
            ):
                break
            body_end += 1

        old_count = 0
        new_count = 0
        for k in range(body_start, body_end):
            bl = lines[k]
            if bl.startswith("\\"):
                # "\ No newline at end of file" — not a counted line.
                continue
            if bl.startswith("+"):
                new_count += 1
            elif bl.startswith("-"):
                old_count += 1
            else:
                # context line: starts with " " or is fully empty
                old_count += 1
                new_count += 1

        new_header = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{tail}\n"
        out.append(new_header)
        out.extend(lines[body_start:body_end])
        i = body_end

    return "".join(out)


def _apply_diff_in_memory(source: str, diff: str, filename: str) -> str | None:
    """Apply a unified diff to source in a temp directory, return patched content.

    Uses GNU `patch -p1 --fuzz=5` rather than `git apply` because LLM-
    generated diffs routinely get hunk line numbers off by a few lines
    while keeping the surrounding context lines accurate. git apply is
    strict and rejects the whole patch when line numbers don't match
    exactly; `patch` with fuzz tolerates offset and slight context
    drift, which is the realistic mode for LLM diff proposers. Live
    evidence: v25 iter=5,6,7,8,9 all had real Claude-authored diffs
    (rationale=llm-diff, ~1500 chars, semantically correct VICReg
    swap) that `git apply` rejected as "corrupt patch" because Claude
    miscounted hunk header line counts. The SAME diffs apply cleanly
    with `patch -p1 --fuzz=5`. See ADR-021 + docs/phase-2-fix-diary.md.

    Returns the modified source string, or None on failure.
    """
    # Preprocess: fix hunk line counts that LLMs reliably get wrong.
    # patch with --fuzz handles position drift; only recount fixes
    # internal hunk-count mismatch. See ADR-021 update 2026-05-18.
    diff = _recount_hunks(diff)

    with tempfile.TemporaryDirectory(prefix="ar-diff-") as tmpdir:
        # patch -p1 strips one path prefix, so place the file under a
        # subdir to match the diff's "a/<filename>" convention.
        src_path = Path(tmpdir) / "a" / filename
        src_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.write_text(source, encoding="utf-8")

        result = subprocess.run(
            [
                "patch",
                "-p1",
                "--fuzz=5",
                "--no-backup-if-mismatch",
                "--silent",
                filename,
            ],
            input=diff,
            text=True,
            capture_output=True,
            cwd=str(src_path.parent),
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning(
                "patch failed (rc=%d): %s",
                result.returncode,
                (result.stderr or result.stdout).strip()[:400],
            )
            return None

        return src_path.read_text(encoding="utf-8")


def _persist_diff(mutable_file: str, diff: str) -> bool:
    """Apply a diff permanently to the mutable file on disk.

    Returns True on success, False on failure.
    """
    source = Path(mutable_file).read_text(encoding="utf-8")
    filename = os.path.basename(mutable_file)
    modified = _apply_diff_in_memory(source, diff, filename)
    if modified is None:
        logger.warning("Failed to persist diff to %s", mutable_file)
        return False
    Path(mutable_file).write_text(modified, encoding="utf-8")
    logger.info("Persisted diff to %s", mutable_file)
    return True


def _rejected(reason: str, run_dir: str) -> Outcome:
    return Outcome(
        status="rejected", metrics={}, stdout="",
        stderr=reason, elapsed_s=0.0, run_dir=run_dir,
    )


# Signal-safe restore for the mutable-file write done inside execute().
# Without this, SIGTERM/SIGINT during target.run() leaves the patched
# diff on disk because the `try/finally` in execute() never reaches
# its restore branch — the engine's ShutdownHandler captures SIGTERM
# but only sets a flag; it cannot interrupt a blocked subprocess.run.
# Subsequent iters then read this dirty state as `source` and stack
# diffs on top. Live evidence: v30 ended with iter=12's fallback diff
# leftover on top of iter=4's kept scheduler. See ADR-024 and
# docs/phase-2-fix-diary.md 2026-05-19.
#
# Signal handlers must be installed from the main thread (Python
# enforces this); the executor is always called from the engine's
# main thread (DiffExecutor is not used by parallel_engine, which
# excludes diff-mode by design — see ADR-011 in continuous.py
# `_run_hybrid_mode`). We guard with threading.main_thread() so a
# misuse from a worker thread becomes a clean no-op rather than a
# crash. The restore-marker sidecar (`<mutable>.autojepa-restore`)
# is also written so a SIGKILL/OOM-kill (uncatchable) leaves a
# durable breadcrumb the next engine boot can recover from.
_RESTORE_MARKER_SUFFIX = ".autojepa-restore"


def _write_restore_marker(mutable_file: str, source: str) -> Path | None:
    marker = Path(mutable_file).with_suffix(
        Path(mutable_file).suffix + _RESTORE_MARKER_SUFFIX
    )
    try:
        marker.write_text(source, encoding="utf-8")
        return marker
    except OSError as exc:
        logger.warning("could not write restore marker: %s", exc)
        return None


def _clear_restore_marker(marker: Path | None) -> None:
    if marker is None:
        return
    try:
        marker.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("could not clear restore marker %s: %s", marker, exc)


@contextmanager
def _restore_on_signal(mutable_file: str, source: str) -> Iterator[Path | None]:
    """Restore `mutable_file` to `source` if SIGTERM/SIGINT fires inside.

    Yields the path to the on-disk restore marker (or None if the
    marker write failed). The marker survives across processes so a
    SIGKILL or container OOM-kill can be cleaned up at next boot.

    Signal handlers chain to the previously-installed handlers (the
    engine's ShutdownHandler for SIGTERM, Python default for SIGINT)
    so shutdown semantics elsewhere in the system are preserved.
    """
    marker = _write_restore_marker(mutable_file, source)

    is_main = threading.current_thread() is threading.main_thread()
    prev_term: SignalHandler = None
    prev_int: SignalHandler = None

    def _handler(signum: int, frame: FrameType | None) -> None:
        try:
            Path(mutable_file).write_text(source, encoding="utf-8")
            logger.warning(
                "DiffExecutor: signal %s restored %s before chaining",
                signal.Signals(signum).name, mutable_file,
            )
        except OSError as exc:
            logger.error("DiffExecutor: restore on signal failed: %s", exc)
        _clear_restore_marker(marker)
        prev = prev_term if signum == signal.SIGTERM else prev_int
        if callable(prev):
            prev(signum, frame)
        elif prev == signal.SIG_DFL:
            # Re-raise default behaviour for the signal.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    if is_main:
        prev_term = signal.signal(signal.SIGTERM, _handler)
        prev_int = signal.signal(signal.SIGINT, _handler)
    try:
        yield marker
    finally:
        if is_main:
            signal.signal(signal.SIGTERM, prev_term or signal.SIG_DFL)
            signal.signal(signal.SIGINT, prev_int or signal.SIG_DFL)
        _clear_restore_marker(marker)


def recover_restore_marker(mutable_file: str) -> bool:
    """Restore from a leftover sidecar marker if one exists.

    Call at engine startup; returns True if a recovery happened. The
    marker is unlinked after restore so we never double-restore.
    """
    marker = Path(mutable_file).with_suffix(
        Path(mutable_file).suffix + _RESTORE_MARKER_SUFFIX
    )
    if not marker.exists():
        return False
    try:
        original = marker.read_text(encoding="utf-8")
        Path(mutable_file).write_text(original, encoding="utf-8")
        marker.unlink(missing_ok=True)
        logger.warning(
            "DiffExecutor: recovered %s from leftover restore marker", mutable_file,
        )
        return True
    except OSError as exc:
        logger.error("DiffExecutor: marker recovery failed: %s", exc)
        return False


class DiffExecutor:
    """Validates a DiffProposal, applies it in-memory, and delegates to target."""

    def __init__(
        self,
        target: TargetAdapter,
        mutable_file: str,
        contract: ContractConfig | None = None,
        required_calls: list[str] | None = None,
    ) -> None:
        self._target = target
        self._mutable_file = mutable_file
        self._contract = contract
        self._required_calls = list(required_calls or [])
        self._filename = os.path.basename(mutable_file)

    def execute(self, proposal: Proposal, run_dir: str) -> Outcome:
        assert isinstance(proposal, DiffProposal)
        diff = proposal.diff

        if not diff.strip():
            return _rejected("empty diff", run_dir)

        validation = validate_diff(diff)
        if not validation.ok:
            return _rejected(validation.reason, run_dir)

        if self._contract:
            ok, reason = validate_diff_against_contract(diff, self._contract)
            if self._contract.strict and not ok:
                return _rejected(reason, run_dir)

        source = Path(self._mutable_file).read_text(encoding="utf-8")
        modified = _apply_diff_in_memory(source, diff, self._filename)
        if modified is None:
            return _rejected("diff apply failed", run_dir)

        if self._required_calls:
            req = validate_required_calls(source, modified, self._required_calls)
            if not req.ok:
                return _rejected(req.reason, run_dir)

        encoded = base64.b64encode(modified.encode("utf-8")).decode("ascii")
        params: dict[str, object] = {
            "AR_MODIFIED_SOURCE": encoded,
            "AR_MODIFIED_TARGET": self._filename,
        }
        # ADR-022: merge engine-set env overrides (notably AR_MODEL_DIR
        # for outcome.json discovery). Without this, the target adapter
        # runs train.py with AR_MODEL_DIR unset and the basilica
        # adapter polls a different path than train.py writes to.
        env_overrides = getattr(proposal, "env_overrides", None)
        if isinstance(env_overrides, dict):
            for k, v in env_overrides.items():
                params[k] = str(v)

        Path(run_dir).mkdir(parents=True, exist_ok=True)
        # Write modified source to disk so local CommandTarget runs it;
        # Basilica targets receive it via AR_MODIFIED_SOURCE bootstrap.
        # The signal-handler restore covers SIGTERM/SIGINT mid-run; the
        # try/finally covers normal completion + exceptions; the
        # restore-marker sidecar covers SIGKILL (recovered on next
        # boot via `recover_restore_marker`).
        with _restore_on_signal(self._mutable_file, source):
            Path(self._mutable_file).write_text(modified, encoding="utf-8")
            try:
                train_out = self._target.run(run_dir=run_dir, params=params)
                outcome = train_out
                if train_out.status == "ok":
                    outcome = self._target.eval(run_dir=run_dir, params=params)
            except Exception as exc:
                Path(self._mutable_file).write_text(source, encoding="utf-8")
                return Outcome(
                    status="failed", metrics={}, stdout="",
                    stderr=str(exc), elapsed_s=0.0, run_dir=run_dir,
                )
            finally:
                # Restore original; on_keep callback persists the diff if accepted.
                Path(self._mutable_file).write_text(source, encoding="utf-8")
        return Outcome(
            status=outcome.status,
            metrics=outcome.metrics,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            elapsed_s=outcome.elapsed_s,
            run_dir=outcome.run_dir,
        )


class HybridExecutor:
    """Dispatches ParamProposal to TargetExecutor, DiffProposal to DiffExecutor."""

    def __init__(
        self,
        target_executor: TargetExecutor,
        diff_executor: DiffExecutor,
    ) -> None:
        self._target_executor = target_executor
        self._diff_executor = diff_executor

    def execute(self, proposal: Proposal, run_dir: str) -> Outcome:
        if isinstance(proposal, DiffProposal):
            return self._diff_executor.execute(proposal, run_dir)
        assert isinstance(proposal, ParamProposal)
        return self._target_executor.execute(proposal, run_dir)
