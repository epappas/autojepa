# ADR-024: DiffExecutor must restore mutable file on SIGTERM/SIGKILL

- **Status:** Accepted
- **Date:** 2026-05-19
- **Deciders:** epappas
- **Source:** v30 end-of-day diary entry (docs/phase-2-fix-diary.md
  2026-05-18); reproduced and fixed in Phase-4 hardening session
  2026-05-19.

## Context

`DiffExecutor.execute` mutates the on-disk mutable file (e.g.
`examples/ijepa-cifar10/train.py`) before delegating to
`target.run`/`target.eval`, and relies on a `try/finally` block to
restore the original content. This works for normal completion and
for Python-level exceptions, but fails for two real cases that hit
us live:

1. **SIGTERM during a hung target.run().** The engine's
   `ShutdownHandler` captures SIGTERM and only sets a flag; it does
   NOT interrupt a blocked `subprocess.run`. If the user sends a
   second SIGTERM (or SIGKILL) to force the issue, the `finally`
   block never executes — the patched file is left on disk.
2. **SIGKILL / OOM-kill / kernel kill.** Uncatchable. No handler
   runs at all.

Live evidence: v30 ended with `examples/ijepa-cifar10/train.py`
containing the iter=12 (in-flight fallback) `use_qk_norm = True`
line appended after iter=4's kept CosineAnnealingLR scheduler.
Two distinct diffs stacked because the kill happened mid-execute
and `finally` never ran. The next campaign would have read this
dirty state as its baseline and built diffs on top of garbage.

## Decision

Two complementary defences in `controller/diff_executor.py`:

1. **`_restore_on_signal` context manager** wraps the
   write-then-run-then-restore block. It installs a SIGTERM/SIGINT
   handler that:
   - Restores the file contents to the original source.
   - Clears the sidecar restore marker.
   - Chains to the previously-installed handler so the engine's
     ShutdownHandler still flips its shutdown flag.

   The handler is only installed when running on the main thread
   (Python enforces this for `signal.signal`); a guard ensures
   misuse from a worker thread degrades to a no-op rather than
   crashing.

2. **Restore marker sidecar** `<mutable_file>.autojepa-restore`
   holds the original source content for the duration of the
   `_restore_on_signal` window. If SIGKILL terminates the process
   (no handler runs at all), this file survives on disk. The next
   engine boot calls `recover_restore_marker(mutable_file)` from
   `_run_diff_mode` / `_run_hybrid_mode` in `controller/continuous.py`
   to restore from the marker before any policy reads the source.

The marker is removed on both normal exit and signal-handler exit
to avoid double-restore on a clean run.

## Consequences

- **Positive:** v30's "leftover dirty train.py" scenario is now
  recoverable: SIGTERM is handled in-process, SIGKILL is recovered
  on next-boot.
- **Positive:** Conversation-state-pollution from a dirty baseline
  (ADR-025's primary symptom) no longer cascades from a kill event.
- **Positive:** Idempotent — applying the fix is harmless on
  already-clean runs; the marker write/clear cycle is cheap.
- **Negative:** Adds two paths the engine must consider on startup
  (the marker file). Documented in `_run_diff_mode` and the
  ADR-024 inline comment.
- **Negative:** Signal-handler chaining is fragile — if a future
  contributor installs their own SIGTERM handler INSIDE the
  execute window, our chain may not reach it. Mitigation:
  comment in `_restore_on_signal` explaining the chaining
  contract; regression tests in `tests/test_diff_executor_signal_cleanup.py`
  assert chain semantics directly.

## How to apply

- Any future executor that mutates files on disk before delegating
  to a subprocess MUST use `_restore_on_signal` (or an equivalent
  guard) for the duration of the mutation.
- Any controller-level startup path that consumes mutable files
  MUST call `recover_restore_marker` before reading them.
- Don't install signal handlers from worker threads inside
  DiffExecutor — `_restore_on_signal` will silently no-op the
  signal half but still write the sidecar marker.
