# ADR-021: Use `patch --fuzz=5` (not `git apply`) for LLM-generated diffs

- **Status:** Accepted
- **Date:** 2026-05-17
- **Deciders:** epappas
- **Source:** v25 post-mortem; see `docs/phase-2-fix-diary.md`
  2026-05-17 "the diff path was never wired" entry.

## Context

`controller/diff_executor.py:_apply_diff_in_memory` originally used
`git apply -` to apply unified diffs from `DiffProposal.diff` to the
mutable file inside an ephemeral git repo. This worked for the
deterministic, hand-crafted diffs the unit tests used.

In production with LLM-authored diffs, every single diff was
rejected. v25 iters 5-9 all failed with:

```
git apply failed: error: corrupt patch at line N
git apply failed: error: patch fragment without header at line M
```

Reproducing locally with the actual Claude response (`/tmp/claude-test.diff`):

- `git apply /tmp/claude-test.diff` → `error: corrupt patch at line 46`
- `patch -p1 --fuzz=5 < /tmp/claude-test.diff` → `Hunk #1 succeeded
  at 146 (offset -2 lines). Hunk #2 succeeded at 184 with fuzz 4
  (offset -12 lines). Hunk #3 succeeded at 249. Hunk #4 succeeded at
  273.`

Inspection: Claude correctly identified the surrounding context (the
`encoder_name=...`, `predictor_depth=...`, etc. inside the IJEPA(...)
constructor) but the hunk header `@@ -195,10 +195,15 @@` claimed
the change started at line 195. The actual line was 184. The context
lines were right; only the line numbers in the hunk header were off.

This is the typical mode of LLM-generated diffs: the model can quote
the surrounding lines accurately but cannot reliably count exact line
positions in a 400-line file. `git apply` is strict and rejects the
patch entirely. `patch` with `--fuzz=N` searches forward/backward
from the claimed position to find a context match and applies the
edit there.

## Decision

`_apply_diff_in_memory` shells out to GNU `patch -p1 --fuzz=5
--no-backup-if-mismatch --silent`, NOT `git apply`. The file is
placed under a `a/<filename>` subdirectory so `-p1` strips the
"a/" prefix that the diff carries.

`--fuzz=5` is the inflection point: zero rejects realistic LLM
output; values >5 risk applying the diff in the wrong place if the
context line is too generic (e.g. a bare `)` line). 5 covers all
observed Claude misalignments (max observed: offset 12 lines, but
the surrounding context was unique enough).

The dependency on `git` (and the `_GIT_ENV` constants) is removed
from the apply path; the project still uses git via the on_keep
callback to persist successful diffs, but that path is separate.

## Consequences

- **Positive:** LLM-authored diffs that get hunk headers wrong-by-a-
  few-lines now apply cleanly instead of being silently rejected as
  "corrupt patch." This was the root cause of v23 / v24 / v25 all
  showing 0-of-N successful diff iters.
- **Positive:** `patch` is in every standard Linux distribution
  (`/usr/bin/patch`, GNU patch 2.7.6+); no new dependency.
- **Positive:** A regression test
  (`test_apply_diff_in_memory_tolerates_offset_hunk_header`) locks
  in this behavior. A future "let's be strict again" change must
  delete the test, which forces the deleter to read this ADR.
- **Negative:** `patch --fuzz=5` will apply a diff to the WRONG
  location if context lines are too generic (e.g. a single `}` line)
  AND that generic context appears somewhere unrelated in the file.
  This is theoretical with our prompt structure (system prompt asks
  for targeted minimal changes); flag for triage if observed.
- **Negative:** If a malicious / hallucinated diff has correct
  context but a subtly wrong addition, it will still apply. The
  AST validator (validate_diff) and contract checker
  (validate_diff_against_contract) are the defenses against that —
  they fire after apply and reject the run if the result violates
  invariants.

## How to apply

- Any new diff-apply path in the codebase should use the same
  `patch --fuzz=N` shell call, not `git apply`. The strict path is
  for known-good diffs (e.g. our own commits), not LLM output.
- If `--fuzz=5` ever becomes the wrong default (e.g., a model
  reliably produces exact line numbers), the fuzz can be tightened
  via a parameter to `_apply_diff_in_memory`. Don't tighten it
  globally without removing the regression test first.
