# Phase-2 Fix Diary

Chronological diary of fixes applied while wrestling Phase-2 toward the
falsifier KPI. Each entry: what was wrong, evidence, what changed, why
that's the right fix, what I'd revert it for.

This is a working diary, not a polished doc. ADRs live in `docs/adr/`
when a change is architectural.

---

## 2026-05-17 — v24 post-mortem + four fixes before v25

### Symptoms across v23 + v24

- v23 iter=3,4: `status=failed`, hung ~3875s before close, identical
  131-char "no-op" diff (`use_qk_norm = True` appended after
  `if __name__ == "__main__":`).
- v24 iter=5,6,7: `status=failed`, INSTANT (elapsed=0s),
  200-char TRUNCATED diff (hunk header + 3 context lines, no `+`
  additions, no closing context).
- v24 iter=8: `status=failed`, INSTANT, 131-char diff identical to
  v23's "Kimi no-op."

I'd been narrating these as "Kimi can't reason / Claude can't reason."
That was wrong. Below is what the evidence actually says.

### Root causes (in order of conviction)

**1. `examples/ijepa-cifar10/train.py:411` had `use_qk_norm = True`
as an UNCOMMITTED local modification.**
- Evidence: `git log -p --follow -- examples/ijepa-cifar10/train.py`
  shows no commit ever touched `use_qk_norm`. `git diff
  examples/ijepa-cifar10/train.py` showed the line as a local
  unstaged addition.
- How it got there: v23 iter=3 was the first diff-mode iter. The diff
  applier in `policy/baselines.py:_build_patch` wrote the change to
  the working tree, the iter "succeeded" in the sense of producing a
  modified file, then v23 was killed before any commit/cleanup. The
  line persisted in the working tree across v23 -> v24.
- Effect: every container in v24 started with this dead line already
  in the base64-injected train.py. Harmless functionally (no consumer
  reads the var) but it polluted every diff comparison.
- Fix: `git checkout -- examples/ijepa-cifar10/train.py`. Done.
- Revert criterion: never.

**2. `llm_timeout_s: 60` in config.yaml was too short for Claude with
the full production prompt.**
- Evidence: with a minimal prompt (`/tmp/or-test.json` test, ~260
  tokens in), OpenRouter+claude-sonnet-4-6 returns a clean 618-char
  diff in <10s, finish_reason=stop, cost $0.005. With the production
  prompt (full train.py source + JEPA_HARD_RULES + history,
  ~10-20K tokens in), Claude takes longer to generate, the urllib
  stream gets cut at 60s, leaving ~200 chars of partial output that
  ends mid-line with trailing whitespace ("lr=LEARNING_RATE,  ").
- The 200-char artifact is reproducible: v24 iter=5,6,7 all show the
  same shape (hunk header + 3 context lines, EOF).
- Fix: bump `policy.llm_timeout_s: 60 -> 300` in
  `examples/ijepa-cifar10/config.yaml`.
- Revert criterion: if a future LLM is faster or a smaller prompt
  is sent, can drop back to 120s. Don't drop to 60s without measuring.

**3. The GreedyLLMPolicy fallback masks LLM failures with a fake diff,
making it impossible to distinguish "LLM produced no-op" from "LLM
call failed and we silently fell back."**
- Evidence: `policy/llm_diff.py:208,212,267` all call
  `_greedy_fallback()` when the LLM is unreachable or all retries
  exhausted. The fallback returns a DiffProposal with rationale set
  (e.g. `"improve_stability_before_fine_tuning"`), but the engine's
  proposal-event emit (`controller/engine.py:321-332`) only includes
  `episode_id`, `iter`, `params` — NOT rationale. So in events.jsonl
  there is no way to tell a real Claude diff from a fallback diff.
- For most of v23, I was attributing the `use_qk_norm = True` no-op
  to Kimi. After the v24 trace I noticed the diff was BYTE-IDENTICAL
  to the string hardcoded in `baselines.py:107-108`. Almost certainly
  the v23 "Kimi no-op" was actually the fallback firing because Kimi
  returned an unparseable response.
- Fix: add `rationale` to the proposal-event payload in both
  `engine.py` and `parallel_engine.py`. Cheap (one line each). High
  signal — the monitor and any future post-mortem can grep
  `"rationale": "llm-diff"` vs other values to see what actually
  generated each diff.
- Not fixing yet: the fallback itself stays. It's correct behavior
  when the LLM is down — we just need to SEE it.
- Revert criterion: if the rationale string somehow leaks PII or
  becomes huge, gate behind a config flag.

**4. `intra_iteration_cancel` killed v24 iter=1 at 3841s under the
new 5400s wall, despite peak probe=0.264 beating best=0.252.**
- Evidence: iter=1 close event has elapsed_s=3841 (which is < the
  bumped timeout_s=5400, so this wasn't wall-cancel). Peak
  probe_auroc=0.264 across the progress series. status=cancelled.
  Same pattern in v21 iter=1, v23 iter=1.
- The forecaster is making bad decisions on rising-probe curves with
  a small dip near the end (typical SSL late-stage noise).
- NOT fixing now. This is a "lose some real wins" bug, not a
  "campaign can't run" bug. Phase 2 falsifier validity is more
  important than capturing every iter. Tracked for Phase 4 hardening.
- Revert criterion: doesn't apply.

### Things I considered but rejected

- **Disable the GreedyLLMPolicy fallback entirely** — tempting for
  "fail loud" but it would mean any transient LLM outage kills the
  whole campaign. Instrumentation (fix 3) is better.
- **Switch from claude-sonnet-4-6 to opus-4-7** — premature.
  Sonnet 4.6 produced a clean diff in the isolated test; the bug was
  ours, not the model's. Will revisit if a properly-timed Sonnet
  still produces garbage.
- **Bump max_tokens beyond 4096** — direct test showed 250 completion
  tokens was enough for a real diff. Not the bottleneck.

### Plan for v25

1. Apply fixes 1, 2, 3 above. (1 already applied; 2+3 next.)
2. Run policy tests.
3. Launch v25 with same config as v24 minus the bugs: explore=3,
   stall=2, model=claude-sonnet-4-6 (fallback opus-4-7),
   timeout_s=5400 (controller), llm_timeout_s=300 (per-LLM-call).
4. First success criterion: a proposal event with
   `rationale="llm-diff"` and a diff that successfully applies and
   the patched container runs to completion. That tells us the
   chain works end-to-end with a real LLM-authored change.
5. Real success criterion: a probe_auroc that beats the best
   param-mode result (v24 had 0.280 from iter=2) BY AT LEAST 0.05
   coming from a diff-mode iter. That's the Phase-2 ratchet.

### Cost so far

- Roughly ~$30-50 Basilica GPU + small LLM costs across v18-v24.
- v25 estimate: ~$25-35.

### Fixes landed (commit-ready)

- `examples/ijepa-cifar10/train.py` — reverted (uncommitted line gone).
- `examples/ijepa-cifar10/config.yaml` — `llm_timeout_s: 60 -> 300`.
- `src/autojepa/controller/engine.py` + `parallel_engine.py` — added
  `rationale` to proposal-event emit (ADR-020).
- `tests/test_ijepa_cifar10_skeleton.py` — replaced
  `hybrid_param_explore_iters >= 25` assertion (which locked in the
  v21 bug) with the meaningful invariant `explore + stall <
  max_iterations`.
- New `docs/adr/020-rationale-in-proposal-events.md` + README index.
- Tests: 151 passed / 5 skipped / 0 failed.

---

## 2026-05-17 (later) — v25 iter=1 falsifies the wall-timeout fix

### What I expected vs what happened

Earlier this session I bumped `timeout_s: 3600 -> 5400` because v21
iter=1 and v23 iter=1 (both max_steps=6000) had cancelled at ~3500s,
under the old 3600s wall by ~100s. Hypothesis: the basilica per-iter
wall was killing iters mid-upload. Bumping the wall would fix it.

v25 iter=1 cancelled at elapsed=2425s. That is **less than half**
of the 5400s wall. The wall was never the problem.

### What's actually happening

Training reaches step 6000 (max_steps) at elapsed=1180s. Then
~1245s elapse with NO progress events before the iteration close
event fires with `status=cancelled`. Same pattern in v21 iter=1
(step 6000 at 1830s, cancelled at 3465s, 1635s gap), v23 iter=1
(step 6000 at 1830s, cancelled at 3465s, 1635s gap), v24 iter=1
(step 6000 at 2029s, cancelled at 3841s, 1812s gap), v25 iter=1
(step 6000 at 1180s, cancelled at 2425s, 1245s gap).

Across 4 separate campaigns with different LLMs, different
hyperparams, and different basilica nodes, every iter where Kimi or
Claude or random proposed `max_steps=6000` got cancelled in this
post-training gap.

The peak probe in each was ABOVE the run's best. v21:
peak 0.295 vs best 0.273. v23: peak 0.295 vs best 0.250. v24: peak
0.264 vs best 0.252. v25: peak 0.264 vs best 0.254. The forecaster
is killing exactly the iters that would have been the next ratchet.

### Likely root cause (high confidence)

`intra_iteration_cancel` is firing on the post-training gap. Either:

1. The forecaster sees the last 2-3 reports (step 5500: 0.264,
   step 6000: 0.256 — a small dip near the very end of training)
   and projects a falling curve. It then concludes "this iter won't
   beat best" and cancels — even though the value at step 6000
   IS already above best.
2. OR the controller's wait loop after step 6000 (waiting for
   outcome.json + model upload) is interpreting "no new progress
   events for X seconds" as "stall" and cancelling.

Either explanation predicts the observed pattern.

### Not fixing in this cycle

This is a "lose some real wins" bug, not a "campaign cannot run"
bug. v25 best is whatever beats 0.254 from non-cancelled iters.
Tracked for a focused investigation in the next fix cycle.

Cost so far on this single bug: ~4 lost wins x ~$2-4 GPU each = ~$10-15
in unrecovered training, plus the deeper damage of falsely
suggesting "param search has plateaued" earlier than it actually has
(making diff-mode look more necessary than it might be).

### Conviction

High that the forecaster / post-train-gap interaction is the bug.
Low that I can fix it correctly without reading the forecaster code
and writing a focused test. Defer until v25 finishes — the campaign
still produces useful evidence at iter rate ~ 1 per 30-40 min, even
losing 1-in-5 to this bug.

---

## 2026-05-17 (end-of-day) — the diff path was never wired

### How I found it

v25 iters 5-9 all failed with `status=failed elapsed=0s
rationale=llm-diff diff_len=200`. The `200` chars looked like an
LLM/network truncation. Direct OpenRouter call with the EXACT same
production prompt returned 1984 chars in 28.3s; the parser
correctly extracted a 1489-char valid VICReg-swap diff (Claude was
producing the right kind of change all along). So the truncation
was NOT in OpenRouter, Claude, urllib, or `_parse_diff_response`.

Grep'd for `[:200]` in src/. Found it in
`controller/continuous.py:123,128`:

```python
def _diff_extractor(proposal):
    return {"diff": proposal.diff[:200]}

def _hybrid_extractor(proposal):
    if isinstance(proposal, DiffProposal):
        return {"diff": proposal.diff[:200], "_type": "diff"}
```

That's a debug-only display truncation. Removing it would have
exposed a deeper hole.

### The deeper hole

`controller/executor.py:46` — `TargetExecutor.execute`:

```python
def execute(self, proposal: Proposal, run_dir: str) -> Outcome:
    assert isinstance(proposal, ParamProposal)  # <-- DiffProposal fails here
    ...
```

The docstring above it: "The legacy SandboxExecutor (which patched
diffs into a git worktree and ran them through sandbox/runner.py)
was removed in batch 7." Removed without a replacement.

And `target/basilica.py:300-302` reads `AR_MODIFIED_SOURCE` /
`AR_MODIFIED_TARGET` from params — but the hybrid extractor outputs
the key `diff`, not `AR_MODIFIED_SOURCE`. There is no translation
step between "DiffProposal contains a unified diff" and "container
gets a patched train.py."

So the complete v25 diff-mode failure mode is:

1. Hybrid policy correctly switches to diff mode after stall.
2. `LLMDiffPolicy.propose` correctly calls Claude.
3. Claude correctly returns a full ~1500-char unified diff that swaps
   the loss to VICReg.
4. `_parse_diff_response` correctly extracts the diff.
5. `_hybrid_extractor` truncates the diff to 200 chars (cosmetic
   bug, doesn't matter because of step 6).
6. `TargetExecutor.execute` asserts `isinstance(proposal,
   ParamProposal)`. DiffProposal hits the assert. Exception caught.
7. `Outcome(status="failed", metrics={}, elapsed_s=0.0, ...)`.
8. Iter close event emitted with no metrics.

Every "failed-diff iter" across v23, v24, v25 was hitting step 6.
The diff path has never functioned in AutoJEPA (or in autoresearch-rl
post batch-7 removal).

### What this means for Phase 2

The Phase-2 falsifier per writeup §12 reads: "the framework either
ratchets meaningfully -> frame succeeds; or doesn't -> Phase-2
fails, falsifier triggers redesign."

Evidence today:

- **Param-mode ratchet:** verified. v25 went from 0.254 baseline ->
  0.274 best in 5 iters (+8%) with all real Claude proposals
  (rationale=llm confirmed). This is the param arm working as
  designed.
- **Diff-mode ratchet:** NOT verifiable because the executor path
  doesn't exist. This is architectural debt inherited from
  autoresearch-rl's batch-7 cleanup, not a model-intelligence
  problem.

The user said earlier: "we can NOT conclude this phase without having
a full end to end evidence the deliverables work as expected, in the
quality we aim for." So accepting "diff path missing" as the verdict
isn't acceptable. Building the missing DiffExecutor is the right
call.

### Scope of the DiffExecutor work

Concrete deliverables:

1. New `controller/executor.py:DiffExecutor` (or extend
   `TargetExecutor` to handle both Proposal subtypes).
2. Diff-apply utility: read `mutable_file` from disk, apply
   `proposal.diff` via `git apply` or `unidiff` library, return the
   patched source string. Fail loud on apply errors.
3. Wire patched source into `params["AR_MODIFIED_SOURCE"]` as
   base64-encoded UTF-8, plus `params["AR_MODIFIED_TARGET"]` as the
   target filename inside the container.
4. Remove the `[:200]` truncation in `_diff_extractor` and
   `_hybrid_extractor`. Replace with a length-aware logger that
   reports diff size in events but stores the FULL diff in a
   sidecar file (e.g. `artifacts/run-{iter}/diff.patch`) to keep
   events.jsonl readable.
5. Unit tests:
   - Apply a known diff to a known source; assert patched content.
   - Round-trip a DiffProposal through `_hybrid_extractor` and
     verify the diff survives in full.
   - Reject a malformed diff loudly (not silently).
6. ADR-021 documenting the executor split.

Time estimate: 2-4 focused hours of code + tests. Not a one-line
fix.

### v25 outcome

Campaign ended at iter=9 after 5 consecutive diff-failures triggered
`failure_rate_limit: 0.7` over window=6. Final:

- best probe_auroc=0.274 (iter=2, param mode, Claude-proposed)
- baseline=0.254 (iter=0)
- ratchet: +8% via param mode
- diff-mode: 0 iters successful (architectural gap)
- LLM calls: 10 (all rationale=llm or llm-diff, no fallbacks fired)
- Cost: ~$5-8 Basilica + ~$0.10 Claude

### Conviction on next move

Build the DiffExecutor. The user's directive "fix forward with
conviction" applies — I'm convinced this is the gap, and the param
ratchet evidence shows the LLM layer is healthy. Without the
DiffExecutor, Phase-2 cannot conclude with the quality the user
asked for.

---

## 2026-05-17 (still later) — the DiffExecutor was already there, and `git apply` was the real bug

### What I assumed vs what I found

I assumed `controller/diff_executor.py` didn't exist (the executor.py
docstring said "legacy SandboxExecutor removed in batch 7"). It DOES
exist, with `DiffExecutor`, `HybridExecutor`, `_apply_diff_in_memory`,
`_persist_diff`, AND 267 lines of unit tests in
`tests/test_diff_executor.py`. `cli.py:177` and
`controller/continuous.py:218` already wire `HybridExecutor` as the
executor when policy is hybrid.

So the diff path IS wired. The 0s-elapsed failures had a different
cause.

### The actual cause

`DiffExecutor._apply_diff_in_memory` shelled out to `git apply -`
inside an ephemeral git repo. v25 iters 5-9 all hit this path with
real Claude diffs. The deploy.log had — invisible to me until I
grepped for "git apply" — lines like:

```
git apply failed: error: corrupt patch at line 46
git apply failed: error: patch fragment without header at line 22:
@@ -239,6 +243,11 @@ def main() -> int:
```

These are `git apply`'s strictness rejections. Claude got the hunk
header line counts wrong by a few lines (the model can quote
surrounding context accurately but cannot exactly count line
positions in a 400-line file). `git apply` refuses to apply such a
patch even though the context lines are unambiguous.

Locally reproduced with the actual saved Claude response
(`/tmp/claude-test.diff`, 1489 chars, real VICReg loss swap):

- `git apply /tmp/claude-test.diff` → "corrupt patch at line 46"
- `patch -p1 --fuzz=5 < /tmp/claude-test.diff` → succeeded, all 4
  hunks applied with small offsets, output is `15665 chars (+365
  vs original)` containing the expected VICReg swap.

### The real fix

Swapped `git apply` for `patch -p1 --fuzz=5 --no-backup-if-mismatch`
in `_apply_diff_in_memory`. The dependency on git in that function
is removed (the `_GIT_ENV` constant is now unused; left in place for
now, can be cleaned up later). Added regression test
`test_apply_diff_in_memory_tolerates_offset_hunk_header` that uses
a diff with a deliberately wrong hunk-start line and asserts it
still applies. ADR-021 captures the decision.

Also removed the `[:200]` truncation in `_diff_extractor` and
`_hybrid_extractor` in `controller/continuous.py`. The full diff
now flows through the trace, which is what makes any future
diff-mode post-mortem possible. The truncation was harmless to
execution (the executor uses `proposal.diff` not the extracted
params) but actively misled multiple past investigations.

Also updated `tests/test_examples_smoke.py` to stub `OPENROUTER_API_KEY`
and `KIMI_API_KEY` (not just the legacy `CHUTES_API_KEY`), so the
validate smoke test continues to pass under the 2026-05-17 example
migration.

### Tests

- `tests/test_diff_executor.py`: 14 passed (was 13 — added the
  offset-tolerance test).
- Full repo sweep: 650 passed, 9 skipped, 0 failed.
- ruff + mypy clean on all changed files.

### What this means for v26

We now have all three pieces in place for a real Phase-2 diff-mode
test:

1. Claude proposing real, syntactically-correct, semantically-relevant
   diffs (v25 evidence + direct OpenRouter repro both confirm).
2. Diff applier (`patch --fuzz=5`) that tolerates LLM line-number
   sloppiness while preserving correctness via context matching
   (this fix + ADR-021).
3. Visible provenance (`rationale` in proposal events, ADR-020) so
   we can tell when the loop is testing the LLM vs the fallback.

If v26 still doesn't ratchet via diff mode, the issue is upstream of
the loop (Claude's choices, prompt design, or training dynamics) —
not the plumbing. That would be the FIRST honest Phase-2 falsifier
verdict we've been able to render.

### Cost so far

- v18-v24: ~$30-50 Basilica + small LLM
- v25: ~$5-8 Basilica + ~$0.10 Claude
- v26 estimate: ~$15-25 (shorter than 20-iter budget since diff iters
  now actually run training, ~30-50 min each)

---

## 2026-05-18 — patch was strict on hunk counts too; recount preprocessing

### What v26 iter=5 showed (live, ~03:30 to ~03:56)

The patch-based applier from ADR-021 made it past `git apply`'s
strict line-number rejection, but `patch` itself rejected v26 iter=5
with:

```
patch failed (rc=2): patch: **** malformed patch at line 15:
@@ -197,6 +203,7 @@ def main() -> int:
```

Inspection of Claude's actual 712-char CosineAnnealingLR diff: the
first hunk header is `@@ -176,6 +176,12 @@` but its body has 5
added lines + 6 context lines = 11 total on the after-side, not 12.
`patch` is strict about this internal count mismatch regardless of
--fuzz (fuzz handles position drift, not body-count drift).

### Two fixes converged on iter=6

1. Independently of any code change, `LLMDiffPolicy.propose` has a
   correction-retry loop: on `_parse_diff_response` or apply failure
   it appends the failure as an assistant message + asks Claude to
   correct. On iter=6, Claude regenerated the diff with `@@ -170,6
   +170,11 @@` — same content but with CORRECT line counts. patch
   accepted it. Container is currently training with the patched
   train.py (basilica pod `226b7d93-...`, 1/1 Running, age 5m39s).
2. Added `_recount_hunks` preprocessor that rewrites each hunk
   header from actual body content BEFORE passing to patch.
   `@@ -176,6 +176,12 @@` becomes `@@ -176,6 +176,11 @@` (auto-
   corrected from body). Position numbers preserved (let --fuzz
   handle drift). This is the durable fix; iter=6 would have
   succeeded the first time with the preprocessor in place.

Both fixes are validated:
- Direct unit test (`test_apply_diff_in_memory_tolerates_wrong_hunk_count`):
  hand-crafted diff with `+1,7` header on a 3-line body now applies
  cleanly. 15/15 diff_executor tests pass.
- Live v26 iter=6: substantive diff (CosineAnnealingLR + scheduler.step)
  applied, base64-injected, basilica pod actively training on it.

### What this means

- The diff-mode pipeline is end-to-end working. Real Claude diff →
  preprocessor → patch → AR_MODIFIED_SOURCE → basilica pod → trained
  on GPU. First time in AutoJEPA's history.
- v26 iter=6's outcome (probe value or canary failure) is the FIRST
  real Phase-2 falsifier signal. If probe > best (0.265) it means a
  Claude-authored code mutation IMPROVED on Claude-authored param
  search.
- If iter=6 produces a healthy probe — even if it doesn't beat best —
  Phase-2 mechanism is fully validated and we can investigate ratchet
  separately.

### NOT yet fixed

- `intra_iteration_cancel` forecaster still kills `max_steps=6000`
  iters with rising-probe-then-small-dip patterns (v26 iter=1
  cancelled at peak 0.265, would have beaten baseline 0.239).
- Conversation state pollution from failed diff attempts: iter=6's
  diff is essentially identical to iter=5's modulo line numbers,
  meaning Claude is reaching the same approach despite "failure
  feedback." Not necessarily bad — might mean Claude is convinced
  CosineAnnealingLR is right.

### Cost so far this session

- Plus a handful of OpenRouter direct-test calls for diagnosis (~$0.02).
- v26 iter=6 will be the first iter to actually train with a diff —
  cost depends on max_steps Claude chose. The diff doesn't override
  the LR/batch/steps params from the iter=2 keep that's still
  active.
