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
