# ADR-025: LLMDiffPolicy conversation hygiene after kept diffs

- **Status:** Accepted
- **Date:** 2026-05-19
- **Deciders:** epappas
- **Source:** v30 end-of-day diary entry (docs/phase-2-fix-diary.md
  2026-05-18) "LLMDiffPolicy correction-retry pollutes the
  conversation permanently"; Phase-4 hardening 2026-05-19.

## Context

`LLMDiffPolicy` maintains a multi-turn conversation across iterations
so the LLM accumulates context about what it's already tried. This is
the right design for exploration BUT has two failure modes when a
diff is kept:

1. **Stale baseline.** After iter=N's diff is kept, the on-disk
   mutable file is now the patched version. Conversation messages
   from iters 0..N-1 reasoned over the PRE-PATCH baseline. They are
   now misleading: "you proposed X over baseline B" doesn't apply
   when the new baseline is B+X.
2. **Anchoring on success.** Live evidence from v30: iter=4's
   CosineAnnealingLR diff was kept. Iters 9, 10, 11 all proposed
   essentially the same scheduler approach, hit validation failures,
   and fell back to `GreedyLLMPolicy` (the "use_qk_norm = True"
   no-op). The conversation context anchored Claude on its previous
   success and prevented it from exploring alternatives.

The previous behaviour: the conversation grew monotonically, never
resetting until process restart. The fallback masked the
diversification failure (and confused the v26 post-mortem until
ADR-020 added rationale visibility).

## Decision

Two complementary fixes in `policy/llm_diff.py`:

1. **Reset on kept-diff observation.** At the top of `propose()`,
   inspect `state["history"]` for the most-recent kept diff iter
   (`_last_kept_diff_iter`). If it differs from the previously-
   acknowledged kept iter (`self._last_seen_kept_diff_iter`), clear
   the conversation. The next call to the API starts fresh with the
   current (post-keep) source as the new baseline.
2. **Anti-repeat prompt section.** Add a "PREVIOUSLY PROPOSED
   APPROACHES (DO NOT propose any of these again)" block to the
   user message, listing one-line gists of recent diffs from history
   (`_extract_prior_diff_summaries`). This survives even within a
   single conversation (e.g. across the correction-retry loop on
   validation failures), so a single propose() call cannot loop on
   the same approach across retries.

The gist is the first substantive added line (after `+++` headers
and bare comments) — distinctive enough to differentiate
"CosineAnnealingLR" from "VICReg" from "EMA-decay tweak" without
spamming the full hunk text.

## Consequences

- **Positive:** v30-style "iter 9/10/11 all propose the same thing"
  is broken. The LLM either picks a different approach to satisfy
  the anti-repeat instruction, or — if it can't — the policy falls
  back to greedy AND the kept-diff reset ensures the next iter
  starts from a clean slate.
- **Positive:** Stale prompt-tokens shrink. A long-running campaign
  no longer carries 10+ baseline-irrelevant conversation pairs.
- **Negative:** Some legitimate cumulative reasoning is lost on
  reset. Mitigation: the prompt always carries the full experiment
  history (`format_history_section`) so the LLM still sees what was
  tried and what worked.
- **Negative:** Heuristic gist matching could miss semantically-
  similar but textually-different diffs (e.g. two ways to write
  the same scheduler addition). The anti-repeat is best-effort;
  the policy reset is the load-bearing guarantee.

## How to apply

- Any new LLM-driven policy whose state spans iterations should
  expose a similar "snapshot of acknowledged kept iter" mechanism
  to avoid the same baseline-staleness bug.
- Prompts that target Claude/GPT-4 class models should consider
  explicit anti-repeat sections — these models can otherwise
  fixate on prior successful approaches even when feedback indicates
  the approach has stopped working in the current iter context.
- Heuristic gists are sufficient when paired with a hard reset;
  do not invest in semantic dedup until evidence of false negatives.
