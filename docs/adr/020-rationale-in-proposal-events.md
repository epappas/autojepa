# ADR-020: Surface proposer rationale in proposal-event payload

- **Status:** Accepted
- **Date:** 2026-05-17
- **Deciders:** epappas
- **Source:** v24 post-mortem (see `docs/phase-2-fix-diary.md`
  2026-05-17 entry).

## Context

For most of the Phase-2 falsifier campaign (v23, v24), I incorrectly
attributed `use_qk_norm = True` no-op diffs to the LLM provider
(first Kimi, then Claude). The diffs were actually emitted by
`policy/baselines.py:GreedyLLMPolicy`, the hardcoded fallback that
`policy/llm_diff.py:LLMDiffPolicy` calls when the LLM is unreachable
or all retries are exhausted.

The fallback was invisible because the engine's proposal-event emit
only carried `episode_id`, `iter`, `params` — never the proposer's
own `rationale`. Both `ParamProposal` and `DiffProposal` carry a
rationale field; `GreedyLLMPolicy` sets it to strings like
`"improve_stability_before_fine_tuning"`, and `LLMParamPolicy` sets it
to `"llm"`, `LLMDiffPolicy` to `"llm-diff"`. These would have been
trivially greppable in events.jsonl, but they were silently dropped.

Effect: every diff iter that resulted in the GreedyLLMPolicy fallback
looked indistinguishable from a real LLM-authored diff in the trace.
The Phase-2 falsifier verdict was being computed against a
contaminated signal.

## Decision

Add `"rationale": getattr(proposal, "rationale", None)` to the
proposal-event payload in both `controller/engine.py` and
`controller/parallel_engine.py`. Use `getattr` with default `None` so
any future proposal type without a rationale field continues to emit
cleanly.

## Consequences

- **Positive:** post-mortem grep
  `jq 'select(.type=="proposal") | .rationale' events.jsonl | sort | uniq -c`
  now answers "how many real LLM diffs vs fallbacks" in one line.
- **Positive:** monitor scripts can flag fallback diffs in real time
  (`!!! FALLBACK FIRED iter=N rationale=improve_stability_*`).
- **Positive:** no behavioral change to the policy/control flow —
  pure observability.
- **Negative:** rationale strings can be long if a future proposer
  decides to dump LLM reasoning into the field. Currently all
  rationales are short (<50 chars), so no truncation needed yet.
  Revisit if a `rationale` string ever exceeds ~1KB.

## How to apply

- Anyone writing a new Proposal subclass should set `rationale` to a
  short, stable identifier (snake_case). Reserved names:
  - `"llm"`, `"llm-diff"` — real LLM authorship
  - `"random_lr_choice"`, `"random_fallback"` — seeded random
  - `"improve_stability_before_fine_tuning"`,
    `"tighten_optimization"`, `"backoff_after_failures"` —
    GreedyLLMPolicy
  - `"llm-diff-fallback-empty"` — LLMDiffPolicy when even
    GreedyLLMPolicy failed
- When adding new fallback paths, choose a snake_case identifier
  that makes the failure mode obvious in events.jsonl.
