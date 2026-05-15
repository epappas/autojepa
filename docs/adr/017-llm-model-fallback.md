# ADR-017: LLM model-name fallback list

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** `docs/phase-2-runtime-evidence.md` (live Chutes 503/404 sequence at v1); `src/autojepa/policy/llm_search.py::_call_chat_api_messages`; `examples/ijepa-cifar10/config.yaml`

## Context

The `policy.llm_model` field is a single string. The Phase-2 falsifier
campaigns at `badbe80` and `82c5e72` configured
`llm_model: "deepseek-ai/DeepSeek-V3-0324"`. Chutes silently renamed
that endpoint to `deepseek-ai/DeepSeek-V3-0324-TEE` between the time
the config was authored and the campaigns ran, so every LLM call
returned HTTP 404. The existing `_call_chat_api_messages`
exponential-backoff path treats 404 as a non-retriable error and
re-raises; the policy's outer `except` then fell back to seeded
random.

The seeded-random path is correct as a final safety net, but losing
the LLM means losing the entire point of the framework — the
proposals are no different from `RandomPolicy(seed=...)`. The
Phase-2 falsifier gate (`probe_auroc > 0.40` after >=1 retained
llm_diff improvement) is structurally untestable when llm_diff
proposals never even reach the model.

## Decision

Allow `policy.llm_model` to be either a string (back-compat) or a
list of strings (new, ordered fallback). On 404 the request loop
advances to the next name and retries the same call. Other 4xx,
5xx, and network errors keep their existing behaviour (retry with
backoff or propagate to the seeded-random fallback).

### Config surface

```yaml
policy:
  llm_model:
    - "deepseek-ai/DeepSeek-V3-0324-TEE"   # current, attempted first
    - "deepseek-ai/DeepSeek-V3-0324"       # legacy name, second-chance
    - "deepseek-ai/DeepSeek-V3"            # base, last resort
```

The string form is unchanged:

```yaml
policy:
  llm_model: "deepseek-ai/DeepSeek-V3-0324-TEE"
```

A comma-separated string is also accepted for shell-friendly
`--override` use:

```
--override 'policy.llm_model=foo,bar,baz'
```

### Implementation

- `PolicyConfig.llm_model` typed `str | list[str] | None` (was
  `str | None`). The model-validator additionally rejects an empty
  list when policy type requires the LLM.
- `_normalise_models(model)` coerces string / comma-string / list
  into an ordered, deduplicated, non-empty list.
- `_call_chat_api_messages` iterates the list. On HTTP 404 with a
  remaining candidate, it logs `LLM model %r returned 404; falling
  back to %r` and continues. On 404 from the last candidate (or any
  non-404 error from any candidate), it re-raises so the existing
  `LLMParamPolicy.propose` / `LLMDiffPolicy.propose` outer-except
  handles it.
- Tracing: each attempt opens its own `llm.chat_completion` span
  with `model_attempt` and `model_total` set, so a Phase-2
  retrospective can see how often the fallback fired.
- `LLMParamPolicy.__init__` and `LLMDiffPolicy.__init__` take
  `model: str | list[str]`; nothing else changes.

## Consequences

- **Positive:** A provider rename or A/B-tested endpoint deprecation
  no longer kills a campaign. The author writes the rename history
  into the list and re-runs.
- **Positive:** The existing single-string config form continues to
  work (back-compat verified by `tests/test_llm_search.py`).
- **Positive:** The retry path is observable via the per-attempt
  span — campaign retrospectives can quantify how often the fallback
  fires.
- **Negative:** A genuinely-bad model name now generates as many
  log lines as the list length. We log a `WARNING` per fallback,
  not per attempt-of-attempt, so volume stays bounded.
- **Negative:** A typoed list element silently fails over to the
  next entry. The validator only rejects an empty list, not a list
  of nonsense. Surface mistakes via the per-iter trace span and the
  `traces/.../events.jsonl` ledger; do not over-engineer the
  validator.

## How to apply

- Existing examples using a single string need no change.
- New examples should prefer the list form so a provider rename is
  a config change, not a campaign re-debug.
- The CHANGELOG entry for this ADR documents the rename history of
  Chutes deepseek endpoints (`-TEE` suffix) and points readers at
  this fallback mechanism.
