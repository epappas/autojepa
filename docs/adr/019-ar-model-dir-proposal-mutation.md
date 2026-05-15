# ADR-019: AR_MODEL_DIR must mutate proposal.params, not just the extractor copy

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** Live debugging of v13 Basilica re-smoke (commit b33d79a) where ADR-018's bootstrap env-inject was correctly inlining env into the container, yet `AR_MODEL_DIR` was still missing from `kubectl exec env`.

## Context

ADR-015 introduced the `outcome.json` iter-done contract. ADR-016
baked deps into a custom Docker image. ADR-018 inlined the env dict
into the bootstrap script as defense-in-depth against the Basilica
SDK silently dropping `env=env` for custom images.

After ADR-018, the bootstrap script now contains:

```python
_os.environ.update(json.loads('{"AR_PARAMS_JSON": ..., "AR_PARAM_LR": ..., ...}'))
```

Verified live (`kubectl exec env`) that AR_PARAMS_JSON, AR_PARAM_*,
HF_TOKEN are all visible inside the container. **Yet AR_MODEL_DIR
was still missing.** The bootstrap server's `_model_dir =
os.environ.get("AR_MODEL_DIR", "")` therefore returned empty,
`/model/files` returned `{"files": []}`, and outcome.json polling
found nothing — every iter still hit the 3600 s timeout fallback.

Root cause traced via three reads:

1. `controller/engine.py:290` — `params = proposal_params_extractor(proposal)`.
   For hybrid mode the extractor returns a fresh dict:
   `{**proposal.params, "_type": "param"}`. So `params` is NOT
   the same object as `proposal.params`.
2. `controller/engine.py:300` — `params["AR_MODEL_DIR"] = model_dir`.
   Mutates the LOCAL extracted dict. The proposal-event log emit
   uses `params` (sees AR_MODEL_DIR ✓). But `proposal.params` is
   unmutated.
3. `controller/engine.py:354` — `executor.execute(proposal, run_dir)`.
   The executor (`controller/executor.py:48`) reads
   `proposal.params` and forwards it to `target.run(params=proposal.params)`.
   The basilica adapter's env construction (`target/basilica.py:296`)
   reads `params` for AR_MODEL_DIR.

So AR_MODEL_DIR appears in the proposal event but never in the env
the basilica adapter assembles for the container. ADR-018's env
inline correctly inlines what's in the env dict — but the dict
itself is missing AR_MODEL_DIR because of the engine bug above.

This bug is inherited from upstream `autoresearch-rl`. Their original
`train.py` examples don't depend on `AR_MODEL_DIR` (they use
`AR_PARAMS_JSON` only), so the bug never surfaced upstream. AutoJEPA's
outcome.json contract (ADR-015) is the first consumer in either
codebase.

## Decision

The engine mutates **both** the extractor-returned dict and
`proposal.params` itself when injecting AR_MODEL_DIR. Done in both
serial `controller/engine.py` and `controller/parallel_engine.py`:

```python
if telemetry.model_output_dir:
    model_dir = str(Path(telemetry.model_output_dir) / f"v{iter_idx:04d}")
    params["AR_MODEL_DIR"] = model_dir
    _params_attr = getattr(proposal, "params", None)
    if isinstance(_params_attr, dict):
        _params_attr["AR_MODEL_DIR"] = model_dir
```

The `getattr(..., None)` + `isinstance(..., dict)` guard cleanly
skips DiffProposal (which has no `params` attribute) and any future
Proposal subtype that doesn't expose a mutable params dict.

## Consequences

- **Positive:** AR_MODEL_DIR reaches the basilica adapter's `env=env`
  build, which then makes it through ADR-018's inline injection into
  the container, which makes the bootstrap server's `_model_dir`
  populated, which makes `/model/files` return outcome.json, which
  makes the controller close the iter via the fast outcome path
  (~60s for canary-fail, ~10-15min for full pretrain) instead of the
  3600s timeout fallback.
- **Positive:** Three regression tests
  (`tests/test_engine_ar_model_dir.py`) lock in the contract:
  `_hybrid_extractor` returns a fresh dict; the mutation reaches
  `proposal.params`; DiffProposal has no `params` attr.
- **Negative:** Mutating `proposal.params` is a side effect on the
  policy's own object. If a policy caches Proposal instances and
  inspects them after `target.run` returns, it will see AR_MODEL_DIR
  in the params it didn't put there. Acceptable: AR_MODEL_DIR is
  an opaque path string; no current policy treats unknown keys as
  errors.
- **Negative:** This is a fix for an upstream `autoresearch-rl` bug.
  We carry it as an AutoJEPA patch; if upstream ever fixes the same
  bug differently (e.g. by making the extractor return the SAME
  dict reference), our defensive double-write becomes a no-op but
  is still correct. The test
  `test_hybrid_extractor_returns_fresh_dict` will fail in that
  world and we can drop the defensive line then.

## How to apply

- Any new param key that needs to appear in the trial env (not just
  the proposal log) must follow the same pattern: mutate both
  `params` (for the emit) and `proposal.params` (for the executor).
- A future refactor that introduces a typed Proposal builder taking
  `(base_params, runtime_overrides)` would let us drop this fix
  entirely. Tracked as a Phase-4 hardening item.
