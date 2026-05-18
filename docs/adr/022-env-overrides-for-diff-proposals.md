# ADR-022: Propagate engine env overrides through DiffProposal

- **Status:** Accepted
- **Date:** 2026-05-18
- **Deciders:** epappas
- **Source:** v29 iter=3 post-mortem; see `docs/phase-2-fix-diary.md`
  2026-05-18 "the diff actually trained, AR_MODEL_DIR didn't reach
  the container" entry.

## Context

ADR-019 made the engine inject `AR_MODEL_DIR` (the iter's versioned
model-output path) into the proposal so the trial knows where to
write `outcome.json` for ADR-015's iter-done contract. For
`ParamProposal`, the engine mutates `proposal.params` directly,
and `BasilicaTarget._deploy_and_collect` reads `AR_MODEL_DIR` out of
`params` as part of building the container env. That path works.

For `DiffProposal`, there is no `proposal.params` dict. The engine's
ADR-019 mutation `_params_attr["AR_MODEL_DIR"] = model_dir` is a
no-op (guarded by `isinstance(_params_attr, dict)`). Then
`HybridExecutor` dispatches to `DiffExecutor.execute`, which builds
its OWN params dict from scratch with just `AR_MODIFIED_SOURCE` and
`AR_MODIFIED_TARGET`. The engine's `AR_MODEL_DIR` never reaches the
container.

Effect on train.py: line 117 falls back to a default model_dir when
`AR_MODEL_DIR` is unset:

```python
ARTIFACT_DIR = Path(os.environ.get("AR_MODEL_DIR", str(Path(__file__).parent / "artifacts")))
```

Container is `/app/train.py`, so the fallback is `/app/artifacts/`.
Training writes `outcome.json` there. The basilica adapter polls the
engine-computed model_dir path (e.g.
`artifacts/ijepa-cifar10/models/v0003/outcome.json`) and finds
nothing. After `timeout_s`, the iter is closed as failed despite
training having completed successfully.

v29 iter=3 evidence (basilica pod log, previous container instance):

```
[probe] ok at step 4000; probe_auroc=0.2074
step=4000 probe_auroc=0.2074
[outcome] wrote /app/artifacts/outcome.json status=ok
final probe_auroc=0.2402 after 4000 steps
```

That is the LLM-authored CosineAnnealingLR scheduler diff running
to completion with probe_auroc=0.2402 — a real Phase-2 ratchet
signal — and the controller silently throwing it away because the
plumbing never told the container where to write the outcome file.

## Decision

Add `env_overrides: dict[str, str]` to the `Proposal` base class
(both `ParamProposal` and `DiffProposal` inherit it). The engine
populates it alongside the existing `params`-style injection. The
`DiffExecutor` merges `proposal.env_overrides` into the params dict
it passes to `target.run`.

`ParamProposal` doesn't need behavior change; its existing
`params`-side AR_MODEL_DIR injection still works. The new
`env_overrides` is a parallel channel that DiffProposal can read.

## Consequences

- **Positive:** AR_MODEL_DIR (and any future engine-set env var)
  reaches the target adapter for DiffProposal iters. v29 iter=3's
  "training succeeded but iter marked failed" failure mode is gone.
- **Positive:** A new regression test
  (`test_diff_executor_forwards_env_overrides_to_target`) locks in
  the contract: any value engine puts in `proposal.env_overrides`
  must appear in `target.last_params` after execute.
- **Positive:** Generic mechanism. Future engine env injections
  (e.g. SEED override, RUN_ID for distributed tracing) can use the
  same channel without touching DiffExecutor.
- **Negative:** Two channels for AR_MODEL_DIR injection
  (`params` for ParamProposal, `env_overrides` for DiffProposal) is
  redundant. A future refactor could collapse to a single
  `proposal.env` channel and have ParamExecutor merge similarly.
  Logged as Phase-4 cleanup.

## How to apply

- Any new env var the engine wants on every iter (regardless of
  proposal type) should be set on `proposal.env_overrides`.
- Executors that build their own params dict (DiffExecutor today,
  any future Executor subclass) must call:
  ```python
  env_overrides = getattr(proposal, "env_overrides", None)
  if isinstance(env_overrides, dict):
      for k, v in env_overrides.items():
          params[k] = str(v)
  ```
  to merge.
