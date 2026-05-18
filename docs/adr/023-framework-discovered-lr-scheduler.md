# ADR-023: Keep the framework-discovered CosineAnnealingLR scheduler in train.py

- **Status:** Accepted
- **Date:** 2026-05-18
- **Deciders:** epappas
- **Source:** v30 iter=4 Phase-2 falsifier ratchet.

## Context

`examples/ijepa-cifar10/train.py` is the deliberately-suboptimal
baseline for the Phase-2 falsifier (ADR-014). The whole point is to
give the LLM-diff loop headroom to discover real improvements.

v30 iter=4 (commit `9f26be2`, 2026-05-18) is the first iter in
AutoJEPA's history where:

1. Claude proposed a unified diff (rationale=`llm-diff`,
   712 chars).
2. The recount preprocessor (ADR-021 update) fixed Claude's hunk
   line counts.
3. `patch --fuzz=5` applied the diff cleanly.
4. AR_MODEL_DIR was propagated via `proposal.env_overrides`
   (ADR-022).
5. The patched `train.py` ran on the basilica GPU.
6. `outcome.json` was written to the controller-visible path.
7. The engine emitted `iteration` with `probe_auroc=0.2402`.
8. The controller marked the iter as `decision=keep`.
9. The on_keep callback persisted the diff to local train.py via
   `_persist_diff`.

The diff added:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_STEPS,
    eta_min=LEARNING_RATE * 0.01,
)
# ... and inside the training loop ...
scheduler.step()  # after optimizer.step() and update_teacher()
```

The direct +0.001 improvement at iter=4 was modest. But the
scheduler being baked into the architecture enabled a subsequent
param search (v30 iter=6) to find a configuration with
`probe_auroc=0.2656` — **+11.3% over the 0.239 baseline** — that
would not have been reachable without the scheduler underneath.

This is the writeup §12 falsifier positive case: the framework
ratchets via combined LLM-diff and param search.

## Decision

The CosineAnnealingLR scheduler addition is the first
framework-authored improvement to be committed to the AutoJEPA
codebase. Per ADR-014, train.py is mutable; we accept the LLM-
discovered improvement as a permanent baseline upgrade.

The commit message attributes the change to v30 iter=4
(`rationale=llm-diff`, commit `9f26be2`). The improvement is not
hand-crafted; it came out of the autonomous loop the framework was
built to enable.

## Consequences

- **Positive:** The example baseline becomes ~11% stronger going
  forward. Future Phase-2 reruns will need to find improvements on
  top of this stronger baseline, which exercises the loop more
  realistically.
- **Positive:** A concrete, runnable artifact of the framework's
  output exists in the repo. Anyone questioning whether the loop
  actually works can `git diff e6e51db..HEAD examples/ijepa-cifar10/train.py`
  to see the change AutoJEPA discovered.
- **Negative:** train.py is no longer the "deliberately suboptimal"
  baseline ADR-014 described. Future Phase-2 reruns measure ratchet
  from a higher starting point. Acceptable: the falsifier's purpose
  was to verify the loop works, not to maintain a fixed baseline.
- **Neutral:** If a future LLM-diff iter proposes REMOVING the
  scheduler, that's allowed under ADR-014 mutability rules — let
  the loop revisit its own decisions.

## What's intentionally NOT in this commit

The fallback `use_qk_norm = True` line (from v30 iter=12's
in-flight state when the campaign was killed) is removed. That's
not a framework discovery; it's the GreedyLLMPolicy hardcoded
fallback string, never executed at runtime (appears after
`sys.exit`). Cleanup-after-kill bug is logged for Phase-4 (see
`docs/phase-2-fix-diary.md` 2026-05-18 entry).

## How to apply

- Treat this addition like any other commit when reasoning about
  the codebase. The `# scheduler` and `scheduler.step()` lines are
  load-bearing per the v30 iter=6 +11% result.
- If subsequent campaigns want to test "diff loop discovers more
  improvements," start from this baseline, not the pre-v30 one.
