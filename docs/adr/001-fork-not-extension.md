# ADR-001: Fork autoresearch-rl rather than extend it

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** Architecture writeup §2 (Motivation: why a fork, not an extension)

## Context

`autoresearch-rl` is a working autonomous-loop framework with a clear
identity ("Autonomous ML experiment loop. An LLM proposes hyperparameters
or code changes, trains on local or cloud GPU, evaluates, keeps or
discards, and repeats"). Its examples — `deberta-prompt-injection`,
`basilica-grpo`, `minimal-trainable-target` — span supervised
classification, RL post-training, and toy targets. The framework's
calibration (forecaster shape, hybrid stall thresholds, default param
ranges, `program.md` priors) reflects this scope.

JEPA pretraining has fundamentally different failure modes
(representation collapse, EMA invariants, mask composability) that
require different defaults and different primitives.

## Options considered

1. Add JEPA support inside `autoresearch-rl` via conditional code paths
   in shared modules (collapse detection, EMA invariants, mask
   composability injected behind feature flags).
2. Add a `contrib/jepa/` or `extensions/jepa/` namespace inside
   `autoresearch-rl`.
3. Per-example boilerplate replicated across every JEPA example.
4. Clean fork named for what it does.

## Decision

Choose option 4: clean fork named **AutoJEPA**.

## Consequences

- **Positive:** Both codebases keep sharp identities. JEPA-specific
  defaults (`probe_auroc` objective, RankMe/LiDAR gates, VICReg loss,
  multi-block masking) become first-class, not feature-flagged.
- **Positive:** Calibration drift between autoresearch-rl's RL/SFT
  curves and AutoJEPA's SSL curves never collides.
- **Negative:** Maintenance tax — bug fixes upstream don't auto-apply.
  Mitigation: cherry-pick log capped at ~1h/wk. The fork does not try
  to track upstream — improvements are pulled only when materially
  relevant. See `docs/cherry-pick-log.md` (to be created).
- **Negative:** Code duplication — modules shared between the two
  projects diverge over time. Acceptable cost for the identity
  clarity gain.

## How to apply

- A future change that adds a JEPA-specific behavior to a shared module
  is a smell. Either the module belongs only in AutoJEPA or the change
  belongs only in autoresearch-rl.
- A change that says "let's also support X-domain (vision, audio,
  trace)" inside AutoJEPA is in scope as long as X is a JEPA workload.
  Anything else is a new fork.

## Rejected alternatives, in detail

**Conditional code paths.** Forcing every shared module to know about
JEPA failure modes (collapse detection, EMA invariants) erodes the
autoresearch-rl identity and creates two test matrices in one repo.

**`contrib/jepa/` namespace.** A known antipattern (cf. Django contrib,
scikit-learn external) that becomes a graveyard of half-supported
optional features.

**Per-example boilerplate.** Probe-eval, collapse-guard, and
mask-scheduler code would be duplicated across every JEPA example, with
inevitable drift between copies.
