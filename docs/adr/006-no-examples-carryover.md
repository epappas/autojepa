# ADR-006: Do not carry over upstream `examples/`

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** Phase-0 bootstrap; upstream `examples/` totals 115 MB of model artifacts and recipes

## Context

`autoresearch-rl/examples/` contains:

- `autoresearch-like/` (98 MB — minimal example)
- `deberta-prompt-injection/` (16 MB — supervised classification)
- `parallel-cancel-showcase/` (784 KB — parallel-iter showcase)
- `security-judge/`, `basilica-grpo/`, `minimal-trainable-target/`
  (40-52 KB each)

All are RL/SFT-shaped; none use JEPA training. Several rely on cached
model weights (DeBERTa, Qwen2.5-0.5B) that bloat the repo.

The Phase-1 inheritance map (`docs/research/AutoresearchRL-Inheritance-Map.md`)
classifies the example tree as out-of-scope for AutoJEPA — Phase-2
ships `examples/ijepa-cifar10/` and Phase-3 ships `examples/trace-jepa/`,
both AutoJEPA-native.

## Decision

Phase-0 carry-over **excludes** `examples/`. AutoJEPA ships
JEPA-native examples in Phases 2-3.

## Consequences

- **Positive:** Repo size stays under 1 MB after Phase-0; no large
  binary artifacts in git history.
- **Positive:** No expectation of maintenance for examples that are
  off-mission.
- **Negative:** Inherited tests that reference `examples/<name>/prepare.py`
  paths fail (9 tests in
  `tests/test_examples_smoke.py`, `tests/test_loop_autonomy.py`,
  `tests/test_scaffold.py`). These are **not skipped** per the user's
  hard rule ("Never re-write tests or code to skip them"). They remain
  failing as a forcing function for Phase-2 to land
  `examples/ijepa-cifar10/` and re-point them.
- **Negative:** No CPU-only smoke target during Phase-0 / early Phase-1.
  Acceptable: all framework code is unit-testable without an example
  fixture.

## How to apply

- New CI checks gate on `make test` with the 9 example-dependent
  failures whitelisted until Phase-2 lands. The whitelist lives in
  `docs/phase-0-baseline.md` §"Test failure analysis" and shrinks as
  AutoJEPA examples are added.
- Phase-2 PR description must include a before/after of the failing
  test count (target: 0 of the 9 still fail after `examples/ijepa-cifar10/`
  re-point).
