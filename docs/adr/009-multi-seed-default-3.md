# ADR-009: Multi-seed scoring with default 3 seeds

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** Architecture writeup §7.5

## Context

JEPA results are seed-sensitive: anti-collapse defenses (EMA momentum
schedule, VICReg variance term, mask sampling) are stochastic. A
single-seed run can land in a collapsed local minimum that another
seed avoids, or vice versa.

`autoresearch-rl` proposes a `params` dict per iteration and runs one
trial per dict. This is correct for low-variance workloads
(prompt-injection classifiers, SFT) but under-resolves the JEPA noise
floor.

## Decision

The AutoJEPA policy proposes `(params, seed_set)` rather than `params`.
Each iteration runs `len(seed_set)` parallel sub-trials with the same
hyperparameters and different seeds. Iteration score is `mean ± std`
across the seed set. Default seed count: **3**.

The trial scheduler launches the K=`len(seed_set)` sub-trials through
the existing `controller/parallel_engine.py::run_experiment_parallel`
machinery; on Basilica with `gpu_count: 1`, this is K=3 deployments.

## Consequences

- **Positive:** Iteration-to-iteration comparison is on `mean(probe_auroc)`,
  which is far less noisy than a single-seed point.
- **Positive:** Std across seeds gives the LLM proposer a free
  uncertainty estimate to feed into its next proposal.
- **Negative:** Per-iteration cost triples. For Basilica $/iter, this
  is direct $/iter * 3.
- **Negative:** Scheduler complexity increases — must wait for K
  sub-trials before declaring iteration complete.
- **Negative:** Some hyperparameters (e.g., LR schedule that has
  knockout failure modes) genuinely benefit from a single high-seed
  run, not three. Mitigation: configurable `seed_count` per campaign
  config; 3 is the framework default, not a hard rule.

## How to apply

- `src/autojepa/config.py::ObjectiveConfig` adds a `seed_count: int = 3`
  field.
- The `propose_batch` Protocol method (already inherited from
  `policy/interface.py`) is the natural fit — it already returns a
  list of proposals and the parallel engine already schedules them.
- `src/autojepa/telemetry/aggregation.py::aggregate_iteration` computes
  `mean ± std` and `min/max` per metric over the seed set; the
  iteration score is `mean(<forecast_target>)`.
- The `program.md` template makes seed-count a declared parameter so
  the LLM proposer can shrink it (e.g., 3→1) when the design space
  is wide and budget is tight, and grow it (3→5) when it wants to
  pin down a marginal improvement.
