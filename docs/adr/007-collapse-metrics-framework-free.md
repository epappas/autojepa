# ADR-007: Collapse metrics are pure tensor math, framework-free

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** Phase-1 design for `autojepa/eval/collapse.py`

## Context

The writeup §6.4 hard-fail gates (latent variance < 0.3, effective rank
< 32, RankMe < 64, LiDAR < 80) need to be computed inside the trial
subprocess so that:

1. The trial can self-cancel (exit 42) before running expensive probe
   evaluation when the representation is collapsed.
2. The controller can read the latest collapse-metric values from
   `progress.jsonl` and feed them to the forecaster's
   cancellation-decision math.

Both consumers are non-Lightning paths. The trial sidecar runs in any
torch process; the controller does not import torch at all.

## Decision

`autojepa/eval/collapse.py` exposes pure-tensor functions:

```python
rankme(embeddings: torch.Tensor) -> float
effective_rank(embeddings: torch.Tensor) -> float
latent_variance(embeddings: torch.Tensor) -> float
```

Each function:
- Takes a 2D tensor of shape `(N, D)`.
- Returns a Python `float` so the value can be JSON-serialized into
  `progress.jsonl` without tensor adapters.
- Does not import or depend on Lightning, stable-pretraining, or any
  callback infrastructure.
- Validates input shape at the boundary (assert early, fail fast).

LiDAR is intentionally excluded because it requires per-class
embeddings and an LDA fit — it lives in the
`stable_pretraining`-wrapping `eval/probes.py` instead (ADR-003).

## Consequences

- **Positive:** Trial cancellation decisions are made on the fastest
  signal available (closed-form SVD over the in-memory batch).
- **Positive:** Unit tests run in <10 seconds with synthetic embeddings;
  no Lightning bootstrap needed.
- **Positive:** Functions are reusable by any external script that
  wants to audit a trained encoder.
- **Negative:** Tensor-shape validation is duplicated across functions.
  Acceptable: each check is one line and centralizing would force a
  shared private helper that obscures the math.

## How to apply

- New collapse-detection metrics that fit the closed-form pure-tensor
  shape go into `eval/collapse.py`.
- Anything that needs label streams, per-class structure, optimizer
  state, or cross-batch queues goes into `eval/probes.py` and wraps
  the matching `spt.callbacks.<X>`.
- Tests for `eval/collapse.py` use synthetic embeddings only (no real
  encoder runs) and live in `tests/eval/test_collapse.py`.
