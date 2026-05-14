# ADR-003: Wrap stable-pretraining for probes/collapse, do not reimplement

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** Architecture writeup §7.1; `docs/research/Stable-Pretraining-V1.md`

## Context

AutoJEPA needs four monitoring signals to feed back to the outer search
policy:

1. Linear-probe accuracy as the campaign objective scalar
   (`probe_auroc`).
2. k-NN accuracy as a cheap eval surrogate.
3. RankMe collapse-detection signal.
4. LiDAR collapse-detection signal as an independent cross-check.

`stable-pretraining-v1` (Balestriero et al., 2025, arXiv:2511.19484)
ships all four as PyTorch Lightning callbacks under
`spt.callbacks.{OnlineProbe, OnlineKNN, RankMe, LiDAR}`. The library
is MIT-licensed, pre-1.0 (v0.1.6 at fetch), and is the same codebase
cited in the JEPA / SSL collapse-metric literature
(`garrido2023rankme`, `thilak2023lidar`).

## Decision

For probe and label-aware collapse signals, AutoJEPA wraps
`stable_pretraining` callbacks via a thin adapter in
`autojepa/eval/probes.py` (forthcoming). For label-free collapse
signals (RankMe, effective rank, latent variance) AutoJEPA implements
the closed-form formulae directly in `autojepa/eval/collapse.py`
because (a) they are <50 LOC each, (b) they must be callable from the
trial sidecar without Lightning bootstrapped, and (c) the formulae are
stable across the SSL literature.

The split is:

| Signal | Source | Why |
|---|---|---|
| RankMe | AutoJEPA `eval/collapse.py` | 5-line closed-form; called from sidecar |
| Effective rank | AutoJEPA `eval/collapse.py` | Participation-ratio formula; same |
| Latent variance | AutoJEPA `eval/collapse.py` | Per-dim std; same |
| LiDAR | `spt.callbacks.LiDAR` wrapper | Requires LDA + per-class structure |
| Linear probe (`probe_auroc`) | `spt.callbacks.OnlineProbe` wrapper | Owns optimizer + label stream |
| k-NN probe | `spt.callbacks.OnlineKNN` wrapper | Owns feature queue |

## Consequences

- **Positive:** AutoJEPA does not maintain its own LDA + nearest-neighbor
  + linear-probe-training stack. Those are non-trivial.
- **Positive:** Collapse signals (writeup §6.4 hard-fail gates) can be
  computed in the trial subprocess without dragging in Lightning,
  enabling fast cancellation decisions before any probe runs.
- **Negative:** stable-pretraining is pre-1.0; callback signatures may
  shift. Mitigation: pin the version (`stable-pretraining==0.1.6`
  initially) and version-gate the wrapper module.
- **Negative:** AutoJEPA is now downstream of two upstreams
  (autoresearch-rl + stable-pretraining). The cherry-pick discipline
  expands to cover stable-pretraining release notes too.
- **Negative:** If AutoJEPA later supports a non-Lightning trainer
  (e.g., raw FSDP), the callback bridge has to be re-implemented.
  Considered acceptable — current scope is Lightning-only.

## How to apply

- Any new collapse metric that requires labels or per-class structure
  goes through the `spt.callbacks` wrapper. Anything label-free and
  closed-form lives in `autojepa.eval.collapse`.
- Wrapper modules expose Python-float scalars to the controller via the
  Lightning logger -> `RunOutcome.metrics` adapter. The controller
  never sees Lightning objects.
