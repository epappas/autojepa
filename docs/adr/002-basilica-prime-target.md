# ADR-002: Basilica is the prime deployment target

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** User instruction; writeup §5 keeps Basilica target intact

## Context

JEPA pretraining is GPU-bound at every meaningful scale (Phase-2 CIFAR
I-JEPA, Phase-3 Trace-JEPA at 25-50M params, future video work).
`autoresearch-rl` ships three target adapters: `command` (local
subprocess), `http` (remote vLLM/sglang), and `basilica` (Basilica GPU
cloud, validated end-to-end on A100-SXM4-80GB).

The user has explicitly named Basilica as the prime deployment target
for AutoJEPA.

## Decision

Promote `basilica-sdk>=0.20` from `[basilica]` optional extra to a
**default dependency** in `pyproject.toml`. All shipped example
configs default to `target: basilica`. README and CLAUDE.md surface
Basilica as the canonical workflow. Local `command` and `http` targets
remain available (inherited unchanged from `autoresearch-rl`) for
testing and prompt-injection style examples.

## Consequences

- **Positive:** New users land on the supported, validated GPU path.
  No "configure this extra and re-sync" tax for the most common case.
- **Positive:** CI gains a clean signal — `import basilica` works
  unconditionally, no try/except dance in `target/registry.py`.
- **Negative:** Local-only users who never run a GPU campaign install
  one extra package they don't use. Acceptable cost (~5 MB, no
  transitive heavy deps).
- **Negative:** Basilica-SDK API changes break the default install.
  Mitigation: pin `basilica-sdk>=0.20` and watch the cherry-pick log
  for upstream `autoresearch-rl` SDK-bump fixes (e.g.
  `55b2570 fix(config_validate): accept BASILICA_API_TOKEN`).

## How to apply

- Every example config under `examples/<name>/config.yaml` defaults to
  `target: basilica` with sane GPU defaults (`gpu_count: 1`,
  `gpu_models: [A100, H100]`).
- Every Phase-N deliverable is gated on a successful Basilica run, not
  a local CPU smoke test. CPU smoke tests remain valid for unit-level
  verification.
- The bootstrap-server contract in `target/basilica.py`
  (`/control`, `/progress`, `/model/files`, `/model/download/<path>`)
  is load-bearing for AutoJEPA and must be preserved across upstream
  cherry-picks.
