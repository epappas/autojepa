# Architecture Decision Records

ADRs are short, dated, immutable records of significant architectural
decisions. Format follows
[Michael Nygard's template](https://github.com/joelparkerhenderson/architecture-decision-record/blob/main/locales/en/templates/decision-record-template-by-michael-nygard/index.md)
with light edits.

Each ADR has one of these statuses:
- **Proposed** — under discussion
- **Accepted** — decided; in effect
- **Deprecated** — replaced or no longer relevant
- **Superseded by ADR-NNN** — replaced by a later record

Never edit an Accepted ADR's substance. To revise a decision, write a
new ADR that **Supersedes** the old one and update the old one's status
line.

## Index

| ADR | Title | Status |
|---|---|---|
| [001](001-fork-not-extension.md) | Fork autoresearch-rl rather than extend it | Accepted |
| [002](002-basilica-prime-target.md) | Basilica is the prime deployment target | Accepted |
| [003](003-wrap-stable-pretraining.md) | Wrap stable-pretraining for probes/collapse, do not reimplement | Accepted |
| [004](004-probe-auroc-objective.md) | Default campaign objective is `probe_auroc`, not training loss | Accepted |
| [005](005-preserve-ar-env-prefix.md) | Preserve `AR_` env-var prefix on the trial-runtime contract | Accepted |
| [006](006-no-examples-carryover.md) | Do not carry over upstream `examples/` | Accepted |
| [007](007-collapse-metrics-framework-free.md) | Collapse metrics are pure tensor math, framework-free | Accepted |
| [008](008-forecaster-recalibration.md) | Recalibrate forecaster for SSL plateau curves | Accepted |
| [009](009-multi-seed-default-3.md) | Multi-seed scoring with default 3 seeds | Accepted |
| [010](010-jepa-extra-not-default.md) | Torch / Lightning live in `[jepa]` extra, not default deps | Accepted |
| [011](011-models-namespace-as-facade.md) | `autojepa.models` is a facade over stable-pretraining (extends ADR-003) | Accepted |
