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
| [012](012-objective-metric-is-forecast-target.md) | `objective.metric` is the forecast target — no separate field (refines ADR-008) | Accepted |
| [013](013-forecaster-plateau-limitation.md) | SSL plateau-then-rise is a known forecaster limitation (refines ADR-008) | Accepted |
| [014](014-deliberately-suboptimal-baseline.md) | Phase-2 example baseline is deliberately suboptimal (gives LLM headroom) | Accepted |
| [015](015-outcome-detection-contract.md) | `outcome.json` iter-done contract between train.py and basilica adapter | Accepted |
| [016](016-custom-docker-image.md) | Bake JEPA deps into a custom Docker image (drops per-iter overhead) | Accepted |
| [017](017-llm-model-fallback.md) | `llm_model` accepts a list — fallback on 404 (Chutes silent renames) | Accepted |
| [018](018-bootstrap-env-inline.md) | Inline AR_* env vars into the Basilica bootstrap script (ADR-015 unblocker) | Accepted |
| [019](019-ar-model-dir-proposal-mutation.md) | AR_MODEL_DIR must mutate proposal.params, not just the extractor copy (ADR-018 unblocker) | Accepted |
| [020](020-rationale-in-proposal-events.md) | Surface proposer rationale in proposal-event payload (distinguish real LLM diffs from baseline fallbacks) | Accepted |
| [021](021-patch-not-git-apply-for-llm-diffs.md) | Use `patch --fuzz=5` (not `git apply`) for LLM-generated diffs — git apply rejects context-correct diffs with wrong line numbers | Accepted |
