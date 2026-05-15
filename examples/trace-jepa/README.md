# Trace-JEPA on synthetic agent traces — AutoJEPA Phase-3

> The Phase-3 thesis test. Pretrain a 25-50 M-param Trace-JEPA model on
> a synthetic LangChain/CrewAI-shaped agent-trace corpus and ask whether
> the linear probe over its embeddings can separate normal sessions
> from synthetic prompt-injection / workflow-disruption attacks at
> `probe_auroc > 0.7` at FPR=0.05. If a 30-iteration hybrid campaign
> cannot hit the bar, the JEPA-for-traces inductive-bias bet is
> reconsidered (writeup §11 Phase-3 kill criterion).

## What this example does

1. `prepare.py` (frozen): generates a synthetic agent-trace corpus
   (default 100k sessions, configurable up to 1 M via `--n-sessions`),
   shards it as WebDataset .tar files, builds a held-out probe split
   with InjecAgent-shaped instruction-hijack overlays and AgentDojo-
   shaped multi-step workflow-disruption overlays (each labelled
   `is_attack: bool`), and writes a 1k-session canary subset.
2. `train.py` (mutable): trains a Trace-JEPA model — a small
   Transformer encoder + smaller Transformer predictor with an EMA
   target encoder via `autojepa.models.ema.build_target_encoder`,
   under a `CompositeMask` of `MultiBlockInfillMask` and
   `FutureBlockMask`. The MTS-JEPA soft codebook bottleneck is the
   Phase-3 search axis (`codebook_size` ∈ {0, 64, 256, 1024} ×
   `codebook_loss_weight` ∈ {0.0, 0.1, 0.5, 1.0}; `codebook_size=0`
   is the vanilla JEPA control row).
3. `program.md` describes the task, JEPA hard rules, codebook search
   axes, the 3-baseline soft gate, and the decision gates to the
   LLM hybrid policy.
4. `config.yaml` declares `target: basilica`, the 30-iter hybrid
   campaign budget (longer than Phase-2's 20 because the codebook
   axes inflate the per-iter search space), and the recalibrated SSL
   forecaster defaults.

## How the codebook bottleneck works

Per `docs/research/mts-jepa.md` the soft codebook serves two roles:
discrete-regime-transition modelling (a learned vocabulary that maps
classes of agent-trace context onto codes) and intrinsic anti-collapse
regularisation. `train.py::SoftCodebookBottleneck` implements it as a
temperature-softmaxed similarity over `n_codes` learnable code vectors;
the entropy of the average usage distribution is added to the loss
scaled by `codebook_loss_weight`. Per ADR-001 / `TODO.md` Phase-3
scope guard, the codebook lives inside `train.py` (not in
`src/autojepa/models/`).

## Phase-3 evaluation — three external baselines (soft gate)

Trace-JEPA must beat each baseline by ≥ 0.05 AUROC on the **same probe
set** or the JEPA inductive-bias bet is reconsidered (writeup §12
escalation). The three baselines (per the `Trace-JEPA-Evaluation`
belief in Alexandria, asserted 2026-05-15):

| Baseline       | Family                              | Reference          |
|----------------|-------------------------------------|--------------------|
| LogLLaMA       | Discrete-AR autoregressive log model | arxiv:2503.14849   |
| GraphIDS / SAFE | MAE-based SSL-IDS (non-JEPA)         | arxiv:2509.16625 / arxiv:2502.07119 |
| MTS-JEPA       | JEPA on time-series                  | arxiv:2602.04643   |

The baselines run as a sibling `examples/baselines/` campaign on the
**same** probe set; the cross-campaign comparison is a post-hoc
analysis step, not in-loop. The `baseline_separation` decision gate in
`program.md` is the in-loop proxy (`probe_auroc > 0.65` after iter 20
WARN-only).

## How to run

```bash
# Validate config (no GPU, no LLM credentials needed)
./run.sh validate

# Generate the synthetic corpus (~50s on CPU for 100k sessions)
./run.sh prepare

# CPU/GPU smoke test: 30 steps, batch 16, tiny encoder
./run.sh smoke

# Local 1-iter campaign (target=command, no Basilica)
./run.sh local

# Full 30-iter campaign on Basilica (REQUIRES BASILICA_API_TOKEN + CHUTES_API_KEY)
./run.sh basilica
```

## Decision gates (writeup §7.6)

```yaml
gates:
  - name: phase3_falsifier
    after_iters: 20
    require:
      probe_auroc: ">0.7"
    on_fail: warn   # preserves run history for analysis
  - name: baseline_separation
    after_iters: 20
    require:
      probe_auroc: ">0.65"
    on_fail: warn
```

Both gates fire WARN not ABORT so the run history is preserved for
post-mortem analysis even on miss — the analysis is the input to a
possible Phase-3 architecture-reconsideration step.

## Outputs

| Path                                              | What                                   |
|---------------------------------------------------|----------------------------------------|
| `data/shards/train-{0000..NNNN}.tar`              | WebDataset training shards             |
| `data/shards/probe-{0000..NNNN}.tar`              | held-out 50/50 normal/attack probe set |
| `data/canary.json`                                | 1k normal-session canary subset        |
| `data/manifest.json`                              | counts + paths + seed                  |
| `artifacts/trace-jepa/checkpoint.json`            | continuous-loop checkpoint state       |
| `artifacts/trace-jepa/results.tsv`                | per-iteration ledger                   |
| `artifacts/trace-jepa/runs/<run_id>/...`          | per-trial run dirs                     |
| `artifacts/trace-jepa/versions/v####/...`         | kept-best model checkpoints            |
| `traces/trace-jepa/events.jsonl`                  | Chrome-trace timeline events           |

## Phase-3 closeout criteria

- [x] `prepare.py` runs end-to-end and produces `data/shards/*.tar`,
      `data/canary.json`, `data/manifest.json`
- [x] `train.py` imports cleanly with `[jepa]` extras installed
- [x] `train.py` smoke run completes end-to-end with codebook off and on
- [x] `autojepa validate config.yaml` exits 0
- [ ] Basilica 30-iter campaign run by the user with a budget of
      ~$50-150 of A100 time
- [ ] At least one trial reaches `probe_auroc > 0.7` at FPR=0.05
- [ ] Trace-JEPA beats each of LogLLaMA, GraphIDS-or-SAFE, and
      MTS-JEPA by ≥ 0.05 AUROC on the same probe set
- [ ] Decision gate `phase3_falsifier` reports `pass`

If the last bullet fails, **the JEPA-for-traces thesis is
reconsidered** — escalate per writeup §12 (architecture
reconsideration). NEVER LIE rule applies: a failed gate is a truth,
not a setback to be papered over.
