# ADR-004: Default campaign objective is `probe_auroc`, not training loss

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** Architecture writeup §6.1; I-JEPA §5 (training loss is not a useful checkpoint-selection signal)

## Context

`autoresearch-rl` defaults `objective.metric: val_bpb` (or `loss`,
depending on example). For supervised fine-tuning and RL post-training,
training loss is monotonic with downstream quality and is therefore a
valid campaign objective.

JEPA training loss is **not** monotonic with downstream quality. The
loss is the L2 distance between predicted embeddings and EMA-target
embeddings; it can decrease while representations collapse to a
constant. Using it as the campaign objective rewards collapse.

Every JEPA paper (I-JEPA, V-JEPA 2, C-JEPA, CNN-JEPA, A-JEPA) reports
results as **probe-based downstream evaluation** scores, not as
training loss.

## Decision

Default `objective.metric: probe_auroc` in AutoJEPA campaign configs.
The metric value is the linear-probe accuracy (binary or multiclass
AUROC depending on the downstream task) computed by
`spt.callbacks.OnlineProbe` and emitted via `emit_progress(...,
metrics={"probe_auroc": float})` from `train.py` at checkpoint
intervals.

The training-loss metric (`L_predict` or `loss`) is still emitted for
diagnostics but does not drive keep/discard or forecaster decisions.

## Consequences

- **Positive:** Eliminates the "loss-collapsing-while-probe-flat"
  failure mode. Trials with collapsed representations score zero on
  probe and are correctly discarded.
- **Positive:** Hybrid policy's keep/discard signal aligns with the
  metric reported in JEPA papers, enabling apples-to-apples comparison.
- **Negative:** Probe evaluation has its own cost (run a fixed-feature
  classifier on a held-out set). Mitigation: probe runs at checkpoint
  intervals, not per step; cost is amortized over many training steps.
- **Negative:** Probe accuracy is noisier than training loss at small
  step counts. Forecaster recalibration (ADR-008) addresses this with
  larger `min_steps` and SSL plateau-aware extrapolation.

## How to apply

- Every shipped `examples/<name>/config.yaml` declares
  `objective.metric: probe_auroc, direction: max`.
- Every shipped `examples/<name>/train.py` calls
  `emit_progress(step, step_target, metrics={"probe_auroc": float, ...})`
  at checkpoint intervals. The `emit_progress` required-call list in
  `program.md` enforces this — the AST validator rejects diffs that
  remove it.
- Trials that fail collapse gates (RankMe<64, LiDAR<80, latent
  variance<0.3, effective rank<32) emit `probe_auroc=0.0` rather than
  attempting to run a probe on a degenerate representation.
