# Trace-JEPA on synthetic agent traces — Phase-3 thesis test

## Task

Pretrain a 25-50M-parameter Trace-JEPA model on a synthetic
LangChain/CrewAI-shaped agent-trace corpus (100k - 1M sessions,
WebDataset-sharded by `prepare.py`) and evaluate on a held-out probe
set whose 50/50 attack overlay uses InjecAgent-shaped instruction
hijacks and AgentDojo-shaped multi-step workflow disruptions. Pass
criterion: `probe_auroc > 0.7` at FPR=0.05 after 20 iterations of a
30-iteration hybrid campaign. Failure to reach the bar is the
Phase-3 falsifier — JEPA-for-traces is reconsidered (writeup §11
Phase-3 kill criterion).

The Trace-JEPA model must additionally beat each of the following
three external baselines by ≥ 0.05 AUROC on the **same probe set**
(soft gate from `TODO.md` Phase-3 Trace-JEPA-Evaluation belief):

| Baseline       | Family                              | Reference |
|----------------|-------------------------------------|-----------|
| LogLLaMA       | Discrete-AR autoregressive log model | arxiv:2503.14849 |
| GraphIDS / SAFE | MAE-based SSL-IDS (non-JEPA)         | arxiv:2509.16625 / arxiv:2502.07119 |
| MTS-JEPA       | JEPA on time-series                  | arxiv:2602.04643 |

If Trace-JEPA cannot separate from any baseline by 0.05 AUROC, the
JEPA inductive-bias bet for agent-trace runtime security is not
earning its keep and the architecture is reconsidered (writeup §12
escalation).

## Objective

Maximise `probe_auroc` — the held-out linear-probe AUROC computed
inside `train.py::_eval_probe_and_collapse` and emitted via
`emit_progress(metrics={"probe_auroc": ...})`. Per ADR-004 this is
the only valid JEPA campaign objective; the JEPA training loss
itself collapses on bad runs and is reported only for diagnostics.

## Mutable file

`train.py` — the Trace-JEPA training script. The hybrid policy
proposes:

- Hyperparameter overrides (param mode, first 25 iterations per ADR
  default).
- Code diffs (diff mode, when param search stalls).

## Frozen file

`prepare.py` — synthetic-trace generator + InjecAgent / AgentDojo
overlay protocol + canary subset. **Must not be modified.** It owns
"what is correct" for the corpus.

## Constraints

- Single-GPU sufficient (A100, H100, L40S, RTX-4090, RTX-A6000).
- Each iteration should complete within `target.timeout_s`. Default
  budget is 4000 steps × ~150 ms / step ≈ 10 min on A100.
- Encoder model: a small Transformer (depth=8, dim=384, heads=6
  default → ~12M params). Diffs may scale up to hit the 25-50M
  writeup target; diffs may NOT scale the predictor above the
  encoder.
- The predictor (Ψ) MUST NOT be deeper or wider than the context
  encoder (Φc). `train.py::main` raises `ValueError` if violated;
  the AST validator additionally rejects diffs setting
  `predictor_depth > encoder_depth` or
  `predictor_embed_dim > encoder_dim`.

## Hyperparameter guidance

| Parameter             | Typical range          | Notes                                                            |
|-----------------------|------------------------|------------------------------------------------------------------|
| learning_rate         | 5e-5 to 5e-4           | AdamW; cosine schedule helps but is currently NOT in baseline    |
| weight_decay          | 0.0 to 0.1             | 0.05 default                                                     |
| batch_size            | 32, 64, 128            | A100 fits 128; smaller helps with long sequences                  |
| max_steps             | 2000 to 8000           | 4000 default; canary forecaster won't decide before step 2000   |
| encoder_depth         | 4 to 12                | 8 default                                                        |
| encoder_dim           | 256, 384, 512          | 384 default; 512 hits ~30M with depth=8                           |
| encoder_heads         | 4, 6, 8                | divisor of encoder_dim                                            |
| predictor_depth       | 1 to encoder_depth     | 2 default; cap = encoder_depth (Ψ <= Φc)                          |
| predictor_embed_dim   | 64, 128, 192, 256      | 128 default                                                      |
| num_targets           | 2 to 8                 | 4 default; matches I-JEPA paper                                   |
| ema_decay_start       | 0.99 to 0.999          | 0.996 default                                                    |
| ema_decay_end         | 0.999 to 1.0           | 1.0 default                                                      |
| codebook_size         | 0, 64, 256, 1024       | 0 = vanilla JEPA control; the MTS-JEPA Phase-3 axis              |
| codebook_loss_weight  | 0.0, 0.1, 0.5, 1.0     | 0 = codebook off as regulariser; MTS-JEPA Phase-3 axis           |
| future_block_weight   | 0.0 to 1.0             | weight for FutureBlockMask in the CompositeMask                  |
| multi_block_weight    | 0.0 to 1.0             | weight for MultiBlockInfillMask in the CompositeMask             |
| probe_eval_every_n_steps | 250, 500, 1000      | 500 default; trade probe-eval cost vs forecaster signal density  |

Suboptimal-vanilla baseline starting point (the LLM should improve on
these — note `codebook_size=0` is the writeup's vanilla JEPA control
row, NOT a known-best setting):

```
learning_rate=3e-4, weight_decay=0.05, batch_size=64, max_steps=4000,
encoder_depth=8, encoder_dim=384, encoder_heads=6,
predictor_depth=2, predictor_embed_dim=128, num_targets=4,
ema_decay_start=0.996, ema_decay_end=1.0,
codebook_size=0, codebook_loss_weight=0.0,
future_block_weight=1.0, multi_block_weight=0.5
```

## Code-diff guidance

After param exploration stalls, the policy switches to code diffs.
High-value targets ranked by ROI:

1. **Sweep the soft codebook bottleneck.** Per `docs/research/mts-jepa.md`
   the only published JEPA-design lever for discrete-regime structure;
   `codebook_size` ∈ {64, 256, 1024} and `codebook_loss_weight` ∈
   {0.1, 0.5, 1.0} are the key axes. The codebook is implemented as
   `SoftCodebookBottleneck` inside `train.py`.
2. **Tune the CompositeMask weights.** `FutureBlockMask` is the
   causal axis (predict future from past); `MultiBlockInfillMask` is
   the non-causal infill axis. Probably future-heavy is better for
   security probes — measure it.
3. **Add VICReg or Barlow Twins anti-collapse loss.** Plug
   `autojepa.models.losses.LOSS_REGISTRY["vicreg"]` into the train
   step alongside the MSE.
4. **Tune the EMA schedule.** Default is linear ramp `0.996 → 1.0`.
   Try a piecewise schedule that holds at 0.996 longer.
5. **Cosine learning-rate schedule.** The baseline uses constant LR.

## Hard rules — collapse gates (writeup §6.4 / `JEPA_HARD_RULES`)

A trial whose representation collapses is hard-failed before
probe-eval runs. Thresholds:

```
latent_variance < 0.3   -> trial fails
effective_rank   < 32   -> trial fails
rankme           < 64   -> trial fails
lidar            < 80   -> trial fails
```

`train.py` reports closed-form `rankme` + `latent_var` from
`autojepa.eval.collapse`; `lidar` is currently aliased to `rankme`
(LiDAR requires a Lightning Trainer queue not wired into the trial
sidecar — a Phase-4 task adds it).

## Required runtime calls (validator-enforced)

The AST validator rejects diffs that remove either of:

```python
emit_progress(step, step_target, metrics={"probe_auroc": ...})
autojepa.models.ema.assert_no_grad_on_target(model.encoder)
```

## Architecture invariants (validator-enforced)

- **Do NOT enable gradients on `encoder.teacher`.** The target encoder
  is a stop-gradient EMA of the student. Backprop through it defeats
  JEPA. `assert_no_grad_on_target(encoder)` is called immediately
  after `build_target_encoder(...)` and again as a sanity check.
- **Do NOT make the predictor (Ψ) deeper or wider than the context
  encoder (Φc).** `train.py::main` raises `ValueError` and the AST
  validator additionally rejects diffs that set
  `predictor_depth > encoder_depth` or
  `predictor_embed_dim > encoder_dim`.

## Decision gates

After 20 iterations the gate engine evaluates two gates:

```yaml
gates:
  - name: phase3_falsifier
    after_iters: 20
    require:
      probe_auroc: ">0.7"
    on_fail: warn   # do NOT abort on this gate; it is documenting
                   # the Phase-3 thesis kill criterion, not enforcing
                   # it. The campaign run history is preserved for
                   # post-mortem analysis even on miss.

  - name: baseline_separation
    after_iters: 20
    require:
      probe_auroc: ">0.65"
    on_fail: warn   # Soft gate from TODO.md Phase-3 Trace-JEPA-
                   # Evaluation belief. The full ≥0.05 separation
                   # check requires running the three baselines
                   # (LogLLaMA / GraphIDS-or-SAFE / MTS-JEPA) on the
                   # same probe set; that lives in a sibling
                   # examples/baselines/ run, not in this campaign.
```

The gates fire WARN not ABORT so the run history is preserved for
analysis even if Trace-JEPA misses the bar — that analysis is the
input to a possible Phase-3 architecture-reconsideration step
(writeup §12 escalation).
