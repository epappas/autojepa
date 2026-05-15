# I-JEPA on CIFAR-10 — Phase-2 framework falsifier

## Task

Pretrain a small I-JEPA model on CIFAR-10 with a deliberately
suboptimal baseline so the hybrid policy has clear headroom to find
improvements. Pass criterion: at least one `llm_diff` proposal must
land a measurable improvement on `probe_auroc` over a 20-iteration
campaign. If the campaign produces zero validated improvements, the
entire AutoJEPA framework approach is dead (writeup §11 Phase-2 kill
criterion).

## Objective

Maximize `probe_auroc` — the linear-probe accuracy on the held-out
CIFAR-10 split, measured at every `probe_eval_every_n_steps` steps and
emitted via `emit_progress(metrics={"probe_auroc": ...})`. Per ADR-004,
this is the only valid JEPA campaign objective; training loss
(`L_predict`) is reported for diagnostics but does not drive
keep/discard.

## Mutable file

`train.py` — the I-JEPA training script. The hybrid policy proposes:

- Hyperparameter overrides (param mode, first 25 iterations per ADR
  default).
- Code diffs (diff mode, when param search stalls).

## Frozen file

`prepare.py` — CIFAR-10 download, probe-eval split, canary subset.
**Must not be modified.** It owns "what is correct".

## Constraints

- Single-GPU sufficient (A100, H100, L40S, RTX-4090, RTX-A6000).
- Each iteration should complete within `target.timeout_s` (default
  3600 s). MAX_STEPS=4000 with batch_size=128 fits this on A100.
- Encoder model: `vit_tiny_patch16_224` (~5.5M params, timm).
- The predictor (Ψ) MUST NOT be deeper or wider than the context
  encoder (Φc). Encoder is 12 blocks @ 192 dim; default predictor is
  2 blocks @ 128 dim. Diffs raising predictor_depth above 12 are
  rejected by the validator.

## Hyperparameter guidance

| Parameter             | Typical range          | Notes                                                            |
|-----------------------|------------------------|------------------------------------------------------------------|
| learning_rate         | 1e-5 to 5e-4           | AdamW; cosine schedule helps but is currently NOT in baseline    |
| weight_decay          | 0.0 to 0.1             | 0.05 default                                                     |
| batch_size            | 64, 128, 256           | A100 fits 256; smaller helps with limited memory                 |
| max_steps             | 2000 to 8000           | 4000 default; canary forecaster won't decide before step 2000   |
| predictor_depth       | 1 to 12 (cap = encoder)| 2 default. Going 3-6 likely improves probe_auroc                 |
| predictor_embed_dim   | 64, 128, 192, 256, 384 | 128 default. 192 matches encoder; 384 violates Ψ<=Φc rule        |
| num_targets           | 1 to 8                 | 2 default; I-JEPA paper uses 4                                   |
| ema_decay_start       | 0.99 to 0.999          | 0.996 default (I-JEPA paper)                                     |
| ema_decay_end         | 0.999 to 1.0           | 1.0 default                                                      |
| probe_eval_every_n_steps | 250, 500, 1000      | 500 default; trade probe-eval cost vs forecaster signal density  |

Suboptimal baseline starting point (the LLM should improve on these):

```
learning_rate=1e-4, weight_decay=0.05, batch_size=128, max_steps=4000,
predictor_depth=2, predictor_embed_dim=128, num_targets=2,
ema_decay_start=0.996, ema_decay_end=1.0
```

## Code-diff guidance

After param exploration stalls, the policy switches to code diffs.
High-value targets ranked by ROI (writeup §6.4):

1. **Add VICReg or Barlow Twins anti-collapse loss.** The baseline
   uses plain L2 (whatever `IJEPA(...)` ships internally); add a
   VICReg term via `autojepa.models.losses.LOSS_REGISTRY["vicreg"]`
   with sim_coeff=25, var_coeff=25, cov_coeff=1.
2. **Replace the masking strategy.** The baseline uses I-JEPA's
   default multi-block. Try `autojepa.masking.CompositeMask`
   weighting two `MultiBlockInfillMask` instances with different
   target_scale ranges.
3. **Tune the EMA schedule.** Default is linear ramp `0.996 → 1.0`.
   Try a cosine schedule or a piecewise schedule that holds at
   0.996 for the first 1000 steps then ramps.
4. **Cosine learning-rate schedule.** The baseline uses constant LR.
   Add `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
   T_max=MAX_STEPS)`.
5. **Probe-eval frequency.** Cheaper probe-eval (k-NN via
   `autojepa.eval.probes.build_knn_probe`) lets you eval more often
   and feed the forecaster a denser signal.

## Hard rules — collapse gates (writeup §6.4 / `JEPA_HARD_RULES`)

A trial whose representation collapses is hard-failed before
probe-eval runs. Thresholds:

```
latent_variance < 0.3   -> trial fails
effective_rank   < 32   -> trial fails
rankme           < 64   -> trial fails
lidar            < 80   -> trial fails
```

## Required runtime calls (validator-enforced)

The AST validator rejects diffs that remove either of:

```python
emit_progress(step, step_target, metrics={"probe_auroc": ...})
autojepa.models.ema.assert_no_grad_on_target(model.encoder)
```

The script must also write `outcome.json` into `$AR_MODEL_DIR` on
exit (any path) per the ADR-015 outcome-detection contract. The
basilica adapter polls for this file via the bootstrap's
`/model/files` endpoint and uses it as the iter-done signal —
without it, the controller waits until `target.timeout_s` and marks
the iter `failed/discard` even if training succeeded. Shape:

```json
{"status": "ok", "metrics": {"probe_auroc": 0.281, "loss": 0.008},
 "elapsed_s": 1417, "completed_steps": 4000, "step_target": 4000,
 "ts": 1778900000}
```

For canary or pretrain failure use `{"status": "failed", "reason":
"<why>", "metrics": {...}, "elapsed_s": ..., "ts": ...}`. Use the
`_write_outcome(...)` helper at the top of `train.py`; do not roll
your own writer.

## Architecture invariants (validator-enforced)

- **Do NOT enable gradients on `model.encoder.teacher`.** The target
  encoder is a stop-gradient EMA of the student. Backprop through it
  defeats JEPA.
- **Do NOT make the predictor (Ψ) deeper than the context encoder
  (Φc).** Encoder depth is 12 (vit_tiny_patch16_224); predictor depth
  cap is therefore 12. Diffs setting `predictor_depth > 12` are
  rejected.

## Decision gate

After 20 iterations, the gate engine evaluates:

```yaml
gates:
  - name: phase2_falsifier
    after_iters: 20
    require:
      probe_auroc: ">0.40"
    on_fail: warn   # do NOT abort the campaign on this gate; it is
                   # documenting the framework's first-real-test pass
                   # criterion, not enforcing it
```

A pass requires probe_auroc > 0.40 (the baseline alone hits ~0.30 at
4k steps; >0.40 means the LLM contributed measurable lift). The gate
fires WARN not ABORT so the run history is preserved for analysis
even if the LLM produced zero useful diffs.
