# I-JEPA on CIFAR-10 — AutoJEPA Phase-2 falsifier

> The framework's first real test. If a 20-iteration hybrid campaign
> against the deliberately suboptimal baseline in `train.py` produces
> zero validated `llm_diff` improvements, the entire AutoJEPA approach
> is dead (writeup §11 Phase-2 kill criterion).

## What this example does

1. `prepare.py` (frozen): downloads CIFAR-10, materialises the
   pretraining set, the linear-probe-eval split (5k train / 5k test),
   and the canary subset (1k samples).
2. `train.py` (mutable): trains an I-JEPA model
   (`stable_pretraining.methods.IJEPA` per ADR-003) using a
   **deliberately suboptimal baseline** (predictor_depth=2,
   predictor_embed_dim=128, num_targets=2, plain L2, no VICReg).
3. `program.md` describes the task and JEPA invariants to the LLM
   diff policy.
4. `config.yaml` declares `target: basilica` (per ADR-002), the 20-iter
   hybrid campaign budget, and the recalibrated SSL forecaster
   defaults (per ADR-008/013).

## Headroom for the LLM (writeup §12.1 mitigation)

The baseline is intentionally weak so the hybrid policy has clear
paths to find improvements:

- Add VICReg / Barlow Twins anti-collapse via `autojepa.models.losses`.
- Deepen the predictor up to (but not exceeding) encoder depth (12).
- Tune EMA momentum schedule.
- Switch masking strategy via `autojepa.masking.CompositeMask`.
- Add a cosine LR schedule (currently constant).

## How to run

```bash
# Validate config (no GPU, no LLM credentials needed)
./run.sh validate

# Download CIFAR-10 (first time only, ~170 MB)
./run.sh prepare

# CPU/GPU smoke test: 50 steps, batch 64, probe every 25 steps
./run.sh smoke

# Local 1-iter campaign (target=command, no Basilica)
./run.sh local

# Full 20-iter campaign on Basilica (REQUIRES BASILICA_API_TOKEN + CHUTES_API_KEY)
./run.sh basilica
```

## Decision gate

The campaign config carries one decision gate (writeup §7.6):

```yaml
gates:
  - name: phase2_falsifier
    after_iters: 20
    require:
      probe_auroc: ">0.40"
    on_fail: warn
```

The baseline alone hits ~0.30 probe_auroc on CIFAR-10 at 4k steps.
> 0.40 means the LLM contributed measurable lift. The gate fires
WARN not ABORT so the run history is preserved for analysis even if
the LLM produced zero useful diffs (the analysis is the point —
identifying *why* the LLM failed informs Phase-3 / Phase-4 work).

## Outputs

| Path                                              | What                                   |
|---------------------------------------------------|----------------------------------------|
| `data/cifar10_{train,test}.pt`                    | uint8 (50000, 3, 32, 32) tensors       |
| `data/probe_eval.pt`                              | 5k/5k probe-eval split                 |
| `data/canary.pt`                                  | 1k overfit-canary subset               |
| `artifacts/ijepa-cifar10/checkpoint.json`         | Continuous-loop checkpoint state       |
| `artifacts/ijepa-cifar10/results.tsv`             | Per-iteration ledger                   |
| `artifacts/ijepa-cifar10/runs/<run_id>/...`       | Per-trial run dirs                     |
| `artifacts/ijepa-cifar10/versions/v####/...`      | Kept-best model checkpoints            |
| `traces/ijepa-cifar10/events.jsonl`               | Chrome-trace timeline events           |

## Phase-2 closeout criteria

- [x] `prepare.py` runs end-to-end and produces `data/*.pt`
- [x] `train.py` imports cleanly with `[jepa]` extras installed
- [x] `autojepa validate config.yaml` exits 0
- [ ] CPU/GPU smoke (`./run.sh smoke`) completes 50 steps without errors
- [ ] Basilica 20-iter campaign run by the user with a budget of
      ~$30-100 of A100 time
- [ ] At least one `llm_diff` proposal lands a measurable
      `probe_auroc` improvement over the baseline
- [ ] Decision gate `phase2_falsifier` reports `pass` (probe_auroc > 0.40)

If the last bullet fails, **the framework approach is dead** — escalate
per writeup §12.1 (mitigation: weaken baseline further, try a different
LLM model, or call it). NEVER LIE rule applies: a failed gate is a
truth, not a setback to be papered over.
