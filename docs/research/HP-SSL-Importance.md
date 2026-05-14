# HP / SSL Importance — Hyperparameters and Augmentation in Self-Supervised Learning

- **Paper:** Wagner, Ferreira, Stoll, Schirrmeister, Mueller, Hutter (2022)
- **arXiv:** 2207.07875
- **URL:** https://arxiv.org/abs/2207.07875
- **Venue:** ICML 2022 Pre-training Workshop
- **Role in our stack:** empirical justification for AutoJEPA's wider hybrid
  search budget and larger parameter space (writeup §6.3).

## 1) One-line thesis

In SSL, the choice of hyperparameters and (especially) data-augmentation
strategy swings downstream linear-probe accuracy by margins large enough to
be a confounder across published results — so SSL HP search must be budgeted
more aggressively than supervised fine-tuning, and must include the
augmentation policy as a first-class search dimension.

## 2) Method (technical)

- Treat the SSL training pipeline as a single optimization problem whose
  variables include the optimizer hyperparameters **and** the augmentation
  policy (operations, magnitudes, sampling probabilities).
- Run Bayesian optimization over that joint space, with linear-probe
  accuracy after pretraining as the objective.
- Backbone: SimSiam. Datasets span CIFAR-10/100, STL-10, and additional
  vision benchmarks.
- Introduce **GroupAugment**: instead of sampling individual augmentations,
  partition operations into semantic groups (color, geometric, blur, ...)
  and learn the per-group sampling distribution. This is jointly optimized
  with the rest of the HPs.

## 3) Results / headline observations

- HP and augmentation choice produce a "dramatic" performance spread. The
  abstract is qualitative on magnitudes; per-dataset numbers are reported in
  the body of the paper. We deliberately do not quote specific deltas
  here — the abstract retrieved via WebFetch did not contain them and the
  PDF was not parsed in detail. See `raw/2207.07875/notes.md`.
- After joint Bayesian optimization, SimSiam's linear-eval accuracy improves
  across every dataset tested vs. the published-default configuration.
- GroupAugment is consistently strong across datasets, outperforming
  supervised auto-augmentation policies (RandAugment, TrivialAugment) when
  used inside an SSL pipeline. The supervised policies' gains do not
  transfer to SSL.

## 4) Why it matters for AutoJEPA

The AutoJEPA writeup §6.3 widens `hybrid_param_explore_iters` from the
DeBERTa default of 5 to 20-30 (target: 25), grows the parameter space from
~5 dims to 10-12 dims, and bumps `hybrid_stall_threshold` from 3 to 5. This
paper is the empirical evidence that backs each of those changes:

| AutoJEPA change                                | This paper's evidence                                                                 |
|------------------------------------------------|---------------------------------------------------------------------------------------|
| `hybrid_param_explore_iters` 5 -> 25           | Per-dim sensitivity is high and interactions are non-trivial — small budgets miss it. |
| `hybrid_stall_threshold` 3 -> 5                | Reward landscape is rugged in SSL; longer stalls before declaring no-improvement.     |
| Param dims 5 -> 10-12                          | Joint HP+augmentation space is necessary; treating augmentation as fixed leaves real gains on the table. |
| Augmentation/masking ratio in the search space | "Likely underestimated role of data augmentation for SSL" — direct quote from abstract. |

Specific dimensions that should be in the AutoJEPA search space because of
this paper:

1. learning rate
2. weight decay
3. EMA / momentum coefficient (the JEPA target-encoder analogue of SimSiam's
   momentum HPs)
4. batch size
5. masking ratio (JEPA's augmentation analogue)
6. masking block size / scaling
7. predictor depth and width
8. loss weights for variance and covariance regularizers
9. augmentation strength (when an augmentation pipeline is used alongside
   masking)
10. seed-of-seeds (the paper does not isolate this, but its variance
    findings imply seed averaging is required to read deltas reliably)

GroupAugment specifically motivates **structuring** the augmentation
sub-search rather than flattening it into independent scalar dims; AutoJEPA's
hybrid-policy search can mirror this by grouping mask-strategy options.

## 5) Caveats

- The empirical study uses **SimSiam** (contrastive-style, joint-embedding
  with stop-gradient). JEPA is non-contrastive and predicts in latent space.
  The qualitative claim (HP+aug matter a lot) transfers; the numerical
  magnitudes do not.
- The search method is Bayesian optimization. AutoJEPA uses a hybrid of
  structured exploration plus a learned policy. The "wider budget"
  recommendation is therefore an analogy from one black-box method to
  another, not a replication of identical conditions.
- "Augmentation" in vision-SSL means image transforms; in JEPA the closest
  analogue is masking strategy. The mapping is conceptual, not literal.
- The paper is a workshop paper — the methodological claims are well
  supported but the parameter-importance ranking has not been replicated
  at ImageNet-1k scale in this specific work.

## 6) Cross-links

- `docs/research/Stable-Pretraining-V1.md` — provides the in-loop probes
  (`OnlineProbe`, `OnlineKNN`) and collapse metrics (`RankMe`, `LiDAR`)
  that AutoJEPA reads as the objective signal for the wider search this
  paper motivates.
- `autojepa-implementation-plan.md` §6.3 — the configuration changes that
  cite this paper directly.

Sources:
- [HP/SSL Importance paper](https://arxiv.org/abs/2207.07875)
