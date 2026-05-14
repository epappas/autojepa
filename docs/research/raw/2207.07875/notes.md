# HP/SSL Importance — extraction notes

**Source used**
- arxiv abstract page: https://arxiv.org/abs/2207.07875 (WebFetch OK)

## Headline claims

- HP and augmentation choices "can have a dramatic impact on performance" in
  SSL — the variance is large enough to be a confounder when comparing SSL
  methods across papers.
- Augmentation is the under-estimated factor; tuning it is at least as
  important as tuning the optimizer/schedule HPs.
- Bayesian optimization over the joint HP+augmentation space improves SimSiam
  across multiple datasets vs. published defaults.
- A new method, GroupAugment, samples augmentation operations from grouped
  buckets and is jointly tuned; it is consistently strong on linear
  evaluation across datasets, unlike supervised auto-augmentation policies
  whose gains do not transfer to SSL.

## Numerical results

The arxiv abstract does not list per-dataset accuracy deltas. The body of
the paper (not parsed here) reports SimSiam linear-eval improvements after
Bayesian optimization across CIFAR-10, CIFAR-100, STL-10, and ImageNet
subsets. Concrete numbers were not retrieved via WebFetch and are
intentionally omitted from the distilled doc rather than guessed.

## Hyperparameter axes the paper treats as load-bearing

- learning rate and schedule
- weight decay
- momentum / EMA coefficient (relevant to JEPA's target encoder)
- batch size
- augmentation operation set, magnitudes, sampling probabilities
- group structure of augmentations (GroupAugment)

## Relevance to AutoJEPA section 6.3

- The paper is the empirical justification for treating SSL HP search as a
  larger-budget problem than supervised fine-tuning. The writeup widens
  `hybrid_param_explore_iters` from 5 to 25 (and grows the parameter space
  from 5 to 10-12 dims) precisely because this paper demonstrates the
  per-dim sensitivity is high and the interactions are non-trivial.
- The augmentation finding directly motivates including masking-ratio and
  augmentation-strength dimensions in the AutoJEPA hybrid search.
- The "underestimated role of augmentation" finding generalizes to JEPA's
  masking strategy, which is the JEPA-side analogue of contrastive-SSL
  augmentation.

## Caveats

- The empirical study targets SimSiam, a contrastive-style SSL method; JEPA
  is non-contrastive and predicts in latent space, so the magnitudes do not
  transfer 1:1. The qualitative claim (HP+aug matter a lot) does transfer.
- Bayesian optimization is the search method used in the paper; AutoJEPA
  uses a hybrid of structured exploration plus a learned policy, so the
  budget recommendation is an analogy, not a replication.
