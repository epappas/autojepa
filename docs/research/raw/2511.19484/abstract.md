# Abstract — stable-pretraining-v1

**Source URL:** https://arxiv.org/abs/2511.19484
**Fetched:** 2026-05-15

> Foundation models and self-supervised learning (SSL) have become central to
> modern AI, yet research in this area remains hindered by complex codebases,
> redundant re-implementations, and the heavy engineering burden of scaling
> experiments. We present stable-pretraining, a modular, extensible, and
> performance-optimized library built on top of PyTorch, Lightning, Hugging
> Face, and TorchMetrics.

Note: the abstract above is reconstructed from the arxiv landing page text
returned by WebFetch. The full paper PDF is at
https://arxiv.org/pdf/2511.19484 and was also retrieved successfully (binary
saved by the harness). The library unifies SSL utilities including
representation probes, collapse-detection metrics, augmentation pipelines,
extensible evaluation routines, and a "log-everything" design principle for
training visibility. Capability is demonstrated through depth-wise
representation analysis and CLIP-degradation studies under synthetic-data
fine-tuning.
