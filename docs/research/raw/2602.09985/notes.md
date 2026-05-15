# JEPA-Automotive-Monitoring — Raw Extraction Notes

Sources:
- Alexandria raw page: `raw/web/arxivorg-abs-260209985.md` (workspace: global)
- Alexandria wiki page: `wiki/JEPA-Security/arxivorg-abs-260209985.md`
- Upstream: https://arxiv.org/abs/2602.09985

## Problem framing
- Online supervisory monitoring of an already-operating autonomous vehicle.
- Anomaly detection without anomaly labels (unknown anomalies have no labels
  by construction).
- Object state representations are the data unit being monitored.

## Method (per abstract)
- Pretraining: a JEPA-based self-supervised prediction task over object
  data; no anomaly labels required.
- Output of pretraining: expressive object embeddings in a latent
  representation space.
- Detection: established (classical) anomaly-detection methods applied on
  top of the JEPA embeddings — i.e. JEPA produces features, classical AD
  produces decisions. The JEPA contribution is representation, not the
  scoring head.

## Empirical
- Validation on the public real-world nuScenes dataset.
- Abstract illustrates "framework capabilities" rather than reporting a
  specific AUROC/F1 number — not paraphrased.

## Why this matters to AutoJEPA / Trace-JEPA
- This is one of only two published JEPA designs we have located in any
  safety/security-adjacent domain (the other is MTS-JEPA arxiv:2602.04643).
- It is a "vanilla JEPA + classical AD on top" architecture. That is
  exactly the architecture Phase-3 trace-jepa would default to if
  `codebook_size=0` (the AutoJEPA control row in the search space).
- The framework split (representation pretraining decoupled from a
  classical detector) is the same split AutoJEPA's
  `eval/probes.py` + `eval/collapse.py` enforce.
