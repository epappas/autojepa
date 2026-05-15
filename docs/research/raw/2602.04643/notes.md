# MTS-JEPA — Raw Extraction Notes

Sources:
- Alexandria raw page: `raw/web/arxivorg-abs-260204643.md` (workspace: global)
- Alexandria wiki page: `wiki/JEPA-Security/arxivorg-abs-260204643.md`
- Upstream: https://arxiv.org/abs/2602.04643

## Problem framing
- Domain: multivariate time series (MTS) anomaly prediction in critical
  infrastructure, framed as proactive (early-warning) rather than post-hoc.
- Two named JEPA failure modes the paper targets:
  1. representation collapse,
  2. inability to capture precursor signals across varying temporal scales.

## Method (per abstract)
- Architecture: JEPA backbone + multi-resolution predictive objective + soft
  codebook bottleneck.
- Multi-resolution objective: explicitly decouples transient shocks from
  long-term trends.
- Soft codebook: captures discrete regime transitions; the abstract calls
  out the codebook constraint as an "intrinsic regularizer" for optimization
  stability — i.e. the codebook is doing double duty as anti-collapse
  regularizer.

## Headline empirical claim
- State-of-the-art performance under the early-warning protocol on standard
  benchmarks. (Specific deltas not in the abstract; not paraphrased.)
- Empirically prevents degenerate solutions (the collapse failure mode the
  paper targets).

## Why this matters to AutoJEPA / Trace-JEPA
- Only published JEPA design we have located that is purpose-built for
  discrete-regime-transition modeling.
- Discrete-regime structure is the load-bearing structural prior for
  agent-trace data (tool-call switches, role boundaries, plan re-entries).
- Soft codebook bottleneck is therefore a candidate Phase-3 search dimension,
  not a core framework primitive.
