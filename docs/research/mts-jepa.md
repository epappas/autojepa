# MTS-JEPA: Multi-Resolution Joint-Embedding Predictive Architecture for Time-Series Anomaly Prediction

**Authors:** Yanan He, Yunshi Wen, Xin Wang, Tengfei Ma
**Venue / Year:** arXiv preprint, Feb 2026
**arXiv:** 2602.04643 — https://arxiv.org/abs/2602.04643
**Status:** Distilled 2026-05-15
**Alexandria raw:** `raw/web/arxivorg-abs-260204643.md` (workspace: global)
**Alexandria wiki:** `wiki/JEPA-Security/arxivorg-abs-260204643.md`
**Local raw:** `raw/2602.04643/`

## 1. One-line thesis

A JEPA backbone whose predictive objective is multi-resolution and whose
latent space is constrained by a soft codebook bottleneck both (a) separates
transient shocks from long-term trends and (b) captures discrete regime
transitions in multivariate time series, while the codebook itself doubles as
the anti-collapse regularizer.

## 2. Method

- Backbone: Joint-Embedding Predictive Architecture for multivariate time
  series.
- Two design departures from vanilla JEPA, both aimed at the failure modes
  the abstract calls out (representation collapse; inability to capture
  precursor signals across varying temporal scales):
  1. **Multi-resolution predictive objective** — explicitly decouples
     transient shocks from long-term trends.
  2. **Soft codebook bottleneck** — a learned discrete-ish latent vocabulary
     that captures regime transitions and acts as an intrinsic regularizer
     to stabilize optimization.
- The soft codebook is doing double duty: it is both the inductive-bias
  carrier (regime structure) and the collapse defense.

## 3. Results

The abstract reports state-of-the-art performance under the early-warning
protocol on standard benchmarks and empirically demonstrates that the soft
codebook prevents degenerate solutions. Specific per-benchmark deltas are
not in the abstract and are deliberately not paraphrased here — see the
upstream paper for numbers.

## 4. Why it matters for AutoJEPA

This is the only published JEPA design we have located that is purpose-built
for discrete-regime-transition modeling, which is the structural prior we
expect to govern LLM-agent traces (tool-call switches, role boundaries, plan
re-entries).

- **Phase-3 trace-jepa search-space dimension.** The soft codebook is added
  to AutoJEPA's Phase-3 `examples/trace-jepa/config.yaml` parameter space as
  two new dimensions, `codebook_size` and `codebook_loss_weight` (see
  `TODO.md` Phase 3). `codebook_size=0` recovers vanilla JEPA, so the search
  contains an honest control.
- **Phase-3 evaluation baseline.** MTS-JEPA is one of the three external
  baselines (LogLLaMA + GraphIDS/SAFE + MTS-JEPA) Trace-JEPA must separate
  from to justify the JEPA inductive-bias bet. See `TODO.md` Phase 3
  evaluation section and the
  `topic=Trace-JEPA-Evaluation` belief in alexandria.
- **Anti-collapse cross-check.** MTS-JEPA's framing of the codebook as an
  intrinsic regularizer is consistent with the C-JEPA argument that EMA +
  stop-gradient alone is not collapse-proof. Treat it as orthogonal evidence,
  not a replacement for the existing
  [C-JEPA](C-JEPA.md) regularizer family.
- **Scope guard.** This is a Phase-3 example concern. The codebook does not
  enter `src/autojepa/models/` — putting it in the core framework would
  repeat the contrib-namespace mistake AutoJEPA explicitly rejected.

## 5. Caveats / known limitations

- The abstract does not pin down the codebook-size or loss-weight values
  used to obtain SOTA — Phase-3 must therefore search the dimension, not
  copy a reported configuration.
- "Soft codebook" is a paper-specific term; the AutoJEPA Phase-3
  implementation should choose a concrete instantiation (e.g. VQ-style with
  EMA updates, or Gumbel-softmax) and document which one is on the search
  axis.
- Time-series MTS structure is not identical to agent-trace structure
  (continuous sensor channels vs structured event records). MTS-JEPA's
  evidence is suggestive, not transferable; Phase-3 must measure the
  soft-codebook benefit on the actual trace probe, not assume it.
- Per the `topic=JEPA-Security-Gap` belief in alexandria
  (asserted 2026-05-15), no JEPA paper targets logs, agent traces,
  prompt-injection detection, eBPF/syscall traces, or container
  observability — MTS-JEPA is the closest published prior art and that is
  exactly why it is a baseline rather than a precedent.

## 6. References to other corpus entries

- [I-JEPA](I-JEPA.md) — vanilla JEPA recipe MTS-JEPA departs from.
- [C-JEPA](C-JEPA.md) — orthogonal collapse-prevention strategy; cross-check
  for the soft-codebook-as-regularizer claim.
- [V-JEPA 2](V-JEPA-2.md) — temporal JEPA on a different modality (video).
- [JEPA Automotive Monitoring](jepa-automotive-monitoring.md) — JEPA
  embeddings + classical AD on automotive time series; orthogonal evidence
  point in the JEPA-for-anomalies family.
- [SSL-IDS Landscape](ssl-ids-landscape.md) — non-JEPA SSL baselines
  (GraphIDS, SAFE, CLAN, transformer-contrastive IDS).
- [JEPA AV Security Survey](jepa-av-security-survey.md) — positioning paper
  on JEPA for AV safety/security; cite-only.

Sources:
- [MTS-JEPA paper](https://arxiv.org/abs/2602.04643)
- Alexandria belief: `topic=Trace-JEPA-Design` (asserted 2026-05-15) — soft
  codebook bottleneck should be a search dimension in Phase-3 AutoJEPA.
