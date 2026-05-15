# JEPA-Automotive-Monitoring: Online Monitoring Framework for Automotive Time Series Data using JEPA Embeddings

**Authors:** Alexander Fertig, Karthikeyan Chandra Sekaran, Lakshman Balasubramanian, Michael Botsch
**Venue / Year:** Accepted at IEEE Intelligent Vehicles Symposium 2026
**arXiv:** 2602.09985 — https://arxiv.org/abs/2602.09985
**Status:** Distilled 2026-05-15
**Alexandria raw:** `raw/web/arxivorg-abs-260209985.md` (workspace: global)
**Alexandria wiki:** `wiki/JEPA-Security/arxivorg-abs-260209985.md`
**Local raw:** `raw/2602.09985/`

## 1. One-line thesis

A vanilla-JEPA self-supervised pretraining produces object embeddings rich
enough that established (classical) anomaly-detection scoring on top of those
embeddings yields a label-free online monitoring framework for autonomous
vehicles, validated on the real-world nuScenes dataset.

## 2. Method

- **Pretraining.** JEPA-based self-supervised prediction task over object
  state data. No anomaly labels are required, which is the load-bearing
  constraint for monitoring unknown anomalies.
- **Embedding.** The trained JEPA encoder maps object data into a latent
  representation space.
- **Detection head.** Established (classical) anomaly-detection methods are
  applied directly on the JEPA embeddings to identify deviations from
  normal operation.
- **Architectural split.** JEPA owns representation; the AD method owns
  scoring. The paper contributes the representation choice, not a new
  scoring rule.

## 3. Results

The abstract reports validation on the publicly available nuScenes dataset
illustrating the framework's capabilities. Specific AUROC / F1 numbers are
not in the abstract and are deliberately not paraphrased here.

## 4. Why it matters for AutoJEPA

- **Architectural confirmation.** This paper instantiates the exact pattern
  AutoJEPA's `eval/` namespace already enforces: representation pretraining
  decoupled from probe-style downstream scoring (see
  `src/autojepa/eval/probes.py` and `src/autojepa/eval/collapse.py`).
  Phase-3 trace-jepa inherits the same split.
- **Phase-3 control row.** With `codebook_size=0` and
  `codebook_loss_weight=0.0` (the search-space control row defined in
  `TODO.md` Phase 3), Trace-JEPA reduces to "vanilla JEPA + classical AD on
  top", which is structurally the same architecture as this paper. That
  makes the paper a direct architectural reference for the control
  configuration.
- **Cross-domain evidence.** Together with [MTS-JEPA](mts-jepa.md), this is
  one of only two published JEPA-for-anomalies architectures in any
  safety-adjacent domain (per the `topic=JEPA-Security-Gap` belief in
  alexandria, asserted 2026-05-15). It is evidence that JEPA embeddings can
  carry an anomaly signal, but it is not evidence for any specific design
  choice we should copy into Trace-JEPA — for that we need the Phase-3
  search.

## 5. Caveats / known limitations

- Domain mismatch: nuScenes object state data is continuous sensor channels
  describing vehicle dynamics. LLM-agent traces are structured event
  records with categorical fields. The transferability of embedding quality
  is conjectural; Phase-3 must measure it.
- The paper offers no claim about adversarial robustness, prompt injection,
  or security-as-attacker-model. It is anomaly detection in the
  reliability-engineering sense, not the security sense.
- The abstract is silent on the specific classical AD methods used. Phase-3
  should not assume a particular downstream scorer is what made the
  framework work.
- This is a non-JEPA-novel design point (the JEPA backbone here is not
  specialized for the modality the way I-JEPA is for images or A-JEPA is
  for audio). It is "JEPA + classical AD", not "AV-JEPA".

## 6. References to other corpus entries

- [MTS-JEPA](mts-jepa.md) — the other JEPA-for-anomalies design point.
  MTS-JEPA targets the architecture (codebook + multi-resolution); this
  paper targets the deployment shape (online monitoring + classical AD).
- [I-JEPA](I-JEPA.md) — the vanilla JEPA recipe this paper applies
  unmodified at the backbone level.
- [JEPA AV Security Survey](jepa-av-security-survey.md) — positioning
  paper that motivates the AV/safety framing this paper instantiates.
- [SSL-IDS Landscape](ssl-ids-landscape.md) — non-JEPA SSL baselines in a
  related anomaly-detection regime (network intrusion).

Sources:
- [JEPA-Automotive-Monitoring paper](https://arxiv.org/abs/2602.09985)
- Alexandria belief: `topic=JEPA-Security-Gap` (asserted 2026-05-15) — no
  JEPA prior art for logs/agent traces; this paper plus MTS-JEPA are the
  closest hits.
