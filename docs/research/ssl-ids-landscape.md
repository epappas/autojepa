# SSL-IDS Landscape (2025) — Non-JEPA Self-Supervised Intrusion-Detection Baselines

**Status:** Distilled 2026-05-15
**Local raw:** `raw/ssl-ids-2025/`
**Alexandria workspace:** global

This is a combined distillation of four published 2025 SSL-for-intrusion-
detection papers. They are grouped because they share three properties: (a)
they target network intrusion / IDS, (b) they are self-supervised, and
(c) **none of them is a JEPA**. The group is the Phase-3 non-JEPA SSL
baseline lineup for Trace-JEPA; see `TODO.md` Phase 3 evaluation.

| Short name                  | arXiv         | Venue                  | SSL family             | Alexandria raw                                  |
|-----------------------------|---------------|------------------------|------------------------|-------------------------------------------------|
| GraphIDS                    | 2509.16625    | NeurIPS 2025           | masked autoencoder     | `raw/web/arxivorg-abs-250916625.md`             |
| SAFE                        | 2502.07119    | AAAI-25 AICS Workshop  | masked autoencoder     | `raw/web/arxivorg-abs-250207119.md`             |
| CLAN                        | 2509.06550    | IEEE CSR 2025          | contrastive            | `raw/web/arxivorg-abs-250906550.md`             |
| Transformer-Contrastive IDS | 2505.08816    | IFIP Networking 2025   | contrastive            | `raw/web/arxivorg-abs-250508816.md`             |

## 1. Per-paper one-line theses

- **GraphIDS** (Guerra et al., NeurIPS 2025). An inductive GNN embeds each
  NetFlow record with its local topological context, then a Transformer
  encoder-decoder reconstructs those embeddings; flows with high
  reconstruction error are flagged as intrusions.
- **SAFE** (Li, Shang, Gungor, Rosing, AAAI-25 AICS). Tabular IDS data is
  reshaped into image-like tensors so a Masked Autoencoder can operate on
  them; MAE features feed a lightweight novelty detector.
- **CLAN** (Wilkie, Hindy, Tachtatzis, Atkinson, IEEE CSR 2025). Inverts the
  contrastive recipe: augmented samples are treated as negative views
  (representing potentially malicious distributions) and other benign
  samples are positive views.
- **Transformer-Contrastive IDS** (Koukoulis, Syrigos, Korakis, IFIP
  Networking 2025). Self-supervised contrastive transformer over **raw
  packet sequences** (not handcrafted NetFlow features), with packet-level
  augmentations.

## 2. Headline numbers (per each paper's abstract)

- **GraphIDS:** up to 99.98% PR-AUC, 99.61% macro F1; 5-25 percentage-point
  gain over baselines on diverse NetFlow benchmarks.
- **SAFE:** up to 26.2% F1 over SLAD (state-of-the-art deep AD), up to
  23.5% F1 over Anomal-E (the prior SSL-IDS reference).
- **CLAN:** surpasses prior SSL and AD techniques in binary classification
  on Lycos2017; superior multi-class accuracy after fine-tuning on a
  limited labeled subset.
- **Transformer-Contrastive IDS:** +3% AUC intra-dataset and +20% AUC
  inter-dataset over prior NetFlow SSL methods; +1.5% AUC over SSL NetFlow
  models after pretraining + supervised fine-tuning. Strong cross-dataset
  transfer.

## 3. What unifies them — and why this matters for AutoJEPA

Per the `topic=JEPA-Security-Gap` belief in alexandria (asserted
2026-05-15), the published SSL-IDS literature is uniformly masked-autoencoder
or contrastive — **not JEPA**. These four papers are the empirical core of
that observation. The gap is load-bearing for the Trace-JEPA novelty claim:
SSL-on-security has been thoroughly explored without anyone reaching for a
JEPA, so Trace-JEPA must out-perform a strong member of this set or its
inductive-bias bet does not earn its keep.

**Phase-3 baseline role.** Per the `topic=Trace-JEPA-Evaluation` belief in
alexandria, the Trace-JEPA Phase-3 evaluation must include three external
baselines: LogLLaMA, MAE-based SSL-IDS (GraphIDS or SAFE), and MTS-JEPA.
This combined note covers the second slot. AutoJEPA selects either GraphIDS
or SAFE as the concrete Phase-3 MAE baseline; SAFE is operationally
simpler (tabular reshape + MAE) while GraphIDS is the stronger reported
result. The choice is recorded in the Phase-3 config when the example is
written.

**Acceptance gate (per `TODO.md` Phase 3).** Trace-JEPA must separate from
each baseline by ≥ 0.05 AUROC on the same probe set, or the JEPA
inductive-bias bet is reconsidered.

## 4. Why none of them is a precedent for Trace-JEPA

- **Modality.** All four target network traffic — NetFlow records or raw
  packets. Trace-JEPA targets LLM-agent execution traces (structured event
  records: tool calls, role boundaries, plan re-entries). Network traffic
  has no analog of "tool-call switch" or "plan re-entry"; the inductive
  prior is different.
- **SSL family.** MAE and contrastive learning both either (a) reconstruct
  in input space (MAE) or (b) impose a similarity geometry over views
  (contrastive). JEPA predicts in latent space without either, which is
  the whole reason it is being investigated for traces.
- **No published JEPA cross-walk to security.** Per the Alexandria belief
  cited above, no JEPA paper targets logs, agent traces, prompt-injection
  detection, eBPF/syscall traces, or container observability as of
  2026-05-15. Cross-domain extrapolation from network IDS to agent traces
  via a non-JEPA SSL recipe is not the same architectural bet as porting a
  JEPA to traces — that is exactly what Phase-3 sets out to test.

## 5. Caveats / known limitations

- All four numbers above are reported by the original abstracts on
  benchmarks of the authors' choice; AutoJEPA does not reproduce them
  here. Phase-3 must measure the chosen baseline (GraphIDS or SAFE) on the
  same probe set as Trace-JEPA — published numbers are not transferable to
  the InjecAgent / AgentDojo overlay protocol.
- GraphIDS's PR-AUC ceiling is so close to 1.0 on NetFlow benchmarks that
  on the much harder agent-trace probe the relevant comparison may be at
  much lower AUROC. Do not assume the reported headroom transfers.
- CLAN's "augmented = negative" inversion is interesting but is a
  contrastive-design point, not a JEPA-design point; it does not feed into
  the AutoJEPA Phase-3 search dimensions, only into baseline selection.
- The transformer-contrastive raw-packet model is the closest in spirit to
  Trace-JEPA's "operate on the raw structured stream" framing, but its
  augmentation-pair construction is fundamentally contrastive; we keep it
  in the baseline lineup specifically as a strong contrastive control.

## 6. References to other corpus entries

- [MTS-JEPA](mts-jepa.md) — the JEPA-on-time-series baseline that completes
  the Trace-JEPA Phase-3 baseline lineup alongside LogLLaMA and one of
  GraphIDS/SAFE.
- [JEPA Automotive Monitoring](jepa-automotive-monitoring.md) — the only
  other published JEPA-for-anomalies design, in a non-security domain.
- [JEPA AV Security Survey](jepa-av-security-survey.md) — positioning
  paper (no implementation) on JEPA for AV safety/security.
- [I-JEPA](I-JEPA.md) — vanilla JEPA recipe whose security-domain
  counterpart these four papers conspicuously do not provide.

Sources:
- [GraphIDS](https://arxiv.org/abs/2509.16625)
- [SAFE](https://arxiv.org/abs/2502.07119)
- [CLAN](https://arxiv.org/abs/2509.06550)
- [Transformer-Contrastive IDS](https://arxiv.org/abs/2505.08816)
- Alexandria belief: `topic=JEPA-Security-Gap` (asserted 2026-05-15) — the
  uniformly-MAE-or-contrastive observation that motivates grouping these
  four together.
- Alexandria belief: `topic=Trace-JEPA-Evaluation` (asserted 2026-05-15) —
  Trace-JEPA Phase-3 baseline lineup including GraphIDS or SAFE.
