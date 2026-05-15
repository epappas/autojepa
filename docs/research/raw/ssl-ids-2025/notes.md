# SSL-IDS Landscape (2025) — Raw Extraction Notes

This is a combined raw note covering the four 2025 SSL-IDS papers grouped
into the `ssl-ids-landscape.md` distillation. None of them is JEPA — they
are listed together as Phase-3 non-JEPA SSL baselines.

## Sources (all from Alexandria workspace `global`)

- GraphIDS (arXiv 2509.16625, NeurIPS 2025)
  - Alexandria raw: `raw/web/arxivorg-abs-250916625.md`
  - Alexandria wiki: `wiki/web/arxivorg-abs-250916625.md`
- SAFE (arXiv 2502.07119, AAAI-25 AICS Workshop)
  - Alexandria raw: `raw/web/arxivorg-abs-250207119.md`
  - Alexandria wiki: `wiki/web/arxivorg-abs-250207119.md`
- CLAN (arXiv 2509.06550, IEEE CSR 2025)
  - Alexandria raw: `raw/web/arxivorg-abs-250906550.md`
  - Alexandria wiki: `wiki/web/arxivorg-abs-250906550.md`
- Transformer-Contrastive IDS (arXiv 2505.08816, IFIP Networking 2025)
  - Alexandria raw: `raw/web/arxivorg-abs-250508816.md`
  - Alexandria wiki: `wiki/web/arxivorg-abs-250508816.md`

## Per-paper extraction

### GraphIDS — arxiv:2509.16625 — Guerra et al., NeurIPS 2025
- SSL family: **masked autoencoder** (NOT JEPA).
- Architecture: inductive graph neural network embeds each NetFlow record
  with its local topological context; a Transformer-based encoder-decoder
  reconstructs the embeddings.
- Training signal: reconstruction error on benign-only training data.
- Inference: flows with unusually high reconstruction error are flagged.
- Reported numbers (from the abstract): up to 99.98% PR-AUC, 99.61% macro
  F1; 5-25 percentage-point gain over baselines on diverse NetFlow
  benchmarks.

### SAFE — arxiv:2502.07119 — Li, Shang, Gungor, Rosing, AAAI-25 AICS
- SSL family: **masked autoencoder** (NOT JEPA).
- Architecture: tabular IDS data is reshaped into image-like tensors so a
  Masked Autoencoder (MAE) can operate on them; MAE features feed a
  lightweight novelty detector.
- Training signal: MAE reconstruction; benign-only.
- Reported numbers: up to 26.2% F1 over SLAD, up to 23.5% F1 over Anomal-E.

### CLAN — arxiv:2509.06550 — Wilkie, Hindy, Tachtatzis, Atkinson, IEEE CSR 2025
- SSL family: **contrastive** (NOT JEPA).
- Inverted-pair design: augmented samples are treated as negative views
  (representing potentially malicious distributions) and other benign
  samples are positive views — opposite to the usual contrastive recipe.
- Reported numbers: surpasses prior SSL and AD techniques in binary
  classification on Lycos2017; superior multi-class performance after
  fine-tuning on a limited labeled subset.
- Code: https://github.com/jackwilkie/CLAN

### Transformer-Contrastive IDS — arxiv:2505.08816 — Koukoulis, Syrigos, Korakis, IFIP Networking 2025
- SSL family: **contrastive** (NOT JEPA).
- Operates on raw packet sequences (not handcrafted NetFlow features).
- Architecture: transformer encoder + packet-level data augmentation; the
  model learns flow representations directly from packets.
- Reported numbers: +3% AUC intra-dataset, +20% AUC inter-dataset over
  prior NetFlow SSL methods; +1.5% AUC over self-supervised NetFlow
  baselines after pretraining + supervised fine-tuning. Strong
  cross-dataset transfer including target-domain-without-benign-data.
- Code: https://github.com/koukipp/contrastive_transformers_ids

## Why grouped

These four papers are the published 2025 SSL-IDS reference set. They are
load-bearing for the Trace-JEPA novelty argument because they prove the
SSL-on-security space has been thoroughly explored — and yet none of them
uses a JEPA. That is the gap recorded in the
`topic=JEPA-Security-Gap` belief in alexandria
(asserted 2026-05-15).
