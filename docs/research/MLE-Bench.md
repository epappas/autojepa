# MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering

**Citation:** Chan, J. S., Chowdhury, N., Jaffe, O., Aung, J., Sherburn, D., Mays, E., Starace, G., Liu, K., Maksin, L., Patwardhan, T., Weng, L., Madry, A. *MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering.* arXiv:2410.07095, 2024 (ICLR 2025).
**Raw material:** `docs/research/raw/2410.07095/`
**AutoJEPA classification:** Comparison benchmark

---

## 1. One-line thesis

Curate 75 Kaggle competitions, calibrate human leaderboards, score AI agents against gate-style medal thresholds, and the best public scaffold (o1-preview + AIDE) reaches at least Kaggle bronze in 16.9% of competitions — establishing an objective, leaderboard-anchored capability ceiling for ML-engineering coding agents.

## 2. Method

- **75 ML-engineering Kaggle competitions** curated for diversity (tabular, vision, NLP, time series; classification, regression, ranking).
- Each competition includes:
  - A held-out test set with public scoring.
  - A calibrated human leaderboard (Kaggle's actual public results).
  - A competition-specific medal threshold (bronze/silver/gold), expressed as percentile on the leaderboard adjusted for participant count.
- **Agent contract:** receive task description + dataset, produce a `submission.csv`. The benchmark scores the submission and bins it into a medal tier (or none).
- **Open-source scaffolds** wrap several frontier models; results are reported per (model, scaffold) pair.
- **Resource scaling and contamination** investigated as auxiliary studies.

## 3. Results

- **Headline:** OpenAI o1-preview + AIDE scaffold reaches at least bronze in 16.9% of competitions — the best published configuration at release.
- More attempts (multiple submissions per competition) measurably help.
- More compute per attempt measurably helps.
- Pre-training contamination inflates some scores; the benchmark surfaces and quantifies the effect.
- Open-source benchmark code at the URL in the abstract enables independent reproduction.

## 4. Why it matters for AutoJEPA — Comparison benchmark

MLE-bench is not AutoJEPA's training task — JEPA pretraining is *upstream* of competition-style ML engineering — but its evaluation philosophy is precisely what AutoJEPA's promotion tracker mirrors:

- **Gate-style decisions, not raw scores.** MLE-bench reduces a noisy continuous metric (Kaggle score) to a calibrated tier (medal). AutoJEPA's keep/discard rule reduces a noisy training-run metric (val_bpb, probe accuracy) to a binary keep/discard. Same shape, same justification: gates are robust to noise in ways that thresholded raw scores are not.
- **Leaderboard-anchored thresholds.** MLE-bench thresholds are calibrated against actual humans. AutoJEPA's thresholds should be calibrated against documented JEPA baselines (I-JEPA, V-JEPA, V-JEPA 2 published numbers) for the same reason — a config that beats the published baseline is the keep signal.
- **Scaffold reporting.** MLE-bench reports (model, scaffold) pairs. AutoJEPA's experiment tracker should similarly report (proposer model, validator config, target adapter) so future comparisons stay apples-to-apples.

The empirical 16.9% bronze rate is also useful as a *capability ceiling*: it documents how good general-purpose coding scaffolds are at ML engineering as of late 2024 / early 2025. AutoJEPA's value proposition rests on specialized primitives doing better than a general scaffold on a narrow task — that comparison only makes sense relative to a published ceiling.

## 5. Caveats / what doesn't transfer

- **Competition style != pretraining.** Kaggle tasks are submission-driven and short (hours-days). JEPA pretraining is multi-day GPU. The gate-design transfers; the operator design (e.g., AIDE's `submission.csv`-centric `improve` operator) does not.
- **Leaderboard calibration is heavy.** MLE-bench had access to Kaggle's full public leaderboard data. AutoJEPA does not have an equivalent crowd-calibrated baseline for JEPA pretraining quality — must rely on published paper numbers, which are sparser and less consistent.
- **Contamination matters here too.** JEPA pretraining objectives are well-documented in public papers; any LLM-proposed mutation may regurgitate published recipes. Worth surfacing in AutoJEPA telemetry as a similar concern.

## 6. Cross-links

- `AIDE.md` — the scaffold half of the headline 16.9% result.
- `Sakana-AI-Scientist-Evaluation.md` — the failure-floor counterpart; MLE-bench measures ceiling, Sakana eval measures floor. AutoJEPA cites both.
- `AlphaEvolve.md` / `CodeEvolve.md` — orthogonal benchmarks (algorithmic discovery, not Kaggle ML engineering); not directly comparable but shape the same "gated decision" intuition.
