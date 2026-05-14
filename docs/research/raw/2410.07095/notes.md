# MLE-bench — extraction notes

## What it is
- 75 curated Kaggle competitions; each has a held-out test set, a held-out submission scorer, and a calibrated human leaderboard.
- Agents must produce a Kaggle-style `submission.csv`; score is then bucketed against the leaderboard into bronze/silver/gold medal cutoffs.

## Gate-style criteria
- Pass/fail thresholds are *gate-style*: bronze medal = top 40%/30%/10% of leaderboard depending on participant count.
- This mirrors AutoJEPA's keep/discard decision gates (val_bpb improvement threshold, probe accuracy floor) — both reduce a noisy continuous metric to a calibrated binary.

## Headline number
- o1-preview + AIDE scaffolding: bronze medal in 16.9% of competitions (best published config at release).
- Resource scaling experiments: more attempts and more compute help; pre-training contamination measurably inflates some scores.

## Why this is the comparison benchmark for AutoJEPA
- Not used as the AutoJEPA training task — JEPA pretraining is upstream of competition-style ML engineering — but the *evaluation philosophy* (gated decisions, leaderboard-calibrated thresholds) is what AutoJEPA's promotion tracker mirrors.
- Useful as the "are coding agents getting better at ML engineering?" yardstick when arguing AutoJEPA should pick up newer underlying LLMs.
