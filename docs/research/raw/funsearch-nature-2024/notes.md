# FunSearch — extraction notes

## Mechanism (LLM-proposer + evaluator)
- Evolutionary procedure: pretrained LLM proposes program candidates; an automated evaluator scores them; top-K return to a pool used to seed subsequent prompts.
- Operates in *program space*: searches for `priority(...)` style functions that describe *how* to solve the problem (interpretable), not raw solutions.
- Uses an islands-based population to maintain diversity and prevent stagnation.
- Underlying LLM in the original paper: PaLM 2 (Codey).

## Empirical results
- Cap set problem: largest improvement on cap-set lower bounds in ~20 years (n=8 finite case; new asymptotic lower bound).
- Online bin packing: discovered heuristics that beat First-Fit and Best-Fit on standard distributions, with fewer bins used.
- DeepMind blog (2024 follow-up) extended to combinatorial competitive-programming tasks.

## What is load-bearing for AutoJEPA
- The two-component pattern (creative LLM proposer + deterministic evaluator that "guards against hallucinations") is precisely the autoresearch loop AutoJEPA inherits from autoresearch-rl.
- The interpretability argument (programs > raw solutions) maps to AutoJEPA emitting JEPA training-recipe deltas (mask schedules, EMA constants, probe layers) rather than weight diffs.

## Citation hygiene
- Verbatim abstract pulled from NSF Public Access mirror (par.nsf.gov/biblio/10499230). Nature URL redirects to authentication; mirror text is the published abstract.
- Authors and venue cross-checked against PubMed (PMID 38096900) and Nature volume 625 (7995):468-475.
