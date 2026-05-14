# FunSearch: Mathematical Discoveries from Program Search with Large Language Models

**Citation:** Romera-Paredes, B., Barekatain, M., Novikov, A., Balog, M., Kumar, M. P., Dupont, E., Ruiz, F. J. R., Ellenberg, J. S., Wang, P., Fawzi, O., Kohli, P., Fawzi, A. *Mathematical discoveries from program search with large language models.* Nature 625, 468-475 (2024). DOI: 10.1038/s41586-023-06924-6.
**Raw material:** `docs/research/raw/funsearch-nature-2024/`
**AutoJEPA classification:** Inherited

---

## 1. One-line thesis

Pair a pretrained LLM proposer with a deterministic evaluator over candidate programs, evolve the population on score, and you get a method that surpasses humans on open mathematical problems without solving the hallucination problem in the LLM itself.

## 2. Method

- **Search target:** programs (Python functions like `priority(...)`), not raw solutions. The LLM proposes; the evaluator runs the program against a fixed scoring procedure.
- **Population:** islands-based pool, each island holds programs clustered by score; periodic reset of the worst islands to maintain diversity.
- **Loop:** sample k high-scoring programs from an island, format them into the prompt as in-context exemplars, ask the LLM for a new variant, run the evaluator, insert into the appropriate island bucket.
- **Underlying LLM:** Codey/PaLM 2 in the original paper; the architecture is model-agnostic.
- **Engineering knobs:** template prompt (problem statement + 1-3 exemplars), per-program timeout, parallel sampling, scoring is a single deterministic function.

## 3. Results

- **Cap set problem (extremal combinatorics):** discovered new constructions of large cap sets going beyond the best-known ones, in both finite-dimensional and asymptotic cases — the largest improvement on cap-set lower bounds in roughly two decades.
- **Online bin packing:** found heuristics that pack the same items into fewer bins than First-Fit / Best-Fit on standard distributions.
- The 2024 follow-up extended the same scaffold to combinatorial competitive-programming tasks, beating top-percentile humans on several.

## 4. Why it matters for AutoJEPA — Inherited

This is the *foundational architectural pattern* AutoJEPA already runs.

The `autoresearch-rl` continuous loop AutoJEPA forks from is structurally a FunSearch instance: a `Policy.propose()` (the proposer) emits hyperparameter / config candidates, a `Target.run()` (the evaluator) executes the candidate end-to-end on real GPU and reports a score, and a `keep/discard` rule maintains the working population. The AST-diff validator AutoJEPA inherits is FunSearch's "automated evaluator that guards against hallucinations" applied at the *code-mutation* layer instead of the *solution-scoring* layer.

What AutoJEPA *does not* take from FunSearch:

- **Islands topology.** AutoJEPA v1 keeps the linear hybrid (random + learned PPO) policy; islands move to v2.
- **Program-space search per se.** The autoresearch loop currently mutates training-recipe deltas (mask schedule entries, EMA decay, probe-layer index), not arbitrary program text. The interpretability argument from FunSearch — that programs are easier to inspect than weights — still applies because the candidates AutoJEPA emits are typed config diffs, not weight tensors.

Per writeup §3 (Lineage), FunSearch is the head of the lineage. Every paper in this corpus is a delta on the FunSearch pattern; AutoJEPA is, too.

## 5. Caveats / what doesn't transfer

- **Single-task assumption.** FunSearch fixes one objective (cap-set count, bin-pack ratio). JEPA pretraining has at least three signals AutoJEPA must combine (val loss, probe accuracy, collapse metric) — the multi-objective Pareto-front approach is explicitly parked per writeup §8.
- **No collapse mode.** FunSearch's evaluator returns a clean score; no equivalent of representation collapse can mislead it. AutoJEPA needs the collapse-guard primitive precisely because the JEPA loss can decrease while representations degenerate.
- **Compute model.** FunSearch evaluations are millisecond-to-second function calls; AutoJEPA evaluations are minute-to-hour GPU runs, which is what motivates the early-stop forecaster (also inherited from `autoresearch-rl`).

## 6. Cross-links

- `AlphaEvolve.md` — direct successor; same authors (Novikov, Balog, Dupont, Ruiz, Kohli) extended FunSearch to multi-LLM, full-codebase mutations.
- `CodeEvolve.md` — open-source reproduction of the AlphaEvolve extension; uses FunSearch's islands.
- `AIDE.md` — substitutes tree search for FunSearch's evolutionary islands.
- `ADAS.md` — substitutes a meta-agent for FunSearch's fixed proposer prompt.
- `Sakana-AI-Scientist-Evaluation.md` — empirical floor for what happens to a FunSearch-style loop *without* a code-level validator.
