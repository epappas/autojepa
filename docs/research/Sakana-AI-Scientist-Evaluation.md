# Evaluating Sakana's AI Scientist: Bold Claims, Mixed Results, and a Promising Future?

**Citation:** Beel, J., Kan, M.-Y., Baumgart, M. *Evaluating Sakana's AI Scientist: Bold Claims, Mixed Results, and a Promising Future?* arXiv:2502.14297, 2025. SIGIR Forum 2025.
**Raw material:** `docs/research/raw/2502.14297/`
**AutoJEPA classification:** Falsification reference

---

## 1. One-line thesis

Run an independent evaluation of Sakana's AI Scientist v1 and the headline numbers are: 42% of experiments fail with coding errors, novelty assessments routinely flag established techniques as novel, code edits are timid (+8% characters per iteration), citations are sparse and outdated, and yet the system still produces full manuscripts for $6-15 in 3.5 hours of human time — undergraduate-quality science at unprecedented throughput.

## 2. Method

- Independent reproduction of Sakana's AI Scientist v1 (Lu et al. 2024, arXiv:2408.06292).
- Evaluation across four dimensions: literature reviews, experiment execution, code modifications, manuscript generation.
- Quantitative metrics:
  - Experiment failure rate (run completion vs. crash).
  - Code-edit magnitude (character delta per iteration).
  - Citation count and recency.
  - Structural manuscript errors (placeholder text, missing figures, repeated sections, hallucinated numbers).
- Cost / wall-clock measurement per generated paper.

## 3. Results

The numbers worth memorizing:

| Metric | Value |
|--------|-------|
| Experiment failures from coding errors | **42%** |
| Avg. code-edit growth per iteration | +8% characters |
| Median citations per paper | 5 |
| Citations from 2020 or later (out of 34) | 5 |
| Cost per generated paper | USD 6-15 |
| Human time per paper | 3.5 hours |

Other findings:

- Novelty checker classifies established concepts (e.g., micro-batching for SGD) as novel.
- Manuscripts contain placeholder text such as "Conclusions Here" that survives to submission.
- Some papers contain hallucinated numerical results.
- Despite all of the above, output is plausible enough that many reviewers may not distinguish it from a rushed undergraduate paper.

## 4. Why it matters for AutoJEPA — Falsification reference

This is *the* citation behind AutoJEPA's validator argument.

The AutoJEPA loop inherits the autoresearch pattern (proposer + target + keep/discard). Without a code-level guard, the empirical floor for that pattern, when the proposer is mutating ML code, is:

> 42% of attempted experiments crash before producing any signal.

That floor is not a JEPA-specific number, it is a *coding-agent-on-ML-code* number, and JEPA-specific code is at least as fragile as the recipes Sakana attempted (mask schedulers, EMA controllers, target-encoder bookkeeping all have non-obvious correctness conditions).

This is exactly why AutoJEPA v1 imports the AST-diff validator from `autoresearch-rl` and runs it *before* any candidate touches GPU:

- The validator catches a large slice of the 42% (broken edits, structural breakage) at zero GPU cost.
- The early-stop forecaster (also inherited) catches a further slice (numeric divergence within the first epochs).
- The collapse-guard (new in AutoJEPA) catches the JEPA-specific failure mode where loss decreases while representations degenerate.

Without this paper as a citable falsification, "we need a validator" reads as paranoid engineering taste. With it, the validator is a measured response to a 42% empirical floor.

## 5. Caveats / what doesn't transfer

- **v1, not v2.** The 42% number is from AI Scientist v1. AI Scientist v2 (`AI-Scientist-V2.md`) introduced agentic tree search; no published independent replication of the failure rate exists. The most defensible AutoJEPA framing is "the v1 floor was 42%; no published evidence that v2 is meaningfully lower; the validator costs ~zero so adopt it."
- **Sakana's framework chose specific recipes.** A different proposer prompt or a stronger underlying LLM would shift the number, but probably not eliminate it.
- **The 8% edit-growth statistic** is noted but less load-bearing for AutoJEPA — it informs the proposer prompt design (encourage larger structured deltas) more than the architecture.

## 6. Cross-links

- `AI-Scientist-V2.md` — the system this paper evaluated v1 of; v2's tree search has not been independently re-evaluated against the 42% floor.
- `FunSearch.md` — the proposer + evaluator pattern works precisely because the evaluator is tight; this paper documents what happens when the evaluator is loose.
- `MLE-Bench.md` — orthogonal evaluation; MLE-bench measures *capability ceiling*, this paper measures *failure floor*. AutoJEPA needs both.
