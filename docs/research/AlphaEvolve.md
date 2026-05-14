# AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery

**Citation:** Novikov, A., Vu, N., Eisenberger, M., Dupont, E., Huang, P.-S., Wagner, A. Z., Shirobokov, S., Kozlovskii, B., Ruiz, F. J. R., Mehrabian, A., Kumar, M. P., See, A., Chaudhuri, S., Holland, G., Davies, A., Nowozin, S., Kohli, P., Balog, M. *AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery.* arXiv:2506.13131, 2025. Google DeepMind white paper.
**Raw material:** `docs/research/raw/2506.13131/`
**AutoJEPA classification:** Parked / explicit non-goal

---

## 1. One-line thesis

Take FunSearch's proposer + evaluator pattern, swap in a *pipeline* of LLMs (cheap fast proposer + slow strong critic), let it edit full code files (not single functions), point it at production codebases, and the system both ships infrastructure improvements (Google datacenter scheduler, TPU circuits, training of its own underlying LLM) and breaks long-standing algorithmic records (4x4 complex matmul in 48 multiplications — first improvement on Strassen since 1969).

## 2. Method

- **Multi-LLM pipeline.** A proposer LLM (e.g. Gemini Flash) generates candidate edits cheaply; a critic/reviser LLM (e.g. Gemini Pro) reviews and refines. Distinct prompts, distinct cost tiers.
- **Evolutionary database.** Programs are stored with full lineage; recombination samples parents from past generations.
- **Multiple evaluators.** Composite scoring across correctness checks, performance benchmarks, and domain-specific signals.
- **Whole-codebase scope.** Unlike FunSearch (single function), AlphaEvolve mutates large source files, sometimes across files.
- **Continuous evaluator feedback.** Each candidate is run through the evaluator pipeline and the score becomes the next-generation selection signal.

## 3. Results

Selected highlights from the white paper:

- **4x4 complex matrix multiplication in 48 scalar multiplications** — first improvement on Strassen's 1969 algorithm in this setting.
- **Datacenter scheduler:** more efficient scheduling algorithm deployed in Google's production stack.
- **TPU circuit design:** found a functionally equivalent simplification in hardware accelerator circuits.
- **Self-bootstrapping:** accelerated training of the LLM underpinning AlphaEvolve itself.
- Multiple new bounds on long-standing combinatorial / mathematical problems.

## 4. Why it matters for AutoJEPA — Parked

Per writeup §8:

> "Proposer + critic is strictly better, but doubles LLM cost."

The proposer/critic split is genuinely better — the critic catches semantic errors the proposer misses, and the cheap-proposer / expensive-critic asymmetry is economically reasonable. AutoJEPA v1 still parks it for two reasons:

1. **Cost doubling.** Two LLM contracts, two retry budgets, two rate-limit policies. For a single-GPU JEPA loop with a 5-minute outer-iteration budget, that overhead is structural, not marginal.
2. **The validator already plays critic.** AutoJEPA's deterministic AST-diff validator catches the bulk of what a critic LLM would flag (broken edits, unsafe mutations, type drift) at zero LLM cost. The remaining critic-class errors (semantic regressions that compile fine) are intercepted by the early-stop forecaster.

The critic-LLM upgrade is the natural v2 path once the AST validator's recall ceiling is empirically measured against a strong-model baseline.

The deeper architectural notes from AlphaEvolve that *do* transfer conceptually to AutoJEPA:

- **Composite multi-evaluator.** AutoJEPA's keep/discard already reads multiple metrics (val_bpb, eval_score, probe accuracy, collapse signals). The AlphaEvolve white paper formalizes this; AutoJEPA v1 keeps the simpler single-objective gating but the multi-evaluator pattern is the v2 upgrade.
- **Lineage tracking.** AutoJEPA already records candidate provenance via the experiment tracker.

## 5. Caveats / what doesn't transfer

- **Closed model dependence.** AlphaEvolve runs on Gemini; replicating exact numbers needs DeepMind compute. `CodeEvolve.md` shows the architecture works with open weights.
- **Whole-codebase mutations.** AlphaEvolve mutates large code surfaces. AutoJEPA v1 mutates typed config deltas — a much narrower surface, intentionally.
- **Benchmarks are bespoke.** Most headline results are on internal Google infrastructure; not a reproducible public benchmark.

## 6. Cross-links

- `FunSearch.md` — direct predecessor, overlapping author list (Novikov, Balog, Dupont, Ruiz, Kohli). AlphaEvolve generalizes from single functions to full codebases and from one LLM to a pipeline.
- `CodeEvolve.md` — open-source reproduction of the AlphaEvolve pattern; uses islands GA + open-weight LLMs.
- `AIDE.md` — alternate scaling axis: tree search instead of multi-LLM pipeline.
- `ADAS.md` — alternate scaling axis: meta-agent instead of fixed pipeline.
