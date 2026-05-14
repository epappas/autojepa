# AIDE: AI-Driven Exploration in the Space of Code

**Citation:** Jiang, Z., Schmidt, D., Srikanth, D., Xu, D., Kaplan, I., Jacenko, D., Wu, Y. *AIDE: AI-Driven Exploration in the Space of Code.* arXiv:2502.13138, 2025. WecoAI tech report.
**Raw material:** `docs/research/raw/2502.13138/`
**AutoJEPA classification:** Parked / explicit non-goal

---

## 1. One-line thesis

Frame ML engineering as code optimization, instantiate trial-and-error as tree search over candidate programs with `draft / debug / improve` operators, and a single LLM scaffold reaches SOTA on MLE-bench, RE-Bench, and internal Kaggle suites.

## 2. Method

- **Node = candidate solution program.** Each node carries the program text, the validation score it achieved, and a debug history.
- **Edge = LLM-proposed edit.** Three operator types:
  - `draft`: produce a new sketch from the task description.
  - `debug`: given a node that crashed, propose a fix.
  - `improve`: given a working node, propose a mutation aimed at higher score.
- **Selection:** greedy on validation metric; promising nodes are expanded preferentially. The search trades compute for coverage of the design space.
- **No human-authored task templates** — task description goes straight to the proposer.

## 3. Results

- **OpenAI MLE-bench:** with o1-preview as the underlying model, AIDE scaffolding reached at least Kaggle bronze in 16.9% of competitions — best published scaffold at release.
- **METR RE-Bench:** SOTA at release.
- **Internal Kaggle evaluations** also reported as SOTA.

## 4. Why it matters for AutoJEPA — Parked

Per writeup §8, AIDE-style tree search is on the explicit park list:

> "Better than linear hybrid for wide design space, but real engineering tax."

AIDE's tree search dominates a linear/random search whenever the candidate space is wide and most candidates fail. JEPA pretraining is the *opposite* setting in v1: the candidate space is narrow (mask schedules, EMA decay, probe layers), most candidates execute (post-validator), and each one costs hours of GPU. Greedy expansion of a tree of failed branches doesn't pay for itself.

AutoJEPA v1 keeps:
- The **linear hybrid policy** (random + learned PPO) inherited from `autoresearch-rl`.
- The **AST-diff validator**, which subsumes AIDE's `debug` operator at zero GPU cost — code that doesn't parse never reaches `target.run()`.
- The **early-stop forecaster**, which serves a similar role to AIDE's prune-bad-branches behaviour but on time, not topology.

AIDE is the natural v2 upgrade once: (a) the JEPA-specific primitives are stable, and (b) the candidate space widens to architectural variations beyond simple hyperparameter deltas.

## 5. Caveats / what doesn't transfer

- **Compute regime mismatch.** AIDE's per-node cost is bounded by Kaggle dataset sizes (minutes-to-hour). JEPA pretraining is hours-to-days. Tree search amortizes badly.
- **`debug` operator vs. validator.** AIDE's runtime debug loop is reactive (run, crash, fix). AutoJEPA's AST validator is preventive — cheaper, but narrower (it catches syntactic and structural bugs, not numeric divergence).
- **Benchmark dependence.** MLE-bench and RE-Bench reward Kaggle-style submission scripts. JEPA training is upstream of that benchmark — AIDE's specific operator design (heavy `improve` weight on `submission.csv`) doesn't map cleanly.

## 6. Cross-links

- `MLE-Bench.md` — the benchmark AIDE scored SOTA on; the gate-style scoring is the comparison protocol AutoJEPA's promotion tracker mirrors.
- `AI-Scientist-V2.md` — also tree search, but over experiment plans rather than code edits.
- `FunSearch.md` — same proposer + evaluator skeleton, evolutionary islands instead of tree search.
- `ADAS.md` — orthogonal: ADAS would learn the AIDE scaffold itself.
