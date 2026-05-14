# ADAS: Automated Design of Agentic Systems

**Citation:** Hu, S., Lu, C., Clune, J. *Automated Design of Agentic Systems.* arXiv:2408.08435, 2024 (ICLR 2025).
**Raw material:** `docs/research/raw/2408.08435/`
**AutoJEPA classification:** Parked / explicit non-goal

---

## 1. One-line thesis

Hand-designed agentic systems (Chain-of-Thought, Self-Reflection, Toolformer) will be replaced by *learned* agentic systems — let a meta agent program new agents in code, score them on downstream tasks, and grow an archive that beats hand-crafted scaffolds.

## 2. Method

- **Representation:** every agent is a Python program with a fixed `forward(task)` interface.
- **Meta Agent Search:** a meta agent (an LLM with the archive in context) iteratively writes new agent programs. Each new agent is scored on a benchmark suite; high-scoring agents are added to the archive and become exemplars in future meta-prompts.
- **Search space includes:** novel prompts, tool-use orderings, workflow graphs, ensembling, self-critique loops — anything expressible as Python.
- **Turing-complete argument:** because the substrate is code, the search space contains every possible agentic system.

## 3. Results

- Across coding (HumanEval), math (GSM8K, MGSM), and science (DROP, MMLU), ADAS-discovered agents outperformed hand-designed scaffolds.
- **Transfer:** agents discovered on one domain or one base model retained their advantage when moved to a different domain or different base model — the learned scaffolds generalize, which is the headline scientific claim.

## 4. Why it matters for AutoJEPA — Parked

Per writeup §8: ADAS is the explicit "meta-meta-search" non-goal. AutoJEPA v1 fixes the *agent program* (the autoresearch loop: proposer + validator + target + keep/discard + forecaster) and only varies *content* (JEPA training-recipe deltas).

ADAS sits one level above AutoJEPA: it would let a meta agent rewrite the autoresearch loop itself. That is a research project of its own and a separate codebase concern. AutoJEPA v1's hypothesis is that the existing autoresearch loop is already adequate when specialized to JEPA — the JEPA-specific primitives (collapse-guard, probe-eval, mask-scheduler) buy more than meta-agent rewrites of the loop topology.

The ADAS pattern is a credible v3+ path: once AutoJEPA accumulates enough run history, a meta agent could propose modifications to the keep/discard rule, the early-stop forecaster, or the mask-scheduler primitive. v1 does not attempt this.

## 5. Caveats / what doesn't transfer

- **Benchmarks are small/cheap.** ADAS evaluates on QA / code / math benchmarks where each agent run is seconds-to-minutes. JEPA pretraining runs are hours-to-days; the meta-search loop becomes prohibitive without ruthless early-stop and surrogate scoring.
- **Transfer claim is on similar tasks.** Cross-domain transfer in ADAS is across QA/coding/math — all token-level reasoning. There is no evidence the same scaffolds transfer to *training*-style tasks like JEPA.
- **Safety surface.** A meta agent that rewrites the autoresearch loop can rewrite the validator. Without a meta-validator, the 42% Sakana failure mode (see `Sakana-AI-Scientist-Evaluation.md`) propagates upward.

## 6. Cross-links

- `AI-Scientist-V2.md` — Hu, Lu, Clune are co-authors; v2's experiment-manager agent applies ADAS-style meta-design within a single research session.
- `FunSearch.md` — ADAS searches *over agent programs*, FunSearch searches *over solution programs*; AutoJEPA v1 is a FunSearch-level instantiation, not an ADAS-level one.
- `AIDE.md` — fixed scaffold, learned policy. The hand-designed AIDE scaffold won MLE-bench at the time, suggesting ADAS-style replacement is not strictly required for SOTA.
