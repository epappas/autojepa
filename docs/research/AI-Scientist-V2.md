# AI Scientist v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search

**Citation:** Yamada, Y., Lange, R. T., Lu, C., Hu, S., Lu, C., Foerster, J., Clune, J., Ha, D. *The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search.* arXiv:2504.08066, 2025.
**Raw material:** `docs/research/raw/2504.08066/`
**AutoJEPA classification:** Parked / explicit non-goal

---

## 1. One-line thesis

Replace v1's hand-authored code templates with a progressive agentic tree search managed by a dedicated experiment-manager sub-agent, and the loop produces a workshop-level paper that crosses the average human acceptance threshold at an ICLR workshop.

## 2. Method

- **End-to-end pipeline:** hypothesize -> design -> execute -> analyze -> visualize -> draft, all autonomously.
- **Progressive agentic tree search:** experiments are nodes in a tree; a manager agent selects which branch to expand next based on intermediate results. Replaces v1's linear template-driven flow.
- **No code templates:** v2 generates experiment scaffolding from scratch per hypothesis, generalizing across ML domains (where v1 was bolted to specific recipes).
- **VLM-in-the-loop reviewer:** a Vision-Language Model inspects generated figures and feeds revisions back to the writer agent for content + aesthetic polish.

## 3. Results

- 3 fully autonomous manuscripts submitted to a peer-reviewed ICLR 2025 workshop.
- 1 of 3 scored above the average human acceptance threshold — first AI-only paper accepted at a peer-reviewed venue.
- Code open-sourced at https://github.com/SakanaAI/AI-Scientist-v2.
- The paper does not publish a head-to-head success-rate vs. v1 on a controlled benchmark; the headline result is a *single* workshop acceptance.

## 4. Why it matters for AutoJEPA — Parked

Per writeup §8, AutoJEPA v1 explicitly does not adopt the agentic tree search. Reasons:

- **Engineering surface area.** Tree search needs a frontier queue, branch-budget accounting, an operator-selection policy (debug vs. draft vs. improve), and rollback semantics for failed branches. AutoJEPA v1 keeps the linear hybrid policy from `autoresearch-rl` — proposer emits one candidate per iteration, validator gates it, target executes, keep/discard, repeat.
- **JEPA scope is narrower.** v2's selling point is *generalization across ML domains*. AutoJEPA fixes the domain (JEPA pretraining) and so doesn't need a generalist tree-search experiment manager — it needs JEPA-specific primitives (collapse-guard, probe-eval, mask-scheduler).
- **The validator argument.** v2 inherits v1's coding-trust problem (see `Sakana-AI-Scientist-Evaluation.md`). v2 does not introduce a static AST-diff validator; it relies on tree search to discard branches that fail at runtime, which is expensive on JEPA-scale GPU jobs.

The v2 manager-agent / VLM-reviewer split is the natural v2-or-v3 upgrade for AutoJEPA, but only after JEPA-specific primitives are validated standalone.

## 5. Caveats / what doesn't transfer

- **Sample size of one.** A single workshop acceptance is anecdote, not benchmark. The 42% coding-error rate documented for v1 has no published refutation for v2.
- **Workshop != main conference.** Workshop review thresholds are looser; the result does not establish that v2 produces main-conference-quality work.
- **Compute opacity.** The paper does not publish per-paper LLM token cost or wall-clock; for AutoJEPA's small-team economics this matters.
- **Research domain.** v2's evals are full ML research projects (~weeks of human work); AutoJEPA's unit of work is a single JEPA pretraining run (~hours-days of GPU). The orchestration overhead amortizes very differently.

## 6. Cross-links

- `Sakana-AI-Scientist-Evaluation.md` — independent eval of v1; the floor that v2's tree search must beat (no published replication yet).
- `AIDE.md` — same tree-search-over-code primitive, narrower scope (ML engineering, not full research).
- `ADAS.md` — Hu, Lu, Clune (overlapping authors) explored meta-agent search; v2's experiment-manager agent is in the same lineage.
- `AlphaEvolve.md` — alternate "agentic LLM pipeline" pattern with stronger emphasis on the proposer/critic split rather than tree search.
