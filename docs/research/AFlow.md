# AFlow — Automating Agentic Workflow Generation

- arxiv: https://arxiv.org/abs/2410.10762
- raw: `docs/research/raw/2410.10762/`
- distilled: 2026-05-15

## 1. Thesis

Treat an LLM agent workflow as a **code-represented graph** (nodes = LLM
invocations, edges = data flow) and search the workflow space with
**Monte Carlo Tree Search**, refining candidate workflows from execution
feedback rather than from human edits.

## 2. Method

- **Workflow encoding.** Each candidate is a Python program where each
  LLM-invoking node is a function call to one of a small library of
  operators (Generate, Review, Ensemble, etc.). Edges are the data
  dependencies between calls.
- **Search.** Standard MCTS — UCB1-style selection, expansion via code
  modification (mutate a node, splice subgraphs, swap an operator),
  rollout = run the workflow on a held-out task slice, backprop the
  empirical score.
- **Tree-structured experience.** Cached evaluations from sibling nodes
  inform the value estimate of unexplored branches, cutting the number of
  full rollouts needed.
- **Execution feedback loop.** Failed test cases are re-injected into the
  next code-modification prompt so mutations target observed weaknesses.

## 3. Results

- 5.7% mean absolute improvement over prior auto-workflow baselines across
  six benchmarks (HumanEval, MBPP, MATH, GSM8K, HotpotQA, DROP).
- Workflows that wrap a smaller model can beat GPT-4o on several tasks at
  ~4.55% of GPT-4o's per-token cost, i.e. workflow design dominates
  base-model size in this regime.
- Authors explicitly compare AFlow against single-prompt and prior
  auto-workflow systems (e.g. Aflow's own ablations against GPTSwarm,
  ADAS-style search).

## 4. Why it matters for AutoJEPA

AutoJEPA's outer loop today is a **fixed**, **linear** pipeline:
`propose params → train → eval → keep/discard → maybe propose diff`.
That whole pipeline is itself a workflow. AFlow's reframing — that the
controller graph is a first-class search target — is therefore directly
applicable; we deliberately chose not to take it on for v1.

**Explicit non-goal in v1**, per the writeup §8:

> "Workflow-graph search (AFlow-style) — the outer loop IS a workflow.
> v1 stays linear."

What we would gain by adopting it in a later version:

- **Variant policies.** MCTS over (param-search → diff vs diff → param-search vs param-only), with the JEPA gate criteria as the reward signal, would let us discover that, e.g., diff-first works better for masking-strategy changes while params-first dominates for LR tuning. Today this is a hand-tuned `hybrid` setting.
- **Operator library.** AutoJEPA's policies (`grid`, `random`, `llm`, `llm_diff`, `hybrid`, `learned`) become a first-class operator set that AFlow can recombine. This is closer to RLix-Phase-5 territory than to anything we have today.
- **Cost-aware search.** AFlow's tree-MCTS over execution cost (Basilica $/iter) would let the framework prefer cheap operators early and expensive ones only on the most promising branches.

What we lose by deferring: a tunable controller. The writeup's call is that
the engineering tax (MCTS state, persistent tree, replay determinism across
parallel branches) is too high relative to the marginal gain on the linear
hybrid baseline. Reopen this when (a) the linear hybrid clearly saturates
on Trace-JEPA scale, OR (b) we want to compare three or more controller
variants on the same campaign. Track in `docs/cherry-pick-log.md`.

## 5. Caveats

- AFlow benchmarks are reasoning/coding tasks with clear per-instance
  rewards; a JEPA campaign's reward is `probe_auroc` measured **once per
  iteration** (~30 min on Basilica), not per workflow rollout. MCTS
  rollouts at that cost are at the edge of feasibility.
- The MCTS reward in AFlow is bounded `[0, 1]` per task. AutoJEPA's
  `probe_auroc` is also `[0, 1]` but with a much narrower useful range
  (0.5–0.85). MCTS exploration constants would need recalibration.
- AFlow assumes a fixed, well-defined operator library. AutoJEPA's
  `policy/learned.py` and `policy/learned_search.py` are still maturing —
  treating them as MCTS leaf operators today would freeze a moving target.

## 6. Cross-links

- Architecture writeup §8 — the explicit "v1 stays linear" decision.
- `docs/research/AgentHPO.md` — single-LLM HP-only precedent for the `llm`
  param mode that AFlow generalizes to a workflow-search problem.
- `docs/research/AutoresearchRL-Inheritance-Map.md` — the carry-over plan
  for the linear controller AFlow would replace.
- Sibling: ADAS, AIDE, AlphaEvolve, CodeEvolve (in `docs/research/`) —
  workflow-search precedents covered in the JEPA-family corpus.
