# AgentHPO — LLM Agent for Hyper-Parameter Optimization

- arxiv: https://arxiv.org/abs/2402.01881
- raw: `docs/research/raw/2402.01881/`
- distilled: 2026-05-15

## 1. Thesis

Replace Bayesian-optimization or grid/random HPO with an **LLM agent** that
reads the task description, the search space, and the running history of
(config → metric) pairs, and proposes the next configuration in
natural-language-grounded reasoning steps.

## 2. Method

- **Two agents, one loop.**
  - **Creator** reads task spec + history + a free-form rationale field;
    emits the next HP configuration as JSON plus an explanation of *why*.
  - **Executor** runs the trial, parses logs, returns metrics + run
    artifacts to the Creator.
- **History as prompt context.** Every prior (config, metric, rationale)
  is appended to the Creator's prompt; reasoning improves as the budget
  is consumed.
- **Termination.** Fixed trial budget OR Creator declares convergence
  (no proposed HP differs meaningfully from prior best).
- **Comparators.** Random search, Bayesian (Optuna / hyperopt), and the
  best human-tuned configuration from the original paper for each task.

## 3. Results

- 12 representative ML tasks (vision, NLP, tabular, light RL).
- Matches or beats the best human-tuned configuration on most tasks while
  using "significantly fewer trials" than Bayesian baselines.
- Provides natural-language rationales for each configuration — a strict
  interpretability gain over Bayesian-opt black box.
- Failure modes: tasks with extreme search-space dimensionality (>20
  continuous dims) lose the LLM's advantage; the agent regresses toward
  random.

## 4. Why it matters for AutoJEPA

AgentHPO is the **direct precedent** for AutoJEPA's `policy.type=llm`
mode and the `param-mode` half of the hybrid policy. The architecture is
the same shape, with one collapse:

| Concept | AgentHPO | AutoJEPA / autoresearch-rl |
|---|---|---|
| Creator agent | Separate LLM call | Folded into `LLMParamPolicy.propose` (single chat call) |
| Executor agent | Separate LLM call | Replaced by `TargetAdapter` (deterministic Python; no LLM) |
| History prompt | Full (config, metric, rationale) trail | `policy/llm_context.py::summarize_history()` returns the same triple |
| Convergence stop | Creator decides | `controller.no_improve_limit` fires deterministically |
| Multiple proposals | Sequential | `propose_batch(k)` — one call asks for `k` diverse proposals (Phase 4) |

Cite as background for:

- The justification that the `llm` param mode in `policy/llm_search.py` is
  not a one-off heuristic but a published-method pattern with a 12-task
  empirical floor.
- The design of `policy/_prompt_fragments.py::PROGRESS_PROTOCOL_RULES` —
  AgentHPO's rationale-required prompt is the closest published analog.
- The hybrid-mode rationale: AgentHPO works for HP only; for **algorithmic**
  improvements the LLM has to write code, which is the `llm_diff` mode
  AutoJEPA stacks on top of AgentHPO-style param search.

What AutoJEPA does **not** import from AgentHPO:

- The Creator/Executor split. AutoJEPA's Executor is a `TargetAdapter`
  (Basilica / command / http) — never an LLM call. This is a security
  boundary: the LLM never executes code, only proposes it.
- Per-trial natural-language rationale exposed to humans. We log the LLM
  response in `traces/events.jsonl` but do not surface it as the primary
  artifact; the primary artifact is the (config, metric) pair.

## 5. Caveats

- AgentHPO's tasks are seconds-to-minutes per trial. AutoJEPA Basilica
  iterations are 5–30 min each. The "few-shot improves quickly" result
  may not transfer at our trial cost; recalibration of trial budgets
  needed and is open work for the Phase-2 CIFAR campaign.
- AgentHPO's continuous search-space ceiling (~20 dims) is very close to
  AutoJEPA's planned 10–12-dim JEPA HP space. We are at the edge of where
  the LLM advantage is reported to disappear; the framework `kill criterion`
  in writeup §Phase-2 is partly the hedge against this.
- Single-author replication of AgentHPO results is sparse in the
  literature; treat the absolute numbers as suggestive, not definitive.
  The pattern is sound; the ranking vs Bayesian is workload-dependent.

## 6. Cross-links

- `docs/research/AFlow.md` — generalization of single-LLM HPO to
  workflow-graph search.
- `docs/research/AutoresearchRL-Inheritance-Map.md` — `policy/llm_search.py`
  and `policy/_prompt_fragments.py` rows.
- Architecture writeup §6 (`What we adapt`) — `program.md` template
  encoding and 10–12-dim JEPA parameter space inherit AgentHPO's
  rationale pattern.
- Sibling FunSearch / AI Scientist v2 entries in `docs/research/` —
  earlier and broader LLM-as-researcher precedents.
