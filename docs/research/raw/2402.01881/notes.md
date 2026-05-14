# AgentHPO — raw notes

Retrieval: WebFetch on https://arxiv.org/abs/2402.01881 (2026-05-15). Single
fetch sufficed.

## Mechanism (as reported in abstract + arxiv landing page)

- Two-agent design: a **Creator** agent reads the task description and
  proposes a hyperparameter configuration; an **Executor** agent runs the
  trial, parses logs, and reports back to the Creator.
- The Creator's prompt includes the full history of prior (config → metric)
  pairs plus a natural-language rationale field, so each new proposal is
  conditioned on history and on the agent's own reasoning trace.
- Loop terminates on a fixed budget (number of trials) or when the Creator
  declares convergence.

## Reported numerics

- 12 ML tasks across vision, NLP, tabular, RL.
- Matches or surpasses best-of-human-trial baseline on most tasks.
- "Significantly fewer trials" than Bayesian / random / grid baselines (paper
  body has the per-task tables; abstract gives the qualitative claim).
- Provides natural-language explanations for each proposed configuration —
  interpretability win over Bayesian-opt black box.

## Relevance pin

Direct precedent for AutoJEPA's `policy.type=llm` mode: an LLM proposes
hyperparameter configs from the experiment history, and the proposals are
fed back into the loop. AutoJEPA's `LLMParamPolicy` (inherited from
autoresearch-rl) is functionally a single-agent variant of AgentHPO with the
Creator/Executor split collapsed into one chat call per proposal batch
(`propose_batch` for parallel mode).
