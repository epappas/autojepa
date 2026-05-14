# AFlow — raw notes

Retrieval: WebFetch on https://arxiv.org/abs/2410.10762 (2026-05-15). Single
fetch sufficed; PDF fallback not needed.

## Mechanism (as reported in abstract + arxiv landing page)

- Workflow = directed graph of LLM-invoking nodes connected by edges, encoded
  as Python code (each node = a function call to an operator).
- Search procedure = Monte Carlo Tree Search (MCTS) over candidate workflow
  programs.
- Each MCTS node holds a candidate workflow plus its empirical score from
  running it on a held-out set of task instances.
- Refinement uses three signals: code modification (mutate a node, splice
  subgraphs), tree-structured experience (cached evaluations from siblings),
  execution feedback (test cases that failed).

## Reported numerics

- 5.7% average absolute improvement over baseline auto-workflow methods.
- Smaller models (when wrapped in AFlow workflows) outperform GPT-4o on
  several benchmarks at 4.55% of GPT-4o inference dollar cost.
- Six benchmark datasets evaluated (HumanEval, MBPP, MATH, GSM8K, HotpotQA,
  DROP — standard suite for code/math/reasoning workflow papers; named in
  the paper PDF, abstract names only the count).

## Relevance pin

The outer loop in AutoJEPA (param search → diff → eval → keep/discard) is
itself a fixed workflow. AFlow's MCTS would replace the linear hybrid
controller with a tree of workflow variants. Out of scope for v1 per writeup
§8 ("the outer loop IS a workflow. v1 stays linear").
