# AIDE — extraction notes

## Mechanism
- ML engineering recast as code optimization. Each node = a candidate solution program; edges = LLM-proposed edits (debug, draft, improve).
- Tree search over candidates; greedy selection on a held-out validation metric; promising nodes are expanded preferentially.
- Operator types: draft (new sketch), debug (fix exception), improve (mutate working solution).

## Benchmarks
- Kaggle internal evals.
- OpenAI MLE-bench: with o1-preview as the underlying model, AIDE achieves bronze medal in 16.9% of competitions (best published scaffold at the time).
- METR RE-Bench.

## Why parked for AutoJEPA v1
- Tree search with debug/draft/improve operators yields better wide-search behaviour, but requires a queue/priority heap, branch budgeting, and operator-selection policy.
- Per writeup §8: "Better than linear hybrid for wide design space, but real engineering tax." AutoJEPA v1 keeps the linear hybrid (random + learned PPO) inherited from autoresearch-rl.
- AIDE is the natural v2 upgrade path; v1 saves it for after the JEPA-specific primitives (collapse-guard, probe-eval, mask-scheduler) are validated.
