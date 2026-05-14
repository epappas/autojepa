# AI Scientist v2 — extraction notes

## Pattern
- Outer agentic loop: hypothesize -> design -> execute -> analyze -> draft.
- Replaces v1's human-authored code templates with a *progressive agentic tree search* managed by an experiment-manager sub-agent.
- VLM-in-the-loop feedback for figure refinement (content + aesthetics).

## Result that everyone cites
- 1 of 3 fully autonomous submissions at an ICLR 2025 workshop scored above the average human acceptance threshold — first AI-only paper accepted at a peer-reviewed venue.

## Caveat — extremely load-bearing for AutoJEPA
- "Peer-review-accepted workshop paper" is not the same as "validated science."
- The companion evaluation paper (arXiv 2502.14297) reports 42% experiment failure due to coding errors in the v1 system; v2 inherits the same coding-trust problem.
- This is exactly why AutoJEPA insists on an AST-diff validator before any candidate touches GPU.

## Relation to AutoJEPA tree-search
- v2's progressive tree search is a strict superset of AutoJEPA v1's linear hybrid policy. AutoJEPA explicitly parks tree search per writeup §8 to keep v1 small.
