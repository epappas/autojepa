# ADAS — extraction notes

## Core idea
- Move one level up the search hierarchy: a *meta agent* writes the *agent program* itself.
- Agents represented as Python code; the meta agent maintains an archive of past agents and proposes novel combinations of prompts, tools, and workflows.

## Key result
- "Meta Agent Search" agents transfer across domains and across underlying foundation models — i.e. an agent discovered on math improves coding too.

## Why it's parked for AutoJEPA v1
- ADAS is a meta-meta-search: it searches over the search policy that AutoJEPA fixes.
- Per writeup §8: "Meta-meta-search. Its own research project." — the engineering surface area is too large for v1.
- AutoJEPA v1 fixes the outer loop topology (linear hybrid policy with proposer + AST validator + early-stop forecaster) and only varies *content* (mask schedule, EMA decay, probe layer).

## Distinction from FunSearch
- FunSearch searches over candidate programs *for one fixed task*; ADAS searches over candidate *agent designs*. The two compose, but AutoJEPA v1 takes only the FunSearch level.
