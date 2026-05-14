# CodeEvolve — extraction notes

## What it is
- Open-source clone / re-implementation of AlphaEvolve.
- Islands-based GA (FunSearch lineage) plus modular LLM orchestration.
- Adaptive meta-prompting; context-aware recombination across islands.

## Why it matters as a reference
- Demonstrates that the AlphaEvolve pattern reproduces with open-weight models at a fraction of closed-source compute, validating the *architecture* not just the proprietary stack.
- Provides public benchmarks and ablations AutoJEPA can compare against without buying Gemini API credits.

## Why parked / non-goal for AutoJEPA v1
- Same reason as AlphaEvolve: islands + multi-LLM orchestration is heavier than the linear hybrid AutoJEPA v1 inherits from autoresearch-rl.
- AutoJEPA may import the evaluation harness and benchmark protocol but does not adopt the islands topology in v1.
