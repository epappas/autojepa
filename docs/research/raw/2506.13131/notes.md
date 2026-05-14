# AlphaEvolve — extraction notes

## Mechanism
- Successor to FunSearch from many of the same authors (Novikov, Balog, Dupont, Ruiz, Kohli).
- Orchestrates a *pipeline of LLMs* — typically a fast-cheap proposer (Gemini Flash) plus a slower-stronger reviser/critic (Gemini Pro) — that mutate full code files (not just single functions).
- Multiple evaluators score candidates; evolutionary database stores program lineage.
- Operates on production codebases, not just isolated functions.

## Headline results
- Matrix multiplication: 4x4 complex matrices in 48 scalar multiplications — first improvement on Strassen since 1969.
- Inside Google: improved data-center scheduling, simplified TPU circuit fragments, sped up training of the Gemini that powers AlphaEvolve itself (self-bootstrapping).
- Improved on multiple long-standing combinatorial bounds.

## Why parked for AutoJEPA v1
- Multi-LLM pipeline doubles inference cost and operational complexity (rate limits, retries, two model-vendor contracts).
- Per writeup §8: "Proposer + critic is strictly better, but doubles LLM cost." AutoJEPA v1 uses a single-model proposer with the deterministic AST-diff validator standing in as the cheap critic.
- The proposer/critic upgrade is the natural follow-on once v1 ROI is proven.
