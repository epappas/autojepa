# CodeEvolve: An Open-Source Evolutionary Coding Agent for Algorithmic Discovery and Optimization

**Citation:** Assumpcao, H., Ferreira, D., Campos, L., Murai, F. *CodeEvolve: an open source evolutionary coding agent for algorithmic discovery and optimization.* arXiv:2510.14150, 2025.
**Raw material:** `docs/research/raw/2510.14150/`
**AutoJEPA classification:** Parked / explicit non-goal (architecture); Comparison benchmark (eval harness)

---

## 1. One-line thesis

Reproduce AlphaEvolve in the open: islands-based genetic algorithm + modular LLM orchestration + execution-feedback scoring, with extensive ablations showing open-weight models match or exceed closed-source baselines on the AlphaEvolve benchmark suite at a fraction of the compute.

## 2. Method

- **Islands-based GA** (FunSearch lineage). Each island maintains its own population and breeding loop; periodic migration mixes islands.
- **Modular LLM orchestration.** Different LLM roles (proposer, recombiner, refiner) implemented as pluggable modules; supports both closed (GPT/Gemini/Claude) and open-weight (Qwen, DeepSeek) models.
- **Adaptive meta-prompting.** Prompt template mutates over generations based on what kinds of edits are scoring well.
- **Context-aware recombination.** Parent selection conditioned on the local island's fitness landscape.
- **Execution-feedback scoring.** Each candidate runs against task-specific metrics (correctness checks + performance benchmarks); scores feed back as selection signal.
- **Open release:** framework, configs, and experimental results at https://github.com/inter-co/science-codeevolve.

## 3. Results

- SOTA on several of the benchmarks AlphaEvolve was originally evaluated on.
- Open-weight models often match or beat closed-source baselines at a fraction of the compute cost — the architectural pattern, not the proprietary stack, is what produces the gains.
- Extensive ablations published over GA hyperparameters, prompt strategies, and model choice.

## 4. Why it matters for AutoJEPA — Parked (architecture); Comparison benchmark (eval harness)

**Parked architecture.** Same reasons as AlphaEvolve (writeup §8):

- Islands + multi-role LLM orchestration is a much heavier scaffold than the linear hybrid policy AutoJEPA inherits from `autoresearch-rl`.
- AutoJEPA v1's hypothesis is that JEPA-specific primitives (collapse-guard, probe-eval, mask-scheduler) buy more than topology upgrades to the search loop.

**Comparison benchmark.** CodeEvolve is the *practically reachable* baseline for measuring AutoJEPA's outer loop quality:

- Closed-source AlphaEvolve isn't reproducible without Gemini access.
- CodeEvolve is open and configurable to run on the same cluster AutoJEPA targets.
- The published ablations give defensible hyperparameter ranges to compare against without a tuning campaign of our own.

The right v2 question for AutoJEPA is not "should we adopt islands?" but "on a fixed JEPA pretraining task, does AutoJEPA v1 match a CodeEvolve scaffold at equivalent GPU budget?" If yes, the JEPA-specific primitives justify their existence. If no, that is the empirical case for upgrading the outer loop.

## 5. Caveats / what doesn't transfer

- **Benchmarks are algorithmic / heuristic discovery,** not deep model training. CodeEvolve's results don't directly say anything about JEPA pretraining quality.
- **Compute model.** Per-candidate evaluation in CodeEvolve is seconds-to-minutes; JEPA training is hours. Islands GA needs many generations to see useful migration; that costs serious GPU.
- **No collapse-mode awareness.** Like FunSearch and AlphaEvolve, the evaluator returns a clean scalar; representation collapse has no analogue in the algorithmic-discovery setting.

## 6. Cross-links

- `AlphaEvolve.md` — the closed-source ancestor that CodeEvolve reproduces.
- `FunSearch.md` — the islands-based GA pattern originated here.
- `MLE-Bench.md` — independent measurement protocol; CodeEvolve does not target MLE-bench but shares the gate-style philosophy.
- `AIDE.md` — alternative open scaffold pattern (tree search instead of islands GA).
