# CodeEvolve: an open source evolutionary coding agent for algorithmic discovery and optimization

Source: https://arxiv.org/abs/2510.14150

## Abstract (verbatim)

We introduce CodeEvolve, an open-source framework that combines large language models (LLMs) with evolutionary search to synthesize high-performing algorithmic solutions. CodeEvolve couples an islands-based genetic algorithm with modular LLM orchestration, using execution feedback and task-specific metrics to guide selection and variation. Exploration and exploitation are balanced through context-aware recombination, adaptive meta-prompting, and targeted refinement of promising solutions. We evaluate CodeEvolve on benchmarks used to assess Google DeepMind's AlphaEvolve, and include direct comparisons with popular open-source frameworks for algorithmic discovery and heuristic design. Our results show that CodeEvolve achieves state-of-the-art (SOTA) performance on several tasks, with open-weight models often matching or exceeding closed-source baselines at a fraction of the compute cost. We provide extensive ablations, practical hyperparameter guidance, and release our framework and experimental results at https://github.com/inter-co/science-codeevolve.
