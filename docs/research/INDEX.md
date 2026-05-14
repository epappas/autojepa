# AutoJEPA Research Corpus — Index

Distilled summaries of papers and external code that AutoJEPA depends on or
draws empirical justification from. Raw artifacts (verbatim abstracts,
metadata, extraction notes) live under `raw/<arxiv-id>/`.

## SSL methodology

- [Stable-Pretraining v1](Stable-Pretraining-V1.md) — Balestriero et al.,
  arXiv 2511.19484. Lightning-callback library that AutoJEPA wraps for
  in-loop linear/k-NN probes and `RankMe`/`LiDAR` collapse metrics. Raw:
  `raw/2511.19484/`.
- [HP/SSL Importance](HP-SSL-Importance.md) — Wagner et al., arXiv
  2207.07875. Empirical evidence that SSL hyperparameter and augmentation
  choices swing downstream accuracy substantially; justifies the writeup
  §6.3 widening of `hybrid_param_explore_iters` 5 -> 25 and the broader
  10-12 dim parameter space. Raw: `raw/2207.07875/`.

## JEPA family

- [I-JEPA](I-JEPA.md) — image-domain block masking; foundational. Defines the
  encoder + EMA target + predictor template, multi-block masking, L2 latent
  loss. Source of `autojepa.masking.MultiBlockInfillMask` and the default
  module layout in `autojepa.models`. Raw: `raw/2301.08243/`.
- [V-JEPA 2](V-JEPA-2.md) — video and world-model scaling. ~300M predictor
  with block-causal attention, two-stage action-free + action-conditioned
  recipe. Motivates `autojepa.models.predictors.BlockCausalPredictor` and
  predictor-depth as a hybrid-policy search axis. Raw: `raw/2506.09985/`.
- [C-JEPA](C-JEPA.md) — VICReg-aware loss that fixes I-JEPA's
  partial-collapse and mean-prediction gaps. AutoJEPA's default loss
  configuration; weights exposed as searchable hyperparameters. Raw:
  `raw/2410.19560/`.
- [CNN-JEPA](CNN-JEPA.md) — JEPA for ResNet-style CNNs via sparse-conv
  encoder and depthwise-separable conv predictor. Backs
  `autojepa.models.encoders.SparseCNNEncoder` and the cheap-compute Phase 2
  validation baseline. Raw: `raw/2408.07514/`.
- [A-JEPA](A-JEPA.md) — audio port with curriculum masking annealing from
  random-block to time-frequency-aware. Source of
  `autojepa.masking.CurriculumMask` and the regularized-attention fine-tuning
  trick. Raw: `raw/2311.15830/`.
- [JEPA Audio Design Choices](JEPA-Audio-Design-Choices.md) — systematic
  ablation showing image-domain JEPA priors fail on audio. Establishes
  unstructured masking, both-branches masking, and smoothed-L1 as audio
  defaults; justifies LLM-driven cross-modal hyperparameter search. Raw:
  `raw/2405.08679/`.

## Workflow/HPO

- [AFlow](AFlow.md) — Zhang et al., arXiv 2410.10762. MCTS over
  code-represented agent workflows; 5.7% mean improvement over prior
  auto-workflow baselines. **Explicit non-goal in AutoJEPA v1**
  (writeup §8 — "the outer loop IS a workflow. v1 stays linear");
  reopen criteria documented in the distilled doc. Raw:
  `raw/2410.10762/`.
- [AgentHPO](AgentHPO.md) — Liu et al., arXiv 2402.01881. LLM agent
  (Creator + Executor) for hyperparameter optimization across 12 ML
  tasks. Direct precedent for AutoJEPA's `policy.type=llm` mode and
  the param-search half of the `hybrid` policy. Raw:
  `raw/2402.01881/`.

## Autonomous-research lineage

Corpus that documents the lineage and explicit non-goals of AutoJEPA's
autonomous loop. Each entry classified per writeup §3 (Lineage and
positioning) and §8 (What we park) of `autojepa-implementation-plan.md`
(draft v0.1, 2026-05-14).

- [FunSearch](FunSearch.md) — Romera-Paredes et al., Nature 2024
  (DOI 10.1038/s41586-023-06924-6). LLM-proposer + deterministic
  evaluator over candidate programs; islands-based GA. **Inherited** —
  the foundational architectural pattern of the `autoresearch-rl` loop
  AutoJEPA forks. Raw: `raw/funsearch-nature-2024/`.
- [AI Scientist v2](AI-Scientist-V2.md) — Yamada et al., arXiv 2504.08066.
  Progressive agentic tree search managed by an experiment-manager
  sub-agent; first AI-only paper accepted at a peer-reviewed (workshop)
  venue. **Parked / explicit non-goal** for v1 (writeup §8). Raw:
  `raw/2504.08066/`.
- [ADAS](ADAS.md) — Hu, Lu, Clune, ICLR 2025 (arXiv 2408.08435).
  Meta-Agent Search programming new agents in code; transferable across
  domains and base models. **Parked / explicit non-goal** — meta-meta
  search is its own research project (writeup §8). Raw: `raw/2408.08435/`.
- [AIDE](AIDE.md) — Jiang et al., arXiv 2502.13138. Tree search over
  candidate programs with `draft / debug / improve` operators; SOTA on
  MLE-bench (16.9% bronze with o1-preview). **Parked / explicit non-goal**
  — engineering tax not justified for narrow JEPA candidate space (writeup
  §8). Raw: `raw/2502.13138/`.
- [AlphaEvolve](AlphaEvolve.md) — Novikov et al., arXiv 2506.13131.
  Multi-LLM pipeline (cheap proposer + strong critic) evolving full
  codebases; 4x4 complex matmul in 48 multiplications, first improvement
  on Strassen since 1969. **Parked / explicit non-goal** — proposer +
  critic doubles LLM cost (writeup §8). Raw: `raw/2506.13131/`.
- [CodeEvolve](CodeEvolve.md) — Assumpcao et al., arXiv 2510.14150.
  Open-source AlphaEvolve reproduction with islands GA + open-weight
  models. **Parked** (architecture, same reasons as AlphaEvolve);
  **Comparison benchmark** (open eval harness reachable on AutoJEPA's
  cluster). Raw: `raw/2510.14150/`.
- [Sakana AI Scientist Evaluation](Sakana-AI-Scientist-Evaluation.md) —
  Beel, Kan, Baumgart, SIGIR Forum 2025 (arXiv 2502.14297). Independent
  evaluation of Sakana AI Scientist v1: **42% of experiments fail with
  coding errors**. **Falsification reference** — the empirical floor
  argument for AutoJEPA's AST-diff validator. Raw: `raw/2502.14297/`.
- [MLE-bench](MLE-Bench.md) — Chan et al., OpenAI, ICLR 2025 (arXiv
  2410.07095). 75 Kaggle competitions with leaderboard-calibrated medal
  thresholds; o1-preview + AIDE reaches bronze in 16.9%. **Comparison
  benchmark** — the gate-style decision protocol AutoJEPA's promotion
  tracker mirrors. Raw: `raw/2410.07095/`.

## Foundation

- [Karpathy/autoresearch — Foundation kernel](Karpathy-Autoresearch-Foundation.md)
  — distillation of `karpathy/autoresearch` (the ~630 LOC `train.py`,
  ~389 LOC frozen `prepare.py`, single-GPU, 5-min iter budget).
  Documents the frozen/mutable contract verbatim, what `autoresearch-rl`
  (and AutoJEPA) add without breaking the kernel, and why these four
  properties (reproducibility, reviewability, autonomy, composability)
  are load-bearing.
- [autoresearch-rl Inheritance Map](AutoresearchRL-Inheritance-Map.md)
  — module-by-module + test-by-test carry-over plan from the sibling
  `autoresearch-rl` repo. Dominant action: **Inherit** (38 modules /
  50 tests inherit; 13 modules / 3 tests adapt; 8 modules / 3 tests
  drop; 1 file replaced). The Phase-0 plan; includes a cherry-pick
  log seed table from `git log --oneline -50`.
