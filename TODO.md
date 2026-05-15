# AutoJEPA — Roadmap & TODO

> Living roadmap. Mirrors the architecture writeup at
> https://gist.github.com/epappas/2567a53350ba0d6ca064c71986a76046
> (`autojepa-implementation-plan.md`, draft v0.1, 2026-05-14).
>
> Every entry is a checkable item. **Done means validated end-to-end with evidence**, not "code written".
> Prime deployment target: **Basilica GPU cloud**.

---

## Phase 0 — Bootstrap (D1-2) — DONE 2026-05-15

- [x] Repo locally initialized
- [x] Carry over `../autoresearch-rl/src/autoresearch_rl/` → `src/autojepa/` with module rename pass
- [x] Carry over `../autoresearch-rl/tests/` → `tests/` with import rename pass
- [x] Carry over `Makefile`, `pyproject.toml`, `CLAUDE.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `scripts/`
- [x] Rename CLI entrypoint `autoresearch-rl` → `autojepa`
- [x] Update `pyproject.toml`: `basilica-sdk` promoted to default dep; `[jepa]` extra added (torch, lightning, torchmetrics, transformers, webdataset)
- [x] Rewrite `README.md` for AutoJEPA identity (Basilica-first)
- [x] `git remote add upstream /home/epappas/workspace/spacejar/autoresearch-rl`
- [x] `uv sync --extra dev` and `--extra jepa` both succeed
- [x] `uv run pytest -q --ignore=tests/eval/test_real_llm.py` → **470 passed, 9 failed, 5 skipped** (the 9 failures all reference `examples/<name>/prepare.py` fixtures deliberately not carried over; documented in `docs/phase-0-baseline.md`)
- [x] **Deliverable**: `autojepa` CLI behaves identically to upstream; baseline verified

---

## Phase 1 — JEPA primitives library (W1)

### `src/autojepa/eval/`

- [x] `probes.py` — `build_linear_probe`, `build_knn_probe`, `build_rankme`, `build_lidar`, `default_probes` factories wrapping `spt.callbacks.{OnlineProbe, OnlineKNN, RankMe, LiDAR}` — 13/13 tests passing
- [x] `collapse.py` — RankMe, latent variance, effective rank (LiDAR via probes wrapper since it requires labels) — 18/18 tests passing
- [ ] `downstream.py` — task-eval suite scaffolding (deferred to Phase 2 — concrete task suites land with examples)
- [x] `canary.py` — `CanaryConfig` + `build_canary_gate` thin layer over `gates.Gate` per writeup §7.4 — 13/13 tests passing
- [x] Tests: `tests/eval/test_collapse.py`, `test_probes.py`, `test_canary.py`, `tests/test_forecaster_ssl.py` (10 cases inc. honest documentation of the SSL plateau over-cancellation per ADR-013)

### `src/autojepa/masking/`

- [x] `primitives.py` — `MaskOutput` + `MultiBlockInfillMask` (I-JEPA paper defaults: n_targets=4, target_scale=(0.15, 0.20), target_aspect=(0.75, 1.5), context_scale=(0.85, 1.0)). Other writeup §7.2 masks (`FutureBlockMask`, `SemanticUnitMask`, `ActorAnonymizedMask`, `TimeFrequencyMask`) deferred to Phases 2-3 (refusing to ship stubs per ZERO-TOLERANCE rule)
- [x] `composite.py` — `CompositeMask([(mask, weight), ...])` weighted delegation
- [x] Tests: `tests/test_masking.py` — 17/17 covering boundary checks, shape invariants, the I-JEPA non-overlap invariant (context ∩ targets = ∅), determinism, weighted-choice coverage

### `src/autojepa/models/`

- [ ] `encoders.py` — ViT, ConvNeXt, generic transformer
- [ ] `predictors.py` — block-causal, full-attention, cross-attention
- [x] `ema.py` — EMAConfig + build_target_encoder (wraps `stable_pretraining.TeacherStudentWrapper`) + `assert_no_grad_on_target` invariant — 8/8 tests passing
- [x] `losses.py` — facade re-exporting `spt.losses.{VICRegLoss, BarlowTwinsLoss, BYOLLoss, DINOv1Loss, NTXEntLoss, NegativeCosineSimilarity}` + closed-form `l1_loss` / `l2_loss` + flat `LOSS_REGISTRY` — 15/15 tests passing
- [x] Tests: `tests/test_models_ema.py`, `tests/test_models_losses.py`

### `src/autojepa/gates.py`

- [x] Decision-gate engine: `Gate`, `GateRequirement`, `GateResult`, `GateEngine.evaluate / should_abort`, plus `GateConfig` Pydantic loader and `build_engine` factory. Supports `<, <=, >, >=, ==, !=` ops and `abort_campaign | warn` on_fail. Per writeup §7.6
- [x] Tests: `tests/test_gates.py` — 24/24 covering eval semantics, validation, after_iters cutoff, abort vs warn, expression parsing, YAML round-trip

### Forecaster recalibration (adapted)

- [x] Recalibrated SSL defaults: `IntraIterationCancelConfig.min_steps: 2000`, `poll_interval_s: 30.0`, `min_reports_before_decide: 10` (writeup §6.2 / ADR-008). Mirror `GuardConfig` dataclass updated to match
- [x] No new `forecast_target` field needed — `ObjectiveConfig.metric` is already wired through to `IntraIterationGuard.metric` at all 3 call sites; per ADR-012 we use it directly
- [ ] `tests/test_forecaster_ssl.py` — SSL plateau-then-improvement curve test (forecaster must NOT cancel during long initial plateau). Will land in batch 5

### Config / `program.md` template

- [x] Default `objective.metric: probe_auroc`, `direction: max` (ADR-004)
- [x] Hybrid widened: `param_explore_iters` 5→25, `stall_threshold` 3→5, `diff_failure_limit` 3→5 (writeup §6.3 / §12.5). Tests in `tests/test_hybrid_jepa_defaults.py`
- [x] `JEPA_HARD_RULES` prompt fragment encoding the full §6.4 program.md invariants (collapse thresholds, forbid target-encoder grads, forbid Ψ over-capacity, forbid removing anti-collapse regularisers, required runtime calls, high-value diff targets) — wired into both `llm_diff._SYSTEM_PROMPT` and `llm_search._SYSTEM_PROMPT`. Tests in `tests/test_jepa_prompt_fragments.py`
- [ ] Per-example `program.md` template — Phase 2 (used by `examples/ijepa-cifar10/program.md`)
- [ ] Param dims expanded to 10-12 JEPA-relevant dims (lr, EMA_start/end, mask_ratio_max, λ_var, λ_cov, predictor_depth/width, target/context_block_scale, loss_type, seed_count) — surfaced in per-example config, not framework default
- [ ] Storage policy: `keep_top_k=5`, encoder-only archive, prune>7d — touches `telemetry/` and config (Phase-4 hardening)

**Deliverable**: importable Python library; no working example yet.

---

## Phase 2 — `examples/ijepa-cifar10/` (W2) — KILL CRITERION

> The Phase-2 falsifier. If a 20-iter hybrid campaign produces zero validated diff
> improvements against a deliberately suboptimal baseline, the entire framework
> approach is dead.

### Code complete (Phase 2 batch 1)

- [x] `examples/ijepa-cifar10/prepare.py` (frozen): CIFAR-10 download + probe-eval split + canary subset
- [x] `examples/ijepa-cifar10/train.py` (mutable): I-JEPA loop wrapping `stable_pretraining.methods.IJEPA` per ADR-003. Deliberately suboptimal baseline per ADR-014 (predictor_depth=2, predictor_embed_dim=128, num_targets=2, plain L2). Calls `emit_progress(metrics={"probe_auroc": ...})` and `assert_no_grad_on_target` at the AST-validator-required boundaries
- [x] `examples/ijepa-cifar10/program.md`: full task spec encoding the §6.4 hard rules, hyperparameter guidance, code-diff guidance, and decision gate
- [x] `examples/ijepa-cifar10/config.yaml`: Basilica target (A100/H100/L40S/RTX-4090/RTX-A6000), 20-iter hybrid policy, recalibrated SSL forecaster defaults (intra_iteration_cancel: enabled, min_steps=2000, poll_interval_s=30, min_reports_before_decide=10)
- [x] `examples/ijepa-cifar10/{README.md,run.sh,requirements.txt}` for orchestration
- [x] `tests/test_examples_smoke.py` re-points TIER2_VALIDATE_ONLY at this example
- [x] ADR-014 documents the deliberately-suboptimal-baseline decision

### Validated locally (Phase 2 batch 2 — DONE 2026-05-15, evidence in `docs/phase-2-runtime-evidence.md`)

- [x] `BASILICA_API_TOKEN=stub CHUTES_API_KEY=stub uv run autojepa validate examples/ijepa-cifar10/config.yaml` -> `OK`
- [x] `BASILICA_API_TOKEN=<real> CHUTES_API_KEY=<real> uv run autojepa validate ...` -> `OK` (real credentials accepted)
- [x] prepare.py downloaded CIFAR-10 in 7s, materialised all 6 documented `.pt` outputs
- [x] Local 1-step IJEPA forward+backward integration test PASS (synthetic batch=2, CPU; commit `555386f`). Caught and fixed a real bug in `_extract_features` (MaskedEncoderOutput.encoded vs .predictions) before the Basilica run
- [x] `examples/ijepa-cifar10/deploy.py` added — base64-injects train.py + prepare.py into the Basilica container per the upstream basilica-grpo pattern
- [x] Full test suite: 574 passed, 6 skipped (no regressions)
- [ ] Full 50-step smoke on CPU — impractical (3-7 hours estimated); skipped in favour of the Basilica smoke

### Phase 2 batch 3 — Basilica execution (in flight)

- [ ] 3-iter Basilica smoke campaign (`--max-iterations 3 --git-ref 555386f`) — IN FLIGHT, monitor `bo1g3492h`
- [ ] If smoke passes: authorise full 20-iter campaign with budget ~$30-100 of A100 time
- [ ] Reserve no-forecaster control group for first 50 iters (per writeup §12.2 / ADR-013)
- [ ] **Decision gate**: ≥1 hybrid-mode diff produces a measurable retained improvement; `phase2_falsifier` gate (probe_auroc > 0.40) reports pass
- [ ] **Deliverable**: reproducible I-JEPA-CIFAR campaign with hybrid-policy improvement, OR documented framework kill (writeup §12.1 escalation)

---

## Phase 3 — `examples/trace-jepa/` (W3-4)

- [ ] Data pipeline: synthetic LangChain/CrewAI traces → 1M sessions, WebDataset shards
- [ ] Tokenizer: structured event schema (`action_name`, `action_type`, `args`, `return_code`, `timestamp`, `actor_id`, `parent_link`)
- [ ] `train.py`: 25-50M-param Trace-JEPA, multi-block + future-block masking
- [ ] `prepare.py`: probe eval on InjecAgent payloads, AgentDojo gate
- [ ] `config.yaml`: Basilica target, 30-iter hybrid campaign
- [ ] Decision gate: `probe AUROC > 0.7 at 5% FPR` after 20 iters → abort if missed
- [ ] **Deliverable**: AutoJEPA validates or kills the Trace-JEPA thesis

### Phase-3 search-space dimensions — codebook bottleneck (per MTS-JEPA)

The Phase-3 `examples/trace-jepa/config.yaml` parameter space (still to be
written) **must** include the following two dimensions, motivated by
[mts-jepa.md](docs/research/mts-jepa.md) and the
`topic=Trace-JEPA-Design` belief in Alexandria (asserted 2026-05-15):

```yaml
codebook_size: [0, 64, 256, 1024]   # 0 = vanilla JEPA, no codebook
codebook_loss_weight: [0.0, 0.1, 0.5, 1.0]
```

Search-dimension rationale (one sentence per dimension, citing the
MTS-JEPA distillation):

- `codebook_size`: per [mts-jepa.md](docs/research/mts-jepa.md), the soft
  codebook bottleneck is the only published JEPA mechanism aimed at
  discrete-regime-transition modeling — varying its vocabulary from `0`
  (vanilla JEPA control) through `1024` measures whether agent-trace
  regime structure (tool-call switches, role boundaries, plan re-entries)
  benefits from a discrete latent vocabulary at all and at what
  granularity.
- `codebook_loss_weight`: per [mts-jepa.md](docs/research/mts-jepa.md),
  the codebook constraint also acts as an intrinsic anti-collapse
  regularizer — sweeping the loss weight from `0.0` (off) through `1.0`
  measures the codebook's contribution as a regularizer separately from
  its role as a latent-vocabulary carrier, so the search can attribute
  any gain to representation structure vs collapse defense.

Phase-2 scope guard (`examples/ijepa-cifar10/`): **do NOT add these
dimensions to Phase-2**. Phase-2 is the kill-criterion run and must stay
vanilla I-JEPA to be a fair reproduction (per ADR-014 — deliberately
suboptimal vanilla baseline).

Framework scope guard: **do NOT add the codebook to the core framework**
(`src/autojepa/models/`). The codebook is a Phase-3 trace-jepa example
concern, not a framework primitive — putting it in core would repeat the
contrib-namespace mistake AutoJEPA explicitly rejected.

---

## Phase 4 — Hardening for Basilica production (W5-6)

- [ ] Multi-GPU contract end-to-end on Basilica (`train.py` owns torchrun; `gpu_count` flows from Basilica via env var)
- [ ] Multi-seed scoring + aggregation (default 3 seeds, mean ± std)
- [ ] Model archive policy enforced (`keep_top_k=5`, encoder-only, prune>7d)
- [ ] Documentation site, tutorials
- [ ] CI smoke tests including canary
- [ ] **Deliverable**: v1.0 release on private repo, internal-usable

---

## Phase 5 — Public release (M2+)

- [ ] Security audit (no leaked keys, no proprietary trace data)
- [ ] Open-source license (MIT or Apache-2.0)
- [ ] Public README, example notebooks
- [ ] PyPI publication
- [ ] Blog post on Trace-JEPA case study

---

## Phase 1 — Legacy module drops (per inheritance map §10.3)

- [x] Drop `controller/loop.py` (legacy entry; live path is `controller/continuous.py`)
- [x] Drop `sandbox/runner.py` (legacy trial runner)
- [x] Drop `eval/judge.py`, `eval/scoring.py` (legacy heuristic scoring)
- [x] Drop `distillation/{__init__,sdft,sink,trainer}.py` (entire legacy directory)
- [x] Drop `SandboxExecutor`, `SandboxExecutorConfig`, `JudgeEvaluator` from `controller/executor.py` (dead classes; live path uses `TargetExecutor` + `MetricEvaluator`)
- [x] Drop tests for dropped modules: `test_loop_autonomy.py`, `test_loop_comparability.py`, `test_scaffold.py`, `test_runner.py`, `test_runner_forecast.py`, `test_distillation_sink.py`, `test_distillation_trainer.py`, `test_sdft.py`
- [x] Adapt `tests/test_examples_smoke.py` parametrize lists to empty until Phase-2 lands AutoJEPA examples

## Standing tasks

- [ ] **Docs sync rule**: every code change affecting architecture, contracts, or campaigns updates the corresponding `docs/` entry in the same commit
- [ ] **Cherry-pick log**: `docs/cherry-pick-log.md` records which `autoresearch-rl` improvements were ported and which were skipped (cap ~1h/wk)
- [ ] **Truth-only reporting**: `CHANGELOG.md` records validated outcomes only — no aspirational claims

---

## Research corpus (docs/research/) — INGESTED 2026-05-15

All 4 ingestion agents complete. Index at `docs/research/INDEX.md`.
Raw arxiv abstracts under `docs/research/raw/<arxiv-id>/`.

- [x] JEPA family (6): I-JEPA, V-JEPA-2, C-JEPA, CNN-JEPA, A-JEPA, JEPA-Audio-Design-Choices
- [x] SSL methodology (2): Stable-Pretraining-V1 (verified import paths), HP-SSL-Importance
- [x] Autonomous-research lineage (8): FunSearch, AI-Scientist-V2, ADAS, AIDE, AlphaEvolve, CodeEvolve, Sakana-AI-Scientist-Evaluation, MLE-Bench
- [x] Workflow/HPO (2): AFlow, AgentHPO
- [x] Foundation (2): Karpathy-Autoresearch-Foundation, AutoresearchRL-Inheritance-Map (the carry-over plan, 297 lines)
