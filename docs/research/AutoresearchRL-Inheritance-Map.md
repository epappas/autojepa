# AutoresearchRL → AutoJEPA Inheritance Map

- sibling repo: `/home/epappas/workspace/spacejar/autoresearch-rl/`
- compiled: 2026-05-15 against `autoresearch-rl` HEAD `ea71c96`
- writeup: https://gist.github.com/epappas/2567a53350ba0d6ca064c71986a76046

This is the carry-over plan for **Phase 0** of AutoJEPA. Every module in
`src/autoresearch_rl/` and every test in `tests/` is classified as
**Inherit** (copy, rename `autoresearch_rl` → `autojepa`, no behavior
change), **Adapt** (rename + targeted modification), **Replace** (functionally
the same role but a new implementation), or **Drop** (not carried into
AutoJEPA).

Classification source: writeup §5 (What we keep), §6 (What we adapt),
§7 (What we add).

---

## 1. Top-level packaging

| File | One-line role | Action | Notes |
|---|---|---|---|
| `pyproject.toml` | Package metadata + entrypoints + extras | **Adapt** | Rename project to `autojepa`, change `[project.scripts]` to `autojepa = "autojepa.cli:app"`, add `basilica` to default install (writeup: Basilica-first), add JEPA-relevant deps (`stable-pretraining`, `torchvision`, `transformers`, `webdataset`) under a new `[jepa]` extra |
| `Makefile` | Common workflows (lint, test, smoke, showcase) | **Adapt** | Rename `autoresearch-rl` invocations to `autojepa`; add JEPA-specific targets: `make canary CONFIG=…`, `make probe-eval CONFIG=…` |
| `README.md` | Public-facing intro | **Replace** | Rewrite for AutoJEPA identity; the autoresearch-rl README's loop diagram and config docs are inheritable but the framing is different |
| `CLAUDE.md` | Agent guidance for this repo | **Adapt** | Keep "do not call done without realistic-config end-to-end run" rule verbatim; update module paths and JEPA-specific failure modes |
| `CONTRIBUTING.md` | Per-area pre-merge checklist | **Adapt** | Rename refs; add JEPA-canary requirement to the checklist |
| `CHANGELOG.md` | Phase-by-phase change log | **Drop** then re-init | History is autoresearch-rl's; AutoJEPA starts a fresh log |
| `uv.lock` | Locked dep versions | **Drop** then regenerate | New deps mean a fresh resolve |
| `docs/ARCHITECTURE.md` | Module-by-module walkthrough | **Adapt** | Inherit the structure verbatim; rewrite the "Phase X" history sections to point to AutoJEPA Phases 1–4 from writeup |

## 2. `src/autoresearch_rl/` — top-level modules

| Module / file | One-line role | Action | Notes |
|---|---|---|---|
| `__init__.py` | Package marker | **Inherit** | Rename only |
| `cli.py` | Typer-based CLI: `run`, `validate`, `print-config`, `status`, `run-one`, `upload` | **Adapt** | Inherit all subcommands. Rename script entry. Add `autojepa probe-eval` subcommand for ad-hoc probe scoring (writeup §Phase-1 `eval/`) |
| `config.py` | Pydantic root models for all config sections | **Adapt** | Inherit all sub-configs. Recalibrate forecaster defaults (`min_steps: 5 → 2000`, `poll_interval_s: 5 → 30`, `min_reports_before_decide: 5 → 10`) per writeup §6. Widen hybrid (`explore_iters: 5 → 25`, `stall_threshold: 3 → 5`). Default `objective.metric` to `probe_auroc` |
| `config_validate.py` | 8 runtime validation checks (env vars, file existence, budget alignment, required_calls presence) | **Adapt** | Inherit all 8 checks. Add a 9th check: when `objective.metric` is one of {`probe_auroc`, `rankme`, `lidar`, `effective_rank`}, the canary file referenced by `prepare.py` must exist (Phase-1 `canary.py`) |
| `forecasting.py` | Power-law `y = a*x^b + c` fitter + `should_early_stop` | **Adapt** | Algorithm is metric-agnostic; recalibrate via `min_points` default and `IntraIterationCancelConfig.min_steps` default for SSL plateau shape. Add unit-test coverage for monotone-then-plateau curves (the SSL signature) |
| `checkpoint.py` | Atomic JSON checkpoint write/read | **Inherit** | Pure infrastructure |
| `mdp.py` | `State`, `Action`, `Reward` dataclasses for MDP framing | **Inherit** | Pure dataclasses |
| `trajectory.py` | Trajectory dataclass | **Inherit** | Pure dataclass |
| `tracking.py` | Best-value + run-counter tracking | **Inherit** | Workload-agnostic |
| `promotion.py` | Promote / discard decision struct | **Inherit** | Workload-agnostic; direction-aware |

## 3. `src/autoresearch_rl/controller/`

| Module | One-line role | Action | Notes |
|---|---|---|---|
| `__init__.py` | Re-exports `run_experiment`, `Evaluator`, etc. | **Inherit** | Path rename only |
| `continuous.py` | Primary loop: propose → train → eval → keep/discard, with stop guards | **Inherit** | Workload-agnostic; loop body is the kernel |
| `engine.py` | `run_experiment(...)` serial entrypoint | **Inherit** | |
| `parallel_engine.py` | Sibling `run_experiment_parallel(...)` with `ThreadPoolExecutor` + ResourcePool | **Inherit** | Inherits intact for Basilica K=N parallel campaigns |
| `executor.py` | `Outcome` dataclass + iteration executor | **Inherit** | |
| `helpers.py` | Stop guards (wall time, no-improve, failure rate) | **Inherit** | |
| `intra_iteration.py` | Cooperative-cancel guard wrapping forecaster + `BestValueRef` | **Adapt** | Inherit `BestValueRef` and the watcher thread verbatim. Bump `GuardConfig` defaults per writeup §6 (SSL recalibration). Add unit test for SSL plateau-then-improvement series (must NOT cancel during a long initial plateau) |
| `resource_pool.py` | Threadsafe bin-packing for parallel iterations | **Inherit** | |
| `diff_executor.py` | Validate diffs (safety + contract + required_calls), apply in-memory, run | **Inherit** | The required_calls plumbing is exactly what JEPA needs to keep `emit_progress(probe_auroc=...)` from being stripped |
| `contract.py` | Frozen/mutable file boundary enforcement | **Inherit** | The `fef66d1` basename-comparison fix is load-bearing — must carry the test alongside |
| `shutdown.py` | SIGTERM/SIGINT handler | **Inherit** | |
| `types.py` | Shared Outcome / decision types | **Inherit** | |
| `loop.py` | **Legacy** sandbox/contract loop (not used by CLI) | **Drop** | Per `CLAUDE.md` §"Two independent loop systems coexist"; CLI uses `continuous.py` only. Drop to remove dead code from the carry-over |
| `one_shot.py` | Single-iteration helper for `cli.py::run_one` | **Inherit** | |

## 4. `src/autoresearch_rl/policy/`

| Module | One-line role | Action | Notes |
|---|---|---|---|
| `interface.py` | `Policy` Protocol + `Proposal` dataclass + `propose_batch` helper | **Inherit** | Stable contract |
| `search.py` | `GridPolicy`, `RandomPolicy`, `StaticPolicy` baselines | **Inherit** | Pure search, workload-agnostic |
| `baselines.py` | Diff-policy baselines for the legacy loop | **Drop** | Tied to legacy `controller/loop.py` which is being dropped |
| `llm_search.py` | `LLMParamPolicy`: param proposals from LLM with history; `propose_batch` for parallel | **Adapt** | Inherit code. Rewrite `_SYSTEM_PROMPT` to teach SSL/JEPA HP shape (LR vs EMA momentum vs masking ratio) and JEPA failure modes (collapse signatures) |
| `llm_diff.py` | `LLMDiffPolicy`: code-diff proposals with correction retry on validation failure | **Adapt** | Inherit code + retry logic. Rewrite `_SYSTEM_PROMPT` to forbid removing EMA `stop_gradient`, forbid making predictor Ψ deeper than encoder Φc, forbid removing collapse-defense regularizers (per writeup §Phase-1 `program.md` template) |
| `hybrid.py` | `HybridPolicy`: param-first → diffs on stall → fallback to params on diff fail | **Adapt** | Inherit code. Pull thresholds from new wider config defaults |
| `llm_context.py` | History summarization, log extraction, error extraction for LLM prompts | **Adapt** | Inherit. Add `summarize_progress_metrics(["probe_auroc", "rankme", "lidar"])` so the LLM sees the full collapse-signal time series, not just the scalar objective |
| `_prompt_fragments.py` | Shared `PROGRESS_PROTOCOL_RULES`, `CANCELLATION_CONTEXT_RULES`, `BATCH_DIVERSITY_RULES`, history renderers | **Adapt** | Inherit. Add a `JEPA_HARD_RULES` fragment encoding the writeup's `program.md` invariants (latent_var<0.3, eff_rank<32, RankMe<64, LiDAR<80 → fail) |
| `learned.py` | `LearnedDiffPolicy` PPO-style learner | **Inherit** | Workload-agnostic; reward signal is the same |
| `learned_search.py` | Learned variant of param search | **Inherit** | |
| `ppo.py` | PPO update math | **Inherit** | Pure math |
| `gae.py` | Generalized Advantage Estimation | **Inherit** | Pure math |
| `sdpo.py` | SDPO update math | **Inherit** | Pure math |

## 5. `src/autoresearch_rl/target/`

| Module | One-line role | Action | Notes |
|---|---|---|---|
| `__init__.py` | Package marker | **Inherit** | |
| `interface.py` | `TargetAdapter` Protocol + `RunOutcome` + `resource_cost` helper | **Inherit** | Stable contract |
| `registry.py` | `build_target(cfg)` dispatch | **Inherit** | |
| `command.py` | Local subprocess target with `AR_PARAMS_JSON` and `AR_PARAM_<NAME>` env-var injection | **Inherit** | |
| `http.py` | Remote vLLM/sglang HTTP target | **Inherit** | Useful for prompt-injection / classifier examples |
| `basilica.py` | Basilica GPU cloud target — bootstrap server, deploy, wait_ready, poll, download_model, cleanup, /control + /progress endpoints | **Adapt** | Inherit verbatim. Add JEPA-specific resource_cost (multi-GPU jobs declare `{"gpu": cfg.gpu_count}`). Confirm bootstrap server preserves `AR_PROBE_DATASET_DIR` env var alongside the existing `AR_MODEL_DIR` |
| `progress.py` | `emit_progress()` trial-side contract; reads `$AR_PROGRESS_FILE`, `$AR_CONTROL_FILE`; `sys.exit(42)` on cancel | **Inherit** | Load-bearing. The metric dict accepts arbitrary float keys, so `metrics={"probe_auroc": 0.71, "rankme": 32.4}` works without code change |
| `progress_reader.py` | Controller-side daemon thread tail of `progress.jsonl` with `drain()` | **Inherit** | |

## 6. `src/autoresearch_rl/sandbox/`

| Module | One-line role | Action | Notes |
|---|---|---|---|
| `validator.py` | `validate_diff()` (forbidden tokens, AST check on added lines) + `validate_required_calls(pre, post, required)` | **Inherit** | Required-calls is exactly the mechanism that protects `emit_progress(probe_auroc=...)` from being deleted by an LLM diff |
| `ast_policy.py` | `validate_python_source()` rejecting forbidden imports/calls | **Inherit** | Security boundary; workload-agnostic |
| `diff_utils.py` | Patch utilities | **Inherit** | |
| `runner.py` | **Legacy** loop runner | **Drop** | Tied to legacy `controller/loop.py` |

## 7. `src/autoresearch_rl/eval/` (legacy)

The directory name collides with the new `autojepa/eval/` (writeup
§Phase-1). Rename it during carry-over to avoid confusion.

| Module | One-line role | Action | Notes |
|---|---|---|---|
| `judge.py` | Heuristic next-state voting (legacy) | **Drop** | Used only by legacy loop |
| `metrics.py` | Stdout parsers for `val_bpb` / `loss` | **Adapt** then move | Move to `src/autojepa/parsing/metrics.py` (or fold into `target/command.py` which already does the equivalent for `outcome.metrics`). Add probe_auroc / rankme parsers |
| `scoring.py` | Composite score computation (legacy) | **Drop** | Replaced by writeup §Phase-1 `autojepa/eval/probes.py` + `eval/collapse.py` |

The `autojepa/eval/` namespace under writeup §Phase-1 is **net new**
content (`probes.py`, `collapse.py`, `downstream.py`, `canary.py`) and
does not inherit anything from this directory.

## 8. `src/autoresearch_rl/distillation/`

| Module | One-line role | Action | Notes |
|---|---|---|---|
| `__init__.py` | Package marker | **Inherit** | |
| `sdft.py` | SDFT loss / sampling math | **Drop** for v1 | No JEPA distillation use case in v1; reopen for Trace-JEPA distillation in Phase 3 if needed. Keep upstream patches via cherry-pick log |
| `sink.py` | Distillation sample sink | **Drop** for v1 | Same |
| `trainer.py` | Distillation trainer | **Drop** for v1 | Same |

## 9. `src/autoresearch_rl/telemetry/`

| Module | One-line role | Action | Notes |
|---|---|---|---|
| `events.py` | JSONL trace emission (`trace_id`, span shape) | **Inherit** | |
| `ledger.py` | TSV results ledger with comparability metadata | **Inherit** | |
| `manifest.py` | Per-run manifest files | **Inherit** | |
| `comparability.py` | Hardware fingerprint + budget-mode checks; strict-mode rejection of mismatched runs | **Inherit** | Already supports `parallel_wallclock` after Phase-4B fix |
| `aggregation.py` | Multi-seed aggregation utilities | **Adapt** | Inherit core. Default to 3-seed mean ± std per writeup §Phase-4 ("Multi-seed scoring + aggregation") |
| `rotation.py` | Log rotation helpers | **Inherit** | |
| `run.py` | Run identity / metadata | **Inherit** | |
| `timeline.py` | Chrome-trace / Perfetto timeline JSON export | **Inherit** | |
| `distill.py` | Distillation sample collection (legacy loop only) | **Drop** | Tied to dropped distillation modules |

---

## 10. `tests/` — full classification

`tests/__init__.py` and `tests/eval/__init__.py` are inherited as
empty marker files (rename only).

### 10.1 Inherit (rename only) — 36 tests

These cover workload-agnostic scaffolding that AutoJEPA imports unchanged.

| Test file | Covers |
|---|---|
| `test_aggregation.py` | `telemetry/aggregation.py` |
| `test_ast_policy.py` | `sandbox/ast_policy.py` |
| `test_basilica_integration.py` | `target/basilica.py` end-to-end (gated on Basilica creds) |
| `test_basilica_unit.py` | `target/basilica.py` units (no creds) |
| `test_checkpoint.py` | `checkpoint.py` |
| `test_cli.py` | `cli.py` core subcommands |
| `test_cli_agent.py` | `cli.py` agent integration |
| `test_command_progress.py` | `target/command.py` + progress reader |
| `test_comparability.py` | `telemetry/comparability.py` |
| `test_config_validate.py` | `config_validate.py` 8 checks |
| `test_contract.py` | `controller/contract.py` (load-bearing — covers the `fef66d1` basename fix) |
| `test_diff_executor.py` | `controller/diff_executor.py` |
| `test_engine_cancel.py` | `controller/engine.py` cancel paths |
| `test_examples_smoke.py` | End-to-end example smoke (Tier 1 + Tier 2) |
| `test_forecasting.py` | `forecasting.py` (will need the new SSL-plateau test added — see Adapt section) |
| `test_gae.py` | `policy/gae.py` |
| `test_hybrid.py` | `policy/hybrid.py` |
| `test_intra_iteration.py` | `controller/intra_iteration.py` |
| `test_learned_search.py` | `policy/learned_search.py` |
| `test_ledger.py` | `telemetry/ledger.py` |
| `test_llm_context.py` | `policy/llm_context.py` |
| `test_llm_diff.py` | `policy/llm_diff.py` mechanics (mocked LLM) |
| `test_llm_search.py` | `policy/llm_search.py` mechanics (mocked LLM) |
| `test_loop_autonomy.py` | `controller/continuous.py` autonomy guarantees |
| `test_loop_comparability.py` | Comparability across loop iters |
| `test_manifest.py` | `telemetry/manifest.py` |
| `test_mdp.py` | `mdp.py` |
| `test_metrics.py` | Metric parsing (will become `tests/parsing/test_metrics.py` after the eval rename) |
| `test_multiturn_llm.py` | Multi-turn LLM conversation |
| `test_parallel_engine.py` | `controller/parallel_engine.py` |
| `test_policy_baselines.py` | `policy/search.py` baselines (NB: file name historic; covers `search.py`, not the legacy `baselines.py`) |
| `test_policy_snapshots.py` | Policy state snapshot/resume |
| `test_ppo.py` | `policy/ppo.py` |
| `test_progress.py` | `target/progress.py` `emit_progress()` contract |
| `test_promotion.py` | `promotion.py` |
| `test_prompt_fragments.py` | `policy/_prompt_fragments.py` |
| `test_propose_batch.py` | `propose_batch` Protocol method |
| `test_required_calls.py` | `sandbox/validator.py::validate_required_calls` |
| `test_resource_pool.py` | `controller/resource_pool.py` |
| `test_runner.py` | Trial runner (continuous loop integration) |
| `test_runner_forecast.py` | Forecaster + runner integration |
| `test_scaffold.py` | Scaffold smoke |
| `test_sdpo.py` | `policy/sdpo.py` |
| `test_showcase_determinism.py` | Two-tier determinism for parallel-cancel-showcase |
| `test_shutdown.py` | `controller/shutdown.py` |
| `test_telemetry_rotation.py` | `telemetry/rotation.py` |
| `test_timeline.py` | `telemetry/timeline.py` |
| `test_tracking.py` | `tracking.py` |
| `test_trajectory.py` | `trajectory.py` |
| `tests/eval/test_prompt_eval.py` | LLM prompt-eval harness with fixtures |
| `tests/eval/test_real_llm.py` | 3 behavioral assertions vs Kimi K2.6 (gated on `MOONSHOT_API_KEY`) |
| `tests/eval/fixtures/*` | Test fixtures (`baseline_history.json`, `baseline_train.py`, `with_progress_train.py`, `real_responses/`) |

### 10.2 Adapt — 3 tests

| Test file | Adaptation |
|---|---|
| `test_forecasting.py` | Add a test case for SSL plateau-then-improvement curves (must NOT early-stop during the initial plateau). Per writeup §6 forecaster recalibration |
| `test_intra_iteration.py` | Add a test for the new SSL-plateau-aware default config (`min_steps=2000`) |
| `test_examples_smoke.py` | Update example list: drop `autoresearch-like`, `basilica-grpo`, `deberta-prompt-injection` from required-pass tier; add `examples/ijepa-cifar10/` (writeup §Phase-2) and `examples/trace-jepa/` (writeup §Phase-3) once those exist; gate them on Tier 2 (`validate` only) until the Phase-2 campaign closes |

### 10.3 Drop — 6 tests

Tied to modules being dropped per the §2-9 tables.

| Test file | Reason |
|---|---|
| `test_distillation_sink.py` | `distillation/sink.py` dropped |
| `test_distillation_trainer.py` | `distillation/trainer.py` dropped |
| `test_sdft.py` | `distillation/sdft.py` dropped |
| (existing tests for `eval/judge.py`, `eval/scoring.py`) | Modules dropped — note: there are no dedicated test files in upstream for these; coverage was via `test_runner.py` which is inherited and will need the legacy-eval references stripped during the import-rename pass |
| (existing tests for `controller/loop.py`, `sandbox/runner.py`) | Same — covered by `test_runner.py`. Strip references during rename pass |

### 10.4 Net new (Phase 1 of writeup) — to be authored, not inherited

These are listed in the writeup TODO under Phase 1; they have no upstream
analog:

- `tests/eval/test_probes.py` — `autojepa/eval/probes.py` linear / attentive / kNN
- `tests/eval/test_collapse.py` — RankMe / LiDAR / variance / effective rank
- `tests/eval/test_canary.py` — sanity-overfit canary
- `tests/test_masking.py` — composite mask invariants
- `tests/test_models_*.py` — encoder, predictor, EMA, loss tests
- `tests/test_gates.py` — decision gate engine
- `tests/test_forecaster_ssl.py` — SSL-recalibrated forecaster (may merge with the adapted `test_forecasting.py`)

---

## 11. Inheritance breakdown summary

| Category | `src/` modules | `tests/` files |
|---|---:|---:|
| **Inherit** (rename only) | 38 | 50 |
| **Adapt** (rename + targeted change) | 13 | 3 |
| **Replace** (same role, new code) | 1 (`README.md`) | 0 |
| **Drop** (not carried over) | 8 | 3 |

Net new in AutoJEPA (per writeup §7) is tracked in `TODO.md` Phase 1 and
not counted above. The dominant action is **Inherit**: the autoresearch-rl
codebase is doing what AutoJEPA needs, and the deltas are concentrated in
prompts, defaults, and one new `eval/` namespace.

---

## 12. Cherry-pick log seeds

Source: `git log --oneline -50` on the sibling repo at HEAD. These are
upstream improvements likely to appear over the next 1–3 months that
AutoJEPA will want to selectively port. Track decisions in
`docs/cherry-pick-log.md` (writeup standing-tasks).

| Recent commit (sibling) | What it adds | Likely AutoJEPA action |
|---|---|---|
| `ea71c96 docs(research): Phase A.3 — Trajectory-Aware-Ablation, n=5 paired` | Empirical paired ablation methodology for the trajectory feature | Port the **methodology** for our own A.3-equivalent (probe_auroc paired ablation between hybrid and llm_diff) |
| `750d9fc scripts(research): A.3 v2 runner` | Parameterized data-dir for ablation runner | Port; useful for our own ablation scripts |
| `f49b61d scripts(research): A.3 ablation runner + analysis` | Ablation harness (no committed data) | Port; same shape |
| `a9f9a9e feat(policy): AR_DISABLE_PROGRESS_SERIES kill switch` | Env var to disable progress-series in prompts (for ablation) | Port — useful for AutoJEPA ablations comparing progress-series vs scalar-only LLM context |
| `52a8b77 docs(research): Phase A.2 — Reproduction-SecurityJudge on Basilica K=4` | Reproduction protocol for K=4 parallel | Port the **protocol** as a template for our Phase 2 CIFAR Basilica reproduction |
| `702f43d docs(research): Phase A.1 — Competitive-Analysis vs 10 adjacent tools` | Competitive analysis methodology | Reference; AutoJEPA's competitive analysis is JEPA-ecosystem, not autoresearch-tool ecosystem |
| `55b2570 fix(config_validate): accept BASILICA_API_TOKEN` | SDK-correct env-var name | Port immediately on next sync |
| `ec680ba fix(basilica+parallel): two real bugs surfaced by probe5` | propose_batch over-fires; bootstrap killed itself before model download | Port immediately — both bugs hit any parallel Basilica campaign |
| `055e894 fix(basilica): make ready_timeout_s configurable` | Removes hardcoded 600s cap | Port immediately |
| `297efa5 fix(basilica): per-run-dir run/eval cache (race fix)` | Parallel `_last_train_outcome` race | Port immediately — silent kept-best attribution corruption |
| `0ae528f / cfaa7bf fix(security-judge): add hf_transfer to setup_cmd` | `HF_HUB_ENABLE_HF_TRANSFER=1` requires `hf_transfer` package | Port to `examples/ijepa-cifar10/deploy.py` and `examples/trace-jepa/deploy.py` setup_cmd if HF transfers are used |
| `5cbe559 fix(test): de-flake test_rewards_arrive_in_submission_order` | Flake fix | Port if the test is inherited (it is, under `test_parallel_engine.py`) |
| `6e6da41 trust: smoke all examples + CONTRIBUTING checklist + CLAUDE hard rule` | The "do not call done without realistic-config end-to-end" rule | **Already inherited** in §1 above; track upstream wording changes |
| `37af70e feat(config_validate): warn when telemetry paths would overwrite tracked data` | New `_check_telemetry_paths_not_overwriting_tracked` validator | Port immediately — same risk class for AutoJEPA campaigns |
| `fef66d1 fix(contract): basename comparison so workdir-prefixed mutable_file works` | The most-lurking bug of the prior arc | **Already inherited** in `controller/contract.py`; the regression test is inherited in `test_contract.py` |
| `f4b8d5a fix(llm): allow per-call temperature + bump default to 1.0 for Kimi compat` | Per-call temperature kwarg | Port — JEPA campaigns will use the same provider abstraction |
| `8b27f9b fix(basilica): hash-based de-dup for cancel control uploads` | SHA-256 caching of upload bodies | Port immediately |
| `d6577ae fix(basilica): propagate cancel control file to deployment (#19)` | `_propagate_control` POSTs to `/control` per poll tick | **Already inherited** in `target/basilica.py` |
| `9d52c3c fix(parallel): wire intra-iteration cancel safely (#24)` | Per-worker progress paths | **Already inherited** |

**Likely future-3-month upstream adds** to watch:

- Multi-LoRA target (deferred Phase-5 in autoresearch-rl, documented in `docs/research/RLix-Phase5-Deferred.md`). If revived upstream, AutoJEPA will need to assess relevance — JEPA encoders are not LoRA-natural but Trace-JEPA fine-tuning passes might be.
- Real-LLM eval expansion (`tests/eval/test_real_llm.py` currently has 3 assertions; likely to grow). Port the harness changes; rewrite the JEPA-specific assertions ourselves.
- Forecaster generalization (autoresearch-rl uses pure power-law; SSL needs a sigmoid-or-piecewise variant). If upstream adds it, port. If not, AutoJEPA owns it under writeup §Phase-1 `forecaster.py` adaptation.
- Cost-aware policies (Basilica $/iter). Likely to land upstream because Basilica is the prime target both projects share.

Cap upstream sync work at ~1h/wk per writeup standing-tasks.

---

## 13. External baseline references

Distilled paper notes added 2026-05-15 to back the Phase-3 `examples/trace-jepa/`
design and evaluation. These are baseline / cite-only references; **none of
them enters the core framework** (`src/autojepa/`). They land in the codebase
at the Phase-3 example level only — `examples/trace-jepa/` config search-space
and `examples/trace-jepa/` evaluation harness — preserving the rule that
contrib-style features stay in examples (writeup §8 / ADR rejection of the
contrib-namespace pattern).

| Distilled note | Upstream | Family | Codebase entry point |
|---|---|---|---|
| [mts-jepa.md](mts-jepa.md) | arxiv:2602.04643 | JEPA / time-series | Phase 3 — `examples/trace-jepa/config.yaml` `codebook_size` and `codebook_loss_weight` search dimensions; Phase-3 evaluation baseline (the JEPA-on-time-series row) |
| [jepa-automotive-monitoring.md](jepa-automotive-monitoring.md) | arxiv:2602.09985 | JEPA / automotive AD | Phase 3 — architectural reference for the `codebook_size=0, codebook_loss_weight=0` control row of the trace-jepa search (vanilla JEPA + classical AD on top, same split as `src/autojepa/eval/`) |
| [jepa-av-security-survey.md](jepa-av-security-survey.md) | ScienceDirect S1474034626002909 | JEPA AV survey / cite-only | Phase 3 — related-work citation only; **not a baseline**, **not a search dimension**. The source page is not ingested in Alexandria as of 2026-05-15; only the `topic=JEPA-Security-Gap` belief is. Documented in the note. |
| [ssl-ids-landscape.md](ssl-ids-landscape.md) | arxiv:2509.16625 (GraphIDS), 2502.07119 (SAFE), 2509.06550 (CLAN), 2505.08816 (Transformer-Contrastive IDS) | Non-JEPA SSL-IDS baselines | Phase 3 — evaluation baselines (the MAE-based row, GraphIDS or SAFE; the contrastive rows CLAN and Transformer-Contrastive IDS as additional controls) |

Scope rules:
- Phase-2 (`examples/ijepa-cifar10/`) is unaffected. It remains the
  vanilla-I-JEPA reproduction kill-criterion run; do not add §13 entries to
  it.
- Phase-3 (`examples/trace-jepa/`) consumes these references via its
  `config.yaml` search dimensions (codebook) and its evaluation harness
  (external baselines). See `TODO.md` Phase 3 for the operational gates.
- Per the `topic=JEPA-Security-Gap` belief in Alexandria (asserted 2026-05-15),
  no published JEPA paper targets logs, agent traces, prompt-injection
  detection, eBPF/syscall traces, or container observability. The four
  references above are the closest published prior art, which is why they
  are baselines rather than precedents.
