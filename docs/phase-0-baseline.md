# Phase 0 Baseline — Bootstrap Verification

**Date:** 2026-05-15
**Status:** Phase 0 complete. AutoJEPA inherits the autoresearch-rl framework as a clean fork.

---

## What was carried over

From `/home/epappas/workspace/spacejar/autoresearch-rl/` (sibling repo, this repo's `git remote upstream`):

| Item | Source | Destination | Notes |
|---|---|---|---|
| Source modules | `src/autoresearch_rl/` (65 files) | `src/autojepa/` | Module rename `autoresearch_rl` → `autojepa` applied via sed |
| Test suite | `tests/` (52 files) | `tests/` | Same import-rename pass |
| Build scripts | `scripts/` (4 files) | `scripts/` | Excluded `scripts/research/` (autoresearch-rl-specific reports) |
| `Makefile`, `pyproject.toml`, `CLAUDE.md`, `CHANGELOG.md`, `CONTRIBUTING.md` | repo root | repo root | Identity strings updated |
| Examples | `examples/` (115 MB of model artifacts + recipes) | **NOT carried over** | Replaced by `examples/ijepa-cifar10/` (Phase 2) and `examples/trace-jepa/` (Phase 3) |

## Identity changes

- Package name: `autoresearch_rl` → `autojepa`
- CLI entrypoint: `autoresearch-rl` → `autojepa`
- Default git author for ephemeral diff worktrees: `autoresearch` → `autojepa`
- HF Hub metadata key: `autoresearch_version.json` → `autojepa_version.json`
- `pyproject.toml` `[project] name`: `autoresearch-rl` → `autojepa`, version reset to `0.1.0`
- **Basilica is now a default dependency** (`basilica-sdk>=0.20`), not optional — Basilica is the prime deployment target
- Added `[project.optional-dependencies] jepa` extra: torch, lightning, torchmetrics, transformers, webdataset (installed only when running JEPA workloads)

## Inherited contract preserved

Per writeup §5 (workload-agnostic primitives), these are NOT renamed:

- `AR_PARAMS_JSON` env var prefix used by `CommandTarget` and `BasilicaTarget` to inject hyperparameters
- `emit_progress(step, step_target, metrics={...})` contract for streaming metrics from `train.py` to the controller
- Frozen-`prepare.py` / mutable-`train.py` split with AST-walking diff validator
- Hybrid policy mechanics, exit-code 42 cooperative cancellation, ResourcePool, ThreadPoolExecutor parallel iters

These will only change when Phase 1 introduces JEPA-specific calibration (forecaster `forecast_target` field, `program.md` template).

## Validation evidence

Verified by running the full inherited suite on 2026-05-15 against the renamed package:

```
$ uv run pytest -q --ignore=tests/eval/test_real_llm.py
9 failed, 470 passed, 5 skipped in 23.70s

$ uv run ruff check src/ tests/
All checks passed!

$ uv run mypy src/
Success: no issues found in 65 source files

$ uv run autojepa --help
Usage: autojepa [OPTIONS] COMMAND [ARGS]...
  Autonomous ML experiment loop.
  Commands: run, validate, print-config, status, run-one, upload
```

### Test failure analysis (the 9)

All 9 failures share a single root cause: tests reference `examples/<name>/prepare.py` paths that point at fixtures from the upstream `autoresearch-rl` `examples/` tree which was deliberately not carried over. **None of the failures are caused by the rename pass or by missing functionality.**

| Test | Missing fixture | Resolution |
|---|---|---|
| `test_examples_smoke.py::test_llm_diff_example_produces_real_best_value[examples/minimal-trainable-target]` | `examples/minimal-trainable-target/` | Will be retired or replaced when AutoJEPA examples land |
| `test_examples_smoke.py::test_llm_diff_example_produces_real_best_value[examples/autoresearch-like]` | `examples/autoresearch-like/` | Same |
| `test_examples_smoke.py::test_example_validates_cleanly_with_stub_credentials[examples/basilica-grpo]` | `examples/basilica-grpo/` | Same |
| `test_examples_smoke.py::test_example_validates_cleanly_with_stub_credentials[examples/security-judge]` | `examples/security-judge/` | Same |
| `test_examples_smoke.py::test_example_validates_cleanly_with_stub_credentials[examples/deberta-prompt-injection]` | `examples/deberta-prompt-injection/` | Same |
| `test_loop_autonomy.py::test_loop_stops_on_no_improve_limit` | `examples/autoresearch-like/prepare.py` | Will be repointed at `examples/ijepa-cifar10/` in Phase 2 |
| `test_loop_autonomy.py::test_loop_stops_on_failure_rate_limit` | Same | Same |
| `test_loop_autonomy.py::test_loop_stops_on_max_wall_time` | Same | Same |
| `test_scaffold.py::test_loop_runs` | Same | Same |

These 9 are the only known issues from the carry-over. **They will be re-pointed in Phase 2** when `examples/ijepa-cifar10/` lands.

Per the user's hard rule (`Never re-write tests or code to skip them`), the failing tests are NOT skipped or stubbed. They remain in the suite as a forcing function for Phase 2.

## What is NOT in this fork yet

- No JEPA-specific code (Phase 1)
- No `examples/` directory (Phase 2 onwards)
- No `docs/architecture.md`, no API reference docs
- `make smoke` target points at `tests/test_examples_smoke.py` which fails until Phase 2

## Cherry-pick log seeds

Upstream `autoresearch-rl` improvements that may want porting (cap attention to ~1h/wk, see writeup §12.6):

- (none yet — fork happened on 2026-05-15)

Tracked going forward in `docs/cherry-pick-log.md` (created by the autoresearch-rl-inheritance-map agent).
