# ADR-005: Preserve `AR_` env-var prefix on the trial-runtime contract

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** Architecture writeup §5 ("workload-agnostic primitives" inherited unchanged)

## Context

`autoresearch-rl` uses an `AR_`-prefixed env-var contract to inject
hyperparameters and runtime state into the trial subprocess:

| Env var | Set by | Read by |
|---|---|---|
| `AR_PARAMS_JSON` | controller | `train.py` (full hyperparameter dict) |
| `AR_PARAM_<NAME>` | controller | `train.py` (per-key shortcut) |
| `AR_PROGRESS_FILE` | controller | `target/progress.py::emit_progress` |
| `AR_CONTROL_FILE` | controller | `target/progress.py` (cancel polling) |
| `AR_MODEL_DIR` | controller | `train.py` (versioned output dir) |

These names are **inherited verbatim** by AutoJEPA. The carry-over
from `autoresearch-rl/src/autoresearch_rl/target/{command,basilica,progress,progress_reader}.py`
contains them; the autoresearch-rl test suite (inherited) asserts
exact-name behaviors.

## Decision

Keep the `AR_` prefix on all trial-runtime env vars. Do not rename to
`AJ_` or `AUTOJEPA_`.

## Consequences

- **Positive:** All inherited tests
  (`tests/test_command_progress.py`, `tests/test_progress.py`,
  `tests/test_basilica_unit.py`, etc.) pass without fixture changes.
- **Positive:** Cherry-picks from upstream `autoresearch-rl` apply
  cleanly; no rename pass needed at the env-var boundary.
- **Positive:** A user moving between `autoresearch-rl` and AutoJEPA
  trial scripts has the same runtime contract.
- **Negative:** Branding leaks — an AutoJEPA `train.py` script reads
  an `AR_`-prefixed env var. Acceptable cost (the prefix is opaque to
  downstream users; the docstrings on these vars explain the lineage).
- **Negative:** If the AutoJEPA contract ever diverges from the
  autoresearch-rl contract (e.g., adds `AR_PROBE_DATASET_DIR`), the
  prefix is misleading. Mitigation: namespace new AutoJEPA-only vars
  with `AR_PROBE_*`, `AR_JEPA_*`, etc., to keep the prefix consistent
  while signaling AutoJEPA-specific scope in the suffix.

## How to apply

- Never write a new env-var contract with a different prefix.
- New AutoJEPA-specific env vars use `AR_<scope>_<name>` form
  (e.g. `AR_PROBE_DATASET_DIR`).
- Documentation refers to "the `AR_` runtime contract inherited from
  autoresearch-rl" rather than rebranding it.
