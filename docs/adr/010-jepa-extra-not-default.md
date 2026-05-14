# ADR-010: Torch / Lightning live in `[jepa]` extra, not default deps

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** Phase-0 dependency layout

## Context

The framework code in `src/autojepa/` (controller, policy, target,
telemetry, sandbox, config) does not import torch. Only the JEPA-specific
modules added in Phase-1 (`eval/`, `masking/`, `models/`) and the
example training scripts under `examples/<name>/train.py` import torch
+ Lightning + transformers + webdataset.

Forcing torch as a default dependency would:

1. Bloat the install for users running framework-level tests
   (CI on a small CPU runner).
2. Force a CUDA toolchain decision (cu130, cu124, cpu) at install time
   even for non-JEPA workflows.
3. Slow down `uv sync` from ~5 s to ~60 s on a fresh checkout.

## Decision

`pyproject.toml` puts torch + Lightning + torchmetrics + transformers +
webdataset under a `[jepa]` optional extra. Default install includes
only framework deps + `basilica-sdk` (per ADR-002).

`uv sync --extra dev` runs the framework test suite (470 tests) without
torch.
`uv sync --extra dev --extra jepa` enables JEPA-specific modules and
their tests (currently +18 collapse tests; will grow with each Phase-1
module).

JEPA-specific tests are marked `@pytest.mark.jepa` so the framework
test suite can be run in isolation when JEPA deps are absent.

## Consequences

- **Positive:** Framework CI stays fast and slim.
- **Positive:** Users who only need the controller/policy plumbing for
  non-JEPA workloads can install AutoJEPA without dragging torch.
- **Positive:** Different Phase-2/3 examples can declare different
  torch versions or accelerators in their own `pyproject.toml` without
  conflicting with the framework dep.
- **Negative:** A test that imports torch directly inside a `tests/`
  module will fail under default install. Mitigation: every JEPA test
  starts with `torch = pytest.importorskip("torch")` and the
  `pytestmark = pytest.mark.jepa` module-level marker.
- **Negative:** Two-tier CI matrix: the JEPA extras run a separate job.
  Acceptable cost.

## How to apply

- Any new module under `src/autojepa/` that imports torch belongs in
  the JEPA-extra-gated namespaces (`eval/`, `masking/`, `models/`,
  forthcoming `parsing/jepa_metrics.py` if needed). The framework
  layers (`controller/`, `policy/`, `target/`, `telemetry/`,
  `sandbox/`, `config.py`, `cli.py`) must remain torch-free.
- A `mypy` boundary check is desirable but not yet enforced; for now
  CI relies on the import-graph being inspectable by hand.
- New JEPA tests follow the `pytest.importorskip` + `pytestmark`
  template from `tests/eval/test_collapse.py`.
