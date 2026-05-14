# Karpathy/autoresearch — the foundation kernel

- upstream: https://github.com/karpathy/autoresearch
- mirror inspected: `/tmp/autoresearch_karpathy/` @ HEAD on 2026-05-15
- distilled: 2026-05-15

This is the load-bearing pattern that `autoresearch-rl` and AutoJEPA both
inherit unchanged. Everything else in either fork is structure built around
this kernel.

## 1. The minimal viable loop

The repo deliberately fits in four files. Sizes (lines, on the inspected HEAD):

| File | LOC | Mutability | Role |
|---|---:|---|---|
| `train.py` | 630 | mutable | full GPT model + Muon/AdamW optimizer + training loop; the **only** file the agent edits |
| `prepare.py` | 389 | frozen | data download, BPE tokenizer training, dataloader, `evaluate_bpb()` ground-truth metric |
| `program.md` | 114 | human-edited | natural-language instructions to the LLM agent |
| `README.md` | 92 | human-edited | repo intro |

Plus `pyproject.toml` (43 LOC) and `analysis.ipynb` (notebook). Total agent-touchable surface: one Python file, ~630 LOC.

**Iteration budget.** Verbatim from the README:

> "training runs for a fixed 5-minute time budget (wall clock, excluding
> startup/compilation), regardless of the details of your compute."

> "approx 12 experiments/hour and approx 100 experiments while you sleep"

**Hardware floor.** Single NVIDIA GPU (tested on H100). No distributed
training, no complex configs. One GPU, one file, one metric.

**Metric.** `val_bpb` (validation bits per byte). Lower is better.
Vocab-size-independent so architectural changes are fairly compared.

This — single file, fixed wall-clock budget, single scalar metric, no
distributed training — is the **kernel**. Everything `autoresearch-rl` adds
is engineering scaffolding around it. AutoJEPA changes the *workload*
(JEPA SSL, Trace-JEPA) but inherits the kernel pattern.

## 2. The frozen / mutable contract

Verbatim from `program.md`:

> "**What you CAN do:**
> - Modify `train.py` — this is the only file you edit. Everything is fair
>   game: model architecture, optimizer, hyperparameters, training loop,
>   batch size, model size, etc.
>
> **What you CANNOT do:**
> - Modify `prepare.py`. It is read-only. It contains the fixed evaluation,
>   data loading, tokenizer, and training constants (time budget, sequence
>   length, etc).
> - Install new packages or add dependencies. You can only use what's
>   already in `pyproject.toml`.
> - Modify the evaluation harness. The `evaluate_bpb` function in
>   `prepare.py` is the ground truth metric."

Why this works as a contract:

1. **Trust boundary.** Evaluation is in `prepare.py`. The LLM physically
   cannot game the eval because it cannot edit the eval.
2. **Comparability.** Sequence length, time budget, dataset, tokenizer all
   live in `prepare.py`. Two iterations are comparable iff they ran
   against the same `prepare.py`.
3. **Diff scope.** The agent's diff surface is exactly one file. Diffs are
   reviewable, cherry-pickable, and resettable.

`autoresearch-rl` enforces this contract programmatically (`controller/contract.py`,
`sandbox/validator.py`); Karpathy's repo enforces it via `program.md` and
trust in the agent.

## 3. The diff "validator" — Karpathy version

Karpathy's autoresearch has **no** AST-walking validator and **no**
`required_calls` mechanism. The contract is enforced via `program.md`
prose ("CANNOT modify prepare.py") plus the human reviewing `git diff`
between iterations. That is the entire "validator" in the kernel.

`autoresearch-rl` adds two strict layers on top of this:

- **`sandbox/validator.py::validate_diff`** rejects diffs that introduce
  a banned token (`import socket`, `requests.`, `subprocess.Popen(`,
  `os.system(`).
- **`sandbox/ast_policy.py::validate_python_source`** AST-walks the
  added lines, rejects forbidden imports (`socket`, `requests`, `httpx`,
  `urllib`, `subprocess`) and forbidden calls (`os.system`,
  `subprocess.Popen`, `subprocess.run`).
- **`sandbox/validator.py::validate_required_calls(pre, post, required)`**
  AST-walks both pre- and post-patch sources; rejects diffs that strip
  *all* calls to any name in `required` (default
  `policy.required_calls = ["emit_progress"]`). Direct enforcement of
  load-bearing instrumentation.

The required_calls mechanism in `autoresearch-rl` is what guarantees that
LLM-edited `train.py` keeps emitting progress — without it, an LLM could
simplify the loop, drop the `emit_progress(...)` call, and silently break
the cooperative-cancel and forecasting paths. This is **net new** vs the
upstream kernel; the kernel doesn't need it because it doesn't have
intra-iteration cancel.

## 4. The `emit_progress` contract

Karpathy's autoresearch has **no** `emit_progress` call. The script runs
to its 5-minute budget, prints a final summary, and the agent
`grep`s `^val_bpb:` from the log:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

`autoresearch-rl` introduces the streaming protocol verbatim
(`src/autoresearch_rl/target/progress.py`):

```python
def emit_progress(
    *,
    step: int,
    step_target: int,
    metrics: dict[str, float] | None = None,
    iter: int | None = None,
    exit_on_cancel: bool = True,
) -> bool:
```

Each call writes one JSON line to `$AR_PROGRESS_FILE` and reads
`$AR_CONTROL_FILE`; on cancel it `sys.exit(42)` (cooperative-cancel exit
code). Without `$AR_PROGRESS_FILE` set the call is a no-op so the trial
script also runs cleanly outside the harness — same `train.py` works in
both the kernel pattern and the wrapped harness.

This is the second piece of net-new scaffolding `autoresearch-rl` adds to
the kernel: **without** `emit_progress`, intra-iteration cancellation
(Phase 2) and the power-law forecaster (`forecasting.py`) cannot work.

## 5. Why this kernel is load-bearing

The Karpathy pattern is the smallest possible thing that gives you all of:

- **Reproducibility.** Frozen `prepare.py` + fixed time budget means two
  iterations on the same hardware are directly comparable.
- **Reviewability.** One mutable file, one scalar metric, one diff per
  iteration. A human can `git log` and read the actual algorithmic story
  of what the agent tried.
- **Autonomy.** No human-in-the-loop required between iterations. The
  agent runs until you stop it.
- **Composability.** The same `train.py` runs unchanged whether you call
  it via `python train.py` or via a wrapper that injects env vars.

`autoresearch-rl` and AutoJEPA both keep all four properties intact.
Every piece of scaffolding either fork adds is justified by *not*
breaking these properties:

- The harness must not require `train.py` to import the harness (preserved:
  `emit_progress` no-ops without env vars).
- The harness must not modify the eval definition (preserved: still in
  `prepare.py`, still frozen).
- The harness must keep iterations bounded in wall-clock (preserved:
  `controller.max_wall_time_s` is the modern equivalent).

If any future feature breaks one of these four, it is *changing* the
kernel, not extending it — and should be rejected on those grounds.

## 6. What `autoresearch-rl` adds (and AutoJEPA inherits)

| Addition | Module(s) in `autoresearch-rl` | Why it doesn't break the kernel |
|---|---|---|
| Run on Basilica GPU cloud | `target/basilica.py`, `target/registry.py` | New `TargetAdapter`; `train.py` is unaware |
| Multiple iterations in parallel | `controller/parallel_engine.py`, `controller/resource_pool.py` | Each worker runs the kernel loop in isolation |
| LLM-driven param search (HPO) | `policy/llm_search.py`, `policy/_prompt_fragments.py` | Replaces "human picks next experiment"; trial script unchanged |
| LLM-driven code diffs | `policy/llm_diff.py`, `controller/diff_executor.py`, `controller/contract.py` | Codifies the "agent edits `train.py`" pattern as patch + retry |
| Hybrid policy | `policy/hybrid.py` | Composition of the above two |
| Cooperative cancel + power-law forecaster | `controller/intra_iteration.py`, `forecasting.py`, `target/progress.py`, `target/progress_reader.py` | Requires `emit_progress` (net new) |
| Diff guardrails | `sandbox/validator.py`, `sandbox/ast_policy.py` | Required because LLM diffs are not human-reviewed inline |
| Learnable policy (PPO) | `policy/learned.py`, `policy/learned_search.py`, `policy/ppo.py`, `policy/gae.py`, `policy/sdpo.py` | Long-campaign optimization on top of the per-iter loop |
| Distillation | `distillation/sdft.py`, `distillation/sink.py`, `distillation/trainer.py` | Off-loop; consumes the per-iter ledger |
| Telemetry | `telemetry/*` | Off-loop; consumes the per-iter ledger |
| Config + validation | `config.py`, `config_validate.py`, `cli.py` | Replaces "edit constants in `train.py`" with declarative YAML |
| Model artifact persistence | `target/basilica.py` (`/model/files`, `/model/download`) | Off-loop; downloads after iter completes |

Everything in this table is **infrastructure**. None of it changes the
kernel: one mutable file, one frozen evaluator, one bounded-time iteration,
one scalar metric, one keep/discard decision.

## 7. The minimal experiment loop, verbatim from `program.md`

```
LOOP FOREVER:
  1. Look at the git state: the current branch/commit we're on
  2. Tune `train.py` with an experimental idea by directly hacking the code.
  3. git commit
  4. Run the experiment: `uv run train.py > run.log 2>&1`
  5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
  6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log`
     to read the Python stack trace and attempt a fix.
  7. Record the results in the tsv
  8. If val_bpb improved (lower), you "advance" the branch, keeping the
     git commit
  9. If val_bpb is equal or worse, you git reset back to where you started
```

This is the entire algorithm. `autoresearch-rl/controller/continuous.py`
is, structurally, the same nine steps with: `(1)` replaced by checkpoint
state, `(2)` by `policy.propose()`, `(3)` skipped (diff applied
in-memory), `(4)` by `target.run()`, `(5)` by `outcome.metrics`, `(6)` by
`outcome.status == "failed"` handling, `(7)` by `ledger.write_row(...)`,
`(8)/(9)` by the keep/discard score comparison.

AutoJEPA's `controller/continuous.py` will be the same loop with
`metrics["probe_auroc"]` swapped for `metrics["val_bpb"]` and the
program-md prose replaced with the JEPA-specific gates from writeup §6.
That's the inheritance.

## 8. Cross-links

- `docs/research/AutoresearchRL-Inheritance-Map.md` — module-by-module
  carry-over plan that this doc justifies at the kernel level.
- `docs/ARCHITECTURE.md` (forthcoming for AutoJEPA; mirror of
  `../autoresearch-rl/docs/ARCHITECTURE.md`).
- Architecture writeup §5 — the explicit "what we keep" list maps 1:1 to
  the inheritance table in §6 above.
- Architecture writeup §Phase-0 — concrete carry-over checklist that
  begins by copying the kernel into AutoJEPA verbatim.
