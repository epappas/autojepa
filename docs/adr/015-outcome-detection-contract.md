# ADR-015: outcome.json contract for iter-done detection

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** `docs/phase-2-runtime-evidence.md` (v1-v10 live debug log); kubectl logs of campaigns at `badbe80` and `82c5e72`; `src/autojepa/target/basilica.py::_poll_for_metrics`; `examples/ijepa-cifar10/train.py`

## Context

The basilica adapter's `_poll_for_metrics` originally relied on two
signals to decide an iter was done:

1. Regex over container stdout matching `\w+=[\d.eE+-]+` against a
   hardcoded `_KNOWN_METRIC_KEYS` allowlist.
2. The container exiting (`status.is_failed` true).

Both signals failed in the live Phase-2 falsifier campaigns at
`badbe80` and `82c5e72`. `train.py` correctly trained the model and
printed `final probe_auroc=0.XXXX after N steps` (peak observed
0.281, well above CIFAR random 0.10), then `main()` returned 0 and
the Python process exited cleanly. But:

- The metric-key allowlist (`probe_auroc`, `canary_loss` etc.) did
  not match the regex pattern in the way the trial logged them
  every iteration — the controller missed many real completions.
- After process exit, k8s restarted the container per its restart
  policy, so `status.is_failed` never went true; the deployment
  stayed Active, the adapter kept polling, and finally hit the
  `target.timeout_s: 3600` ceiling.

Result: every iter was marked `failed/discard`, `best_value` stayed
`null`, the LLM-diff ratchet never engaged, and the campaign
`{"iterations": 3, "best_value": null, "best_score": Infinity}`
was unfalsifiable for Phase-2 purposes.

## Decision

Introduce an explicit **outcome.json contract** that the trial
script writes and the adapter reads. The contract is one file at
`$AR_MODEL_DIR/outcome.json` with a fixed schema:

```json
{
  "status": "ok" | "failed",
  "metrics": {"probe_auroc": 0.281, "loss": 0.008, ...},
  "elapsed_s": 1417,
  "completed_steps": 4000,
  "step_target": 4000,
  "ts": 1778900000,
  "reason": "<short string>"  // required when status == "failed"
}
```

### Trial side (e.g., `examples/ijepa-cifar10/train.py`)

- A `_write_outcome(...)` helper writes the file atomically (`*.tmp`
  then `replace`) at every exit path:
  - canary failure → `status="failed"`, embeds `canary_loss` + reason.
  - successful pretrain → `status="ok"`, embeds best `probe_auroc`,
    final `loss`, `canary_loss`.
  - import error → `status="failed"`, reason `import_error: ...`.
  - any uncaught exception (`_entrypoint` last-resort) →
    `status="failed"`, reason `unhandled_exception: ...`.

### Adapter side (`src/autojepa/target/basilica.py`)

- `_poll_for_metrics` calls `_fetch_outcome(deployment)` on every
  poll cycle. That helper hits the existing bootstrap endpoints
  `/model/files` and `/model/download/<rel>` to check for and pull
  `outcome.json` — no new HTTP routes needed.
- When present, `_finalize_outcome(...)` downloads the rest of the
  model dir, deletes the deployment, and returns a `RunOutcome`
  with the embedded status + metrics in <= one poll interval.
- Legacy log-pattern path stays intact for back-compat — examples
  inherited from `autoresearch-rl` (and `examples/trace-jepa/`)
  that don't yet write outcome.json keep working unchanged.
- Final timeout path runs one last `_fetch_outcome` check to catch
  the case where the trial wrote the file during the final
  `post_trial_sleep_s` window.

## Consequences

- **Positive:** Iter completion is a single explicit event, not an
  inferred one. The 3600s timeout no longer dominates iter
  elapsed_s; observed iter time drops to actual training time
  (~5-25 min per the Phase-2 budget in `program.md`).
- **Positive:** Failure status is structured. A canary-failed iter
  reports `status=failed reason=canary_loss=...` instead of being
  indistinguishable from a network outage.
- **Positive:** `best_value` updates correctly because real
  metrics arrive via `RunOutcome.metrics` instead of being lost to
  timeout-induced discard.
- **Negative:** Examples must opt in by writing the file. The
  adapter's fallback path keeps inherited examples working but a
  new example author has to remember the contract. `program.md`
  documents it; the per-example README references the ADR.
- **Negative:** The atomic-write requires `$AR_MODEL_DIR` to exist
  and be writable. The trial helper creates the directory; targets
  that don't set `AR_MODEL_DIR` (rare) skip the write and fall back
  to log-pattern detection.

## How to apply

- New examples adopt the contract by copying the `_write_outcome`
  helper from `examples/ijepa-cifar10/train.py` and calling it on
  every exit path.
- The `program.md` for the example documents the contract under
  "Required runtime calls".
- The basilica adapter requires no per-example code; the polling
  loop checks for `outcome.json` regardless of which example is
  running.
- Other targets (`target.command`, `target.http`) do not need this
  contract — they wait for the subprocess to exit, so iter-done is
  unambiguous already.
