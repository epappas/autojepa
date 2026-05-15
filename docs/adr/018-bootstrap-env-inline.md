# ADR-018: Inline AR_* env vars into the Basilica bootstrap script

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** Live debugging of v12 Basilica re-smoke (commit b3fbad1) where the
  outcome.json contract from ADR-015 was working on the train.py side but the
  controller-side polling never found the file.

## Context

ADR-015 introduced the `outcome.json` iter-done contract: train.py
writes `<AR_MODEL_DIR>/outcome.json` on exit, the basilica adapter
polls the bootstrap server's `/model/files` endpoint, and the iter is
marked done as soon as outcome.json appears. This eliminates the
3600 s `target.timeout_s` wait that plagued v9/v10 and gives the
LLM-diff ratchet a working `best_value` to advance.

ADR-016 (custom Docker image baked with all deps) made deployments
boot in seconds instead of cold-installing for ~30 min per iter.

The v12 re-smoke (commit b3fbad1, against the freshly-public
`ghcr.io/epappas/autojepa-runtime:phase2`) ran the full pipeline:
container ready in seconds, full 6000-step pretrain on GPU, peak
`probe_auroc=0.296`, train.py wrote outcome.json correctly. But the
controller still hit the 3600 s timeout and marked the iter failed.
events.jsonl recorded only proposal events for v12, no `iteration`
event.

Diagnosis (via `kubectl exec`):

```
$ kubectl exec -n u-github-434149 <pod> -- env | grep ^AR_
(empty)

$ kubectl exec -n u-github-434149 <pod> -- cat /app/artifacts/outcome.json
{"status":"ok","metrics":{"probe_auroc":0.2946,...},"completed_steps":6000,...}
```

Train.py wrote outcome.json to `/app/artifacts/` (its hard-coded
default fallback when `AR_MODEL_DIR` is unset). The bootstrap script
reads `_model_dir = os.environ.get("AR_MODEL_DIR", "")` at startup;
with the env var missing, `_model_dir = ""` and the `/model/files`
endpoint short-circuits to `{"files": []}`. The adapter's polling
therefore never sees the file even though it exists on disk.

Why was `AR_MODEL_DIR` empty in the container? The basilica adapter
DOES build an `env` dict (basilica.py line 258) and DOES pass it to
`client.create_deployment(..., env=env)` (line 320). Either:

- the Basilica SDK's `env=` parameter is silently dropped for custom
  Docker images, or
- the cluster's pod-spec rendering loses env vars when `image:` points
  outside the default registry, or
- some other layer between SDK and pod-spec eats them.

We did not pin the exact cause — the Basilica SDK is closed-source on
that path and the bug only reproduces against custom images. But the
empirical fact (env empty in container despite `env=env` passed) is
verified.

## Decision

Inline the env dict into the bootstrap script SOURCE itself. Add a
`$env_inject` substitution to `_BOOTSTRAP_TEMPLATE` that produces a
literal `_os.environ.update(json.loads('{"AR_MODEL_DIR": "...",
"AR_PARAMS_JSON": "...", ...}'))` call at the top of the script,
before any `os.environ.get()` reads.

The `env=env` argument to `create_deployment` STAYS — this is
defense-in-depth, not a replacement. If/when Basilica fixes the
SDK-level propagation, the inline form remains correct (it just
overwrites with the same values).

`_build_bootstrap_cmd` gains an `env: dict[str, str] | None = None`
parameter. The single call site in `BasilicaTarget` passes the env
dict through. JSON encoding (rather than Python repr) handles
arbitrary string values safely (quotes, backslashes, unicode all
round-trip cleanly).

## Consequences

- **Positive:** AR_MODEL_DIR / AR_PARAMS_JSON / AR_PARAM_* /
  AR_PROGRESS_FILE / AR_CONTROL_FILE are reliably visible to the
  bootstrap process and all its child processes (setup_cmd via
  `subprocess.check_call`, train_cmd via `subprocess.call(env=os.environ)`),
  regardless of cluster-side env-propagation behaviour.
- **Positive:** outcome.json polling (ADR-015) now actually has
  `_model_dir` populated → `/model/files` returns the real file
  listing → adapter detects iter completion → `best_value` advances
  → LLM-diff ratchet engages → Phase-2 falsifier becomes testable.
- **Positive:** Three new unit tests in `tests/test_basilica_unit.py`
  lock in the inject-before-read invariant.
- **Negative:** Env values are baked into the bootstrap script string
  at deployment time. Sensitive secrets (HF_TOKEN, etc.) end up in
  the deployment command argv visible via `kubectl describe pod`.
  Mitigation: the existing code only inlines what it would have set
  via `env=env` anyway; secrets exposure surface is identical.
- **Negative:** A larger bootstrap script (~+200 chars per env entry).
  Negligible — the script is Python source code, not a hot path.

## How to apply

- Any new env var the controller wants the trial to see goes into
  the `env` dict in `BasilicaTarget._deploy_and_collect`. The
  bootstrap injection picks it up automatically.
- If a secret needs to NOT be in the bootstrap script (e.g. if we
  later host the script publicly), add a separate `secret_env` path
  that uses k8s `Secret` references instead. Not needed today —
  HF_TOKEN is already in env=env via the SDK and always was.
- The inline contract is verified by `tests/test_basilica_unit.py::TestBootstrapEnvInjection`.
  Any future refactor of `_build_bootstrap_cmd` that breaks the
  inject-before-read order will fail the
  `test_env_dict_is_inlined_as_json_literal::inject_pos < read_pos`
  assertion.
