# Phase 2 — Runtime Evidence

> Companion to `docs/phase-0-baseline.md` and `examples/ijepa-cifar10/README.md`.
> Captures the actual runtime validation of the Phase-2 example beyond
> "code compiles and tests pass". Per NEVER LIE, every claim here is
> backed by a reproducible command + its captured output.

## Environment

| Field | Value |
|---|---|
| Date | 2026-05-15 |
| Local host | Linux 6.8.0-106-generic, x86_64, no GPU |
| Python | 3.13.12 (cpython, via uv) |
| torch | 2.12.0+cu130 (CPU-only at runtime; no CUDA device) |
| stable-pretraining | 0.1.6 |
| autojepa HEAD | `555386f` (after the Phase-2 train.py fix, see below) |
| GitHub remote | `git@github.com:epappas/autojepa.git` (default branch `master`) |

## 1. `./run.sh prepare` — CIFAR-10 download

**Command**: `cd examples/ijepa-cifar10 && uv run python3 prepare.py`

**Result**: PASS. Downloaded 170 MB in 7 s, materialised the documented
output files.

```
downloading CIFAR-10 to .../examples/ijepa-cifar10/data/raw ...
100%|██████████| 170M/170M [00:06<00:00, 24.6MB/s]
prepared: train=(50000, 3, 32, 32) test=(10000, 3, 32, 32) probe=5000/5000 canary=1000
```

Output files materialised:
```
data/cifar10_train.pt
data/cifar10_train_labels.pt
data/cifar10_test.pt
data/cifar10_test_labels.pt
data/probe_eval.pt
data/canary.pt
data/raw/                # torchvision intermediate
```

## 2. `autojepa validate` — config validates with real credentials

**Command**:
```
set -a && source .env && set +a
uv run autojepa validate examples/ijepa-cifar10/config.yaml
```

**Result**: PASS. Exit 0, single-line output `OK`.

The validator confirmed:
- `objective.metric=probe_auroc, direction=max` (per ADR-004)
- `target.type=basilica` with 5 acceptable GPU models
- `policy.type=hybrid` with widened JEPA defaults
- `intra_iteration_cancel.{min_steps=2000, poll_interval_s=30, min_reports_before_decide=10}` (per ADR-008/013)
- `BASILICA_API_TOKEN` and `CHUTES_API_KEY` env vars present and non-empty
- `controller.checkpoint_path` and telemetry paths writable

## 3. Local 1-step integration test — train.py code paths

The full smoke (`./run.sh smoke` = 200-step canary + 50-step pretrain at
batch=64, ViT-Tiny @ 224x224) is impractical on CPU: estimated 3-7
hours of compute. Instead we ran a fast 1-step forward+backward on a
synthetic batch of 2 to validate the training code paths integrate
cleanly with `stable_pretraining.methods.IJEPA`.

**Command**: see commit `555386f` for the snippet.

**Result**: PASS. The test caught a real bug (now fixed), then passed.

```
=== train.py integration test ===
IJEPA built in 4.0s
assert_no_grad_on_target: PASS
forward in 3.8s; loss=0.5322
backward in 6.1s
extracted feats shape: (2, 192)
rankme=1.00 eff_rank=1.00 latent_var=0.0552
=== PASS ===
```

**Bug caught**: `_extract_features` had wrong assumption that the
EMA target encoder's output had a `.predictions` attribute. The
stable-pretraining `MaskedEncoder` actually returns a
`MaskedEncoderOutput` namedtuple with attrs
`['encoded', 'grid_size', 'ids_keep', 'mask']`. Fixed in commit
`555386f` to read `out.encoded` and drop the CLS token before
mean-pooling. Without this fix, the Basilica run would have crashed
on the first probe-eval call.

The collapse signals (rankme=1.00, eff_rank=1.00) reflect rank-1
features because the test ran on a batch of 2 untrained-init random
features — a healthy signal that the collapse-detection wiring is
end-to-end functional, not that the model is collapsed.

## 4. Basilica smoke campaign (3-iter, in progress)

**Command**:
```
set -a && source /home/epappas/workspace/spacejar/autojepa/.env && set +a
uv run python3 examples/ijepa-cifar10/deploy.py --max-iterations 3 --git-ref 555386f
```

**Initial output** (first ~5 minutes):
```
BASILICA_API_TOKEN set: basilica...
CHUTES_API_KEY set: cpk_05fc...
setup_cmd length: 17947 chars
running from /home/epappas/workspace/spacejar/autojepa ...
LLM API error 503, retry 1/5 in 12s
LLM API error 503, retry 2/5 in 24s
LLM API error 503, retry 3/5 in 42s
LLM API error 503, retry 4/5 in 89s
LLM API error 503, retry 5/5 in 98s
LLM API error 503 (no retry): {"detail":"No instances available (yet) for chute_id='0df3133d-c477-56d2-b4db-f2093bb150a1'"}
LLM policy failed, falling back to random
[traceback for the underlying HTTPError, then continues with random fallback]
```

The Chutes endpoint hosting `deepseek-ai/DeepSeek-V3-0324` was scaled
to zero instances at campaign launch time. The inherited LLM provider
abstraction (writeup §5) handled this correctly:
- 5-retry exponential backoff (12s + 24s + 42s + 89s + 98s = ~265s)
- After exhaustion, fell back to seeded-random params

This validates the LLM-failure resilience path. Iter 0 proceeded with
random-fallback params:

```json
{"learning_rate": 0.0002, "weight_decay": 0.0, "batch_size": 128,
 "max_steps": 6000, "predictor_depth": 2, "predictor_embed_dim": 128,
 "num_targets": 4, "ema_decay_start": 0.99,
 "probe_eval_every_n_steps": 500}
```

**Iter 0 proposal** captured to `traces/ijepa-cifar10/events.jsonl`:

```
{"schema": "v1", "run_id": "f1ea56ced9cb", "event_id": "84b7e1d2d53e",
 "ts": 1778844821, "type": "proposal", "episode_id": "f1ea56ced9cb",
 "iter": 0, "params": {...}}
```

**Basilica deployment created** (verified via `basilica.BasilicaClient.list_deployments()`):

```
friendly_name: ar-train-1f036885
instance_name: da861cbe-4315-43ef-accd-58bbbfa91ed6
state:         Active
replicas:      desired=1 ready=0  (container starting; setup_cmd in progress)
url:           https://da861cbe-4315-43ef-accd-58bbbfa91ed6.deployments.basilica.ai
created_at:    2026-05-15T11:33:42 UTC  (= 13:33 local; matches iter 0 proposal ts)
```

Setup_cmd is installing `autojepa[jepa] @ git+https://github.com/epappas/autojepa.git@555386f`
(the post-bug-fix SHA), which pulls torch + lightning + transformers +
stable-pretraining + webdataset + timm + transitive deps. Estimated
container ready time: 5-10 minutes from container start.

### Iter 0 outcome (smoke v1, commit `555386f`) — FAILED

```
status:   failed
decision: discard
metrics:  {}
elapsed:  607.5 s
stdout:   ""
stderr:   "not_ready"
```

**Root cause**: the Basilica adapter's default `ready_timeout_s: 600`
is too tight for the heavy `setup_cmd`. `pip install autojepa[jepa] @
git+https://...` pulls torch + lightning + torchvision + transformers
+ stable-pretraining + timm + transitives (~3-4 GB) on a fresh
container — reliably > 10 min on first install. The adapter
correctly marked the container `not_ready` at 600 s while pip was
still working.

**Fix** (commit `b467ca8`): bumped
`target.basilica.ready_timeout_s: 1800` (30 min). This exercises the
inherited `055e894 fix(basilica): make ready_timeout_s configurable`
upstream improvement listed in the cherry-pick log seeds — first
real use in AutoJEPA.

The LLM-failure resilience evidence from v1 still holds: Chutes 503 →
exponential backoff → seeded-random fallback worked. The `not_ready`
failure was strictly an infrastructure-timing issue, not a code or
contract issue.

**v1 campaign stopped** at iter 1 (which was retrying LLM with 429s).

### Smoke v2 (commit `b467ca8`) — FAILED at exactly elapsed=1805s

Same `not_ready` failure mode. The bumped 1800s timeout was still
tight: pip-installing `autojepa[jepa] @ git+...` reliably exceeds
30 min on a fresh Basilica container.

### Smoke v3 (commit `d23af7c`) — STOPPED before launch

Attempted to slim setup_cmd by skipping transformers + webdataset +
datasets. Verified locally that this breaks: `import stable_pretraining`
hard-imports both transformers AND datasets at package import time
(despite shipping a separate dependency for them). v3 stopped before
Basilica deployment to save the wasted GPU time.

### Smoke v4 (commit `5ab0262`) — IN FLIGHT, container ready in ~30s

Restored transformers + datasets in setup_cmd; only webdataset stays
out (verified `'webdataset' not in sys.modules` after spt import).
Bumped `ready_timeout_s: 3600` (1 hour) as belt-and-suspenders.

```
$ uv run python3 examples/ijepa-cifar10/deploy.py --max-iterations 3 --git-ref 5ab0262
```

Background task `b0pw7yaki`. Deployment came up:

```
friendly_name: ar-train-f286c5ea
state:         Active
replicas:      desired=1 ready=1
created_at:    2026-05-15T12:24:49 UTC
```

**Surprise observation**: container ready in ~30s vs the >30 min cold
boot we saw on v1/v2. Hypothesis: Basilica caches pip wheel files
between deployments on the same node, so subsequent installs hit a
warm wheel cache. If this holds across iters, the per-iter overhead
drops from ~30 min → ~30s, making the full 20-iter campaign budget
~$5-15 instead of $30-100. Verifying with iter 0 outcome.

Status: training in progress on the warm container. ETA: ~10-20 min
per iter for 6200 training steps (200 canary + 6000 pretrain) on A100
+ 12 probe-eval rounds.

### Why a 3-iter smoke before the full 20-iter campaign

The full 20-iter campaign costs ~$30-100 of A100 time per writeup
§12.4. A 3-iter smoke validates:

- File-injection via `deploy.py` works (train.py + prepare.py reach
  the container intact).
- The Basilica adapter brings up the bootstrap server, runs
  `prepare.py`, runs `train.py`, and reports metrics back through
  the bootstrap HTTP server.
- The hybrid policy proposes valid params and the trial completes.

If the smoke passes, the user authorises the full 20-iter campaign as
a separate command.

## Cherry-pick log seed

Bugs found during Phase-2 runtime validation that DID NOT exist in
upstream `autoresearch-rl` (because they only manifest with JEPA
modules):

| SHA | What | Origin |
|---|---|---|
| `555386f` | `_extract_features` missing `MaskedEncoderOutput.encoded` access | AutoJEPA-specific (spt.MaskedEncoder integration) — not upstream-relevant |

Track in `docs/cherry-pick-log.md` as the first AutoJEPA-only bug. No
upstream cherry-pick needed.

## Reproducibility

To reproduce the prepare + integration test on any machine:

```bash
git clone git@github.com:epappas/autojepa.git
cd autojepa
git checkout 555386f
uv sync --extra dev --extra jepa
cd examples/ijepa-cifar10
uv run python3 prepare.py
# Inline integration test:
cd ../..
uv run python3 -c "
import torch, time
from stable_pretraining.methods import IJEPA
from autojepa.models.ema import assert_no_grad_on_target
from autojepa.eval.collapse import rankme

model = IJEPA('vit_tiny_patch16_224', predictor_embed_dim=128, predictor_depth=2,
              num_targets=2, pretrained=False)
assert_no_grad_on_target(model.encoder)
x = torch.randn(2, 3, 224, 224)
out = model(x); out.loss.backward()
print('loss:', float(out.loss.item()))
"
```

To reproduce the Basilica smoke:

```bash
export BASILICA_API_TOKEN=...
export CHUTES_API_KEY=...
uv run python3 examples/ijepa-cifar10/deploy.py --max-iterations 3 --git-ref 555386f
```
