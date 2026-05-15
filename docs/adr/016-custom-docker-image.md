# ADR-016: Custom Docker image with deps baked

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** epappas
- **Source:** `docs/phase-2-runtime-evidence.md` (v1-v10 lessons); kubectl logs of crash-loops at v4/v5; `examples/ijepa-cifar10/deploy.py::_build_setup_cmd` (pre-change); `examples/ijepa-cifar10/Dockerfile`

## Context

Every Basilica iter ran the same heavy `setup_cmd` from a cold base
image (`pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel`):

1. `apt-get install git` — base image ships without git, the
   `pip install ... @ git+https://...` form needs it (root cause of
   the v1-v5 crash-loop, only diagnosed via kubectl per the
   runtime-evidence doc).
2. `pip install torch lightning torchvision transformers==4.47.x
   datasets stable-pretraining timm basilica-sdk autojepa @ git+...`
   ~2-2.5 GB of wheels resolved over the network on first install.

Cost per iter:
- Cold container: 5-10 min just for setup_cmd.
- Warm container (k8s reuses the wheel cache occasionally): ~1-2 min.

The 1800s `ready_timeout_s` from the inherited config was tight; we
bumped it to 3600s after live failures (commit `5ab0262`). That
moved the bottleneck off ready-detection but did not address the
underlying waste: the same install runs every iter, of every
campaign, on every node. Across a 20-iter Phase-2 campaign that is
~2 hours of pure setup overhead — comparable to the actual training
budget.

## Decision

Bake the heavy stack into a custom image published to
`ghcr.io/epappas/autojepa-runtime:phase2` (built by
`examples/ijepa-cifar10/build_image.sh`).

### Baked layers

| Layer | Contents |
|---|---|
| Base | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel` |
| apt | `git ca-certificates curl` (git for `pip install ... @ git+`) |
| pip heavy | `torch>=2.4 lightning>=2.4 torchmetrics>=1.4 torchvision`, `transformers>=4.47,<4.48`, `datasets`, `stable-pretraining>=0.1.6,<0.2`, `timm` |
| pip core | `numpy>=1.24 pydantic>=2.7 pyyaml>=6.0 typer>=0.12 basilica-sdk>=0.20` |
| sanity | RUN-time import check fails the build, not the iter |

### Per-iter setup_cmd shrinks to

1. `pip install --no-deps autojepa @ git+https://...@<sha>` — small
   wheel, no transitive resolution because `--no-deps` skips the
   already-baked dependency tree.
2. Base64-inject `train.py` + `prepare.py` into `/app/`.
3. Sanity import.

Empirically <60s on a warm container; `ready_timeout_s` drops from
3600s to 600s in `config.yaml`.

### Pinning policy

- `transformers <4.48` is a hard pin: 4.50+ uses
  `from __future__ import annotations` in `integrations/moe.py`,
  which produces lazy-string type hints that torch 2.4's
  `infer_schema()` cannot resolve during
  `torch.library.custom_op` registration. Verified via kubectl
  logs of v6 (commit `954ea70`).
- `stable-pretraining <0.2` is a hard pin: 0.2 has not been
  validated against the autojepa.models EMA helpers. Bump in a
  follow-up ADR after a smoke run.
- `torch>=2.4` is the floor — earlier versions miss the
  `compile`-friendly attention path the IJEPA wrapper uses.

## Consequences

- **Positive:** Per-iter overhead drops by ~5-10 min. A 20-iter
  campaign saves ~2 hours of A100 time, ~$30-100 by the writeup
  §12.4 cost model.
- **Positive:** `ready_timeout_s: 600` is back to the upstream
  `BasilicaConfig` default (no per-example bump needed in
  `config.yaml`).
- **Positive:** Pin choices live in one Dockerfile, not duplicated
  across `deploy.py::_build_setup_cmd` and the ad-hoc `pip install`
  comments. Bumping a pin is one PR, not a campaign-long debug
  loop.
- **Negative:** Image is large (~7 GB after the baked layers). Per
  GHCR storage quota (10 GB free tier) we keep one tag (`phase2`),
  not per-commit tags.
- **Negative:** The Dockerfile + image are now part of the
  reproducibility surface. A reader cloning the repo must run
  `build_image.sh` (or use the published `ghcr.io/...:phase2` tag)
  before deploying.
- **Negative:** Phase-3 (`examples/trace-jepa/`) will likely need
  its own image because the dep set differs (webdataset shards).
  Plan: `ghcr.io/epappas/autojepa-runtime:phase3` with a sibling
  Dockerfile. Cross-example sharing of the heavy layer happens
  automatically via Docker layer caching.

## How to apply

- Author the Dockerfile next to the example's `train.py`.
- `build_image.sh` builds + pushes to `ghcr.io/epappas/<name>:<tag>`.
  The author needs `gh auth login --scopes write:packages` and
  `docker login ghcr.io` (one-time).
- `deploy.py::_build_setup_cmd` is restricted to the layers that
  vary per-iter (the autojepa SHA + the example files).
- `config.yaml::target.basilica.image` references the published tag.
- `ready_timeout_s` drops to <=600s; bumps mean a regression in
  setup_cmd or image pull and should be investigated, not absorbed.
