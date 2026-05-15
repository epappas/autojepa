# AutoJEPA Phase-3 trace-jepa runtime image.
#
# Bakes the heavy GPU/ML dependency stack (torch, lightning,
# stable-pretraining, transformers, timm, webdataset) ahead of time
# so each Basilica iter does not pay the ~5-10 min cold-install
# penalty. With this image, deploy.py's setup_cmd shrinks to a single
# `pip install --no-deps autojepa @ git+...` (<60 s on a warm net).
#
# Trace-jepa adds `webdataset` to the Phase-2 dep set; everything else
# matches examples/ijepa-cifar10/Dockerfile so the two examples can
# share the same baked image when ADR-016's
# `ghcr.io/epappas/autojepa-runtime:phase2` is published. Until then,
# building from this Dockerfile produces a phase3-tagged sibling.

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# git is required for `pip install foo @ git+https://...` form (the
# Basilica base image ships without it). curl + ca-certificates are
# defensive — some pip indexes need them.
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        git ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Heavy GPU/ML stack. transformers <4.48 dodges torch 2.4 infer_schema
# crash; stable-pretraining <0.2 because 0.2 is not yet validated.
# webdataset is the trace-jepa-specific addition vs Phase-2.
RUN pip install --no-cache-dir \
        "torch>=2.4" "lightning>=2.4" "torchmetrics>=1.4" "torchvision" \
        "transformers>=4.47,<4.48" "datasets" \
        "stable-pretraining>=0.1.6,<0.2" "timm" "webdataset>=0.2"

# autojepa core deps (the source itself is git-installed at runtime
# so the campaign can pin a specific SHA per iter).
RUN pip install --no-cache-dir \
        "numpy>=1.24" "pydantic>=2.7" "pyyaml>=6.0" "typer>=0.12" \
        "basilica-sdk>=0.20"

# Sanity import — fail the build, not the iter, on a broken pin.
RUN python3 -c "import torch, lightning, transformers, stable_pretraining, timm, webdataset; \
    print('built:', torch.__version__, lightning.__version__, \
    transformers.__version__, stable_pretraining.__version__, timm.__version__)"

WORKDIR /app
