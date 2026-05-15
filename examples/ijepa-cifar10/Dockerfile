# AutoJEPA Phase-2 runtime image.
#
# Bakes the heavy GPU/ML dependency stack ahead of time so each
# Basilica iter does not pay the ~5-10 min cold-install penalty
# (apt-install git + pip-install torch+lightning+transformers+spt+timm).
# With this image, deploy.py's setup_cmd shrinks to a single
# `pip install --no-deps autojepa @ git+...` (<60 s on a warm net).
#
# See ADR-016 for the rationale and pin choices.

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

# Heavy GPU/ML stack. Pins match deploy.py::_build_setup_cmd as of
# commit 994489a (transformers <4.48 to dodge torch 2.4 infer_schema
# crash) and stable-pretraining 0.1.6 (spt 0.2 not yet validated).
RUN pip install --no-cache-dir \
        "torch>=2.4" "lightning>=2.4" "torchmetrics>=1.4" "torchvision" \
        "transformers>=4.47,<4.48" "datasets" \
        "stable-pretraining>=0.1.6,<0.2" "timm"

# autojepa core deps (the source itself is git-installed at runtime
# so the campaign can pin a specific SHA per iter).
RUN pip install --no-cache-dir \
        "numpy>=1.24" "pydantic>=2.7" "pyyaml>=6.0" "typer>=0.12" \
        "basilica-sdk>=0.20"

# Sanity import — fail the build, not the iter, on a broken pin.
RUN python3 -c "import torch, lightning, transformers, stable_pretraining, timm; \
    print('built:', torch.__version__, lightning.__version__, \
    transformers.__version__, stable_pretraining.__version__, timm.__version__)"

WORKDIR /app
