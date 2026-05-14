"""SSL loss zoo for AutoJEPA.

Per ADR-003 we wrap `stable_pretraining.losses` rather than reimplement
the standard SSL objectives. This module exposes:

1. The closed-form latent-distance helpers `l1_loss` and `l2_loss`
   used by the I-JEPA-shaped predictor objective (writeup §I-JEPA
   distilled doc — L2 distance between predictor output and EMA
   target embeddings).
2. Re-exports of the SSL collapse-defense losses
   (`VICRegLoss`, `BarlowTwinsLoss`, `BYOLLoss`, `DINOv1Loss`,
   `NTXEntLoss`, `NegativeCosineSimilarity`) so a `train.py` only
   needs to import from `autojepa.models.losses`.
3. A flat `LOSS_REGISTRY` mapping string keys to factory callables —
   the LLM diff policy can switch losses by editing the config field
   `loss_type` instead of rewriting import statements.

The C-JEPA paper (writeup §6.4) recommends VICReg as the AutoJEPA
default because its variance + covariance terms train the model to
satisfy the same `latent_variance > 0.3` invariant the program.md
validator enforces.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from stable_pretraining.losses import (
    BarlowTwinsLoss,
    BYOLLoss,
    DINOv1Loss,
    NegativeCosineSimilarity,
    NTXEntLoss,
    VICRegLoss,
)

__all__ = [
    "l1_loss",
    "l2_loss",
    "VICRegLoss",
    "BarlowTwinsLoss",
    "BYOLLoss",
    "DINOv1Loss",
    "NTXEntLoss",
    "NegativeCosineSimilarity",
    "LOSS_REGISTRY",
    "build_loss",
]


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-element L1 distance between predictor output and target
    embeddings. Mean reduction matches the I-JEPA loss specification.
    """
    _check_pair(prediction, target)
    return F.l1_loss(prediction, target, reduction="mean")


def l2_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-element L2 (mean squared) distance. The default loss in
    I-JEPA and V-JEPA 2 (writeup `docs/research/I-JEPA.md` §2).
    """
    _check_pair(prediction, target)
    return F.mse_loss(prediction, target, reduction="mean")


def _check_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}"
        )
    if prediction.numel() == 0:
        raise ValueError("prediction tensor is empty")


LOSS_REGISTRY: dict[str, Callable[..., Any]] = {
    "l1": l1_loss,
    "l2": l2_loss,
    "vicreg": VICRegLoss,
    "barlow_twins": BarlowTwinsLoss,
    "byol": BYOLLoss,
    "dino_v1": DINOv1Loss,
    "ntxent": NTXEntLoss,
    "neg_cosine": NegativeCosineSimilarity,
}


def build_loss(name: str, **kwargs: Any) -> Any:
    """Resolve a loss by name from the registry.

    Closures (`l1`, `l2`) are returned unwrapped so the caller can call
    them directly. Class entries (`VICRegLoss`, `BarlowTwinsLoss`, ...)
    are instantiated with the keyword arguments — the caller owns the
    instance lifecycle.

    Raises KeyError with the supported names if `name` is unknown so
    the LLM diff policy gets a useful error message.
    """
    if name not in LOSS_REGISTRY:
        raise KeyError(
            f"unknown loss '{name}'; supported: {sorted(LOSS_REGISTRY.keys())}"
        )
    entry = LOSS_REGISTRY[name]
    if isinstance(entry, type):
        return entry(**kwargs)
    if kwargs:
        raise TypeError(
            f"loss '{name}' is a function and takes no constructor kwargs; got {kwargs}"
        )
    return entry
