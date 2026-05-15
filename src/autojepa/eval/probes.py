"""Probe + collapse Lightning callbacks for AutoJEPA.

Per ADR-003 these are thin factories over `stable_pretraining.callbacks`
rather than reimplementations:

| AutoJEPA factory     | Wraps                          | Purpose                          |
|----------------------|--------------------------------|----------------------------------|
| `build_linear_probe` | `spt.callbacks.OnlineProbe`    | probe_auroc / probe_acc scalar   |
| `build_knn_probe`    | `spt.callbacks.OnlineKNN`      | k-NN scalar (cheap surrogate)    |
| `build_rankme`       | `spt.callbacks.RankMe`         | RankMe collapse signal           |
| `build_lidar`        | `spt.callbacks.LiDAR`          | LiDAR collapse signal            |

`default_probes(num_classes, embed_dim)` returns the four-callback
bundle a typical AutoJEPA `train.py` registers on its Lightning Trainer.

The closed-form label-free collapse signals (`rankme`,
`effective_rank`, `latent_variance`) live in
`autojepa.eval.collapse` and do NOT require Lightning — see ADR-007.
This module's wrappers are for in-training Lightning-callback use only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torchmetrics
from stable_pretraining.callbacks import LiDAR, OnlineKNN, OnlineProbe, RankMe

# spt v0.1.x default queue lengths used in the published examples.
_DEFAULT_QUEUE_LENGTH = 10000
_DEFAULT_KNN_QUEUE_LENGTH = 20000
_DEFAULT_KNN_K = 10


def build_linear_probe(
    *,
    module: Any,
    embed_dim: int,
    num_classes: int,
    name: str = "linear_probe",
    input_key: str = "embedding",
    target_key: str = "label",
    optimizer: Any = None,
    scheduler: Any = None,
) -> OnlineProbe:
    """OnlineProbe with a single linear layer — the standard SSL probe.

    Reports `probe_auroc` (binary) or `probe_acc` (multiclass) into the
    Lightning logger; the AutoJEPA train.py extracts the scalar and
    forwards via `emit_progress(metrics={"probe_auroc": ...})`.
    """
    if embed_dim <= 0:
        raise ValueError(f"embed_dim must be positive; got {embed_dim}")
    if num_classes <= 1:
        raise ValueError(f"num_classes must be >= 2; got {num_classes}")
    metric = (
        torchmetrics.classification.BinaryAUROC()
        if num_classes == 2
        else torchmetrics.classification.MulticlassAccuracy(num_classes)
    )
    metric_key = "probe_auroc" if num_classes == 2 else "probe_acc"
    return OnlineProbe(
        module=module,
        name=name,
        input=input_key,
        target=target_key,
        probe=torch.nn.Linear(embed_dim, num_classes),
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        metrics={metric_key: metric},
    )


def build_knn_probe(
    *,
    name: str = "knn_probe",
    input_key: str = "embedding",
    target_key: str = "label",
    queue_length: int = _DEFAULT_KNN_QUEUE_LENGTH,
    metrics: dict | None = None,
    k: int = _DEFAULT_KNN_K,
) -> OnlineKNN:
    """OnlineKNN cheap surrogate. Used as a sanity gate alongside the
    linear probe when the dataset is small enough that linear-probe
    fitting is itself noisy.
    """
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    if queue_length <= 0:
        raise ValueError(f"queue_length must be positive; got {queue_length}")
    metrics_dict = metrics if metrics is not None else {"knn_acc": torchmetrics.classification.BinaryAUROC()}
    return OnlineKNN(
        name=name,
        input=input_key,
        target=target_key,
        queue_length=queue_length,
        metrics=metrics_dict,
        k=k,
    )


def build_rankme(
    *,
    target_shape: int | tuple[int, ...],
    name: str = "rankme",
    target_key: str = "embedding",
    queue_length: int = _DEFAULT_QUEUE_LENGTH,
) -> RankMe:
    """RankMe Lightning callback (effective-rank collapse signal).

    Mirrors the closed-form `autojepa.eval.collapse.rankme` but runs
    in-training as a callback. AutoJEPA uses both: collapse.rankme()
    inside the trial sidecar for fast cancellation, RankMe callback
    inside Lightning for live monitoring + logger emission.
    """
    if queue_length <= 0:
        raise ValueError(f"queue_length must be positive; got {queue_length}")
    return RankMe(
        name=name,
        target=target_key,
        queue_length=queue_length,
        target_shape=target_shape,
    )


def build_lidar(
    *,
    target_shape: int | tuple[int, ...],
    n_classes: int,
    name: str = "lidar",
    target_key: str = "embedding",
    queue_length: int = _DEFAULT_QUEUE_LENGTH,
    samples_per_class: int = 10,
) -> LiDAR:
    """LiDAR Lightning callback (LDA-based collapse signal).

    LiDAR requires per-class structure (it fits LDA over a queued
    sample of embeddings) so unlike RankMe it cannot be computed in
    the trial sidecar. This is why LiDAR lives in `eval/probes.py`
    rather than `eval/collapse.py` (ADR-007).
    """
    if n_classes <= 1:
        raise ValueError(f"n_classes must be >= 2; got {n_classes}")
    if samples_per_class <= 0:
        raise ValueError(f"samples_per_class must be positive; got {samples_per_class}")
    if queue_length <= 0:
        raise ValueError(f"queue_length must be positive; got {queue_length}")
    return LiDAR(
        name=name,
        target=target_key,
        queue_length=queue_length,
        target_shape=target_shape,
        n_classes=n_classes,
        samples_per_class=samples_per_class,
    )


@dataclass(frozen=True)
class ProbeBundle:
    """The four callbacks a typical AutoJEPA train.py registers."""

    linear_probe: OnlineProbe
    knn_probe: OnlineKNN
    rankme: RankMe
    lidar: LiDAR

    def as_list(self) -> list[Any]:
        return [self.linear_probe, self.knn_probe, self.rankme, self.lidar]


def default_probes(
    *,
    module: Any,
    embed_dim: int,
    num_classes: int,
) -> ProbeBundle:
    """Construct the standard AutoJEPA probe + collapse callback bundle.

    Caller registers `bundle.as_list()` on the Lightning Trainer's
    `callbacks=` argument.
    """
    return ProbeBundle(
        linear_probe=build_linear_probe(
            module=module, embed_dim=embed_dim, num_classes=num_classes
        ),
        knn_probe=build_knn_probe(),
        rankme=build_rankme(target_shape=embed_dim),
        lidar=build_lidar(target_shape=embed_dim, n_classes=num_classes),
    )
