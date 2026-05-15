"""Unit tests for autojepa.eval.probes.

Validates the factory builders return correctly-typed
stable-pretraining callbacks with AutoJEPA-flavored defaults.

Marked `jepa` because the module imports torch + stable_pretraining.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
spt = pytest.importorskip("stable_pretraining")
pl = pytest.importorskip("lightning.pytorch")

from autojepa.eval.probes import (  # noqa: E402
    ProbeBundle,
    build_knn_probe,
    build_lidar,
    build_linear_probe,
    build_rankme,
    default_probes,
)

pytestmark = pytest.mark.jepa


class _MinimalModule(pl.LightningModule):
    """Smallest Lightning module that satisfies OnlineProbe's wrap_configure_model.

    OnlineProbe wraps `pl_module.configure_model` to register the
    probe submodule into the parent module hierarchy. We need a real
    LightningModule subclass; an attribute-only stub fails because
    OnlineProbe.__init__ accesses LightningModule-specific hooks.
    """

    def __init__(self) -> None:
        super().__init__()


class TestBuildLinearProbe:
    def test_returns_online_probe(self) -> None:
        cb = build_linear_probe(
            module=_MinimalModule(),
            embed_dim=128,
            num_classes=10,
        )
        assert isinstance(cb, spt.callbacks.OnlineProbe)

    def test_binary_picks_auroc(self) -> None:
        cb = build_linear_probe(
            module=_MinimalModule(),
            embed_dim=64,
            num_classes=2,
        )
        # OnlineProbe stores its metrics dict; key reflects what the
        # AutoJEPA train.py later forwards via emit_progress.
        assert "probe_auroc" in cb.metrics

    def test_multiclass_picks_accuracy(self) -> None:
        cb = build_linear_probe(
            module=_MinimalModule(),
            embed_dim=64,
            num_classes=10,
        )
        assert "probe_acc" in cb.metrics

    def test_rejects_zero_embed_dim(self) -> None:
        with pytest.raises(ValueError, match="embed_dim"):
            build_linear_probe(module=_MinimalModule(), embed_dim=0, num_classes=2)

    def test_rejects_single_class(self) -> None:
        with pytest.raises(ValueError, match="num_classes"):
            build_linear_probe(module=_MinimalModule(), embed_dim=8, num_classes=1)


class TestBuildKnnProbe:
    def test_returns_online_knn(self) -> None:
        cb = build_knn_probe()
        assert isinstance(cb, spt.callbacks.OnlineKNN)

    def test_rejects_zero_k(self) -> None:
        with pytest.raises(ValueError, match="k"):
            build_knn_probe(k=0)

    def test_rejects_zero_queue_length(self) -> None:
        with pytest.raises(ValueError, match="queue_length"):
            build_knn_probe(queue_length=0)


class TestBuildRankMe:
    def test_returns_rankme(self) -> None:
        cb = build_rankme(target_shape=128)
        assert isinstance(cb, spt.callbacks.RankMe)

    def test_rejects_zero_queue_length(self) -> None:
        with pytest.raises(ValueError, match="queue_length"):
            build_rankme(target_shape=128, queue_length=0)


class TestBuildLidar:
    def test_returns_lidar(self) -> None:
        cb = build_lidar(target_shape=128, n_classes=10)
        assert isinstance(cb, spt.callbacks.LiDAR)

    def test_rejects_one_class(self) -> None:
        with pytest.raises(ValueError, match="n_classes"):
            build_lidar(target_shape=128, n_classes=1)

    def test_rejects_zero_samples_per_class(self) -> None:
        with pytest.raises(ValueError, match="samples_per_class"):
            build_lidar(target_shape=128, n_classes=10, samples_per_class=0)


class TestDefaultProbes:
    def test_returns_full_bundle(self) -> None:
        bundle = default_probes(
            module=_MinimalModule(),
            embed_dim=128,
            num_classes=10,
        )
        assert isinstance(bundle, ProbeBundle)
        assert isinstance(bundle.linear_probe, spt.callbacks.OnlineProbe)
        assert isinstance(bundle.knn_probe, spt.callbacks.OnlineKNN)
        assert isinstance(bundle.rankme, spt.callbacks.RankMe)
        assert isinstance(bundle.lidar, spt.callbacks.LiDAR)

    def test_as_list_yields_four_callbacks(self) -> None:
        bundle = default_probes(
            module=_MinimalModule(),
            embed_dim=64,
            num_classes=10,
        )
        out = bundle.as_list()
        assert len(out) == 4
        # Lightning's Callback base class is the contract spt callbacks
        # extend. The bundle is what the AutoJEPA train.py registers
        # directly on Trainer(callbacks=...).
        assert all(isinstance(cb, pl.Callback) for cb in out)
