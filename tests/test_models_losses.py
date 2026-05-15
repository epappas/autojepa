"""Unit tests for autojepa.models.losses.

Validates:
- l1_loss / l2_loss closed-form behavior on synthetic tensors
- shape-mismatch rejection
- LOSS_REGISTRY surface (every name resolves to something callable)
- build_loss factory dispatch and error path

Marked `jepa` because the module imports torch + stable_pretraining
(installed via the [jepa] extra).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
spt = pytest.importorskip("stable_pretraining")

from autojepa.models.losses import (  # noqa: E402
    LOSS_REGISTRY,
    BarlowTwinsLoss,
    BYOLLoss,
    DINOv1Loss,
    NTXEntLoss,
    VICRegLoss,
    build_loss,
    l1_loss,
    l2_loss,
)

pytestmark = pytest.mark.jepa


class TestL2Loss:
    def test_zero_when_identical(self) -> None:
        a = torch.randn(8, 16)
        assert l2_loss(a, a).item() == pytest.approx(0.0)

    def test_matches_torch_mse(self) -> None:
        a = torch.randn(8, 16)
        b = torch.randn(8, 16)
        expected = torch.nn.functional.mse_loss(a, b, reduction="mean").item()
        assert l2_loss(a, b).item() == pytest.approx(expected)

    def test_returns_scalar_tensor(self) -> None:
        a = torch.randn(4, 8)
        b = torch.randn(4, 8)
        out = l2_loss(a, b)
        assert out.ndim == 0
        assert out.dtype == torch.float32

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            l2_loss(torch.randn(4, 8), torch.randn(4, 16))

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            l2_loss(torch.empty(0, 8), torch.empty(0, 8))


class TestL1Loss:
    def test_zero_when_identical(self) -> None:
        a = torch.randn(8, 16)
        assert l1_loss(a, a).item() == pytest.approx(0.0)

    def test_matches_torch_l1(self) -> None:
        a = torch.randn(8, 16)
        b = torch.randn(8, 16)
        expected = torch.nn.functional.l1_loss(a, b, reduction="mean").item()
        assert l1_loss(a, b).item() == pytest.approx(expected)

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            l1_loss(torch.randn(4, 8), torch.randn(8, 4))


class TestRegistry:
    def test_registry_contains_writeup_required_losses(self) -> None:
        # writeup §6.4 program.md template lists VICReg, Barlow, DINO-center
        # as the high-value diff targets; L2 is the I-JEPA default.
        for required in ("l1", "l2", "vicreg", "barlow_twins", "dino_v1"):
            assert required in LOSS_REGISTRY, f"registry missing required loss {required}"

    def test_class_entries_match_spt(self) -> None:
        assert LOSS_REGISTRY["vicreg"] is VICRegLoss
        assert LOSS_REGISTRY["barlow_twins"] is BarlowTwinsLoss
        assert LOSS_REGISTRY["byol"] is BYOLLoss
        assert LOSS_REGISTRY["dino_v1"] is DINOv1Loss
        assert LOSS_REGISTRY["ntxent"] is NTXEntLoss


class TestBuildLoss:
    def test_returns_function_for_l2(self) -> None:
        fn = build_loss("l2")
        assert callable(fn)
        out = fn(torch.randn(4, 8), torch.randn(4, 8))
        assert out.ndim == 0

    def test_returns_function_for_l1(self) -> None:
        fn = build_loss("l1")
        assert callable(fn)
        out = fn(torch.randn(4, 8), torch.randn(4, 8))
        assert out.ndim == 0

    def test_instantiates_class_loss(self) -> None:
        loss_module = build_loss("vicreg")
        assert isinstance(loss_module, VICRegLoss)

    def test_unknown_name_raises_with_helpful_list(self) -> None:
        with pytest.raises(KeyError, match="vicreg"):
            build_loss("not_a_real_loss")

    def test_kwargs_to_function_raise(self) -> None:
        with pytest.raises(TypeError, match="no constructor kwargs"):
            build_loss("l2", reduction="sum")
