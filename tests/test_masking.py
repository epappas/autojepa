"""Unit tests for autojepa.masking.

Validates:
- MultiBlockInfillMask boundary checks and shape invariants
- Determinism with seeded torch.Generator
- Non-overlapping context vs targets (the I-JEPA invariant)
- CompositeMask weight handling and delegation

Marked `jepa` because the module imports torch.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from autojepa.masking import (  # noqa: E402
    CompositeMask,
    FutureBlockMask,
    MaskOutput,
    MultiBlockInfillMask,
)

pytestmark = pytest.mark.jepa


def _seeded_gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class TestMultiBlockInfillMaskValidation:
    def test_default_construction_succeeds(self) -> None:
        m = MultiBlockInfillMask()
        assert m.n_targets == 4
        assert m.target_scale == (0.15, 0.20)

    def test_rejects_zero_n_targets(self) -> None:
        with pytest.raises(ValueError, match="n_targets"):
            MultiBlockInfillMask(n_targets=0)

    def test_rejects_invalid_scale(self) -> None:
        with pytest.raises(ValueError, match="target_scale"):
            MultiBlockInfillMask(target_scale=(0.5, 0.2))
        with pytest.raises(ValueError, match="target_scale"):
            MultiBlockInfillMask(target_scale=(0.0, 0.5))
        with pytest.raises(ValueError, match="target_scale"):
            MultiBlockInfillMask(target_scale=(0.5, 1.5))

    def test_rejects_invalid_aspect(self) -> None:
        with pytest.raises(ValueError, match="target_aspect"):
            MultiBlockInfillMask(target_aspect=(2.0, 1.0))


class TestMultiBlockInfillMaskShape:
    def test_sample_returns_correct_shapes(self) -> None:
        m = MultiBlockInfillMask(n_targets=4)
        out = m.sample(grid_h=14, grid_w=14, generator=_seeded_gen(1))
        assert isinstance(out, MaskOutput)
        n = 14 * 14
        assert out.context.shape == (n,)
        assert out.context.dtype == torch.bool
        assert len(out.targets) == 4
        for t in out.targets:
            assert t.shape == (n,)
            assert t.dtype == torch.bool

    def test_n_targets_is_respected(self) -> None:
        m = MultiBlockInfillMask(n_targets=7)
        out = m.sample(grid_h=8, grid_w=8, generator=_seeded_gen(2))
        assert len(out.targets) == 7

    def test_rejects_invalid_grid(self) -> None:
        m = MultiBlockInfillMask()
        with pytest.raises(ValueError, match="grid dims"):
            m.sample(grid_h=0, grid_w=8)
        with pytest.raises(ValueError, match="grid dims"):
            m.sample(grid_h=8, grid_w=-1)


class TestMultiBlockInfillMaskInvariants:
    def test_context_does_not_overlap_targets(self) -> None:
        """The I-JEPA invariant: target patches are removed from context."""
        m = MultiBlockInfillMask(n_targets=4)
        for seed in range(10):
            out = m.sample(grid_h=14, grid_w=14, generator=_seeded_gen(seed))
            for t in out.targets:
                overlap = (out.context & t).any()
                assert not overlap, f"context overlaps a target on seed {seed}"

    def test_context_is_non_empty(self) -> None:
        m = MultiBlockInfillMask(n_targets=4)
        for seed in range(20):
            out = m.sample(grid_h=14, grid_w=14, generator=_seeded_gen(seed))
            assert out.context.any(), f"empty context on seed {seed}"

    def test_each_target_is_non_empty(self) -> None:
        m = MultiBlockInfillMask(n_targets=4)
        for seed in range(20):
            out = m.sample(grid_h=14, grid_w=14, generator=_seeded_gen(seed))
            for i, t in enumerate(out.targets):
                assert t.any(), f"target {i} empty on seed {seed}"

    def test_seeded_runs_are_deterministic(self) -> None:
        m = MultiBlockInfillMask(n_targets=4)
        a = m.sample(grid_h=14, grid_w=14, generator=_seeded_gen(42))
        b = m.sample(grid_h=14, grid_w=14, generator=_seeded_gen(42))
        assert torch.equal(a.context, b.context)
        for ta, tb in zip(a.targets, b.targets, strict=True):
            assert torch.equal(ta, tb)

    def test_different_seeds_produce_different_masks(self) -> None:
        m = MultiBlockInfillMask(n_targets=4)
        a = m.sample(grid_h=14, grid_w=14, generator=_seeded_gen(1))
        b = m.sample(grid_h=14, grid_w=14, generator=_seeded_gen(2))
        # extremely unlikely to coincide on a 196-position grid
        assert not torch.equal(a.context, b.context)


class TestCompositeMaskValidation:
    def test_rejects_empty_samplers(self) -> None:
        with pytest.raises(ValueError, match="at least one sampler"):
            CompositeMask(samplers=[])

    def test_rejects_negative_weights(self) -> None:
        m = MultiBlockInfillMask()
        with pytest.raises(ValueError, match="non-negative"):
            CompositeMask(samplers=[(m, -0.1)])

    def test_rejects_zero_weight_sum(self) -> None:
        m = MultiBlockInfillMask()
        with pytest.raises(ValueError, match="positive value"):
            CompositeMask(samplers=[(m, 0.0)])


class TestCompositeMaskDelegation:
    def test_single_sampler_delegates_directly(self) -> None:
        primary = MultiBlockInfillMask(n_targets=3)
        composite = CompositeMask(samplers=[(primary, 1.0)])
        out = composite.sample(grid_h=10, grid_w=10, generator=_seeded_gen(7))
        assert isinstance(out, MaskOutput)
        assert len(out.targets) == 3

    def test_weighted_choice_picks_both_branches(self) -> None:
        a = MultiBlockInfillMask(n_targets=2)
        b = MultiBlockInfillMask(n_targets=8)
        composite = CompositeMask(samplers=[(a, 1.0), (b, 1.0)])
        seen_lens = set()
        # Many seeded calls should hit both branches.
        for s in range(40):
            out = composite.sample(grid_h=10, grid_w=10, generator=_seeded_gen(s))
            seen_lens.add(len(out.targets))
        assert seen_lens == {2, 8}, f"only saw target counts {seen_lens}"


class TestFutureBlockMaskValidation:
    def test_default_construction_succeeds(self) -> None:
        m = FutureBlockMask()
        assert m.n_targets == 4
        assert m.context_fraction == (0.4, 0.6)

    def test_rejects_zero_n_targets(self) -> None:
        with pytest.raises(ValueError, match="n_targets"):
            FutureBlockMask(n_targets=0)

    def test_rejects_invalid_context_fraction(self) -> None:
        with pytest.raises(ValueError, match="context_fraction"):
            FutureBlockMask(context_fraction=(0.7, 0.3))

    def test_rejects_invalid_target_time_scale(self) -> None:
        with pytest.raises(ValueError, match="target_time_scale"):
            FutureBlockMask(target_time_scale=(0.0, 0.2))

    def test_rejects_invalid_target_channel_scale(self) -> None:
        with pytest.raises(ValueError, match="target_channel_scale"):
            FutureBlockMask(target_channel_scale=(0.5, 1.5))

    def test_rejects_negative_horizon_gap(self) -> None:
        with pytest.raises(ValueError, match="min_horizon_gap"):
            FutureBlockMask(min_horizon_gap=-1)


class TestFutureBlockMaskShape:
    def test_sample_returns_correct_shapes(self) -> None:
        m = FutureBlockMask(n_targets=3)
        out = m.sample(grid_h=32, grid_w=8, generator=_seeded_gen(1))
        assert isinstance(out, MaskOutput)
        n = 32 * 8
        assert out.context.shape == (n,)
        assert out.context.dtype == torch.bool
        assert len(out.targets) == 3
        for t in out.targets:
            assert t.shape == (n,)
            assert t.dtype == torch.bool

    def test_rejects_invalid_grid(self) -> None:
        m = FutureBlockMask()
        with pytest.raises(ValueError, match="grid dims"):
            m.sample(grid_h=0, grid_w=8)
        with pytest.raises(ValueError, match="FutureBlockMask requires grid_h"):
            m.sample(grid_h=1, grid_w=8)

    def test_one_dim_channel_grid_supported(self) -> None:
        """grid_w=1 models a flat 1D event sequence."""
        m = FutureBlockMask(n_targets=2)
        out = m.sample(grid_h=64, grid_w=1, generator=_seeded_gen(0))
        assert out.context.shape == (64,)
        for t in out.targets:
            assert t.shape == (64,)


class TestFutureBlockMaskFutureOnlyInvariant:
    """The load-bearing invariant: every target position is strictly in
    the future of every context position on the time axis."""

    def _max_context_time(self, ctx_flat: torch.Tensor, grid_h: int, grid_w: int) -> int:
        ctx = ctx_flat.view(grid_h, grid_w)
        rows_with_true = ctx.any(dim=1)
        idxs = torch.nonzero(rows_with_true, as_tuple=False)
        return int(idxs.max().item()) if idxs.numel() else -1

    def _min_target_time(self, tgt_flat: torch.Tensor, grid_h: int, grid_w: int) -> int:
        tgt = tgt_flat.view(grid_h, grid_w)
        rows_with_true = tgt.any(dim=1)
        idxs = torch.nonzero(rows_with_true, as_tuple=False)
        return int(idxs.min().item()) if idxs.numel() else -1

    def test_targets_strictly_after_context(self) -> None:
        m = FutureBlockMask(n_targets=4)
        for seed in range(20):
            out = m.sample(grid_h=64, grid_w=4, generator=_seeded_gen(seed))
            ctx_max = self._max_context_time(out.context, 64, 4)
            for i, t in enumerate(out.targets):
                tgt_min = self._min_target_time(t, 64, 4)
                assert tgt_min > ctx_max, (
                    f"seed={seed} target {i} starts at t={tgt_min} but context "
                    f"extends to t={ctx_max}"
                )

    def test_min_horizon_gap_is_respected(self) -> None:
        gap = 5
        m = FutureBlockMask(n_targets=2, min_horizon_gap=gap)
        for seed in range(20):
            out = m.sample(grid_h=64, grid_w=4, generator=_seeded_gen(seed))
            ctx_max = self._max_context_time(out.context, 64, 4)
            for i, t in enumerate(out.targets):
                tgt_min = self._min_target_time(t, 64, 4)
                assert tgt_min >= ctx_max + 1 + gap, (
                    f"seed={seed} target {i} at t={tgt_min} violates min_horizon_gap={gap} "
                    f"(context_end={ctx_max + 1})"
                )

    def test_context_and_targets_are_disjoint(self) -> None:
        m = FutureBlockMask(n_targets=4)
        for seed in range(20):
            out = m.sample(grid_h=64, grid_w=4, generator=_seeded_gen(seed))
            for i, t in enumerate(out.targets):
                overlap = (out.context & t).any()
                assert not overlap, f"seed={seed} target {i} overlaps context"

    def test_context_is_non_empty(self) -> None:
        m = FutureBlockMask(n_targets=4)
        for seed in range(20):
            out = m.sample(grid_h=64, grid_w=4, generator=_seeded_gen(seed))
            assert out.context.any(), f"empty context on seed {seed}"

    def test_each_target_is_non_empty(self) -> None:
        m = FutureBlockMask(n_targets=4)
        for seed in range(20):
            out = m.sample(grid_h=64, grid_w=4, generator=_seeded_gen(seed))
            for i, t in enumerate(out.targets):
                assert t.any(), f"target {i} empty on seed {seed}"


class TestFutureBlockMaskDeterminism:
    def test_seeded_runs_are_deterministic(self) -> None:
        m = FutureBlockMask(n_targets=3)
        a = m.sample(grid_h=48, grid_w=4, generator=_seeded_gen(42))
        b = m.sample(grid_h=48, grid_w=4, generator=_seeded_gen(42))
        assert torch.equal(a.context, b.context)
        for ta, tb in zip(a.targets, b.targets, strict=True):
            assert torch.equal(ta, tb)

    def test_different_seeds_produce_different_masks(self) -> None:
        m = FutureBlockMask(n_targets=3)
        a = m.sample(grid_h=48, grid_w=4, generator=_seeded_gen(1))
        b = m.sample(grid_h=48, grid_w=4, generator=_seeded_gen(2))
        # extremely unlikely to coincide on 192 positions
        assert not torch.equal(a.context, b.context)
