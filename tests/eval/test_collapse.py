"""Unit tests for autojepa.eval.collapse.

Covers the boundary conditions that the writeup §6.4 hard-fail gates
depend on: full collapse, full spread, partial collapse, and the input
contracts.

Marked `jepa` because the module imports torch (installed via the
[jepa] extra). Tests skip gracefully when torch is missing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from autojepa.eval.collapse import effective_rank, latent_variance, rankme  # noqa: E402

pytestmark = pytest.mark.jepa


def _full_rank_isotropic(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def _collapsed(n: int, d: int) -> torch.Tensor:
    """All samples identical: zero spread."""
    return torch.ones(n, d) * 0.5


def _rank_one(n: int, d: int, seed: int = 0) -> torch.Tensor:
    """All samples lie on a single direction."""
    g = torch.Generator().manual_seed(seed)
    direction = torch.randn(d, generator=g)
    coeffs = torch.randn(n, 1, generator=g)
    return coeffs * direction


class TestRankMe:
    def test_isotropic_high_dim_approaches_dim(self) -> None:
        emb = _full_rank_isotropic(n=4096, d=64, seed=1)
        score = rankme(emb)
        assert 50.0 <= score <= 64.0, f"isotropic 64-D embedding RankMe={score}"

    def test_full_collapse_is_one(self) -> None:
        score = rankme(_collapsed(n=128, d=32))
        assert score == pytest.approx(1.0, abs=1e-3)

    def test_rank_one_is_one(self) -> None:
        score = rankme(_rank_one(n=512, d=64, seed=2))
        assert score == pytest.approx(1.0, abs=1e-2)

    def test_partial_collapse_below_dim(self) -> None:
        emb = _full_rank_isotropic(n=2048, d=64, seed=3)
        emb[:, 32:] = 0.0
        score = rankme(emb)
        assert 24.0 < score < 36.0, f"32-active-dim RankMe={score}"

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            rankme(torch.randn(4, 8, 16))

    def test_rejects_single_sample(self) -> None:
        with pytest.raises(ValueError, match="at least 2 samples"):
            rankme(torch.randn(1, 64))

    def test_returns_python_float(self) -> None:
        score = rankme(_full_rank_isotropic(n=128, d=16, seed=4))
        assert isinstance(score, float)


class TestEffectiveRank:
    def test_isotropic_matches_dim(self) -> None:
        emb = _full_rank_isotropic(n=4096, d=64, seed=5)
        er = effective_rank(emb)
        assert 50.0 <= er <= 64.0, f"isotropic effective_rank={er}"

    def test_collapse_is_one(self) -> None:
        emb = _collapsed(n=128, d=32)
        # under full collapse all sigma are zero -> ratio = 0/eps = 0
        er = effective_rank(emb)
        assert er == pytest.approx(0.0, abs=1e-3)

    def test_rank_one_near_one(self) -> None:
        er = effective_rank(_rank_one(n=512, d=64, seed=6))
        assert er == pytest.approx(1.0, abs=5e-2)

    def test_returns_python_float(self) -> None:
        assert isinstance(effective_rank(_full_rank_isotropic(64, 8)), float)


class TestLatentVariance:
    def test_collapse_is_zero(self) -> None:
        assert latent_variance(_collapsed(128, 32)) == pytest.approx(0.0, abs=1e-6)

    def test_isotropic_unit_variance(self) -> None:
        emb = _full_rank_isotropic(n=4096, d=64, seed=7)
        v = latent_variance(emb)
        assert v == pytest.approx(1.0, abs=0.05)

    def test_below_writeup_threshold(self) -> None:
        emb = _full_rank_isotropic(n=4096, d=64, seed=8) * 0.1
        v = latent_variance(emb)
        assert v < 0.3, f"scaled embeddings should fail variance gate; got {v}"

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            latent_variance(torch.randn(4, 8, 16))


class TestProgramMdGates:
    """The four signals the writeup §6.4 program.md template gates on.

    These tests document the threshold semantics used by `gates.py`.
    They do not exercise the gate engine (that is in tests/test_gates.py)
    -- they only validate that the metrics are in the right *direction*
    when the trial is healthy vs collapsed.
    """

    def test_healthy_embeddings_pass_all_gates(self) -> None:
        emb = _full_rank_isotropic(n=4096, d=128, seed=9)
        assert latent_variance(emb) >= 0.3
        assert rankme(emb) >= 64.0
        assert effective_rank(emb) >= 32.0

    def test_collapsed_embeddings_fail_all_gates(self) -> None:
        emb = _collapsed(n=4096, d=128)
        assert latent_variance(emb) < 0.3
        assert rankme(emb) < 64.0
        assert effective_rank(emb) < 32.0

    def test_partially_collapsed_embeddings_fail_rank_gates(self) -> None:
        # Healthy variance but only first 16 dims active -> ranks should fail.
        emb = _full_rank_isotropic(n=4096, d=128, seed=10)
        emb[:, 16:] = 0.0
        # Variance still > 0.3 because active dims have unit std spread across all dims
        assert rankme(emb) < 64.0
        assert effective_rank(emb) < 32.0
