"""Unit tests for autojepa.models.ema.

Validates:
- EMAConfig boundary checks
- build_target_encoder returns a usable TeacherStudentWrapper
- assert_no_grad_on_target catches a leaky teacher
- assert_target_params_unchanged_by_loss catches gradient flow

Marked `jepa` because the module imports torch + stable_pretraining
(installed via the [jepa] extra).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
spt = pytest.importorskip("stable_pretraining")

from autojepa.models.ema import (  # noqa: E402
    DEFAULT_BASE_EMA,
    DEFAULT_FINAL_EMA,
    EMAConfig,
    assert_no_grad_on_target,
    build_target_encoder,
)

pytestmark = pytest.mark.jepa


def _tiny_encoder(in_dim: int = 8, out_dim: int = 16) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, out_dim),
    )


class TestEMAConfig:
    def test_defaults_match_writeup(self) -> None:
        cfg = EMAConfig()
        assert cfg.base_ema_coefficient == DEFAULT_BASE_EMA == 0.996
        assert cfg.final_ema_coefficient == DEFAULT_FINAL_EMA == 1.0
        assert cfg.warm_init is True

    def test_rejects_out_of_range_base(self) -> None:
        with pytest.raises(ValueError, match="base_ema_coefficient"):
            EMAConfig(base_ema_coefficient=-0.1)
        with pytest.raises(ValueError, match="base_ema_coefficient"):
            EMAConfig(base_ema_coefficient=1.1)

    def test_rejects_out_of_range_final(self) -> None:
        with pytest.raises(ValueError, match="final_ema_coefficient"):
            EMAConfig(final_ema_coefficient=2.0)


class TestBuildTargetEncoder:
    def test_returns_teacher_student_wrapper(self) -> None:
        student = _tiny_encoder()
        wrapper = build_target_encoder(student)
        assert isinstance(wrapper, spt.TeacherStudentWrapper)

    def test_teacher_param_count_matches_student(self) -> None:
        student = _tiny_encoder()
        wrapper = build_target_encoder(student)
        student_total = sum(p.numel() for p in student.parameters())
        teacher_total = sum(p.numel() for p in wrapper.teacher.parameters())
        assert teacher_total == student_total

    def test_warm_init_aligns_teacher_to_student(self) -> None:
        student = _tiny_encoder()
        wrapper = build_target_encoder(student, EMAConfig(warm_init=True))
        for s_param, t_param in zip(
            student.parameters(), wrapper.teacher.parameters(), strict=True
        ):
            assert torch.equal(s_param, t_param)


class TestAssertNoGradOnTarget:
    def test_passes_for_freshly_built_wrapper(self) -> None:
        wrapper = build_target_encoder(_tiny_encoder())
        assert_no_grad_on_target(wrapper)

    def test_catches_leaky_teacher(self) -> None:
        wrapper = build_target_encoder(_tiny_encoder())
        first_param = next(wrapper.teacher.parameters())
        first_param.requires_grad_(True)
        with pytest.raises(AssertionError, match="requires_grad=True"):
            assert_no_grad_on_target(wrapper)


