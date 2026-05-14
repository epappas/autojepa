"""EMA target-encoder primitives.

Per ADR-003 we do not reimplement EMA wrapping. The actual
`teacher = EMA(student)` update lives in
`stable_pretraining.TeacherStudentWrapper`, which is the same primitive
cited in I-JEPA, V-JEPA 2, BYOL, and DINO. This module provides a
thin AutoJEPA-flavoured factory + the invariant assertions the
`program.md` validator (writeup §6.4) refers to:

- `assert_no_grad_on_target(teacher)` — guards that
  `target_encoder.parameters()` carry `requires_grad=False`. Diffs that
  enable gradients on the target encoder are rejected.

The module-level constants encode the writeup §6 default schedule
(I-JEPA values: 0.996 → 1.0 over the full training run).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from stable_pretraining import TeacherStudentWrapper

DEFAULT_BASE_EMA = 0.996
DEFAULT_FINAL_EMA = 1.0


@dataclass(frozen=True)
class EMAConfig:
    """EMA hyperparameters surfaced to the AutoJEPA hybrid policy.

    Listed in the writeup §6.3 as default param-search dimensions
    (`EMA_start`, `EMA_end`).
    """

    base_ema_coefficient: float = DEFAULT_BASE_EMA
    final_ema_coefficient: float = DEFAULT_FINAL_EMA
    warm_init: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.base_ema_coefficient <= 1.0:
            raise ValueError(
                f"base_ema_coefficient must be in [0, 1]; got {self.base_ema_coefficient}"
            )
        if not 0.0 <= self.final_ema_coefficient <= 1.0:
            raise ValueError(
                f"final_ema_coefficient must be in [0, 1]; got {self.final_ema_coefficient}"
            )


def build_target_encoder(
    student: torch.nn.Module,
    config: EMAConfig | None = None,
) -> TeacherStudentWrapper:
    """Wrap a student encoder with an EMA-tracked target encoder.

    The returned wrapper exposes `forward_student(x)` /
    `forward_teacher(x)` per the stable-pretraining contract. EMA
    updates are driven by `spt.callbacks.TeacherStudentCallback`
    registered on the Lightning Trainer; the wrapper itself is not
    responsible for stepping the EMA.
    """
    cfg = config or EMAConfig()
    return TeacherStudentWrapper(
        student=student,
        warm_init=cfg.warm_init,
        base_ema_coefficient=cfg.base_ema_coefficient,
        final_ema_coefficient=cfg.final_ema_coefficient,
    )


def assert_no_grad_on_target(wrapper: TeacherStudentWrapper) -> None:
    """Hard-fail guard from writeup §6.4 program.md: the target encoder
    must not receive gradients. Used by the AST validator's
    required-call list and by the canary smoke test.
    """
    teacher = _get_teacher_module(wrapper)
    if teacher is None:
        raise ValueError("wrapper has no teacher module to inspect")
    for name, param in teacher.named_parameters():
        if param.requires_grad:
            raise AssertionError(
                f"target encoder parameter has requires_grad=True: {name}"
            )


def _get_teacher_module(wrapper: TeacherStudentWrapper) -> torch.nn.Module | None:
    """Resolve the teacher submodule from the stable-pretraining
    wrapper. The attribute name is `teacher` in v0.1.x; this helper
    centralizes the lookup so a future spt rename only touches one
    place.
    """
    return getattr(wrapper, "teacher", None)
