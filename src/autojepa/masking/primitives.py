"""Mask primitives operating on a 2D patch grid.

Conventions:
- A *mask sampler* is anything with a `.sample(grid_h, grid_w, generator=None)
  -> MaskOutput` method.
- `MaskOutput.context` is a flat boolean tensor of shape `(grid_h * grid_w,)`
  marking which patches participate in the context encoder forward pass.
- `MaskOutput.targets` is a list of flat boolean tensors of the same
  shape, one per target block. The model predicts each target block's
  embedding from the context.

`MultiBlockInfillMask` follows the I-JEPA recipe (writeup
`docs/research/I-JEPA.md` §2):
- Sample one context block at scale 0.85-1.0, aspect 1.0.
- Sample M=4 target blocks at scale 0.15-0.20, aspect 0.75-1.5.
- Remove target overlaps from context.

The masking ablation in I-JEPA (multi-block 54.2 vs single-block 20.2
on 1% ImageNet) is the empirical justification for putting masking
strategy in the AutoJEPA hybrid policy's default search dimensions
(writeup §6.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch


class MaskOutput(NamedTuple):
    """Boolean masks over a flattened 2D patch grid.

    `context` and each entry in `targets` have shape (grid_h * grid_w,).
    True means the patch is included in that role.
    """

    context: torch.Tensor
    targets: list[torch.Tensor]


@dataclass(frozen=True)
class MultiBlockInfillMask:
    """I-JEPA multi-block masking over a 2D patch grid.

    The defaults match Assran et al. 2023 (writeup
    `docs/research/I-JEPA.md` §2). Hyperparameters are exposed as
    fields so the AutoJEPA hybrid policy can search over them per
    writeup §6.3 (`mask_ratio_max`, `target_block_scale`,
    `context_block_scale`).
    """

    n_targets: int = 4
    target_scale: tuple[float, float] = (0.15, 0.20)
    target_aspect: tuple[float, float] = (0.75, 1.5)
    context_scale: tuple[float, float] = (0.85, 1.0)
    max_attempts: int = 20

    def __post_init__(self) -> None:
        _validate_scale("target_scale", self.target_scale)
        _validate_scale("context_scale", self.context_scale)
        _validate_aspect("target_aspect", self.target_aspect)
        if self.n_targets <= 0:
            raise ValueError(f"n_targets must be positive; got {self.n_targets}")
        if self.max_attempts <= 0:
            raise ValueError(f"max_attempts must be positive; got {self.max_attempts}")

    def sample(
        self,
        grid_h: int,
        grid_w: int,
        generator: torch.Generator | None = None,
    ) -> MaskOutput:
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"grid dims must be positive; got ({grid_h}, {grid_w})")
        target_blocks = [
            self._sample_block(
                grid_h,
                grid_w,
                self.target_scale,
                self.target_aspect,
                generator,
            )
            for _ in range(self.n_targets)
        ]
        targets_union = _union(target_blocks, grid_h, grid_w)
        context = self._sample_context(grid_h, grid_w, targets_union, generator)
        return MaskOutput(context=context.flatten(), targets=[t.flatten() for t in target_blocks])

    def _sample_block(
        self,
        grid_h: int,
        grid_w: int,
        scale: tuple[float, float],
        aspect: tuple[float, float],
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        n_positions = grid_h * grid_w
        s_lo, s_hi = scale
        a_lo, a_hi = aspect
        scale_v = _uniform(s_lo, s_hi, generator)
        aspect_v = _uniform(a_lo, a_hi, generator)
        block_area = scale_v * n_positions
        h = max(1, int(round(math.sqrt(block_area * aspect_v))))
        w = max(1, int(round(math.sqrt(block_area / aspect_v))))
        h = min(h, grid_h)
        w = min(w, grid_w)
        top = int(_uniform_int(0, grid_h - h + 1, generator))
        left = int(_uniform_int(0, grid_w - w + 1, generator))
        mask = torch.zeros(grid_h, grid_w, dtype=torch.bool)
        mask[top : top + h, left : left + w] = True
        return mask

    def _sample_context(
        self,
        grid_h: int,
        grid_w: int,
        targets_union: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        for _ in range(self.max_attempts):
            block = self._sample_block(
                grid_h, grid_w, self.context_scale, (1.0, 1.0), generator
            )
            block &= ~targets_union
            if block.any():
                return block
        # Fallback: any non-target patch.
        fallback = ~targets_union
        if fallback.any():
            return fallback
        raise RuntimeError(
            "could not sample a non-empty context block; targets cover the entire grid"
        )


def _validate_scale(name: str, scale: tuple[float, float]) -> None:
    lo, hi = scale
    if not 0.0 < lo <= hi <= 1.0:
        raise ValueError(f"{name} must satisfy 0 < lo <= hi <= 1; got {scale}")


def _validate_aspect(name: str, aspect: tuple[float, float]) -> None:
    lo, hi = aspect
    if not 0.0 < lo <= hi:
        raise ValueError(f"{name} must satisfy 0 < lo <= hi; got {aspect}")


def _union(masks: list[torch.Tensor], grid_h: int, grid_w: int) -> torch.Tensor:
    out = torch.zeros(grid_h, grid_w, dtype=torch.bool)
    for m in masks:
        out |= m
    return out


def _uniform(lo: float, hi: float, generator: torch.Generator | None) -> float:
    if lo == hi:
        return lo
    u = torch.rand((), generator=generator).item()
    return lo + u * (hi - lo)


def _uniform_int(lo: int, hi: int, generator: torch.Generator | None) -> int:
    """Half-open [lo, hi). Returns lo when hi == lo + 1 (degenerate)."""
    if hi <= lo:
        return lo
    return int(torch.randint(lo, hi, (1,), generator=generator).item())
