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


@dataclass(frozen=True)
class FutureBlockMask:
    """Causal future-block masking over a 2D (time, channel) grid.

    Used by Phase-3 trace-jepa: agent traces are sequences of structured
    events ordered in time, and the task-relevant prediction is "given
    the past, predict a contiguous block of future events". This is the
    causal counterpart of `MultiBlockInfillMask` (which is non-causal /
    infill), and is the only mask in the writeup §7.2 list that enforces
    a *temporal* relationship between context and target.

    Convention (matches `MultiBlockInfillMask.sample`):

    - The first grid axis (`grid_h` from `.sample(grid_h, grid_w, ...)`)
      is interpreted as the **time** axis.
    - The second axis (`grid_w`) is the channel / feature axis.
    - The grid is split along time at a sampled `context_end` index.
      Context = all positions with `time < context_end`.
      Targets = `n_targets` blocks sampled strictly inside
                `time >= context_end + min_horizon_gap`.

    Hyperparameters:

    - `n_targets` — number of future blocks to predict.
    - `context_fraction` — half-open uniform draw `(lo, hi)` for the
      fraction of the time axis that becomes context. e.g. `(0.4, 0.6)`
      means the context spans 40-60% of the timeline, future is the rest.
    - `target_time_scale` — fraction of the *future* region one target
      block spans on the time axis. e.g. `(0.05, 0.20)` means each
      target block is 5-20% of the future window.
    - `target_channel_scale` — fraction of the channel axis a target
      block covers. Default `(1.0, 1.0)` (full-channel slab) matches
      the trace-jepa case where every event has the same fields.
    - `min_horizon_gap` — number of timesteps to skip between context
      end and the earliest legal target start. Defaults to 0; set >0
      to force the predictor to span an explicit prediction horizon.
    - `max_attempts` — retries per target block when sampling collides
      with another target.

    Invariants the validator + tests enforce:

    - For every target block t, every "True" position in t has a time
      index strictly greater than every "True" position in context.
      (the future-only invariant)
    - Context and any target are disjoint.
    - `sample()` is deterministic under a seeded `torch.Generator`.

    Out-of-scope (v1):

    - Overlap between distinct target blocks is permitted (matches
      I-JEPA's multi-block convention; the model sees them as separate
      prediction targets even when they share positions). A future
      `disjoint_targets=True` switch can be added when a Phase-3 ablation
      asks for it.
    - Future-block sampling on a 1D sequence (no channel axis) is
      modelled by `grid_w=1`.
    """

    n_targets: int = 4
    context_fraction: tuple[float, float] = (0.4, 0.6)
    target_time_scale: tuple[float, float] = (0.05, 0.20)
    target_channel_scale: tuple[float, float] = (1.0, 1.0)
    min_horizon_gap: int = 0
    max_attempts: int = 20

    def __post_init__(self) -> None:
        _validate_scale("context_fraction", self.context_fraction)
        _validate_scale("target_time_scale", self.target_time_scale)
        _validate_scale("target_channel_scale", self.target_channel_scale)
        if self.n_targets <= 0:
            raise ValueError(f"n_targets must be positive; got {self.n_targets}")
        if self.min_horizon_gap < 0:
            raise ValueError(
                f"min_horizon_gap must be non-negative; got {self.min_horizon_gap}"
            )
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
        if grid_h < 2:
            raise ValueError(
                f"FutureBlockMask requires grid_h>=2 to split context vs future; got {grid_h}"
            )

        ctx_lo, ctx_hi = self.context_fraction
        ctx_frac = _uniform(ctx_lo, ctx_hi, generator)
        # Reserve at least 1 timestep for context and (1 + min_horizon_gap)
        # for the future region; clamp to satisfy both.
        min_future = max(1, self.min_horizon_gap + 1)
        max_ctx_end = grid_h - min_future
        if max_ctx_end < 1:
            raise RuntimeError(
                f"grid_h={grid_h} too small for min_horizon_gap={self.min_horizon_gap}; "
                f"need grid_h >= {self.min_horizon_gap + 2}"
            )
        context_end = max(1, min(max_ctx_end, int(round(ctx_frac * grid_h))))

        future_start = context_end + self.min_horizon_gap
        future_len = grid_h - future_start
        if future_len <= 0:
            raise RuntimeError(
                f"no future region left after context_end={context_end} + "
                f"min_horizon_gap={self.min_horizon_gap} on grid_h={grid_h}"
            )

        context = torch.zeros(grid_h, grid_w, dtype=torch.bool)
        context[:context_end, :] = True

        targets: list[torch.Tensor] = [
            self._sample_future_block(
                grid_h, grid_w, future_start, future_len, generator
            )
            for _ in range(self.n_targets)
        ]
        return MaskOutput(context=context.flatten(), targets=[t.flatten() for t in targets])

    def _sample_future_block(
        self,
        grid_h: int,
        grid_w: int,
        future_start: int,
        future_len: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        ts_lo, ts_hi = self.target_time_scale
        cs_lo, cs_hi = self.target_channel_scale

        time_frac = _uniform(ts_lo, ts_hi, generator)
        chan_frac = _uniform(cs_lo, cs_hi, generator)

        h = max(1, min(future_len, int(round(time_frac * future_len))))
        w = max(1, min(grid_w, int(round(chan_frac * grid_w))))
        # Top is constrained to lie inside [future_start, grid_h - h].
        top_lo = future_start
        top_hi = grid_h - h + 1
        if top_hi <= top_lo:
            top = top_lo
        else:
            top = int(_uniform_int(top_lo, top_hi, generator))
        left_hi = grid_w - w + 1
        left = 0 if left_hi <= 0 else int(_uniform_int(0, left_hi, generator))

        mask = torch.zeros(grid_h, grid_w, dtype=torch.bool)
        mask[top : top + h, left : left + w] = True
        return mask


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
