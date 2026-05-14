"""Composite mask: weighted sampling over multiple mask primitives.

Use case: the AutoJEPA hybrid policy proposes a campaign in which each
trial draws a masking strategy from a weighted distribution, e.g.

    mask = CompositeMask([
        (MultiBlockInfillMask(n_targets=4), 0.7),
        (MultiBlockInfillMask(n_targets=8, target_scale=(0.05, 0.10)), 0.3),
    ])

The `sample(...)` method delegates to one of the wrapped maskers
according to the weight distribution. This keeps the LLM diff policy
from having to author per-strategy boilerplate every iteration —
proposing a new masking schedule becomes an edit to the weight list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from autojepa.masking.primitives import MaskOutput


class _MaskSampler(Protocol):
    def sample(
        self,
        grid_h: int,
        grid_w: int,
        generator: torch.Generator | None = None,
    ) -> MaskOutput: ...


@dataclass
class CompositeMask:
    """Weighted union of mask samplers.

    `samplers` is a list of `(sampler, weight)` pairs. Weights need not
    sum to 1; they are normalized internally. Each `sample()` call
    picks exactly one sampler to delegate to.
    """

    samplers: list[tuple[_MaskSampler, float]]

    def __post_init__(self) -> None:
        if not self.samplers:
            raise ValueError("CompositeMask requires at least one sampler")
        if any(w < 0 for _, w in self.samplers):
            raise ValueError("CompositeMask weights must be non-negative")
        total = sum(w for _, w in self.samplers)
        if total <= 0:
            raise ValueError("CompositeMask weights must sum to a positive value")
        self._weights = torch.tensor(
            [w / total for _, w in self.samplers], dtype=torch.float32
        )

    def sample(
        self,
        grid_h: int,
        grid_w: int,
        generator: torch.Generator | None = None,
    ) -> MaskOutput:
        idx = int(torch.multinomial(self._weights, num_samples=1, generator=generator).item())
        sampler, _ = self.samplers[idx]
        return sampler.sample(grid_h, grid_w, generator)
