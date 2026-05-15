"""Composable masking primitives for AutoJEPA.

Net-new namespace introduced in writeup §7.2. Exposes typed mask
factories the LLM diff policy composes.

The full writeup §7.2 list (`FutureBlockMask`, `MultiBlockInfillMask`,
`SemanticUnitMask`, `ActorAnonymizedMask`, `TimeFrequencyMask`,
`CompositeMask`) will fill in over Phases 2 and 3 as concrete examples
land. The remaining `SemanticUnitMask`, `ActorAnonymizedMask`, and
`TimeFrequencyMask` are deferred until a concrete example needs them
(per the ZERO-TOLERANCE-for-stubs hard rule).

Currently shipped:
- `MultiBlockInfillMask` — I-JEPA multi-block sampling on a 2D patch
  grid (writeup `docs/research/I-JEPA.md` §2). Drives Phase-2
  ijepa-cifar10.
- `FutureBlockMask` — causal future-block sampling on a 2D
  (time, channel) grid. Drives Phase-3 trace-jepa: agent traces are
  ordered event sequences and the predictive task is "predict a
  contiguous future block from a past context".
- `CompositeMask` — combines multiple mask samplers with weights so
  the LLM can probe the masking design space without re-authoring
  primitives.
- `MaskOutput` — typed return for any mask sampler.
"""

from autojepa.masking.composite import CompositeMask
from autojepa.masking.primitives import (
    FutureBlockMask,
    MaskOutput,
    MultiBlockInfillMask,
)

__all__ = [
    "CompositeMask",
    "FutureBlockMask",
    "MaskOutput",
    "MultiBlockInfillMask",
]
