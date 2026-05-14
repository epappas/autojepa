"""Composable masking primitives for AutoJEPA.

Net-new namespace introduced in writeup §7.2. Exposes typed mask
factories the LLM diff policy composes.

The full writeup §7.2 list (`FutureBlockMask`, `MultiBlockInfillMask`,
`SemanticUnitMask`, `ActorAnonymizedMask`, `TimeFrequencyMask`,
`CompositeMask`) will fill in over Phases 2 and 3 as concrete examples
land. Phase 1 ships only what `examples/ijepa-cifar10/` (Phase 2) and
`examples/trace-jepa/` (Phase 3) require, per the ZERO-TOLERANCE-for-stubs
hard rule.

Currently shipped:
- `MultiBlockInfillMask` — I-JEPA multi-block sampling on a 2D patch
  grid (writeup `docs/research/I-JEPA.md` §2). Drives Phase-2.
- `CompositeMask` — combines multiple mask samplers with weights so
  the LLM can probe the masking design space without re-authoring
  primitives.
- `MaskOutput` — typed return for any mask sampler.
"""

from autojepa.masking.composite import CompositeMask
from autojepa.masking.primitives import MaskOutput, MultiBlockInfillMask

__all__ = [
    "CompositeMask",
    "MaskOutput",
    "MultiBlockInfillMask",
]
