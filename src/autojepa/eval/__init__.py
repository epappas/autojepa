"""AutoJEPA evaluation primitives.

Net-new namespace introduced in writeup Phase-1 §7.1. Houses probe
runners, collapse-detection metrics, downstream-eval suites, and the
sanity-overfit canary.

The legacy modules `judge.py`, `metrics.py`, `scoring.py` belong to the
inherited `autoresearch-rl` controller/legacy-loop path and are kept
unchanged for now. They will be relocated to `autojepa/parsing/` and
some dropped per the inheritance map (`docs/research/AutoresearchRL-Inheritance-Map.md` §7).
"""

from autojepa.eval.collapse import (
    effective_rank,
    latent_variance,
    rankme,
)

__all__ = [
    "effective_rank",
    "latent_variance",
    "rankme",
]
