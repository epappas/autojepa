"""AutoJEPA reference building blocks.

Net-new namespace introduced in writeup Phase-1 §7.3. Provides:

- `ema`        — EMA / target-encoder primitives (wraps stable-pretraining
                 TeacherStudentWrapper per ADR-003)
- `losses`     — SSL loss zoo (wraps stable-pretraining spt.losses +
                 adds L1/L2 distance helpers and a flat registry the
                 LLM diff policy can reach into)
- `encoders`   — encoder factories (ViT, ConvNeXt, generic transformer)
- `predictors` — predictor factories (block-causal, full-attention,
                 cross-attention)

The subpackages import torch and stable-pretraining and therefore live
behind the [jepa] extra (ADR-010). Importing this module without the
extra raises ImportError immediately rather than half-loading.
"""

from autojepa.models import ema, losses

__all__ = ["ema", "losses"]
