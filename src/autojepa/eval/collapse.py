"""Representation-collapse detection metrics for JEPA pretraining.

These are the hard fail-gate signals from the writeup §6.4 program.md
template: a JEPA trial that drops below the configured thresholds on any
of these metrics is rejected before downstream-probe evaluation runs.

Sources:
- RankMe: Garrido et al., "RankMe: Assessing the downstream performance
  of pretrained self-supervised representations by their rank" (2023),
  arXiv:2210.02885.
- Effective rank: Roy & Vetterli, "The effective rank: A measure of
  effective dimensionality" (2007), EUSIPCO. Participation-ratio form
  used here (a.k.a. PR(sigma^2)).
- Latent variance: standard per-feature std summary used by VICReg-style
  defenses (Bardes et al., 2022).

All functions are pure tensor math, take no labels, and require only a
batch of embeddings of shape (N, D). They are intentionally
framework-free (no Lightning, no callbacks) so the controller can call
them from the trial sidecar to decide cancellation.

For the LiDAR signal (Thilak et al., 2024), which requires per-class
embedding splits, AutoJEPA wraps `stable_pretraining.callbacks.LiDAR`
inside `autojepa.eval.probes` rather than re-implementing the LDA math.
"""

from __future__ import annotations

import torch

_RANK_EPS = 1e-7


def _singular_values(embeddings: torch.Tensor) -> torch.Tensor:
    """SVD singular values of a 2D embedding batch.

    Centers per-feature so RankMe and effective_rank measure spread of
    the zero-mean embedding distribution rather than its translation.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2D (N, D); got shape {tuple(embeddings.shape)}"
        )
    if embeddings.shape[0] < 2:
        raise ValueError("need at least 2 samples to estimate spread")
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    sigma = torch.linalg.svdvals(centered.float())
    return sigma


def rankme(embeddings: torch.Tensor) -> float:
    """RankMe score (Garrido et al., 2023).

    Defined as exp(H(p)) where p_i = sigma_i / sum(sigma) + eps and H is
    Shannon entropy in nats. Equals the dimensionality D under perfectly
    uniform singular spectrum and 1.0 under full collapse.

    Returns a Python float so the controller can write it to
    progress.jsonl without tensor serialization.
    """
    sigma = _singular_values(embeddings)
    norm = sigma.sum()
    if norm <= 0:
        return 1.0
    p = sigma / norm + _RANK_EPS
    entropy = -(p * p.log()).sum()
    return float(entropy.exp().item())


def effective_rank(embeddings: torch.Tensor) -> float:
    """Participation-ratio effective rank: (sum sigma)^2 / sum(sigma^2).

    This is a different summary statistic from RankMe. Both rise and
    fall together but the writeup §6.4 thresholds (eff_rank < 32 vs
    RankMe < 64) suggest cross-checking on both — when they disagree,
    the spectrum is heavy-tailed.
    """
    sigma = _singular_values(embeddings)
    num = sigma.sum() ** 2
    den = (sigma ** 2).sum() + _RANK_EPS
    return float((num / den).item())


def latent_variance(embeddings: torch.Tensor) -> float:
    """Mean per-feature standard deviation, the VICReg variance term.

    Below the writeup §6.4 hard threshold of 0.3 the trial is failed
    before probe-eval runs.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2D (N, D); got shape {tuple(embeddings.shape)}"
        )
    if embeddings.shape[0] < 2:
        raise ValueError("need at least 2 samples to estimate variance")
    std_per_dim = embeddings.float().std(dim=0, unbiased=True)
    return float(std_per_dim.mean().item())
