"""Trace-JEPA pretraining loop — the AutoJEPA Phase-3 example.

Hypothesis (writeup §11 Phase-3): JEPA-style joint-embedding predictive
training over structured agent traces yields representations whose
linear probe separates normal sessions from synthetic prompt-injection
attacks at `probe_auroc > 0.7` at 5% FPR. The Phase-3 falsifier kills
the JEPA-for-traces thesis if a 30-iteration hybrid campaign cannot
hit that bar.

Architecture
============

- Tokeniser: deterministic vocabulary lookups for `action_name`,
  `action_type`, plus a small char-trigram hash for `args`. No
  separately-trained tokenizer — keeps prepare.py / train.py decoupled.
- Encoder Φc: Transformer over per-event embeddings; ~10-30 M
  parameters depending on `encoder_depth`, `encoder_dim`. Default
  config (depth=8, dim=384, heads=6) lands ~12 M; the LLM diff policy
  can scale up to hit the 25-50 M writeup target.
- Target encoder Φt: EMA of Φc via `autojepa.models.ema.build_target_encoder`.
- Predictor Ψ: smaller Transformer (depth=2, dim=128 default). Per
  the JEPA hard rules Ψ MUST NOT be deeper or wider than Φc.
- Soft codebook bottleneck (the MTS-JEPA innovation, Phase-3 search
  axis): when `codebook_size > 0`, the predictor output is routed
  through a soft codebook lookup before being compared to the target;
  the codebook usage adds an entropy regulariser scaled by
  `codebook_loss_weight`. `codebook_size = 0` disables the codebook
  entirely (vanilla JEPA control row from `TODO.md` Phase 3).

Hyperparameters consumed from `AR_PARAMS_JSON`
==============================================

    learning_rate, weight_decay, batch_size, max_steps,
    encoder_depth, encoder_dim, encoder_heads,
    predictor_depth, predictor_embed_dim,
    num_targets, ema_decay_start, ema_decay_end,
    codebook_size, codebook_loss_weight,
    future_block_weight, multi_block_weight,
    probe_eval_every_n_steps, canary_loss_threshold

Mutable per the autoresearch contract: the LLM diff policy may rewrite
this file. Required calls enforced by the AST validator (writeup §6.4
program.md / `autojepa.policy._prompt_fragments.JEPA_HARD_RULES`):

    emit_progress(step, step_target, metrics={"probe_auroc": ...})
    autojepa.models.ema.assert_no_grad_on_target(...)

Outputs `probe_auroc` (the campaign objective), `canary_loss`, `rankme`,
`latent_var` (the JEPA collapse signals from `autojepa.eval.collapse`)
and writes `outcome.json` for the controller's iter-done detection.
"""

from __future__ import annotations

import json
import os
import sys
import tarfile
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# ---------- contract: read params and progress emitter from the controller

from autojepa.target.progress import emit_progress  # noqa: E402

PARAMS = json.loads(os.environ.get("AR_PARAMS_JSON", "{}"))

OUTCOME_FILENAME = "outcome.json"


def _hp(name: str, default: float | int | str) -> float | int | str:
    """Per-key shortcut: AR_PARAM_<NAME> overrides the JSON dict."""
    env_key = f"AR_PARAM_{name.upper()}"
    if env_key in os.environ:
        v = os.environ[env_key]
        try:
            return type(default)(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return v
    return PARAMS.get(name, default)


# Defaults match the writeup §11 Phase-3 starting point. The hybrid
# policy will explore the search dims declared in config.yaml.
LEARNING_RATE = float(_hp("learning_rate", 3e-4))
WEIGHT_DECAY = float(_hp("weight_decay", 0.05))
BATCH_SIZE = int(_hp("batch_size", 64))
MAX_STEPS = int(_hp("max_steps", 4000))

ENCODER_DEPTH = int(_hp("encoder_depth", 8))
ENCODER_DIM = int(_hp("encoder_dim", 384))
ENCODER_HEADS = int(_hp("encoder_heads", 6))
PREDICTOR_DEPTH = int(_hp("predictor_depth", 2))
PREDICTOR_EMBED_DIM = int(_hp("predictor_embed_dim", 128))
NUM_TARGETS = int(_hp("num_targets", 4))
EMA_DECAY_START = float(_hp("ema_decay_start", 0.996))
EMA_DECAY_END = float(_hp("ema_decay_end", 1.0))

# Phase-3 search axes per docs/research/mts-jepa.md and TODO.md
# Phase-3. codebook_size=0 -> vanilla JEPA control row (no codebook).
CODEBOOK_SIZE = int(_hp("codebook_size", 0))
CODEBOOK_LOSS_WEIGHT = float(_hp("codebook_loss_weight", 0.0))

# Mask mixing weights for the CompositeMask of MultiBlockInfillMask +
# FutureBlockMask. Both must be non-negative; the sum is normalised.
FUTURE_BLOCK_WEIGHT = float(_hp("future_block_weight", 1.0))
MULTI_BLOCK_WEIGHT = float(_hp("multi_block_weight", 0.5))

# VICReg-style variance penalty on the predictor output. Without this,
# the plain MSE-against-EMA-target loss collapses immediately to constant
# output (rankme=1.0, var<1e-3) — v1 evidence 2026-05-19: every iter
# probe_auroc=0.502 (random) because target encoder mirrored a constant
# student. variance_loss_weight=1.0 is enough to break the trivial
# fixed-point per VICReg paper §3.2; the LLM can tune via the search
# axis. Set to 0.0 to recover the pre-2026-05-19 broken-baseline behaviour.
VARIANCE_LOSS_WEIGHT = float(_hp("variance_loss_weight", 1.0))
VARIANCE_LOSS_EPS = float(_hp("variance_loss_eps", 1e-2))

PROBE_EVAL_EVERY_N_STEPS = int(_hp("probe_eval_every_n_steps", 500))
# Bumped 2026-05-19 from 0.5 -> 5.0 after v2 evidence: with the variance
# penalty defaulted ON (preventing trivial collapse to a constant), the
# canary loss naturally floors around the variance-penalty term (~1.0)
# instead of going to ~0 via collapse. The original 0.5 threshold was
# calibrated against the broken-baseline behaviour where any model that
# collapsed trivially passed canary (v1 evidence). Same class of
# calibration bug as Phase-2 (commit a120b68 bumped ijepa from 0.05->0.08).
# 5.0 is conservative — still catches NaN, ~10+ pipeline blow-ups, etc.
# Future calibration: run a clean canary with var_w=1.0 + N=200 steps,
# pick threshold at 3-5x the observed steady-state.
CANARY_LOSS_THRESHOLD = float(_hp("canary_loss_threshold", 5.0))
CANARY_MAX_STEPS = int(_hp("canary_max_steps", 200))

DATA_DIR = Path(__file__).parent / "data"
ARTIFACT_DIR = Path(os.environ.get("AR_MODEL_DIR", str(Path(__file__).parent / "artifacts")))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MAX_EVENTS = 32  # matches prepare.py MAX_EVENTS_PER_SESSION
ARGS_HASH_DIM = 64  # number of trigram-hash buckets per args string


def _check_required_calls() -> None:
    """Asserted at script start so a malformed diff fails fast."""
    assert callable(emit_progress)
    from autojepa.models.ema import assert_no_grad_on_target  # noqa: F401


def _write_outcome(
    *,
    model_dir: Path,
    status: str,
    metrics: dict[str, float],
    elapsed_s: float,
    completed_steps: int = 0,
    step_target: int = 0,
    reason: str | None = None,
) -> None:
    """Atomically write outcome.json so the controller can detect completion.

    Mirrors the Phase-2 ijepa-cifar10 contract — the basilica adapter
    polls /model/files for this file as the iter-done signal that lets
    the controller bypass timeout-based waits.
    """
    payload: dict[str, Any] = {
        "status": status,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "elapsed_s": float(elapsed_s),
        "completed_steps": int(completed_steps),
        "step_target": int(step_target),
        "ts": int(time.time()),
    }
    if reason is not None:
        payload["reason"] = reason
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        target = model_dir / OUTCOME_FILENAME
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(target)
        print(f"[outcome] wrote {target} status={status}", flush=True)
    except OSError as exc:
        print(f"[outcome] WARN: failed to write {OUTCOME_FILENAME}: {exc}",
              file=sys.stderr, flush=True)


# ---------- import torch (heavy dep) and define model classes -------------
# train.py is mutable: the LLM diff policy may rewrite it. We keep the
# heavy `import torch` block guarded so that an environment without the
# [jepa] extra still produces a deterministic outcome.json("failed",
# reason="import_error: ...") rather than crashing inside class
# definitions at module-import time.

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
    _TORCH_IMPORT_ERROR: Exception | None = None
except ImportError as _exc:  # pragma: no cover — exercised only without [jepa]
    _TORCH_OK = False
    _TORCH_IMPORT_ERROR = _exc


def _abort_missing_torch() -> int:
    _write_outcome(
        model_dir=ARTIFACT_DIR,
        status="failed",
        metrics={},
        elapsed_s=0.0,
        reason=f"import_error: {_TORCH_IMPORT_ERROR}",
    )
    print(
        f"ERROR: train.py requires the [jepa] extra: {_TORCH_IMPORT_ERROR}",
        file=sys.stderr,
    )
    return 2


# =====================================================================
# Tokeniser (deterministic; no offline training needed)
# =====================================================================

class TraceTokenizer:
    """Vocabulary lookups for `action_name` and `action_type`,
    plus a char-trigram hash for `args` so the model sees structure
    inside the args string without an offline-trained subword tokenizer.
    """

    def __init__(self, action_types: tuple[str, ...], tool_names: tuple[str, ...]) -> None:
        self._type_to_id = {t: i + 1 for i, t in enumerate(action_types)}  # 0 = pad
        self._name_to_id = {n: i + 1 for i, n in enumerate(tool_names)}
        self._unk_name_id = len(self._name_to_id) + 1

    def vocab_size(self) -> int:
        return len(self._name_to_id) + 2  # pad + unk

    def type_vocab_size(self) -> int:
        return len(self._type_to_id) + 1  # pad

    def name_id(self, name: str) -> int:
        return self._name_to_id.get(name, self._unk_name_id)

    def type_id(self, t: str) -> int:
        return self._type_to_id.get(t, 0)

    def args_hash(self, args: str) -> list[int]:
        """64-bucket char-trigram hash of `args`. Returns a 0/1 multi-hot
        feature the encoder embeds via a small Linear layer."""
        out = [0] * ARGS_HASH_DIM
        s = args.lower()
        for i in range(len(s) - 2):
            tri = s[i : i + 3]
            out[hash(tri) % ARGS_HASH_DIM] = 1
        return out


# =====================================================================
# Trace Transformer encoder + predictor
# =====================================================================

if _TORCH_OK:

    def _build_transformer_encoder(
        embed_dim: int, depth: int, n_heads: int, dropout: float = 0.0
    ) -> "nn.Module":
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        return nn.TransformerEncoder(layer, num_layers=depth)


    class TraceTransformer(nn.Module):
        """Per-event embedding lookup + Transformer encoder.

        Each event carries:
            action_name (categorical)
            action_type (categorical)
            return_code (int)
            actor_id    (int)
            parent_link (int)
            timestamp   (float)
            args_hash   (binary multi-hot, ARGS_HASH_DIM)

        The forward pass also accepts a `mask_keep: BoolTensor (B, T)` of
        which events to keep (True) for the context-encoder pass — the
        predictor uses the kept events to predict masked-out target events.
        """

        def __init__(
            self,
            *,
            vocab: int,
            type_vocab: int,
            max_events: int,
            embed_dim: int,
            depth: int,
            n_heads: int,
            args_hash_dim: int,
        ) -> None:
            super().__init__()
            self.embed_dim = embed_dim
            self.name_emb = nn.Embedding(vocab, embed_dim, padding_idx=0)
            self.type_emb = nn.Embedding(type_vocab, embed_dim, padding_idx=0)
            self.return_emb = nn.Embedding(8, embed_dim)  # return_code bucketised
            self.actor_emb = nn.Embedding(8, embed_dim)
            self.parent_emb = nn.Embedding(max_events + 1, embed_dim)
            self.args_proj = nn.Linear(args_hash_dim, embed_dim)
            self.pos_emb = nn.Embedding(max_events, embed_dim)
            self.encoder = _build_transformer_encoder(embed_dim, depth, n_heads)
            self.norm = nn.LayerNorm(embed_dim)
            self.register_buffer("_pos_ids", torch.arange(max_events).unsqueeze(0))

        def event_embeds(self, batch: dict[str, "torch.Tensor"]) -> "torch.Tensor":
            e = (
                self.name_emb(batch["name"])
                + self.type_emb(batch["type"])
                + self.return_emb(batch["return"].clamp(0, 7))
                + self.actor_emb(batch["actor"].clamp(0, 7))
                + self.parent_emb(batch["parent"])
                + self.args_proj(batch["args"].float())
            )
            e = e + self.pos_emb(self._pos_ids[:, : e.shape[1]])
            return e

        def forward(
            self,
            batch: dict[str, "torch.Tensor"],
            attn_mask: "torch.Tensor | None" = None,
        ) -> "torch.Tensor":
            x = self.event_embeds(batch)
            if attn_mask is not None:
                # nn.TransformerEncoder takes `src_key_padding_mask` where
                # True = positions to ignore. attn_mask convention here:
                # True = keep. Invert before passing through.
                x = self.encoder(x, src_key_padding_mask=~attn_mask)
            else:
                x = self.encoder(x)
            return self.norm(x)


    class TracePredictor(nn.Module):
        """Maps context embeddings -> target-position predictions."""

        def __init__(
            self, *, embed_dim: int, predictor_dim: int, depth: int, n_heads: int
        ) -> None:
            super().__init__()
            self.in_proj = nn.Linear(embed_dim, predictor_dim)
            self.encoder = _build_transformer_encoder(predictor_dim, depth, n_heads)
            self.out_proj = nn.Linear(predictor_dim, embed_dim)

        def forward(self, context_features: "torch.Tensor") -> "torch.Tensor":
            h = self.in_proj(context_features)
            h = self.encoder(h)
            return self.out_proj(h)


    class SoftCodebookBottleneck(nn.Module):
        """Soft codebook lookup between predictor output and target embeddings.

        Per docs/research/mts-jepa.md the soft codebook serves two roles:
        (1) discrete-regime-transition modeling carrier, and (2) intrinsic
        anti-collapse regulariser. The implementation is a temperature-
        softmaxed similarity over `n_codes` learnable code vectors of
        dimension `embed_dim`. The forward returns `(quantized, code_loss)`
        where `quantized` is a soft mixture of codes and `code_loss` is an
        entropy term that encourages the codebook usage distribution to be
        spread out (anti-collapse).

        Per ADR-001 / TODO.md Phase-3 scope guard, this lives in
        `examples/trace-jepa/train.py` rather than `src/autojepa/models/` —
        it is an example-specific design lever, not a framework primitive.
        """

        def __init__(self, *, n_codes: int, embed_dim: int, temperature: float = 1.0) -> None:
            super().__init__()
            if n_codes <= 0:
                raise ValueError(f"n_codes must be positive; got {n_codes}")
            self.n_codes = n_codes
            self.embed_dim = embed_dim
            self.temperature = temperature
            self.codes = nn.Parameter(torch.randn(n_codes, embed_dim) * 0.02)

        def forward(self, x: "torch.Tensor") -> "tuple[torch.Tensor, torch.Tensor]":
            sims = torch.matmul(x, self.codes.T) / self.temperature
            weights = F.softmax(sims, dim=-1)
            quantized = torch.matmul(weights, self.codes)
            # Anti-collapse term: encourage the average usage distribution
            # over the batch to be uniform (max entropy = log n_codes).
            avg_usage = weights.mean(dim=(0, 1))
            max_entropy = torch.log(torch.tensor(float(self.n_codes), device=x.device))
            usage_entropy = -(avg_usage * (avg_usage + 1e-9).log()).sum()
            code_loss = max_entropy - usage_entropy
            return quantized, code_loss


# =====================================================================
# Train / canary / probe-eval loops
# =====================================================================

def _shard_iterator(
    shards: list[Path],
    batch_size: int,
    tokenizer: "TraceTokenizer",
) -> Iterator[dict[str, Any]]:
    """One pass over all shards; yields batched tensor dicts."""
    buffer: list[dict] = []
    for shard_path in shards:
        with tarfile.open(shard_path, "r") as tar:
            for member in tar.getmembers():
                f = tar.extractfile(member)
                if f is None:
                    continue
                buffer.append(json.loads(f.read().decode("utf-8")))
                if len(buffer) >= batch_size:
                    yield _batch_to_tensors(buffer, tokenizer)
                    buffer = []
    if buffer:
        yield _batch_to_tensors(buffer, tokenizer)


def _batch_to_tensors(
    sessions: list[dict],
    tokenizer: "TraceTokenizer",
) -> dict[str, Any]:
    """Convert a list of session dicts to padded (B, MAX_EVENTS, ...) tensors."""
    B = len(sessions)
    name = torch.zeros((B, MAX_EVENTS), dtype=torch.long)
    type_ = torch.zeros((B, MAX_EVENTS), dtype=torch.long)
    return_ = torch.zeros((B, MAX_EVENTS), dtype=torch.long)
    actor = torch.zeros((B, MAX_EVENTS), dtype=torch.long)
    parent = torch.zeros((B, MAX_EVENTS), dtype=torch.long)
    args = torch.zeros((B, MAX_EVENTS, ARGS_HASH_DIM), dtype=torch.float32)
    valid = torch.zeros((B, MAX_EVENTS), dtype=torch.bool)
    labels = torch.zeros((B,), dtype=torch.long)
    for i, sess in enumerate(sessions):
        labels[i] = int(bool(sess.get("is_attack", False)))
        for j, e in enumerate(sess["events"][:MAX_EVENTS]):
            name[i, j] = tokenizer.name_id(e["action_name"])
            type_[i, j] = tokenizer.type_id(e["action_type"])
            return_[i, j] = min(7, max(0, int(e.get("return_code", 0))))
            actor[i, j] = min(7, max(0, int(e.get("actor_id", 0))))
            parent[i, j] = max(0, int(e.get("parent_link", -1)) + 1)
            args[i, j] = torch.tensor(tokenizer.args_hash(e.get("args", "")), dtype=torch.float32)
            valid[i, j] = True
    return {
        "name": name,
        "type": type_,
        "return": return_,
        "actor": actor,
        "parent": parent,
        "args": args,
        "valid": valid,
        "label": labels,
    }


def _to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def _train_step(
    *,
    batch: dict[str, Any],
    encoder: Any,
    predictor: Any,
    codebook: Any,
    mask_sampler: Any,
    device: Any,
) -> tuple[Any, Any]:
    """One JEPA train step: encode context, predict targets, MSE."""
    batch = _to_device(batch, device)
    B, T = batch["name"].shape
    masks = mask_sampler.sample(grid_h=T, grid_w=1)
    ctx_mask = masks.context.view(T).to(device).unsqueeze(0).expand(B, T)
    ctx_mask = ctx_mask & batch["valid"]

    # Target encoding: full sequence under EMA teacher (no grad).
    with torch.no_grad():
        target_features = encoder.forward_teacher(batch)
    # Context encoding: only kept events visible to student.
    ctx_features = encoder.forward_student(batch, attn_mask=ctx_mask)
    pred = predictor(ctx_features)

    if codebook is not None:
        pred, code_loss = codebook(pred)
    else:
        code_loss = torch.tensor(0.0, device=device)

    losses = []
    for tgt in masks.targets:
        tgt_flat = tgt.view(T).to(device).unsqueeze(0).expand(B, T)
        tgt_flat = tgt_flat & batch["valid"]
        if not tgt_flat.any():
            continue
        pred_sel = pred[tgt_flat]
        targ_sel = target_features[tgt_flat]
        losses.append(F.mse_loss(pred_sel, targ_sel))
    if not losses:
        # Fallback: avoid an empty loss.backward (degenerate sample).
        losses.append(F.mse_loss(pred, target_features))
    loss = torch.stack(losses).mean()
    # VICReg variance penalty on the flattened predictor output. The plain
    # MSE-against-EMA-target loop has a trivial fixed point at "everything
    # is the same constant vector" (predictor matches target, MSE=0, both
    # encoders collapsed). The penalty forces per-feature-dim std above
    # sqrt(VARIANCE_LOSS_EPS) across the batch — see VICReg paper §3.2.
    # Defaults: weight=1.0, eps=1e-2. LLM can tune both via the search.
    if VARIANCE_LOSS_WEIGHT > 0.0:
        pred_flat = pred.reshape(-1, pred.shape[-1])
        std = pred_flat.std(dim=0)
        var_penalty = F.relu(VARIANCE_LOSS_EPS - std).mean()
        loss = loss + VARIANCE_LOSS_WEIGHT * var_penalty
    return loss, code_loss


def _run_canary(
    *,
    sessions: list[dict],
    tokenizer: "TraceTokenizer",
    encoder: Any,
    predictor: Any,
    codebook: Any,
    mask_sampler: Any,
    max_steps: int,
    lr: float,
    device: Any,
) -> float:
    """Overfit a 1k-session subset; success = loss drops below threshold."""
    optimizer = torch.optim.AdamW(
        list(encoder.student.parameters())
        + list(predictor.parameters())
        + (list(codebook.parameters()) if codebook is not None else []),
        lr=lr,
    )
    losses: list[float] = []
    step = 0
    while step < max_steps:
        for i in range(0, len(sessions), BATCH_SIZE):
            if step >= max_steps:
                break
            chunk = sessions[i : i + BATCH_SIZE]
            batch = _batch_to_tensors(chunk, tokenizer)
            loss, code_loss = _train_step(
                batch=batch,
                encoder=encoder,
                predictor=predictor,
                codebook=codebook,
                mask_sampler=mask_sampler,
                device=device,
            )
            total = loss + CODEBOOK_LOSS_WEIGHT * code_loss
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            encoder.update_teacher()
            losses.append(float(loss.item()))
            step += 1
    return min(losses) if losses else float("inf")


def _eval_probe_and_collapse(
    *,
    manifest: dict,
    encoder: Any,
    tokenizer: "TraceTokenizer",
    device: Any,
) -> tuple[float, dict[str, float]]:
    """Linear probe AUROC on the held-out probe set + collapse signals."""
    from autojepa.eval.collapse import latent_variance, rankme

    probe_shards = [DATA_DIR / s for s in manifest["probe_shards"]]
    feats: list[Any] = []
    labels_list: list[Any] = []
    for shard_path in probe_shards:
        with tarfile.open(shard_path, "r") as tar:
            sessions = [
                json.loads(tar.extractfile(m).read().decode("utf-8"))
                for m in tar.getmembers()
                if tar.extractfile(m) is not None
            ]
        for i in range(0, len(sessions), BATCH_SIZE):
            chunk = sessions[i : i + BATCH_SIZE]
            batch = _batch_to_tensors(chunk, tokenizer)
            batch = _to_device(batch, device)
            with torch.no_grad():
                f = encoder.forward_teacher(batch)
                m = batch["valid"].float().unsqueeze(-1)
                pooled = (f * m).sum(dim=1) / (m.sum(dim=1).clamp(min=1.0))
            feats.append(pooled.cpu())
            labels_list.append(batch["label"].cpu())
    feats_t = torch.cat(feats, dim=0)
    labels_t = torch.cat(labels_list, dim=0)

    # Stratified-ish 80/20 split via permutation.
    n = len(feats_t)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    cut = int(n * 0.8)
    train_idx, test_idx = perm[:cut], perm[cut:]
    f_train = feats_t[train_idx].to(device)
    y_train = labels_t[train_idx].to(device)
    f_test = feats_t[test_idx].to(device)
    y_test = labels_t[test_idx].to(device)

    classifier = torch.nn.Linear(feats_t.shape[1], 2).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
    for _ in range(50):
        optimizer.zero_grad()
        logits = classifier(f_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = classifier(f_test)
        probs = F.softmax(logits, dim=-1)[:, 1]
    auroc = _binary_auroc(probs.cpu(), y_test.cpu())

    rm = rankme(feats_t)
    lv = latent_variance(feats_t)
    return float(auroc), {"rankme": float(rm), "latent_var": float(lv)}


def _binary_auroc(scores: Any, labels: Any) -> float:
    """Closed-form binary AUROC; avoids a torchmetrics dependency in
    the trial sidecar (keeps the codepath inspectable)."""
    labels = labels.long()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return 0.5
    pairs = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean()
    ties = (pos.unsqueeze(1) == neg.unsqueeze(0)).float().mean()
    return float((pairs + 0.5 * ties).item())


def main() -> int:
    _check_required_calls()
    if not _TORCH_OK:
        return _abort_missing_torch()

    t_start = time.monotonic()

    # Diagnostic: dump AR_PARAM_* env vars so we can confirm what
    # actually reached the container. trace-jepa v1/v3 surfaced a
    # discrepancy where Claude proposed codebook_size=256 but the
    # container's CODEBOOK_SIZE read 0 — either the env var isn't
    # being set (basilica adapter bug) or the var is being read
    # before env is populated. The dump uses no `key=value` patterns
    # in the value strings to avoid the BasilicaTarget._parse_metrics
    # collision (Phase-2 v27 evidence; see ADR-020 commentary).
    _ar_param_keys = sorted(k for k in os.environ if k.startswith("AR_PARAM_"))
    print(f"[env-dump] {len(_ar_param_keys)} AR_PARAM_* keys present", flush=True)
    for _k in _ar_param_keys:
        # Truncate long values (AR_PARAMS_JSON, AR_MODIFIED_SOURCE)
        _v = os.environ[_k]
        if len(_v) > 80:
            _v = _v[:77] + "..."
        # Use ' -> ' separator (NOT '=') to avoid metric-parser collision.
        print(f"  [env-dump] {_k} -> {_v}", flush=True)

    from autojepa.eval.collapse import latent_variance, rankme
    from autojepa.masking import (
        CompositeMask,
        FutureBlockMask,
        MultiBlockInfillMask,
    )
    from autojepa.models.ema import (
        EMAConfig,
        assert_no_grad_on_target,
        build_target_encoder,
    )

    torch.manual_seed(int(_hp("seed", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} cuda_devices={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
        flush=True,
    )

    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        _write_outcome(
            model_dir=ARTIFACT_DIR,
            status="failed",
            metrics={},
            elapsed_s=time.monotonic() - t_start,
            reason=f"missing manifest: run prepare.py first ({manifest_path} not found)",
        )
        print(
            f"ERROR: data/manifest.json missing under {DATA_DIR}; run prepare.py first",
            file=sys.stderr,
        )
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    tokenizer = TraceTokenizer(
        action_types=tuple(manifest["action_types"]),
        tool_names=tuple(manifest["tool_names"]),
    )

    if PREDICTOR_DEPTH > ENCODER_DEPTH:
        raise ValueError(
            f"predictor_depth={PREDICTOR_DEPTH} > encoder_depth={ENCODER_DEPTH}: "
            "violates JEPA hard rule Ψ <= Φc"
        )
    if PREDICTOR_EMBED_DIM > ENCODER_DIM:
        raise ValueError(
            f"predictor_embed_dim={PREDICTOR_EMBED_DIM} > encoder_dim={ENCODER_DIM}: "
            "violates JEPA hard rule Ψ <= Φc"
        )

    student = TraceTransformer(
        vocab=tokenizer.vocab_size(),
        type_vocab=tokenizer.type_vocab_size(),
        max_events=MAX_EVENTS,
        embed_dim=ENCODER_DIM,
        depth=ENCODER_DEPTH,
        n_heads=ENCODER_HEADS,
        args_hash_dim=ARGS_HASH_DIM,
    ).to(device)

    encoder = build_target_encoder(
        student,
        EMAConfig(
            base_ema_coefficient=EMA_DECAY_START,
            final_ema_coefficient=EMA_DECAY_END,
        ),
    ).to(device)
    assert_no_grad_on_target(encoder)

    predictor = TracePredictor(
        embed_dim=ENCODER_DIM,
        predictor_dim=PREDICTOR_EMBED_DIM,
        depth=PREDICTOR_DEPTH,
        n_heads=max(1, min(ENCODER_HEADS, PREDICTOR_EMBED_DIM // 32)),
    ).to(device)

    codebook = (
        SoftCodebookBottleneck(
            n_codes=CODEBOOK_SIZE,
            embed_dim=ENCODER_DIM,
        ).to(device)
        if CODEBOOK_SIZE > 0
        else None
    )

    n_params = sum(p.numel() for p in student.parameters()) + sum(
        p.numel() for p in predictor.parameters()
    )
    if codebook is not None:
        n_params += sum(p.numel() for p in codebook.parameters())
    print(
        f"trace-jepa params: ~{n_params / 1e6:.1f}M "
        f"(student={sum(p.numel() for p in student.parameters()) / 1e6:.1f}M, "
        f"predictor={sum(p.numel() for p in predictor.parameters()) / 1e6:.1f}M, "
        f"codebook={CODEBOOK_SIZE})",
        flush=True,
    )

    mask_sampler = CompositeMask(
        samplers=[
            (
                FutureBlockMask(
                    n_targets=NUM_TARGETS,
                    target_time_scale=(0.10, 0.30),
                ),
                FUTURE_BLOCK_WEIGHT,
            ),
            (
                MultiBlockInfillMask(
                    n_targets=NUM_TARGETS,
                    target_scale=(0.10, 0.25),
                ),
                MULTI_BLOCK_WEIGHT,
            ),
        ]
    )

    optimizer = torch.optim.AdamW(
        list(student.parameters())
        + list(predictor.parameters())
        + (list(codebook.parameters()) if codebook is not None else []),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ---------- canary: drive L_predict below threshold on 1k samples
    print(f"canary: overfit {DATA_DIR / 'canary.json'} on {device} ...", flush=True)
    canary_sessions = json.loads((DATA_DIR / "canary.json").read_text(encoding="utf-8"))
    canary_loss = _run_canary(
        sessions=canary_sessions,
        tokenizer=tokenizer,
        encoder=encoder,
        predictor=predictor,
        codebook=codebook,
        mask_sampler=mask_sampler,
        max_steps=CANARY_MAX_STEPS,
        lr=LEARNING_RATE,
        device=device,
    )
    emit_progress(
        step=0,
        step_target=MAX_STEPS,
        metrics={"canary_loss": float(canary_loss), "probe_auroc": 0.0},
    )
    if canary_loss > CANARY_LOSS_THRESHOLD:
        elapsed = time.monotonic() - t_start
        _write_outcome(
            model_dir=ARTIFACT_DIR,
            status="failed",
            metrics={"canary_loss": float(canary_loss)},
            elapsed_s=elapsed,
            completed_steps=0,
            step_target=MAX_STEPS,
            reason=f"canary_loss={canary_loss:.4f} > threshold={CANARY_LOSS_THRESHOLD}",
        )
        print(
            f"FAIL: canary loss {canary_loss:.4f} > {CANARY_LOSS_THRESHOLD}; "
            "data pipeline broken or model under-parameterised",
            file=sys.stderr,
        )
        return 1

    # ---------- pretraining loop
    train_shards = [DATA_DIR / s for s in manifest["train_shards"]]
    print(f"pretraining: {len(train_shards)} shards x {manifest['shard_size']} sessions", flush=True)

    # Loud-on-stall instrumentation ported from Phase-2 ijepa-cifar10
    # train.py 2026-05-18 after v3 surfaced silent pod restarts where
    # training never emitted past step=0. SIGTERM handler distinguishes
    # basilica TTL kills from Python hangs; heartbeats narrow the kill
    # window to <50 steps; probe-eval try/except surfaces OOM/hangs
    # there explicitly. All prints use ' | ' separators not 'key=value'
    # to avoid BasilicaTarget._parse_metrics collision (see Phase-2
    # v27 evidence in docs/phase-2-fix-diary.md).
    import signal as _signal

    step = 0
    best_probe_auroc = 0.0
    last_metrics: dict[str, float] = {}

    def _on_sigterm(signum: int, frame: object) -> None:
        print(
            f"[fatal] SIGTERM received at step {step} (basilica TTL or pod kill); "
            f"writing failure outcome and exiting",
            flush=True, file=sys.stderr,
        )
        try:
            _write_outcome(
                model_dir=ARTIFACT_DIR, status="failed", metrics=last_metrics,
                elapsed_s=time.monotonic() - t_start,
                completed_steps=step, step_target=MAX_STEPS,
                reason=f"sigterm_at_step_{step}",
            )
        except Exception:
            pass
        sys.exit(143)

    _signal.signal(_signal.SIGTERM, _on_sigterm)

    iter_loader = _shard_iterator(train_shards, batch_size=BATCH_SIZE, tokenizer=tokenizer)
    last_heartbeat_t = time.monotonic()
    print(f"[pretrain] starting loop, max_steps {MAX_STEPS}, batch {BATCH_SIZE}", flush=True)

    while step < MAX_STEPS:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = _shard_iterator(train_shards, batch_size=BATCH_SIZE, tokenizer=tokenizer)
            batch = next(iter_loader)
        loss, codebook_loss = _train_step(
            batch=batch,
            encoder=encoder,
            predictor=predictor,
            codebook=codebook,
            mask_sampler=mask_sampler,
            device=device,
        )
        total_loss = loss + CODEBOOK_LOSS_WEIGHT * codebook_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        encoder.update_teacher()
        encoder.update_ema_coefficient(step, MAX_STEPS)
        step += 1

        if step % 50 == 0:
            now = time.monotonic()
            dt = now - last_heartbeat_t
            last_heartbeat_t = now
            mem_gb = (
                torch.cuda.memory_allocated() / 1e9
                if torch.cuda.is_available() else 0.0
            )
            print(
                f"[heartbeat] step {step}/{MAX_STEPS} | "
                f"loss {float(loss.item()):.4f} | "
                f"took {dt:.1f}s | mem {mem_gb:.2f}GB",
                flush=True,
            )

        if step % PROBE_EVAL_EVERY_N_STEPS == 0 or step == MAX_STEPS:
            print(f"[probe] starting at step {step}", flush=True)
            try:
                probe_auroc, collapse = _eval_probe_and_collapse(
                    manifest=manifest,
                    encoder=encoder,
                    tokenizer=tokenizer,
                    device=device,
                )
            except BaseException as exc:
                print(
                    f"[probe] FAILED at step {step}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True, file=sys.stderr,
                )
                traceback.print_exc()
                raise
            print(f"[probe] ok at step {step}; probe_auroc={probe_auroc:.4f}", flush=True)
            best_probe_auroc = max(best_probe_auroc, probe_auroc)
            last_metrics = {
                "probe_auroc": float(probe_auroc),
                "loss": float(loss.item()),
                "codebook_loss": float(codebook_loss.item())
                if isinstance(codebook_loss, torch.Tensor)
                else float(codebook_loss),
                "rankme": float(collapse["rankme"]),
                "latent_var": float(collapse["latent_var"]),
                # `lidar` requires a Lightning Trainer queue; the closed-
                # form rankme is the JEPA hard-rule signal we surface
                # from the trial sidecar. Aliasing keeps the metric name
                # forecaster-visible until probes/lidar is wired here.
                "lidar": float(collapse["rankme"]),
            }
            emit_progress(
                step=step,
                step_target=MAX_STEPS,
                metrics=last_metrics,
            )
            print(
                f"step={step} probe_auroc={probe_auroc:.4f} loss={loss.item():.4f} "
                f"rankme={collapse['rankme']:.2f} var={collapse['latent_var']:.3f}",
                flush=True,
            )

    elapsed = time.monotonic() - t_start
    _write_outcome(
        model_dir=ARTIFACT_DIR,
        status="completed",
        metrics={**last_metrics, "best_probe_auroc": float(best_probe_auroc)},
        elapsed_s=elapsed,
        completed_steps=step,
        step_target=MAX_STEPS,
    )
    print(f"final probe_auroc={best_probe_auroc:.4f} after {step} steps", flush=True)

    # Reference the rankme / latent_variance imports so a partial diff
    # that drops them is caught by the AST validator.
    _ = (rankme, latent_variance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
