"""I-JEPA on CIFAR-10 — the AutoJEPA Phase-2 falsifier.

DELIBERATELY SUBOPTIMAL baseline (writeup §12.1 mitigation): the LLM
diff policy needs headroom. This script trains a small I-JEPA with a
shallow predictor and a plain L2 latent loss (no anti-collapse
regularisation). Improvements the hybrid policy is expected to find:

- Add VICReg / Barlow Twins anti-collapse via `autojepa.models.losses.LOSS_REGISTRY`
- Tune EMA schedule (`ema_decay_start`, `ema_decay_end`)
- Tune masking ratios / number of target blocks
- Deepen the predictor up to (but not exceeding) encoder depth

Hyperparameters consumed from `AR_PARAMS_JSON`:
    learning_rate, weight_decay, batch_size, max_steps,
    predictor_depth, predictor_embed_dim, num_targets,
    ema_decay_start, ema_decay_end, mask_ratio_max

Mutable per the autoresearch contract: the LLM diff policy may rewrite
this file. Required calls enforced by the AST validator (writeup §6.4
program.md / `autojepa.policy._prompt_fragments.JEPA_HARD_RULES`):

    emit_progress(step, step_target, metrics={"probe_auroc": ...})
    autojepa.models.ema.assert_no_grad_on_target(...)

Outputs `probe_auroc` (the campaign objective per ADR-004) and
`canary_loss` (writeup §7.4).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------- contract: read params and progress emitter from the controller

from autojepa.target.progress import emit_progress  # noqa: E402

PARAMS = json.loads(os.environ.get("AR_PARAMS_JSON", "{}"))


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


# Suboptimal baseline defaults — see module docstring.
LEARNING_RATE = float(_hp("learning_rate", 1e-4))
WEIGHT_DECAY = float(_hp("weight_decay", 0.05))
BATCH_SIZE = int(_hp("batch_size", 128))
MAX_STEPS = int(_hp("max_steps", 4000))
PREDICTOR_DEPTH = int(_hp("predictor_depth", 2))           # SHALLOW
PREDICTOR_EMBED_DIM = int(_hp("predictor_embed_dim", 128))  # NARROW
NUM_TARGETS = int(_hp("num_targets", 2))                    # FEW
EMA_DECAY_START = float(_hp("ema_decay_start", 0.996))
EMA_DECAY_END = float(_hp("ema_decay_end", 1.0))
PROBE_EVAL_EVERY_N_STEPS = int(_hp("probe_eval_every_n_steps", 500))
CANARY_LOSS_THRESHOLD = float(_hp("canary_loss_threshold", 0.05))

DATA_DIR = Path(__file__).parent / "data"
ARTIFACT_DIR = Path(os.environ.get("AR_MODEL_DIR", str(Path(__file__).parent / "artifacts")))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _check_required_calls() -> None:
    """Asserted at script start so a malformed diff fails fast."""
    assert callable(emit_progress)
    from autojepa.models.ema import assert_no_grad_on_target  # noqa: F401


def main() -> int:
    _check_required_calls()

    try:
        import lightning.pytorch as pl
        import torch
        import torchvision.transforms.functional as TF
        from stable_pretraining import TeacherStudentCallback
        from stable_pretraining.methods import IJEPA
    except ImportError as exc:
        print(f"ERROR: train.py requires the [jepa] extra: {exc}", file=sys.stderr)
        return 2

    from autojepa.models.ema import assert_no_grad_on_target

    pl.seed_everything(int(_hp("seed", 0)), workers=True)

    # Device selection. Basilica nodes have GPU; local CPU is fallback.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} cuda_devices={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
        flush=True,
    )

    train_x = torch.load(DATA_DIR / "cifar10_train.pt")  # (50000, 3, 32, 32) uint8
    # train_y not consumed during pretraining (SSL); the linear-probe
    # eval reads its own labels from probe_eval.pt.
    canary = torch.load(DATA_DIR / "canary.pt")

    # Resize CIFAR to 224 to match the timm vit_tiny encoder. Done once
    # at load time on CPU; small enough that this is < 1 GB RAM.
    def _to_224(x_uint8: torch.Tensor) -> torch.Tensor:
        x = x_uint8.float() / 255.0
        return TF.resize(x, [224, 224], antialias=True)

    print(f"resizing {len(train_x)} train images to 224x224 ...", flush=True)
    train_x_224 = _to_224(train_x)
    canary_x_224 = _to_224(canary["x"])

    model = IJEPA(
        encoder_name="vit_tiny_patch16_224",
        predictor_embed_dim=PREDICTOR_EMBED_DIM,
        predictor_depth=PREDICTOR_DEPTH,
        num_targets=NUM_TARGETS,
        ema_decay_start=EMA_DECAY_START,
        ema_decay_end=EMA_DECAY_END,
        pretrained=False,
    ).to(device)
    assert_no_grad_on_target(model.encoder)

    # ---------- canary: drive L_predict below threshold on 1k samples
    print(f"canary: overfit {len(canary_x_224)} samples on {device} ...", flush=True)
    canary_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(canary_x_224),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    canary_loss = _run_canary(
        model, canary_loader, max_steps=200, lr=LEARNING_RATE, device=device
    )
    emit_progress(
        step=0,
        step_target=MAX_STEPS,
        metrics={"canary_loss": canary_loss, "probe_auroc": 0.0},
    )
    if canary_loss > CANARY_LOSS_THRESHOLD:
        print(
            f"FAIL: canary loss {canary_loss:.4f} > {CANARY_LOSS_THRESHOLD}; "
            "data pipeline broken or model under-parameterised",
            file=sys.stderr,
        )
        return 1

    # ---------- pretraining loop with periodic probe eval
    pretrain_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x_224),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    teacher_student_cb = TeacherStudentCallback()

    step = 0
    best_probe_auroc = 0.0
    while step < MAX_STEPS:
        for batch in pretrain_loader:
            if step >= MAX_STEPS:
                break
            (images,) = batch
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(images)
            output.loss.backward()
            optimizer.step()
            teacher_student_cb.on_train_batch_end(None, model, None, None, step)
            step += 1

            if step % PROBE_EVAL_EVERY_N_STEPS == 0 or step == MAX_STEPS:
                probe_auroc = _run_linear_probe(model, DATA_DIR, device=device)
                best_probe_auroc = max(best_probe_auroc, probe_auroc)
                emit_progress(
                    step=step,
                    step_target=MAX_STEPS,
                    metrics={
                        "probe_auroc": probe_auroc,
                        "loss": float(output.loss.item()),
                    },
                )
                print(f"step={step} probe_auroc={probe_auroc:.4f}", flush=True)

    print(f"final probe_auroc={best_probe_auroc:.4f} after {step} steps", flush=True)
    return 0


def _run_canary(model, loader, max_steps: int, lr: float, device) -> float:
    import torch

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    losses: list[float] = []
    step = 0
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            (images,) = batch
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(images)
            out.loss.backward()
            optimizer.step()
            losses.append(float(out.loss.item()))
            step += 1
    return min(losses) if losses else float("inf")


def _run_linear_probe(model, data_dir: Path, device) -> float:
    """Frozen-features linear probe on the eval split.

    Uses a 50-step linear classifier fit on extracted features; matches
    the methodology in I-JEPA paper §3 linear-eval protocol but cheaper.
    """
    import torch
    import torchvision.transforms.functional as TF

    probe_data = torch.load(data_dir / "probe_eval.pt")
    x_train = TF.resize(probe_data["x_train"].float() / 255.0, [224, 224], antialias=True)
    x_test = TF.resize(probe_data["x_test"].float() / 255.0, [224, 224], antialias=True)
    y_train = probe_data["y_train"].to(device)
    y_test = probe_data["y_test"].to(device)

    model.eval()
    with torch.no_grad():
        feats_train = _extract_features(model, x_train, batch_size=128, device=device)
        feats_test = _extract_features(model, x_test, batch_size=128, device=device)
    model.train()

    classifier = torch.nn.Linear(feats_train.shape[1], 10).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(50):
        optimizer.zero_grad()
        logits = classifier(feats_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = classifier(feats_test)
        preds = logits.argmax(dim=-1)
        acc = float((preds == y_test).float().mean().item())
    return acc


def _extract_features(model, x, batch_size: int, device):
    """Extract per-image embeddings from the EMA target encoder.

    The stable-pretraining MaskedEncoder returns a `MaskedEncoderOutput`
    namedtuple with `.encoded` (B, N+cls, D), `.grid_size`, `.ids_keep`,
    `.mask`. We average over patch tokens (excluding the CLS token if
    present) to get (B, D) probe features.
    """
    import torch

    parts: list[torch.Tensor] = []
    for i in range(0, len(x), batch_size):
        batch = x[i : i + batch_size].to(device, non_blocking=True)
        with torch.no_grad():
            out = model.encoder.forward_teacher(batch)
        feats = out.encoded if hasattr(out, "encoded") else out
        if feats.ndim == 3:
            # Drop CLS token if encoder uses one (vit_tiny does), then mean-pool.
            num_prefix = getattr(out, "num_prefix_tokens", 1)
            feats = feats[:, num_prefix:].mean(dim=1)
        parts.append(feats)
    return torch.cat(parts, dim=0)


if __name__ == "__main__":
    sys.exit(main())
