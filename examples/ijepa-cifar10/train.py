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
import time
import traceback
from pathlib import Path
from typing import Any

# ---------- contract: read params and progress emitter from the controller

from autojepa.target.progress import emit_progress  # noqa: E402

PARAMS = json.loads(os.environ.get("AR_PARAMS_JSON", "{}"))

OUTCOME_FILENAME = "outcome.json"


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

    The basilica adapter polls /model/files for this file; presence is the
    iter-done signal that lets the controller bypass timeout-based waits.
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
        print(f"[outcome] WARN: failed to write {OUTCOME_FILENAME}: {exc}", file=sys.stderr, flush=True)


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
CANARY_LOSS_THRESHOLD = float(_hp("canary_loss_threshold", 0.08))
# Threshold empirically calibrated against the deliberately suboptimal
# ViT-Tiny baseline (ADR-014). Original 0.05 was a guess — verified live
# across v9-v16 runs that the baseline's actual L_predict floor on a 1k-
# sample overfit set is ~0.058-0.061 regardless of learning rate. The
# 0.08 ceiling still catches pipeline breaks (loss=NaN, loss=0.5+) which
# is the canary's actual purpose per writeup §7.4, without falsely
# rejecting healthy-but-suboptimal trials.

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
        from stable_pretraining.methods import IJEPA
    except ImportError as exc:
        _write_outcome(
            model_dir=ARTIFACT_DIR,
            status="failed",
            metrics={},
            elapsed_s=0.0,
            reason=f"import_error: {exc}",
        )
        print(f"ERROR: train.py requires the [jepa] extra: {exc}", file=sys.stderr)
        return 2

    from autojepa.models.ema import assert_no_grad_on_target

    pl.seed_everything(int(_hp("seed", 0)), workers=True)

    t_start = time.monotonic()

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
    canary_x = canary["x"]  # (1000, 3, 32, 32) uint8

    # Eager resize of all 50k CIFAR images to (50000, 3, 224, 224) float32
    # would be ~30 GB and OOM the 32 GiB container. Resize lazily per-batch
    # via a custom Dataset wrapper instead — the per-batch tensor at
    # batch=128 is ~75 MB, fits easily.
    print(f"using lazy 224x224 resize over {len(train_x)} train + {len(canary_x)} canary samples", flush=True)

    class _LazyResizeDataset(torch.utils.data.Dataset):
        def __init__(self, x_uint8: torch.Tensor) -> None:
            self._x = x_uint8

        def __len__(self) -> int:
            return self._x.shape[0]

        def __getitem__(self, idx: int) -> torch.Tensor:
            x = self._x[idx].float() / 255.0
            return TF.resize(x, [224, 224], antialias=True)

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
    print(f"canary: overfit {len(canary_x)} samples on {device} ...", flush=True)
    canary_loader = torch.utils.data.DataLoader(
        _LazyResizeDataset(canary_x),
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

    # ---------- pretraining loop with periodic probe eval
    # num_workers=0 (single-process) because the Basilica container ships
    # the Docker default /dev/shm of 64 MB. multi-process DataLoader
    # workers serialise tensors via shared memory and crash with
    # "Bus error" / "DataLoader worker killed" on the first fetch.
    # Verified live on v8 (commit 6657977): canary loader with
    # num_workers=0 succeeded, pretrain loader with num_workers=2
    # crashed immediately. The lazy 224x224 resize is cheap enough on
    # the main process that the GPU isn't starved.
    pretrain_loader = torch.utils.data.DataLoader(
        _LazyResizeDataset(train_x),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # We call wrapper.update_teacher() / update_ema_coefficient() directly
    # rather than via spt.callbacks.TeacherStudentCallback because the
    # callback expects a real Lightning Trainer (`trainer.global_step`)
    # and we run a manual training loop. Verified live on v9
    # (commit badbe80) which crashed in the callback with
    # `AttributeError: 'NoneType' object has no attribute 'global_step'`.
    # Loud-on-stall instrumentation (added 2026-05-18 after v26 iter=6/7/8
    # all silently stopped emitting progress around step 2000-2500 with no
    # traceback). Heartbeat every 50 steps so silence between heartbeats
    # narrows the kill window to <50 steps. Probe-eval try/except so OOM
    # or hang in probe code surfaces a typed exception instead of looking
    # like training froze. SIGTERM trap so basilica TTL kills become
    # distinguishable from Python-side hangs.
    import signal as _signal

    def _on_sigterm(signum: int, frame: object) -> None:
        print(
            f"[fatal] SIGTERM received at step={step} (basilica TTL or pod kill); "
            f"writing failure outcome and exiting",
            flush=True, file=sys.stderr,
        )
        try:
            _write_outcome(
                model_dir=ARTIFACT_DIR, status="failed", metrics={},
                elapsed_s=time.monotonic() - t_start,
                completed_steps=step, step_target=MAX_STEPS,
                reason=f"sigterm_at_step_{step}",
            )
        except Exception:
            pass
        sys.exit(143)

    _signal.signal(_signal.SIGTERM, _on_sigterm)

    step = 0
    best_probe_auroc = 0.0
    last_heartbeat_t = time.monotonic()
    print(f"[pretrain] starting loop max_steps={MAX_STEPS} batch={BATCH_SIZE}", flush=True)
    while step < MAX_STEPS:
        for images in pretrain_loader:
            if step >= MAX_STEPS:
                break
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(images)
            output.loss.backward()
            optimizer.step()
            model.encoder.update_teacher()
            model.encoder.update_ema_coefficient(step, MAX_STEPS)
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
                    f"[heartbeat] step={step}/{MAX_STEPS} loss={float(output.loss.item()):.4f} "
                    f"dt={dt:.1f}s mem={mem_gb:.2f}GB",
                    flush=True,
                )

            if step % PROBE_EVAL_EVERY_N_STEPS == 0 or step == MAX_STEPS:
                print(f"[probe] starting at step={step}", flush=True)
                try:
                    probe_auroc = _run_linear_probe(model, DATA_DIR, device=device)
                except BaseException as exc:
                    print(
                        f"[probe] FAILED at step={step}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True, file=sys.stderr,
                    )
                    traceback.print_exc()
                    raise
                print(f"[probe] ok step={step} probe_auroc={probe_auroc:.4f}", flush=True)
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

    elapsed = time.monotonic() - t_start
    _write_outcome(
        model_dir=ARTIFACT_DIR,
        status="ok",
        metrics={
            "probe_auroc": float(best_probe_auroc),
            "loss": float(output.loss.item()),
            "canary_loss": float(canary_loss),
        },
        elapsed_s=elapsed,
        completed_steps=step,
        step_target=MAX_STEPS,
    )
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
        for images in loader:
            if step >= max_steps:
                break
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
    Probe set is 5k+5k uint8 (~120 MB resized lazily inside _extract_features
    rather than eagerly to fit the container's 32 GiB memory budget).
    """
    import torch

    probe_data = torch.load(data_dir / "probe_eval.pt")
    y_train = probe_data["y_train"].to(device)
    y_test = probe_data["y_test"].to(device)

    model.eval()
    with torch.no_grad():
        feats_train = _extract_features(model, probe_data["x_train"], batch_size=128, device=device)
        feats_test = _extract_features(model, probe_data["x_test"], batch_size=128, device=device)
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


def _extract_features(model, x_uint8, batch_size: int, device):
    """Extract per-image embeddings from the EMA target encoder.

    Accepts a uint8 (N, 3, 32, 32) CIFAR tensor; resizes per-batch to
    (B, 3, 224, 224) float to fit memory. The stable-pretraining
    MaskedEncoder returns a `MaskedEncoderOutput` namedtuple with
    `.encoded` (B, N+cls, D); we drop the CLS token then mean-pool to
    get (B, D) probe features.
    """
    import torch
    import torchvision.transforms.functional as TF

    parts: list[torch.Tensor] = []
    for i in range(0, len(x_uint8), batch_size):
        chunk = x_uint8[i : i + batch_size].float() / 255.0
        batch = TF.resize(chunk, [224, 224], antialias=True).to(device, non_blocking=True)
        with torch.no_grad():
            out = model.encoder.forward_teacher(batch)
        feats = out.encoded if hasattr(out, "encoded") else out
        if feats.ndim == 3:
            num_prefix = getattr(out, "num_prefix_tokens", 1)
            feats = feats[:, num_prefix:].mean(dim=1)
        parts.append(feats)
    return torch.cat(parts, dim=0)


def _entrypoint() -> int:
    """Wrap main() so any uncaught exception still writes outcome.json.

    Without this, a crash before the success/failure path leaves the
    basilica adapter waiting on the timeout — exactly the failure mode
    that motivated this contract.
    """
    t_start = time.monotonic()
    try:
        return main()
    except SystemExit:
        raise
    except BaseException as exc:
        elapsed = time.monotonic() - t_start
        _write_outcome(
            model_dir=ARTIFACT_DIR,
            status="failed",
            metrics={},
            elapsed_s=elapsed,
            reason=f"unhandled_exception: {type(exc).__name__}: {exc}",
        )
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(_entrypoint())
