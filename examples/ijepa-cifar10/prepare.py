"""Frozen-side data + eval pipeline for the I-JEPA CIFAR-10 example.

Per writeup §Phase-2 / autoresearch contract:
- This file is FROZEN. The LLM diff policy may not modify it.
- It owns "what is correct": dataset download, the probe-eval split,
  and the canary-overfit subset (writeup §7.4).
- `train.py` reads from `data/` and emits `probe_auroc` via
  `emit_progress`; this file does not run training itself.

Outputs (relative to this file's directory):
    data/cifar10_train.pt       full pretraining set (50000, 3, 32, 32) uint8
    data/cifar10_test.pt        held-out set         (10000, 3, 32, 32) uint8
    data/cifar10_train_labels.pt (50000,) int64
    data/cifar10_test_labels.pt  (10000,) int64
    data/probe_eval.pt           {x_train, y_train, x_test, y_test} small
                                 dict for the linear-probe evaluator
    data/canary.pt               first 1k samples of the train set —
                                 used by the sanity-overfit canary

Idempotent: re-running re-uses the already-downloaded data.
"""

from __future__ import annotations

import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PROBE_TRAIN_N = 5000
PROBE_TEST_N = 5000
CANARY_N = 1000


def main() -> int:
    try:
        import torch
        from torchvision import datasets, transforms
    except ImportError as exc:
        print(f"ERROR: prepare.py requires torch + torchvision: {exc}", file=sys.stderr)
        return 2

    train_path = DATA_DIR / "cifar10_train.pt"
    test_path = DATA_DIR / "cifar10_test.pt"

    if train_path.exists() and test_path.exists():
        print(f"data already prepared at {DATA_DIR}; skipping download")
        return 0

    print(f"downloading CIFAR-10 to {DATA_DIR}/raw ...")
    transform = transforms.Compose([transforms.PILToTensor()])
    train = datasets.CIFAR10(
        root=str(DATA_DIR / "raw"), train=True, download=True, transform=transform
    )
    test = datasets.CIFAR10(
        root=str(DATA_DIR / "raw"), train=False, download=True, transform=transform
    )

    train_x = torch.stack([train[i][0] for i in range(len(train))])
    train_y = torch.tensor([train[i][1] for i in range(len(train))], dtype=torch.long)
    test_x = torch.stack([test[i][0] for i in range(len(test))])
    test_y = torch.tensor([test[i][1] for i in range(len(test))], dtype=torch.long)

    torch.save(train_x, train_path)
    torch.save(train_y, DATA_DIR / "cifar10_train_labels.pt")
    torch.save(test_x, test_path)
    torch.save(test_y, DATA_DIR / "cifar10_test_labels.pt")

    probe_eval = {
        "x_train": train_x[:PROBE_TRAIN_N].clone(),
        "y_train": train_y[:PROBE_TRAIN_N].clone(),
        "x_test": test_x[:PROBE_TEST_N].clone(),
        "y_test": test_y[:PROBE_TEST_N].clone(),
    }
    torch.save(probe_eval, DATA_DIR / "probe_eval.pt")

    canary = {
        "x": train_x[:CANARY_N].clone(),
        "y": train_y[:CANARY_N].clone(),
    }
    torch.save(canary, DATA_DIR / "canary.pt")

    print(
        f"prepared: train={tuple(train_x.shape)} test={tuple(test_x.shape)} "
        f"probe={PROBE_TRAIN_N}/{PROBE_TEST_N} canary={CANARY_N}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
