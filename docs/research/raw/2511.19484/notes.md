# stable-pretraining-v1 — extraction notes

**Sources used**
- arxiv abstract page: https://arxiv.org/abs/2511.19484 (WebFetch OK)
- arxiv PDF: https://arxiv.org/pdf/2511.19484 (WebFetch OK, 451.9 KB)
- GitHub repo: https://github.com/rbalestr-lab/stable-pretraining (WebFetch OK,
  README contained the callback example used below)
- Mirror at galilai-group/stable-pretraining surfaced by WebSearch
- Docs site https://rbalestr-lab.github.io/stable-pretraining.github.io/dev/
  returned HTTP 404 at fetch time

## Package layout (verified from README)

- Distribution and import name: `stable_pretraining`
- Conventional alias used in docs: `import stable_pretraining as spt`
- License: MIT
- Latest release at fetch time: v0.1.6 (2026-03-16)
- Built on PyTorch, PyTorch Lightning, Hugging Face Transformers/Datasets,
  TorchMetrics

## Callback import paths (load-bearing for AutoJEPA wrapping)

| Callback                       | Import path                          |
|--------------------------------|--------------------------------------|
| Online linear probe            | `spt.callbacks.OnlineProbe`          |
| Online k-NN probe              | `spt.callbacks.OnlineKNN`            |
| RankMe (Garrido et al. 2023)   | `spt.callbacks.RankMe`               |
| LiDAR (Thilak et al. 2023)     | `spt.callbacks.LiDAR`                |
| CLIP zero-shot evaluation      | `spt.callbacks.CLIPZeroShot`         |
| Image-retrieval evaluation     | `spt.callbacks.ImageRetrieval`       |
| Latent visualization           | `spt.callbacks.LatentViz`            |

## README usage example (verbatim from repo README)

```python
import stable_pretraining as spt
import torch, torchmetrics

linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(512, 10),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
        "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    k=10,
)
```

## Citations the paper relies on for collapse metrics

- `garrido2023rankme` — RankMe (referenced in PDF body)
- `thilak2023lidar` — LiDAR (referenced in PDF body)

## Relevance to AutoJEPA

- `RankMe` and `LiDAR` are the recommended collapse-detection signals for the
  representation-quality monitor in the AutoJEPA writeup (sect. 7.1).
- `OnlineProbe` (linear) and `OnlineKNN` are the in-loop probes that AutoJEPA
  needs to wrap so that its hybrid search policy can read scalar surrogates
  for downstream quality without offline eval.
- The Lightning-callback contract (`on_train_batch_end`,
  `on_validation_epoch_end`, etc.) means AutoJEPA can register these in the
  same `Trainer(callbacks=[...])` it already builds for JEPA training; no
  wrapper layer is mandatory beyond a small adapter that forwards
  callback-emitted scalars into AutoJEPA's `RunOutcome.metrics` dict.

## Caveats

- The arxiv PDF was not fully OCR-decoded by WebFetch; the callback list was
  cross-checked against the GitHub README rather than the paper body.
- v0.1.6 is the most recent release at the time of fetch; the API may break
  before a v1.0.0 stable release.
