# Stable-Pretraining v1 — Foundation Model Research Made Simple

- **Paper:** Balestriero, Van Assel, BuGhanem, Maes (2025)
- **arXiv:** 2511.19484
- **URL:** https://arxiv.org/abs/2511.19484
- **Repo:** https://github.com/rbalestr-lab/stable-pretraining
  (mirror: galilai-group/stable-pretraining)
- **License:** MIT
- **Latest release at fetch:** v0.1.6 (2026-03-16)
- **Role in our stack:** AutoJEPA wraps stable-pretraining's Lightning
  callbacks (probes + collapse metrics) instead of re-implementing them.

## 1) One-line thesis

A modular PyTorch-Lightning library that consolidates the recurring
"plumbing" of self-supervised pretraining — augmentation pipelines,
representation probes, and collapse metrics — behind a small, composable
callback API so SSL research stops re-implementing the same primitives in
every paper repo.

## 2) Method (technical)

Built on PyTorch + PyTorch Lightning + Hugging Face + TorchMetrics. Three
load-bearing primitives:

1. **Manager** — orchestrates training alongside Lightning's `Trainer`,
   absorbing checkpointing, cluster-reload, and environment glue.
2. **Forward modules** — encode the SSL view-construction and joint-embedding
   forward graph as composable modules (so a JEPA, SimCLR, or DINO forward
   pass is a small dataclass-like spec).
3. **Callbacks** — every monitoring signal is a `pytorch_lightning.Callback`
   that observes the model without altering the training loop. The library
   ships:

| Callback                              | Purpose                                |
|---------------------------------------|----------------------------------------|
| `spt.callbacks.OnlineProbe`           | Online linear (or arbitrary) probe     |
| `spt.callbacks.OnlineKNN`             | Online k-NN probe with feature queue   |
| `spt.callbacks.RankMe`                | Effective-rank collapse signal         |
| `spt.callbacks.LiDAR`                 | LDA-based latent-discrimination signal |
| `spt.callbacks.CLIPZeroShot`          | CLIP-style zero-shot eval              |
| `spt.callbacks.ImageRetrieval`        | Image-retrieval probe                  |
| `spt.callbacks.LatentViz`             | Embedding visualization                |

The "log everything" design point: scalars from every callback land in the
Lightning logger and TorchMetrics state, so any external scheduler — including
an autoresearch-style controller — can read them without instrumentation.

## 3) Results / headline observations

The paper is a systems contribution rather than a SOTA-chasing benchmark
result. The reported demonstrations are:

- Depth-wise representation analysis showing where collapse appears across
  layers, made trivial by stacking `RankMe` on multiple intermediate outputs.
- A CLIP-degradation study under synthetic-data fine-tuning, exercising the
  zero-shot / retrieval probes alongside collapse metrics.

The numerical magnitudes are paper-specific and not load-bearing for our
adoption decision; the load-bearing claim is that the callbacks are the same
ones cited in the JEPA / SSL collapse-metric literature
(`garrido2023rankme`, `thilak2023lidar`).

## 4) Why it matters for AutoJEPA

AutoJEPA is a hybrid HP search wrapped around a JEPA training loop. The
inner loop already runs Lightning. Stable-pretraining gives us drop-in
callbacks for every metric we need to feed back to the outer search:

| AutoJEPA need                          | Stable-pretraining callback              |
|----------------------------------------|------------------------------------------|
| Linear-probe accuracy as scalar `eval_score` | `spt.callbacks.OnlineProbe` with `torch.nn.Linear` |
| k-NN accuracy as a cheap eval surrogate | `spt.callbacks.OnlineKNN`               |
| Representation-collapse early-stop signal | `spt.callbacks.RankMe`                |
| Independent collapse cross-check       | `spt.callbacks.LiDAR`                    |

Concretely, the AutoJEPA Lightning module should register all four in its
`Trainer(callbacks=[...])` list, and a thin adapter pulls the per-epoch
scalars out of the Lightning logger and into `RunOutcome.metrics` so the
outer search policy can rank trials. No primitive needs re-implementing.

This also means AutoJEPA's collapse-detection guard (writeup §6.x) is a
thresholding rule on `RankMe` and `LiDAR` rather than custom code, which
keeps the implementation auditable.

Reference call shape (from the upstream README):

```python
import stable_pretraining as spt

linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(embed_dim, num_classes),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={"top1": torchmetrics.classification.MulticlassAccuracy(num_classes)},
)
knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    k=10,
)
rankme = spt.callbacks.RankMe(name="rankme", input="embedding")
lidar = spt.callbacks.LiDAR(name="lidar", input="embedding")
trainer = pl.Trainer(callbacks=[linear_probe, knn_probe, rankme, lidar])
```

## 5) Caveats

- The library is pre-1.0 (v0.1.6 at fetch); callback signatures may shift
  before a stable release. AutoJEPA should pin a version and version-gate
  the wrapper module.
- Stable-pretraining's forward-module abstraction is contrastive-friendly;
  for JEPA's predictor + target encoder split, AutoJEPA still owns the
  forward graph. We borrow callbacks, not the forward DSL.
- The library assumes Lightning. If AutoJEPA later supports a non-Lightning
  trainer (e.g., raw FSDP), the callback bridge has to be re-implemented.
- `OnlineProbe` and `OnlineKNN` need a labeled validation stream; for purely
  unlabeled pretraining datasets, only `RankMe` and `LiDAR` apply.

## 6) Cross-links

- `docs/research/HP-SSL-Importance.md` — empirical justification for budgeting
  the SSL HP search aggressively; the probes/metrics here are the read-side
  of that search loop.
- `autojepa-implementation-plan.md` §6.3 — hybrid-search dimensionality
  expansion that relies on these callbacks for cheap per-trial scoring.
- `autojepa-implementation-plan.md` §7.1 — explicit dependency on
  stable-pretraining for probes and collapse metrics.

Sources:
- [stable-pretraining-v1 paper](https://arxiv.org/abs/2511.19484)
- [stable-pretraining GitHub](https://github.com/rbalestr-lab/stable-pretraining)
