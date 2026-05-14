# AutoJEPA

> Autonomous design-space search over Joint-Embedding Predictive Architecture (JEPA) pretraining recipes.
> A clean fork of [`autoresearch-rl`](../autoresearch-rl), purpose-built for self-supervised pretraining.
> **Prime deployment target: [Basilica](https://basilica.ai) GPU cloud.**

```
prepare.py  -->  [data + probe-eval]  -->  train.py  -->  [probe_auroc]  -->  keep/discard  -->  repeat
 (frozen)                                  (mutable)                            |
                                                ^                               |
                                                |    LLM proposes next          |
                                                +-------- params or diff -------+
```

## Identity

AutoJEPA inherits the autoresearch pattern — frozen `prepare.py` + mutable `train.py`, AST-validated LLM-proposed diffs, hybrid (param → diff on stall) policy — and replaces RL/SFT-shaped defaults with JEPA-shaped defaults:

- **Probe-based downstream evaluation** as the campaign objective (`probe_auroc`, not training loss — JEPA loss collapses).
- **RankMe / LiDAR / latent-variance / effective-rank** as hard fail gates against representation collapse.
- **VICReg-aware loss defaults** ([C-JEPA](https://arxiv.org/abs/2410.19560)).
- **Composable mask primitives** as first-class building blocks the LLM combines.
- **Forecaster recalibrated for SSL learning curves** (long plateau where only probe score moves).
- **Multi-seed scoring** because JEPA outcomes are seed-sensitive.

## Quickstart

```bash
uv sync --extra dev --extra jepa
uv run autojepa run examples/ijepa-cifar10/config.yaml
```

Common workflows are wrapped in `Makefile`:

```bash
make help        # list targets
make check       # lint + typecheck + full tests
make test-fast   # tests excluding slow integration suite
```

## Basilica-first

AutoJEPA targets GPU pretraining; Basilica is the prime deployment target and `basilica-sdk` is a default dependency. Local `command` and `http` targets remain available (inherited from `autoresearch-rl`) but campaign configs default to `target: basilica`.

```yaml
target:
  type: basilica
  image: pytorch:2.4.1-cuda12.4
  gpu_count: 1
  gpu_models: [A100, H100]
  memory: 32Gi
```

## The two scripts

Every campaign has two scripts connected by the filesystem, never by imports:

**`prepare.py`** (frozen) — runs once via `prepare_cmd`. Produces data shards, defines the probe-eval pipeline and collapse-detection callbacks. The LLM cannot modify this file. Trust boundary: evaluation integrity is guaranteed by freezing it.

**`train.py`** (mutable) — runs each iteration. Reads prepared data, trains the JEPA model (Φc context encoder + Φt EMA target encoder + Ψ predictor), prints metrics to stdout via `emit_progress`. The LLM proposes diffs in `llm_diff` or `hybrid` mode.

## Roadmap

See [TODO.md](TODO.md) for the live phased plan and [docs/research/](docs/research/) for the cited research corpus. Architecture writeup: [gist 2567a53](https://gist.github.com/epappas/2567a53350ba0d6ca064c71986a76046).

The Phase-2 falsifier (CIFAR I-JEPA) is the kill criterion for the framework approach: if a 20-iter hybrid campaign produces zero validated diff improvements against a deliberately suboptimal baseline, the entire AutoJEPA approach is dead.

## Lineage

FunSearch → AI Scientist v1/v2 → ADAS → AIDE → AlphaEvolve → karpathy/autoresearch → `autoresearch-rl` → **AutoJEPA**.

Sibling upstream: `../autoresearch-rl` (added as `git remote upstream` for cherry-pick reference only — capped at ~1h/wk).

## License

TBD pending Phase 5 public release. Internal use only until then.
