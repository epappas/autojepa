#!/usr/bin/env bash
# Convenience runner for the Phase-3 trace-jepa example.
#
# Modes:
#   ./run.sh validate           autojepa validate config.yaml (no GPU needed)
#   ./run.sh prepare            run prepare.py once (writes data/)
#   ./run.sh smoke              run train.py with a tiny step budget (CPU/GPU)
#   ./run.sh local              run a 1-iter campaign locally (target=command)
#   ./run.sh basilica           run the full 30-iter campaign on Basilica
#
# Requires: autojepa installed with [jepa] extra, plus OPENROUTER_API_KEY
# for the LLM-backed campaign mode (validate / prepare / smoke don't need
# it). BASILICA_API_TOKEN required only for the basilica mode.
set -euo pipefail

MODE="${1:-validate}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
CONFIG="$HERE/config.yaml"

cd "$REPO_ROOT"

case "$MODE" in
  validate)
    uv run autojepa validate "$CONFIG"
    ;;
  prepare)
    cd "$HERE" && python3 prepare.py
    ;;
  smoke)
    AR_PARAMS_JSON='{"max_steps": 30, "batch_size": 16, "encoder_depth": 2, "encoder_dim": 64, "encoder_heads": 4, "predictor_embed_dim": 64, "probe_eval_every_n_steps": 15, "canary_max_steps": 15, "canary_loss_threshold": 100.0}' \
      AR_PROGRESS_FILE="/tmp/autojepa_trace_progress.jsonl" \
      uv run python3 "$HERE/train.py"
    ;;
  local)
    uv run autojepa run "$CONFIG" \
      --override target.type=command \
      --override controller.max_iterations=1
    ;;
  basilica)
    : "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set for the Claude/OpenRouter LLM policy}"
    : "${BASILICA_API_TOKEN:?BASILICA_API_TOKEN must be set for the GPU target}"
    # Use deploy.py per Phase-2 pattern: it base64-encodes train.py +
    # prepare.py and injects them into /app/ via setup_cmd. Plain
    # `autojepa run` would only install the autojepa wheel (which
    # ships src/, not examples/) and train_cmd would fail with
    # "No such file: /app/train.py" inside the container.
    SHA=$(git -C "$REPO_ROOT" rev-parse HEAD)
    exec uv run python "$HERE/deploy.py" --git-ref "$SHA"
    ;;
  *)
    echo "usage: $0 {validate|prepare|smoke|local|basilica}" >&2
    exit 2
    ;;
esac
