#!/usr/bin/env bash
# Convenience runner for the I-JEPA CIFAR-10 example.
#
# Modes:
#   ./run.sh validate           autojepa validate config.yaml (no GPU needed)
#   ./run.sh prepare            run prepare.py once (downloads CIFAR-10)
#   ./run.sh smoke              run train.py with a tiny step budget (CPU/GPU)
#   ./run.sh local              run a 1-iter campaign locally (target=command)
#   ./run.sh basilica           run the full 20-iter campaign on Basilica
#
# Requires: autojepa installed with [jepa] extra, plus CHUTES_API_KEY for
# LLM-backed campaign modes (validate / prepare / smoke do not need it).
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
    AR_PARAMS_JSON='{"max_steps": 50, "batch_size": 64, "probe_eval_every_n_steps": 25}' \
      AR_PROGRESS_FILE="/tmp/autojepa_progress.jsonl" \
      uv run python3 "$HERE/train.py"
    ;;
  local)
    uv run autojepa run "$CONFIG" \
      --override target.type=command \
      --override controller.max_iterations=1
    ;;
  basilica)
    : "${CHUTES_API_KEY:?CHUTES_API_KEY must be set for the LLM-backed campaign}"
    : "${BASILICA_API_TOKEN:?BASILICA_API_TOKEN must be set for the GPU target}"
    uv run autojepa run "$CONFIG"
    ;;
  *)
    echo "usage: $0 {validate|prepare|smoke|local|basilica}" >&2
    exit 2
    ;;
esac
