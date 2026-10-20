#!/usr/bin/env bash
# Optional offline demo of the RL policy's DQN fine-tuning stage.
#
# The main demo's three cycles accumulate a few hundred replay transitions,
# short of the paper's 1,000-transition training threshold, so every cycle
# there runs behaviour cloning only. This script reruns the same bundled
# fixtures with only that threshold lowered (to 100), so cycles 2 and 3 run
# behaviour cloning followed by DQN fine-tuning. It prints the per-cycle
# training table and writes it to `output/dqn_demo/dqn_training.json`.
#
# It shows that the stage executes and updates the policy; it makes no claim
# that the fine-tuned policy is better (a few hundred transitions are far too
# few for that). It writes only under its own output root: the main demo's
# output/report.json and viewer data are left alone.
#
# Exits non-zero unless a cycle ran DQN fine-tuning with the configured
# number of episodes and a finite loss trace.
#
# Runnable from any working directory.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: demo/run_dqn_demo.sh [--output-root DIR]

  --output-root DIR   Where the run's cycle tree and dqn_training.json go
                      (default: output/dqn_demo).
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$ROOT"

OUTPUT_ROOT="output/dqn_demo"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --output-root)
      [ "$#" -ge 2 ] || { echo "error: --output-root requires a directory" >&2; exit 2; }
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Deterministic offline run, pinned exactly as in demo/run_demo.sh.
export TIBLENDER_MODE=offline
export TIBLENDER_CLOCK=auto
export PYTHONHASHSEED=0
export PYTHONPATH="$ROOT"

if command -v tiblender >/dev/null 2>&1 && python3 -c "import tiblender_runtime.cli" >/dev/null 2>&1; then
  TIBLENDER="tiblender"
elif [ -x "$ROOT/.venv/bin/tiblender" ]; then
  TIBLENDER="$ROOT/.venv/bin/tiblender"
else
  echo "==> tiblender is not installed; creating $ROOT/.venv with install.sh --offline"
  bash "$ROOT/install.sh" --offline
  TIBLENDER="$ROOT/.venv/bin/tiblender"
fi

echo "==> Running the DQN fine-tuning demo (3 cycles, training threshold 100)"
"$TIBLENDER" dqn-demo --output-root "$OUTPUT_ROOT"

echo "==> Wrote $OUTPUT_ROOT/dqn_training.json"
