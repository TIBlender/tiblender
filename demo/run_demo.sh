#!/usr/bin/env bash
# One-command offline demo.
#
# Runs the bundled three-cycle pipeline from recorded evidence only: no
# credentials, no network, no external services. It ends by writing every
# report the run produced, the per-cycle counters and the `verify-sample`
# summary into a single document at `output/report.json`.
#
# Exits non-zero if `verify-sample` finds any difference from the committed
# expected output, so a green run is a checked run, not just a finished one.
#
# Runnable from any working directory.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: demo/run_demo.sh [--output-root DIR]

  --output-root DIR   Where the run's cycle tree and report.json go
                      (default: output).

Environment:
  DATASET_DIR         Dataset location. Not needed by the demo, which
                      reads only fixtures/; resolved for consistency with
                      evaluation/run_evaluation.sh.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT="output"
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

REPORT_JSON="$OUTPUT_ROOT/report.json"

# The demo reads only `fixtures/`; the dataset is not needed. Resolved
# anyway so both entry-point scripts answer DATASET_DIR the same way.
DATASET_DIR="$(bash "$ROOT/tools/dataset_dir.sh" "$ROOT")"
export DATASET_DIR

# Deterministic offline run: the same environment the Dockerfile sets, pinned
# here so a host with network access does not silently make live tool calls.
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

echo "==> Running the offline demo (3 cycles, verify-sample, viewer)"
"$TIBLENDER" demo \
  --output-root "$OUTPUT_ROOT" \
  --cycles 3 \
  --interval 0 \
  --report-json "$REPORT_JSON"

echo "==> Wrote $REPORT_JSON"
