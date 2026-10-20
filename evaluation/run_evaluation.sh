#!/usr/bin/env bash
# Recompute the paper's evaluation figures from the released records.
#
# Executes a COPY of the dataset's notebook under `output/evaluation/`
# and compares the `evaluation_figures.json` it produces with the copy
# committed in the dataset. Prints `evaluation: ok` and exits 0 when the
# two are byte-identical; prints the differing keys and exits 1 otherwise.
#
# The dataset is never written to: the notebook is copied out and its
# `../raw_dataset` input is resolved through a link created beside the copy.
#
# No credentials and no external services. Network access is needed only if
# the Jupyter dependencies still have to be installed.
#
# Runnable from any working directory.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: evaluation/run_evaluation.sh [--output-root DIR]

  --output-root DIR   Where the notebook copy and its output go
                      (default: output/evaluation).

Environment:
  DATASET_DIR         Dataset location. Defaults to `dataset/` in this
                      repository.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT="output/evaluation"
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

DATASET_DIR="$(bash "$ROOT/tools/dataset_dir.sh" "$ROOT")"
NOTEBOOK="$DATASET_DIR/notebook/evaluation_figures.ipynb"
EXPECTED="$DATASET_DIR/notebook/evaluation_figures.json"

if [ ! -f "$NOTEBOOK" ] || [ ! -f "$EXPECTED" ]; then
  echo "error: no dataset at $DATASET_DIR (looked for notebook/evaluation_figures.ipynb)." >&2
  echo "       Keep the released dataset at dataset/ in this repository, or set DATASET_DIR." >&2
  exit 2
fi

export PYTHONHASHSEED=0

PYTHON="python3"
if ! python3 -c "import nbclient, nbformat" >/dev/null 2>&1; then
  if [ ! -x "$ROOT/.venv/bin/python" ]; then
    echo "==> Jupyter is not installed; creating $ROOT/.venv with install.sh --offline --with-dev"
    bash "$ROOT/install.sh" --offline --with-dev
  fi
  PYTHON="$ROOT/.venv/bin/python"
  if ! "$PYTHON" -c "import nbclient, nbformat" >/dev/null 2>&1; then
    echo "==> Installing the dev extra (jupyter) into $ROOT/.venv"
    "$ROOT/.venv/bin/pip" install --no-cache-dir ".[dev]"
  fi
fi

# The notebook resolves its inputs as `../raw_dataset` relative to its own
# directory, so the copy needs a sibling of that name. A symlink keeps the
# ~100 MB of records in one place; a plain copy is the fallback where symlinks
# are unavailable.
WORK="$OUTPUT_ROOT"
rm -rf "$WORK"
mkdir -p "$WORK/notebook"
cp "$NOTEBOOK" "$WORK/notebook/evaluation_figures.ipynb"
if ! ln -s "$(cd -- "$DATASET_DIR/raw_dataset" && pwd)" "$WORK/raw_dataset" 2>/dev/null; then
  cp -r "$DATASET_DIR/raw_dataset" "$WORK/raw_dataset"
fi

echo "==> Executing $WORK/notebook/evaluation_figures.ipynb"
"$PYTHON" -m jupyter nbconvert \
  --to notebook \
  --execute \
  --output evaluation_figures_executed.ipynb \
  "$WORK/notebook/evaluation_figures.ipynb"

PRODUCED="$WORK/notebook/evaluation_figures.json"
if [ ! -f "$PRODUCED" ]; then
  echo "error: the notebook did not write $PRODUCED" >&2
  exit 1
fi

"$PYTHON" "$ROOT/tools/compare_evaluation_figures.py" --expected "$EXPECTED" --actual "$PRODUCED"
