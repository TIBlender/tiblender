#!/usr/bin/env bash
# Resolve the TIBlender dataset directory and print it.
#
# Usage: tools/dataset_dir.sh [package-root]
#
# The dataset sits in the repository as `dataset/`, so that is the
# default. `DATASET_DIR` overrides it and is echoed unchanged, so anyone who
# keeps the dataset elsewhere can point at it.
#
# Always prints a path, even when nothing is there: the caller decides whether
# a missing dataset is fatal (`evaluation/run_evaluation.sh` says so with
# its own message; `demo/run_demo.sh` never needs it at all).
set -euo pipefail

ROOT="${1:-.}"

if [ -n "${DATASET_DIR:-}" ]; then
  printf '%s\n' "$DATASET_DIR"
  exit 0
fi

printf '%s\n' "$ROOT/dataset"
