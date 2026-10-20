#!/usr/bin/env bash
# Local (non-Docker) install for the TIBlender runtime.
#
# Usage:
#   bash install.sh [--offline] [--with-browser] [--with-ocr] [--with-reference]
#                  [--with-dev] [--python PATH] [--venv PATH]
#
# --offline is the default and only mode: it creates a virtualenv, installs
# the pinned requirements.txt, then installs this package with --no-deps (no
# network access beyond pip/PyPI is required). The optional extras add the
# playwright browser, OCR, reference embedding/RL, and dev-tooling (pytest,
# coverage) dependencies on top of the same venv.
#
# The venv location defaults to ./.venv; override it with --venv PATH or the
# TIBLENDER_VENV_DIR env var (the flag takes precedence) for scratch installs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="python3"
WITH_BROWSER=false
WITH_OCR=false
WITH_REFERENCE=false
WITH_DEV=false
VENV_DIR="${TIBLENDER_VENV_DIR:-$SCRIPT_DIR/.venv}"

usage() {
  cat <<'EOF'
Usage: install.sh [--offline] [--with-browser] [--with-ocr] [--with-reference] [--with-dev] [--python PATH] [--venv PATH]

  --offline          Install for the offline/deterministic demo (default,
                     and currently the only mode; accepted as a no-op for
                     documentation and forward compatibility).
  --with-browser      Also install playwright + Chromium for JS-rendered web crawling.
  --with-ocr           Also install the pytesseract/Pillow OCR extra.
  --with-reference   Also install the sentence-transformers/torch reference embedding+RL extra.
  --with-dev           Also install the pytest/coverage dev extra (needed for `make test`/`make coverage`).
  --python PATH       Python interpreter to use for the venv (default: python3).
  --venv PATH         Directory for the virtualenv (default: ./.venv).

Environment:
  TIBLENDER_VENV_DIR   Directory for the virtualenv (default: ./.venv); overridden by --venv.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --offline)
      # Accepted as a documented no-op: offline is the only install mode
      # this script supports today, so there is nothing to toggle.
      shift
      ;;
    --with-browser)
      WITH_BROWSER=true
      shift
      ;;
    --with-ocr)
      WITH_OCR=true
      shift
      ;;
    --with-reference)
      WITH_REFERENCE=true
      shift
      ;;
    --with-dev)
      WITH_DEV=true
      shift
      ;;
    --python)
      if [ "$#" -lt 2 ]; then
        echo "error: --python requires a path argument" >&2
        exit 1
      fi
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      if [ "$#" -lt 2 ]; then
        echo "error: --venv requires a path argument" >&2
        exit 1
      fi
      VENV_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: python interpreter '$PYTHON_BIN' not found on PATH" >&2
  exit 1
fi

# Ubuntu 22.04 ships python3.10 and 24.04 ships python3.12; the runtime
# targets python3.11+ (pyproject.toml requires-python), so fail early with a
# clear message rather than a confusing pip/import error later.
PY_VERSION_OK=$("$PYTHON_BIN" -c '
import sys
print("1" if sys.version_info >= (3, 11) else "0")
')
if [ "$PY_VERSION_OK" != "1" ]; then
  PY_VERSION_STR=$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
  echo "error: $PYTHON_BIN is $PY_VERSION_STR; TIBlender requires Python 3.11 or newer." >&2
  echo "       Install python3.11+ (e.g. 'sudo apt install python3.11 python3.11-venv' on" >&2
  echo "       Ubuntu 22.04/24.04) and re-run with '--python python3.11'." >&2
  exit 1
fi

echo "==> Creating virtualenv at $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

echo "==> Upgrading pip"
"$VENV_PYTHON" -m pip install --upgrade pip --quiet

echo "==> Installing pinned requirements (requirements.txt)"
"$VENV_PIP" install --no-cache-dir -r requirements.txt

echo "==> Installing the tiblender package (--no-deps)"
"$VENV_PIP" install --no-cache-dir --no-deps .

if [ "$WITH_BROWSER" = "true" ]; then
  echo "==> Installing browser extra (playwright + chromium)"
  "$VENV_PIP" install --no-cache-dir ".[browser]"
  "$VENV_PYTHON" -m playwright install --with-deps chromium
fi

if [ "$WITH_OCR" = "true" ]; then
  echo "==> Installing OCR extra (pytesseract, Pillow)"
  "$VENV_PIP" install --no-cache-dir ".[ocr]"
fi

if [ "$WITH_REFERENCE" = "true" ]; then
  echo "==> Installing reference extra (sentence-transformers, torch)"
  "$VENV_PIP" install --no-cache-dir ".[reference]"
fi

if [ "$WITH_DEV" = "true" ]; then
  echo "==> Installing dev extra (pytest, coverage; needed for 'make test'/'make coverage')"
  "$VENV_PIP" install --no-cache-dir ".[dev]"
fi

echo "==> Install complete. Verifying with 'tiblender doctor':"
"$VENV_DIR/bin/tiblender" doctor
