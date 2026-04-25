#!/usr/bin/env bash
# Convenience launcher.
#   ./run.sh install   -> create venv + install deps
#   ./run.sh app       -> launch the Streamlit UI
#   ./run.sh test      -> quick sanity check on the backend (no LLM needed)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

case "${1:-app}" in
  install)
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
    echo "✓ Installed. Activate with: source .venv/bin/activate"
    ;;
  app)
    [ -d .venv ] && source .venv/bin/activate || true
    exec streamlit run frontend/app.py
    ;;
  test)
    [ -d .venv ] && source .venv/bin/activate || true
    exec python3 -m backend.smoketest
    ;;
  *)
    echo "Usage: $0 {install|app|test}" ; exit 1 ;;
esac
