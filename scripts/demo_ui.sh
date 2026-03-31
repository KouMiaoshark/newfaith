#!/usr/bin/env bash
set -euo pipefail

HOST=${1:-127.0.0.1}
PORT=${2:-6006}

PYTHONPATH=$(pwd) python -u demo_qa_ui.py --host "$HOST" --port "$PORT"
