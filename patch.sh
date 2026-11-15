#!/bin/sh
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
exec "$PYTHON" "$DIR/patch.py" "$@"
