#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env PYTHONPATH="$REPO_ROOT" \
    uv run --project "$REPO_ROOT/experiments/kdd_env" \
    python -m experiments.kdd_bench "$@"
