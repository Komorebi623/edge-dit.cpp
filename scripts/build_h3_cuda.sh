#!/usr/bin/env bash
# Offline, non-overwriting H3 CUDA build driver. See --help for explicit pins.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Preview/validation must not leave Python caches in the checkout.
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$ROOT/scripts/h3_cuda_build_config.py" "$@"
