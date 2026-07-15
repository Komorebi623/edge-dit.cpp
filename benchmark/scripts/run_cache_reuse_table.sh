#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DEVICES="3"
OUTPUT_ROOT="benchmark/results/cache-reuse-table-$(date -u +%Y%m%d)"
SITE="benchmark/configs/local/site-h200-readme-build.yaml"
EVAL_DEVICE="cuda:0"
DRY_RUN_ONLY=0

usage() {
  cat <<'EOF'
Usage: bash benchmark/scripts/run_cache_reuse_table.sh [options]

Options:
  --devices IDS           GPU ids to expose to each single-GPU run. Default: 3
  --output-root DIR       Output root for result.json files.
  --site FILE             Local site config with model/binary paths.
  --eval-device DEVICE    Device used for PSNR/LPIPS evaluation. Default: cuda:0
  --dry-run               Preview expanded suites without running models.
  -h, --help              Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --devices)
      DEVICES="${2:?--devices requires a value}"
      shift
      ;;
    --devices=*)
      DEVICES="${1#*=}"
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?--output-root requires a value}"
      shift
      ;;
    --output-root=*)
      OUTPUT_ROOT="${1#*=}"
      ;;
    --site)
      SITE="${2:?--site requires a value}"
      shift
      ;;
    --site=*)
      SITE="${1#*=}"
      ;;
    --eval-device)
      EVAL_DEVICE="${2:?--eval-device requires a value}"
      shift
      ;;
    --eval-device=*)
      EVAL_DEVICE="${1#*=}"
      ;;
    --dry-run)
      DRY_RUN_ONLY=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

resolve_eval_python() {
  python3 - "$SITE" <<'PY'
from pathlib import Path
import sys

import yaml

site = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
print((site.get("paths") or {}).get("diffusers_python") or "python3")
PY
}

run_suite() {
  local suite="$1"
  local output_root="$2"
  BENCHMARK_CUDA_VISIBLE_DEVICES="$DEVICES" \
    python3 benchmark/orchestration/run_suite.py \
      --suite "$suite" \
      --site "$SITE" \
      --execute \
      --resume \
      --systems edge-dit.cpp \
      --output-root "$output_root"
}

dry_run_suite() {
  local suite="$1"
  local dry_run_json
  dry_run_json="$(mktemp)"
  BENCHMARK_CUDA_VISIBLE_DEVICES="$DEVICES" \
    python3 benchmark/orchestration/run_suite.py \
      --suite "$suite" \
      --site "$SITE" \
      --dry-run \
      --systems edge-dit.cpp > "$dry_run_json"
  python3 - "$suite" "$dry_run_json" <<'PY'
import json
import sys

suite = sys.argv[1]
with open(sys.argv[2], "r", encoding="utf-8") as f:
    data = json.load(f)
runs = data.get("runs", [])
workloads = sorted({run["workload"] for run in runs})
print(f"{suite}: {len(runs)} runs, workloads={','.join(workloads)}")
PY
  rm -f "$dry_run_json"
}

link_result_dir() {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  ln -sf "$(realpath "$src/result.json")" "$dst/result.json"
  ln -sf "$(realpath "$src/metrics.json")" "$dst/metrics.json"
  ln -sf "$(realpath "$src/runner_metrics.json")" "$dst/runner_metrics.json"
}

make_matched_eval_root() {
  local join_root="$1"
  local reference_root="$2"
  local target_root="$3"
  rm -rf "$join_root"
  mkdir -p "$join_root"
  local d
  for d in "$reference_root"/*full_compute_p*_s* "$target_root"/*; do
    [ -f "$d/result.json" ] || continue
    link_result_dir "$d" "$join_root/$(basename "$d")"
  done
}

BASE_ROOT="$OUTPUT_ROOT/base"
MAGCACHE_ROOT="$OUTPUT_ROOT/magcache-dicache-retuned"
SENCACHE_CAL_ROOT="$OUTPUT_ROOT/sencache-calibration-s50"
SENCACHE_ROOT="$OUTPUT_ROOT/sencache-retuned"

if [ "$DRY_RUN_ONLY" -eq 1 ]; then
  dry_run_suite benchmark/configs/suites/cache-quality.yaml
  dry_run_suite benchmark/configs/suites/cache-quality-magcache-retuned.yaml
  dry_run_suite benchmark/configs/suites/cache-calibration-s50.yaml
  dry_run_suite benchmark/configs/suites/cache-quality-sencache-retuned.yaml
  cat <<EOF
Dry run only. The script will also score PSNR/LPIPS for the base cache matrix,
retuned MagCache/DiCache rows, and tuned SenCache row after execution.
EOF
  exit 0
fi

EVAL_PYTHON="$(resolve_eval_python)"
if [ ! -x "$EVAL_PYTHON" ] && ! command -v "$EVAL_PYTHON" >/dev/null 2>&1; then
  echo "quality evaluator Python is not executable: $EVAL_PYTHON" >&2
  exit 2
fi

run_suite benchmark/configs/suites/cache-quality.yaml "$BASE_ROOT"
run_suite benchmark/configs/suites/cache-quality-magcache-retuned.yaml "$MAGCACHE_ROOT"
run_suite benchmark/configs/suites/cache-calibration-s50.yaml "$SENCACHE_CAL_ROOT"
run_suite benchmark/configs/suites/cache-quality-sencache-retuned.yaml "$SENCACHE_ROOT"

BASE_RESULTS="$BASE_ROOT/cache-quality"
REFERENCE_RUNS="$BASE_RESULTS/edge-dit.cpp/flux1-dev-t2i-1024-s50"

"$EVAL_PYTHON" benchmark/analysis/evaluate_cache_quality.py \
  --results-root "$BASE_RESULTS" \
  --reference-method full_compute \
  --metrics psnr,lpips \
  --device "$EVAL_DEVICE" \
  --eval-root "$OUTPUT_ROOT/eval-base-cache-quality" \
  --skip-if-json-exists

MAGCACHE_RESULTS="$MAGCACHE_ROOT/cache-quality-magcache-retuned/edge-dit.cpp/flux1-dev-t2i-1024-s50"
MAGCACHE_EVAL_INPUT="$OUTPUT_ROOT/eval-input-magcache-dicache"
make_matched_eval_root "$MAGCACHE_EVAL_INPUT" "$REFERENCE_RUNS" "$MAGCACHE_RESULTS"
"$EVAL_PYTHON" benchmark/analysis/evaluate_cache_quality.py \
  --results-root "$MAGCACHE_EVAL_INPUT" \
  --reference-method full_compute \
  --metrics psnr,lpips \
  --device "$EVAL_DEVICE" \
  --eval-root "$OUTPUT_ROOT/eval-magcache-dicache" \
  --skip-if-json-exists

SENCACHE_RESULTS="$SENCACHE_ROOT/cache-quality-sencache-retuned/edge-dit.cpp/flux1-dev-t2i-1024-s50"
SENCACHE_EVAL_INPUT="$OUTPUT_ROOT/eval-input-sencache-t060"
make_matched_eval_root "$SENCACHE_EVAL_INPUT" "$REFERENCE_RUNS" "$SENCACHE_RESULTS"
"$EVAL_PYTHON" benchmark/analysis/evaluate_cache_quality.py \
  --results-root "$SENCACHE_EVAL_INPUT" \
  --reference-method full_compute \
  --metrics psnr,lpips \
  --device "$EVAL_DEVICE" \
  --eval-root "$OUTPUT_ROOT/eval-sencache-t060" \
  --skip-if-json-exists

cat <<EOF
Computation reuse table run complete.

Result roots:
  base cache-quality:        $BASE_RESULTS
  MagCache/DiCache retuned:  $MAGCACHE_ROOT/cache-quality-magcache-retuned
  SenCache t=0.60 retuned:   $SENCACHE_ROOT/cache-quality-sencache-retuned

Quality summaries:
  base cache-quality:        $OUTPUT_ROOT/eval-base-cache-quality
  MagCache/DiCache retuned:  $OUTPUT_ROOT/eval-magcache-dicache
  SenCache t=0.60 retuned:   $OUTPUT_ROOT/eval-sencache-t060
EOF
