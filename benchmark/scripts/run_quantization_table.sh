#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PREFER_GPUS="${BENCHMARK_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
OUTPUT_ROOT="benchmark/results/perf-quantization-$(date -u +%Y%m%d)"
SITE="benchmark/configs/local/site-h200-readme-build.yaml"
EDGE_BUILD_DIR=""
WARMUP_RUNS="1"
MEASURED_RUNS="5"
DRY_RUN_ONLY=0
WAIT_FOR_GPU=1
MIN_FREE_MIB=120000
MAX_UTILIZATION=10
WAIT_INTERVAL_SEC=60
WAIT_TIMEOUT_SEC=0
DISCOVER_CUDNN_RUNTIME=1
CUDNN_LOG_CHECK=1
CUDNN_PROFILE="${ED_PROFILE_CUDNN_SDPA:-1}"

SUITE="benchmark/configs/suites/quantization.yaml"

usage() {
  cat <<'EOF'
Usage: bash benchmark/scripts/run_quantization_table.sh [options]

Options:
  --devices IDS           GPU ids to consider for the single-GPU suite.
                          Default: BENCHMARK_CUDA_VISIBLE_DEVICES or 0,1,2,3,4,5,6,7
  --prefer-gpus IDS       Alias for --devices.
  --output-root DIR       Output root for quantization results.
  --site FILE             Local site config with model/binary paths.
  --edge-build-dir DIR    Override edge_dit_cli/sample paths with DIR/bin.
  --warmup-runs N         Warm-up runs per benchmark case. Default: 1
  --measured-runs N       Measured runs per benchmark case. Default: 5
  --min-free-mib N        Required free memory for the selected GPU. Default: 120000
  --max-utilization N     Maximum selected-GPU utilization percent. Default: 10
  --wait-interval-sec N   Poll interval while waiting for a GPU. Default: 60
  --wait-timeout-sec N    Wait timeout. Default: 0, meaning wait forever
  --no-wait               Run immediately on the first listed visible GPU.
  --dry-run               Preview and validate the expanded suite only.
  --no-cudnn-discovery    Do not prepend Python NVIDIA CUDA runtime libraries.
  --no-cudnn-log-check    Do not fail on cuDNN SDPA fallback log checks.
  -h, --help              Show this help.

This script reproduces the docs/performance.md quantization trade-off table.
The suite is FLUX.1-dev 1024x1024, 50 steps, batch 1, seed 0, load-once
generation, and five single-GPU BF16/Q8_0/Q6_K/Q4_K precision configurations.

The default GPU wait threshold is intentionally strict for local H200 table
captures, because external GPU allocations would pollute peak VRAM numbers.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --devices|--prefer-gpus)
      PREFER_GPUS="${2:?$1 requires a value}"
      shift
      ;;
    --devices=*|--prefer-gpus=*)
      PREFER_GPUS="${1#*=}"
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
    --edge-build-dir)
      EDGE_BUILD_DIR="${2:?--edge-build-dir requires a value}"
      shift
      ;;
    --edge-build-dir=*)
      EDGE_BUILD_DIR="${1#*=}"
      ;;
    --warmup-runs)
      WARMUP_RUNS="${2:?--warmup-runs requires a value}"
      shift
      ;;
    --warmup-runs=*)
      WARMUP_RUNS="${1#*=}"
      ;;
    --measured-runs)
      MEASURED_RUNS="${2:?--measured-runs requires a value}"
      shift
      ;;
    --measured-runs=*)
      MEASURED_RUNS="${1#*=}"
      ;;
    --min-free-mib)
      MIN_FREE_MIB="${2:?--min-free-mib requires a value}"
      shift
      ;;
    --min-free-mib=*)
      MIN_FREE_MIB="${1#*=}"
      ;;
    --max-utilization)
      MAX_UTILIZATION="${2:?--max-utilization requires a value}"
      shift
      ;;
    --max-utilization=*)
      MAX_UTILIZATION="${1#*=}"
      ;;
    --wait-interval-sec)
      WAIT_INTERVAL_SEC="${2:?--wait-interval-sec requires a value}"
      shift
      ;;
    --wait-interval-sec=*)
      WAIT_INTERVAL_SEC="${1#*=}"
      ;;
    --wait-timeout-sec)
      WAIT_TIMEOUT_SEC="${2:?--wait-timeout-sec requires a value}"
      shift
      ;;
    --wait-timeout-sec=*)
      WAIT_TIMEOUT_SEC="${1#*=}"
      ;;
    --no-wait)
      WAIT_FOR_GPU=0
      ;;
    --dry-run)
      DRY_RUN_ONLY=1
      ;;
    --no-cudnn-discovery)
      DISCOVER_CUDNN_RUNTIME=0
      ;;
    --no-cudnn-log-check)
      CUDNN_LOG_CHECK=0
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

make_effective_site() {
  if [ -z "$EDGE_BUILD_DIR" ]; then
    printf '%s\n' "$SITE"
    return
  fi

  mkdir -p benchmark/tmp
  local tmp_site="benchmark/tmp/quantization-site-$(date -u +%Y%m%dT%H%M%SZ)-$$.yaml"
  python3 - "$SITE" "$EDGE_BUILD_DIR" "$tmp_site" <<'PY'
from pathlib import Path
import sys

import yaml

site_path = Path(sys.argv[1]).resolve()
build_dir = Path(sys.argv[2]).resolve()
out_path = Path(sys.argv[3]).resolve()

site = yaml.safe_load(site_path.read_text(encoding="utf-8")) or {}
paths = site.setdefault("paths", {})
paths["edge_dit_cli"] = str(build_dir / "bin" / "ed-cli")
paths["edge_dit_sample"] = str(build_dir / "bin" / "ed-sample")
paths["edge_dit_cli_cudnn"] = str(build_dir / "bin" / "ed-cli")
paths["edge_dit_sample_cudnn"] = str(build_dir / "bin" / "ed-sample")

notes = site.setdefault("notes", [])
notes.append(f"Generated by run_quantization_table.sh from {site_path}.")
notes.append(f"edge-dit.cpp binaries overridden with {build_dir}/bin.")
out_path.write_text(yaml.safe_dump(site, sort_keys=False), encoding="utf-8")
print(out_path)
PY
}

resolve_site_python() {
  python3 - "$EFFECTIVE_SITE" <<'PY'
from pathlib import Path
import os
import sys

import yaml

override = os.environ.get("PYTHON_BIN")
if override:
    print(override)
    raise SystemExit

site = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
print((site.get("paths") or {}).get("diffusers_python") or "python3")
PY
}

discover_nvidia_runtime_paths() {
  "$1" - <<'PY'
from pathlib import Path
import importlib.util
import os

modules = [
    "nvidia.cublas",
    "nvidia.cuda_nvrtc",
    "nvidia.cuda_runtime",
    "nvidia.cudnn",
]
paths = []

def add(path: Path) -> None:
    text = str(path)
    if path.is_dir() and text not in paths:
        paths.append(text)

for module in modules:
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.submodule_search_locations:
        continue
    root = Path(list(spec.submodule_search_locations)[0])
    add(root / "lib")

cudnn_root = os.environ.get("CUDNN_ROOT")
if cudnn_root:
    add(Path(cudnn_root) / "lib")
    add(Path(cudnn_root) / "lib64")

cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if cuda_home:
    add(Path(cuda_home) / "lib64")
    add(Path(cuda_home) / "lib")

print(":".join(paths))
PY
}

dry_run_suite() {
  local dry_run_json="$1"
  BENCHMARK_CUDA_VISIBLE_DEVICES="$PREFER_GPUS" \
    python3 benchmark/orchestration/run_suite.py \
      --suite "$SUITE" \
      --site "$EFFECTIVE_SITE" \
      --dry-run \
      --systems edge-dit.cpp \
      --warmup-runs "$WARMUP_RUNS" \
      --measured-runs "$MEASURED_RUNS" > "$dry_run_json"
}

validate_quantization_dry_run() {
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

def row(run):
    options = run.get("run_options") or {}
    return (
        run.get("workload"),
        run.get("gpu_count"),
        run.get("parallel_mode"),
        options.get("weight_type_label"),
        options.get("precision"),
        options.get("tensor_type_rules"),
    )

got = [row(run) for run in data.get("runs", [])]
expected = [
    ("flux1-dev-t2i-1024-s50", 1, None, "BF16", "bf16", None),
    ("flux1-dev-t2i-1024-s50", 1, None, "Q8_0", "q8_0", None),
    ("flux1-dev-t2i-1024-s50", 1, None, "Q6_K", "q6_k", None),
    ("flux1-dev-t2i-1024-s50", 1, None, "Q4_K", "q4_k", None),
    ("flux1-dev-t2i-1024-s50", 1, None, "Q4_K + precision rules", "q4_k", "norm=f16,bias=f32"),
]
if got != expected:
    print("quantization expansion mismatch", file=sys.stderr)
    print(f"expected: {expected}", file=sys.stderr)
    print(f"got:      {got}", file=sys.stderr)
    raise SystemExit(2)
PY
}

summarize_dry_run() {
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

runs = data.get("runs", [])
print(f"quantization: {len(runs)} runs")
for run in runs:
    options = run.get("run_options") or {}
    policy = options.get("tensor_type_rules") or "-"
    print(
        "  - "
        f"{options.get('weight_type_label')}:"
        f"{options.get('precision')}:"
        f"policy={policy}"
    )
PY
}

select_device() {
  if [ "$WAIT_FOR_GPU" -eq 0 ]; then
    printf '%s\n' "$PREFER_GPUS"
    return
  fi

  python3 benchmark/orchestration/wait_for_gpus.py \
    --count 1 \
    --min-free-mib "$MIN_FREE_MIB" \
    --max-utilization "$MAX_UTILIZATION" \
    --interval-sec "$WAIT_INTERVAL_SEC" \
    --timeout-sec "$WAIT_TIMEOUT_SEC" \
    --prefer "$PREFER_GPUS"
}

run_suite() {
  BENCHMARK_CUDA_VISIBLE_DEVICES="$RUN_DEVICES" \
  LD_LIBRARY_PATH="$RUNTIME_LD_LIBRARY_PATH" \
  ED_PROFILE_CUDNN_SDPA="$CUDNN_PROFILE" \
    python3 benchmark/orchestration/run_suite.py \
      --suite "$SUITE" \
      --site "$EFFECTIVE_SITE" \
      --execute \
      --resume \
      --systems edge-dit.cpp \
      --warmup-runs "$WARMUP_RUNS" \
      --measured-runs "$MEASURED_RUNS" \
      --output-root "$OUTPUT_ROOT"
}

verify_cudnn_logs() {
  if [ "$CUDNN_LOG_CHECK" -eq 0 ]; then
    return
  fi

  local fallback_pattern="Could not load library libnvrtc|ED_CUDNN_SDPA build failed|No execution plans support the graph"
  if find "$OUTPUT_ROOT" -name stderr.log -print0 | xargs -0 -r rg -n "$fallback_pattern"; then
    echo "cuDNN SDPA fallback or runtime-load failure detected in benchmark stderr logs." >&2
    exit 1
  fi

  if [ "$CUDNN_PROFILE" != "0" ]; then
    if ! find "$OUTPUT_ROOT" -name stderr.log -print0 | xargs -0 -r rg -q "ED_CUDNN_SDPA (success|async build .*result=success)"; then
      echo "ED_PROFILE_CUDNN_SDPA is enabled, but no successful ED_CUDNN_SDPA log entry was found." >&2
      exit 1
    fi
  fi
}

EFFECTIVE_SITE="$(make_effective_site)"
DRY_RUN_JSON="$(mktemp)"
trap 'rm -f "$DRY_RUN_JSON"' EXIT

dry_run_suite "$DRY_RUN_JSON"
validate_quantization_dry_run "$DRY_RUN_JSON"
summarize_dry_run "$DRY_RUN_JSON"

if [ "$DRY_RUN_ONLY" -eq 1 ]; then
  cat <<EOF
Dry run only.

Effective site: $EFFECTIVE_SITE
Output root:    $OUTPUT_ROOT
GPU candidates: $PREFER_GPUS
EOF
  exit 0
fi

RUNTIME_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [ "$DISCOVER_CUDNN_RUNTIME" -eq 1 ]; then
  PYTHON_FOR_RUNTIME="$(resolve_site_python)"
  NVIDIA_RUNTIME_PATHS="$(discover_nvidia_runtime_paths "$PYTHON_FOR_RUNTIME")"
  if [ -z "$NVIDIA_RUNTIME_PATHS" ]; then
    echo "could not discover Python NVIDIA CUDA runtime libraries with $PYTHON_FOR_RUNTIME" >&2
    echo "use --no-cudnn-discovery only if the required cuDNN runtime libs are already on LD_LIBRARY_PATH" >&2
    exit 2
  fi
  RUNTIME_LD_LIBRARY_PATH="${NVIDIA_RUNTIME_PATHS}${RUNTIME_LD_LIBRARY_PATH:+:${RUNTIME_LD_LIBRARY_PATH}}"
  echo "Using Python NVIDIA CUDA runtime libraries from: $NVIDIA_RUNTIME_PATHS"
fi

RUN_DEVICES="$(select_device)"

echo "Effective site: $EFFECTIVE_SITE"
echo "Output root:    $OUTPUT_ROOT"
echo "Devices:        $RUN_DEVICES"

run_suite
verify_cudnn_logs

SUMMARY="$OUTPUT_ROOT/summary.json"
TABLES="$OUTPUT_ROOT/tables.md"
python3 benchmark/analysis/aggregate.py \
  --results-dir "$OUTPUT_ROOT" \
  --include-workload flux1-dev-t2i-1024-s50 \
  --suite-id quantization \
  --output "$SUMMARY"

python3 benchmark/analysis/generate_tables.py \
  "$SUMMARY" \
  --output "$TABLES"

cat <<EOF
Quantization table run complete.

Result root: $OUTPUT_ROOT
Runs:        $OUTPUT_ROOT/quantization
Summary:     $SUMMARY
Tables:      $TABLES
EOF
