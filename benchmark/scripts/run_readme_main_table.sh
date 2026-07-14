#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DEVICES="3"
OUTPUT_ROOT="benchmark/results/readme-main-table-$(date -u +%Y%m%d)"
SITE="benchmark/configs/local/site-h200-readme-build.yaml"
EXPECT_MODELS="3"

SUITE="benchmark/configs/suites/readme-main-table.yaml"
DRY_RUN_ONLY=0

usage() {
  cat <<'EOF'
Usage: bash benchmark/scripts/run_readme_main_table.sh [options]

Options:
  --devices IDS           GPU ids to expose to each single-GPU run. Default: 3
  --output-root DIR       Output root for result.json files.
  --site FILE             Local site config with model/binary paths.
  --expect-models N       Expected number of model workloads. Default: 3
  --dry-run               Preview expanded workloads without running models.
  -h, --help              Show this help.

The current frozen README table expands to 3 real local T2I model workloads.
Use --expect-models 4 after adding a fourth real 1024x1024, 50-step
text-to-image workload to benchmark/configs/suites/readme-main-table.yaml.
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
    --expect-models)
      EXPECT_MODELS="${2:?--expect-models requires a value}"
      shift
      ;;
    --expect-models=*)
      EXPECT_MODELS="${1#*=}"
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

SUMMARY="$OUTPUT_ROOT/summary.json"
TABLES="$OUTPUT_ROOT/tables.md"

DRY_RUN_JSON="$(mktemp)"
trap 'rm -f "$DRY_RUN_JSON"' EXIT

BENCHMARK_CUDA_VISIBLE_DEVICES="$DEVICES" \
python3 benchmark/orchestration/run_suite.py \
  --suite "$SUITE" \
  --site "$SITE" \
  --dry-run > "$DRY_RUN_JSON"

MODEL_COUNT="$(python3 - "$DRY_RUN_JSON" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1]))
print(len({run["workload"] for run in data.get("runs", [])}))
PY
)"

if [ "$MODEL_COUNT" -ne "$EXPECT_MODELS" ]; then
  echo "README main table expands to $MODEL_COUNT model workloads, expected $EXPECT_MODELS." >&2
  echo "Workloads:" >&2
  python3 - "$DRY_RUN_JSON" <<'PY' >&2
import json
import sys
data = json.load(open(sys.argv[1]))
for workload in sorted({run["workload"] for run in data.get("runs", [])}):
    print(f"  - {workload}")
PY
  exit 2
fi

echo "README main table model workload count: $MODEL_COUNT"
echo "Output root: $OUTPUT_ROOT"

if [ "$DRY_RUN_ONLY" -eq 1 ]; then
  echo "Dry run only. Expanded workloads:"
  python3 - "$DRY_RUN_JSON" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1]))
for workload in sorted({run["workload"] for run in data.get("runs", [])}):
    systems = sorted({run["system"] for run in data.get("runs", []) if run["workload"] == workload})
    print(f"  - {workload}: {', '.join(systems)}")
PY
  exit 0
fi

BENCHMARK_CUDA_VISIBLE_DEVICES="$DEVICES" \
python3 benchmark/orchestration/run_suite.py \
  --suite "$SUITE" \
  --site "$SITE" \
  --execute \
  --resume \
  --output-root "$OUTPUT_ROOT"

python3 benchmark/analysis/aggregate.py \
  --results-dir "$OUTPUT_ROOT" \
  --suite-id readme-main-table \
  --output "$SUMMARY"

python3 benchmark/analysis/generate_tables.py \
  "$SUMMARY" \
  --output "$TABLES"

cat <<EOF
README main table run complete.

Result root: $OUTPUT_ROOT
Summary:     $SUMMARY
Tables:      $TABLES
EOF
