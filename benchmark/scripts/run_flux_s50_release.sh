#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TARGET="${1:-single}"
RESULTS_ROOT="${RESULTS_ROOT:-benchmark/results/release-v1-s50-$(date -u +%Y%m%dT%H%M%SZ)}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-30}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-60}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-0}"
PREFER_GPUS="${PREFER_GPUS:-${BENCHMARK_CUDA_VISIBLE_DEVICES:-}}"
FORCE_EXTERNAL_UPDATE="${FORCE_EXTERNAL_UPDATE:-1}"

wait_for_gpus() {
  local count="$1"
  python3 benchmark/orchestration/wait_for_gpus.py \
    --count "${count}" \
    --min-free-mib "${MIN_FREE_MIB}" \
    --max-utilization "${MAX_UTILIZATION}" \
    --interval-sec "${WAIT_INTERVAL_SEC}" \
    --timeout-sec "${WAIT_TIMEOUT_SEC}" \
    ${PREFER_GPUS:+--prefer "${PREFER_GPUS}"}
}

aggregate_report() {
  python3 benchmark/analysis/aggregate.py \
    --results-dir "${RESULTS_ROOT}" \
    --include-workload flux1-dev-t2i-1024-s50 \
    --suite-id "$(basename "${RESULTS_ROOT}")" \
    --output benchmark/reports/v0.1.0-alpha/summary.json

  python3 benchmark/analysis/generate_tables.py \
    benchmark/reports/v0.1.0-alpha/summary.json \
    --output benchmark/reports/v0.1.0-alpha/tables.md

  python3 benchmark/analysis/generate_report.py \
    benchmark/reports/v0.1.0-alpha/summary.json \
    --tables benchmark/reports/v0.1.0-alpha/tables.md \
    --output benchmark/reports/v0.1.0-alpha/report.md
}

run_single_gpu() {
  local devices
  devices="$(wait_for_gpus 1)"
  echo "[flux-s50-release] single-GPU devices: ${devices}" >&2
  local status=0
  BENCHMARK_CUDA_VISIBLE_DEVICES="${devices}" \
  python3 benchmark/orchestration/run_suite.py \
    --suite benchmark/configs/suites/flux-e2e.yaml \
    --site benchmark/configs/local/site-h200.yaml \
    --execute \
    --systems edge-dit.cpp diffusers \
    --output-root "${RESULTS_ROOT}" || status=$?
  aggregate_report
  return "${status}"
}

run_parallel() {
  local devices
  devices="$(wait_for_gpus 4)"
  echo "[flux-s50-release] parallel devices: ${devices}" >&2
  local status=0
  local force_args=()
  if [[ "${FORCE_EXTERNAL_UPDATE}" == "1" ]]; then
    force_args+=(--force-external-update)
  fi
  BENCHMARK_CUDA_VISIBLE_DEVICES="${devices}" \
  python3 benchmark/orchestration/run_suite.py \
    --suite benchmark/configs/suites/flux-parallel-e2e.yaml \
    --site benchmark/configs/local/site-h200.yaml \
    --execute \
    --systems edge-dit.cpp xdit \
    "${force_args[@]}" \
    --output-root "${RESULTS_ROOT}" || status=$?
  aggregate_report
  return "${status}"
}

case "${TARGET}" in
  single)
    run_single_gpu
    ;;
  parallel)
    run_parallel
    ;;
  all)
    single_status=0
    parallel_status=0
    run_single_gpu || single_status=$?
    run_parallel || parallel_status=$?
    if [[ "${single_status}" != "0" ]]; then
      exit "${single_status}"
    fi
    exit "${parallel_status}"
    ;;
  *)
    echo "usage: $0 [single|parallel|all]" >&2
    exit 2
    ;;
esac
