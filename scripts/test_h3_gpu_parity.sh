#!/usr/bin/env bash
# Opt-in only. No build/download, model load, process stop, or GPU reservation.
# nvidia-smi queries below are read-only and run ONLY after --run --device.
# Main agent must coordinate with its baseline: this guard is a snapshot,
# not an exclusive reservation against other users starting work afterwards.
set -euo pipefail

usage() {
    cat <<'HELP'
Tiny real-CUDA vs CPU-reference parity; default is NOT RUN (exit 2).

  bash scripts/test_h3_gpu_parity.sh --run --device INDEX_OR_FULL_GPU_UUID \
      --binary /absolute/path/to/h3-cuda-gpu-parity-test

Requires: one explicit physical GPU, nvidia-smi, idle readings in two snapshots,
zero compute/graphics processes, <=512 MiB used and 0% utilization. Unknown telemetry,
MIG identifiers, comma lists and occupied devices fail closed. UUID binding
avoids CUDA-vs-nvidia-smi index ordering differences. No waiting for a busy GPU.
Coordinate with the main agent first; do NOT overlap its full-model baseline.
No --force / unchecked preflight bypass. The executable also checks opt-in,
a matching single UUID and a fresh (<=30s) guard timestamp before CUDA init.

Compile ONLY (does not run/discover GPUs), using this checkout's existing build:
  ROOT=/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation
  BUILD=/absolute/path/to/matching/cuda/build
  c++ -std=c++17 -O2 -DGGML_MAX_NAME=128 \
      -I"$ROOT/third_party/ggml/include" -I"$ROOT/src/core" \
      "$ROOT/tests/h3_cuda_gpu_parity_test.cpp" \
      -L"$BUILD/bin" -L"$BUILD/lib" \
      -Wl,-rpath,"$BUILD/bin" -Wl,-rpath,"$BUILD/lib" \
      -Wl,-rpath-link,"$BUILD/bin" \
      -lggml-cuda -lggml-cpu -lggml-base -pthread \
      -o "$BUILD/bin/h3-cuda-gpu-parity-test"

Use matching ggml headers/libraries, GGML_MAX_NAME=128, GGML_BACKEND_DL=OFF,
ED_GGML_CUDA=ON and ED_ENABLE_CUDA_MODULATION=ON. This command assumes shared
libraries in bin/lib; static/custom layouts should use the existing CMake
library targets instead. It does not link or run the full model. Do not use
old build-cuda libraries from another ggml revision. Preserve existing binaries
by choosing a fresh output path. No CTest registration is required.

Exit: 0 = entire real-CUDA matrix passed; 1 = parity/operation failure;
2 = not opted in/invalid invocation; 3 = missing backend/device/prerequisite;
75 = occupied GPU or unreliable telemetry. Loader/crash errors remain nonzero.
F16 modulation/gating is rejection-only (currently F32-only); F16 SwiGLU is
numerical CUDA coverage with half-rounded inputs and an unfused F32 CPU graph.
HELP
}
fail() { printf '%s\n' "NOT RUN: $2" >&2; exit "$1"; }
run=0
device=''
binary=''
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --run) ((run == 0)) || fail 2 'duplicate --run'; run=1; shift ;;
        --device)
            (($# >= 2)) && [[ -z "$device" ]] || fail 2 'missing/duplicate --device'
            device="$2"; shift 2 ;;
        --binary)
            (($# >= 2)) && [[ -z "$binary" ]] || fail 2 'missing/duplicate --binary'
            binary="$2"; shift 2 ;;
        *) fail 2 "unknown argument: $1" ;;
    esac
done
((run == 1)) || fail 2 'default off; pass --run, --device and --binary explicitly (--help for compilation)'
uuid_pattern='GPU-[[:xdigit:]]{8}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{12}'
[[ "$device" =~ ^[0-9]+$ || "$device" =~ ^${uuid_pattern}$ ]] ||
    fail 2 '--device must select exactly one index or full physical GPU UUID'
[[ "$binary" == /* && -f "$binary" && -x "$binary" ]] || fail 3 '--binary must be an absolute executable file'
command -v nvidia-smi >/dev/null 2>&1 || fail 3 'nvidia-smi absent; GPU occupancy cannot be verified'

# Never inherit a wider/multiple-device mask. Resolve the explicit physical
# selector to a UUID before starting a CUDA process. nvidia-smi ignores CVD.
export CUDA_VISIBLE_DEVICES="$device"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
uuid=''
for snapshot in 1 2; do
    if ! telemetry=$(nvidia-smi --id="$device" \
        --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits); then
        fail 75 'GPU absent or read-only telemetry query failed'
    fi
    telemetry="${telemetry//$'\r'/}"
    [[ -n "$telemetry" && "$telemetry" != *$'\n'* ]] || fail 75 'expected exactly one GPU telemetry row'
    IFS=, read -r seen used utilization extra <<< "$telemetry"
    seen="${seen//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    [[ "$seen" =~ ^${uuid_pattern}$ && -z "$extra" &&
       "$used" =~ ^[0-9]{1,9}$ && "$utilization" =~ ^[0-9]{1,3}$ ]] ||
        fail 75 'unknown/invalid GPU identity, memory or utilization telemetry'
    [[ -z "$uuid" || "$uuid" == "$seen" ]] || fail 75 'GPU identity changed between snapshots'
    [[ ! "$device" =~ ^${uuid_pattern}$ || "$device" == "$seen" ]] || fail 75 'selected UUID mismatch'
    uuid="$seen"
    ((10#$used <= 512 && 10#$utilization == 0)) ||
        fail 75 "GPU occupied: $uuid memory=${used}MiB utilization=${utilization}%"
    if ! processes=$(nvidia-smi --id="$uuid" --query-compute-apps=pid --format=csv,noheader,nounits); then
        fail 75 'cannot inspect GPU compute-process occupancy'
    fi
    [[ -z "${processes//[[:space:]]/}" ]] || fail 75 "GPU compute processes present or query inconclusive: $uuid"
    # Tiny graphics contexts also count as occupied; low memory is not consent.
    if ! all_processes=$(nvidia-smi -q -x -i "$uuid"); then
        fail 75 'cannot inspect GPU graphics-process occupancy'
    fi
    if ! printf '%s' "$all_processes" | python3 -c '
import sys, xml.etree.ElementTree as ET
try:
    devices = ET.fromstring(sys.stdin.read()).findall("gpu")
    assert len(devices) == 1 and devices[0].findtext("uuid") == sys.argv[1]
    processes = devices[0].find("processes")
    assert processes is not None and not list(processes) and not (processes.text or "").strip()
except (ET.ParseError, AssertionError):
    sys.exit(1)
' "$uuid"; then
        fail 75 'GPU compute/graphics processes present or all-process telemetry inconclusive'
    fi
    printf 'PREFLIGHT snapshot=%s device=%s memory=%sMiB utilization=%s%% compute_processes=0\n' \
        "$snapshot" "$uuid" "$used" "$utilization"
    if ((snapshot == 1)); then sleep 1; fi
done
export CUDA_VISIBLE_DEVICES="$uuid"
export H3_GPU_PARITY_ALLOW=1
export H3_GPU_PARITY_PREFLIGHT_DEVICE="$uuid"
export H3_GPU_PARITY_PREFLIGHT_EPOCH="$(date +%s)"
export OMP_NUM_THREADS=1
printf 'RUN tiny CUDA parity only; no model/baseline; CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
exec "$binary" --run
