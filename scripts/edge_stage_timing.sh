#!/usr/bin/env bash
# edge-dit 单次跑测 + 抓 breakdown。口径与 diffusers_stage_timing_multi.py 严格对齐。
# 用法: MODEL_KEY=flux BIN=onednn STEPS=4 W=512 H=512 ./scripts/edge_stage_timing.sh
set -u
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT_DIR"

MODEL_KEY="${MODEL_KEY:-flux}"
BIN_KIND="${BIN:-onednn}"          # onednn | baseline
STEPS="${STEPS:-4}"; W="${W:-512}"; H="${H:-512}"; SEED="${SEED:-42}"
PROMPT="${PROMPT:-a cat}"
OUTDIR="${OUTDIR:-bench_results/edge}"
TYPE_ARG="${TYPE_ARG:-}"           # 例如 "--type bf16" (SD3 需要)
EXTRA="${EXTRA:-}"                 # 额外参数

declare -A MODELS=(
  [sd3]=/mnt/cfs/9n-das-admin/llm_models/stable-diffusion-3-medium-diffusers
  [flux]=/mnt/cfs/9n-das-admin/llm_models/flux-dev
  [qwen-image]=/mnt/cfs/9n-das-admin/llm_models/qwen-image
  [qwen-image-edit]=/mnt/cfs/9n-das-admin/llm_models/Qwen-Image-Edit-2509
)
MODEL="${MODELS[$MODEL_KEY]}"

if [ "$BIN_KIND" = "onednn" ]; then
  BINPATH=./build-cpu-onednn/bin/ed-cli
  export LD_LIBRARY_PATH=/tmp/onednn-tp-install/lib64:${LD_LIBRARY_PATH:-}
else
  BINPATH=./build-cpu/bin/ed-cli
fi

mkdir -p "$OUTDIR"
TAG="${MODEL_KEY}_${BIN_KIND}_${W}x${H}_${STEPS}step"
LOG="$OUTDIR/${TAG}.log"
OUT="$OUTDIR/${TAG}.png"

echo "=== $TAG | bin=$BINPATH | type='$TYPE_ARG' ==="
echo "load before: $(cut -d' ' -f1 /proc/loadavg)"
T0=$(date +%s.%N)
$BINPATH --backend cpu --model "$MODEL" $TYPE_ARG $EXTRA \
  -p "$PROMPT" -W "$W" -H "$H" --steps "$STEPS" -s "$SEED" \
  -o "$OUT" > "$LOG" 2>&1
RC=$?
T1=$(date +%s.%N)
WALL=$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.2f", b-a}')
echo "load after : $(cut -d' ' -f1 /proc/loadavg)  exit=$RC  进程墙钟(含加载)=${WALL}s"
grep "generate breakdown" "$LOG" | tail -1
python3 -c "
from PIL import Image; import numpy as np
try:
    a=np.asarray(Image.open('$OUT').convert('RGB')).astype(float)
    print(f'  out mean={a.mean():.2f} std={a.std():.2f}')
except Exception as e: print('  [出图失败]', e)
"
