#!/usr/bin/env bash
# 带 load 门控的三方对照批量跑测。每组实验前等 load 落回阈值，彻底消除累积污染。
# 严格串行。每组记录 load。结果汇总到 bench_results/matrix/summary.tsv
set -u
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT_DIR"

LOAD_GATE="${LOAD_GATE:-10}"       # load 落回此值以下才开跑
GATE_TIMEOUT="${GATE_TIMEOUT:-300}" # 最多等这么久(秒)
SEED=42; W=512; H=512
OUTDIR=bench_results/matrix
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.tsv"
LD_ONEDNN=/tmp/onednn-tp-install/lib64

declare -A MODELS=(
  [sd3]=/mnt/cfs/9n-das-admin/llm_models/stable-diffusion-3-medium-diffusers
  [flux]=/mnt/cfs/9n-das-admin/llm_models/flux-dev
  [qwen-image]=/mnt/cfs/9n-das-admin/llm_models/qwen-image
  [qwen-image-edit]=/mnt/cfs/9n-das-admin/llm_models/Qwen-Image-Edit-2509
)

# SD3 需要 --type bf16 才能吃 AMX (原生 f16)
declare -A TYPEARG=( [sd3]="--type bf16" [flux]="" [qwen-image]="" [qwen-image-edit]="" )

wait_load() {
  local start=$(date +%s)
  while :; do
    local l=$(cut -d' ' -f1 /proc/loadavg)
    local li=${l%.*}
    if [ "${li:-99}" -lt "$LOAD_GATE" ]; then echo "  [gate] load=$l < $LOAD_GATE, 开跑"; return 0; fi
    local now=$(date +%s)
    if [ $((now-start)) -gt "$GATE_TIMEOUT" ]; then echo "  [gate] 等待超时($GATE_TIMEOUT s), load=$l 仍偏高, 强行开跑"; return 0; fi
    sleep 8
  done
}

run_edge() {  # $1=model_key $2=bin_kind(onednn|baseline) $3=steps
  local mk=$1 bk=$2 st=$3
  local bin ld=""
  if [ "$bk" = onednn ]; then bin=./build-cpu-onednn/bin/ed-cli; ld=$LD_ONEDNN; else bin=./build-cpu/bin/ed-cli; fi
  local tag="${mk}_edge-${bk}_${st}step"
  local log="$OUTDIR/${tag}.log" out="$OUTDIR/${tag}.png"
  # qwen-image-edit 需要输入图
  local imgarg=""
  if [ "$mk" = "qwen-image-edit" ]; then imgarg="-i bench_results/edit_input.png"; fi
  echo ">>> $tag"; wait_load
  local lb=$(cut -d' ' -f1 /proc/loadavg)
  LD_LIBRARY_PATH=$ld:${LD_LIBRARY_PATH:-} $bin --backend cpu --model "${MODELS[$mk]}" ${TYPEARG[$mk]} $imgarg \
    -p "a cat" -W $W -H $H --steps $st -s $SEED -o "$out" > "$log" 2>&1
  local rc=$?
  local bd=$(grep "generate breakdown" "$log" | tail -1 | grep -oE "total=[0-9.]+s.*")
  local ms=$(python3 -c "from PIL import Image;import numpy as np;a=np.asarray(Image.open('$out').convert('RGB')).astype(float);print(f'{a.mean():.2f}/{a.std():.2f}')" 2>/dev/null || echo "NA")
  echo "    load_before=$lb rc=$rc | $bd | mean/std=$ms"
  printf "%s\t%s\t%s\t%s\t%s\tload=%s\t%s\n" "$mk" "edge-$bk" "$st" "$bd" "mean/std=$ms" "$lb" "rc=$rc" >> "$SUMMARY"
}

run_diff() {  # $1=model_key $2=steps
  local mk=$1 st=$2
  local tag="${mk}_diffusers_${st}step"
  local log="$OUTDIR/${tag}.log"
  echo ">>> $tag"; wait_load
  local lb=$(cut -d' ' -f1 /proc/loadavg)
  MODEL_KEY=$mk STEPS=$st W=$W H=$H SEED=$SEED OUTDIR=$OUTDIR python3 scripts/diffusers_stage_timing_multi.py > "$log" 2>&1
  local rc=$?
  local e2e=$(grep "END-TO-END" "$log" | grep -oE "[0-9.]+s" | head -1)
  local txt=$(grep "text_encode" "$log" | grep -oE "[0-9.]+s" | head -1)
  local dit=$(grep "sampling(DiT)" "$log" | grep -oE "[0-9.]+s" | head -1)
  local vae=$(grep "vae_decode" "$log" | grep -oE "[0-9.]+s" | head -1)
  local ms=$(grep "out mean" "$log" | grep -oE "mean=[0-9.]+ std=[0-9.]+")
  echo "    load_before=$lb rc=$rc | e2e=$e2e text=$txt dit=$dit vae=$vae | $ms"
  printf "%s\tdiffusers\t%s\ttotal=%s text=%s dit=%s vae=%s\t%s\tload=%s\t%s\n" "$mk" "$st" "$e2e" "$txt" "$dit" "$vae" "$ms" "$lb" "rc=$rc" >> "$SUMMARY"
}

echo "=== 三方对照矩阵 开始 $(date +%T) | load_gate=$LOAD_GATE ===" | tee "$SUMMARY"
# 顺序: 先 4 步(快)全模型三方, 再 20 步. 每模型内部 onednn->baseline->diffusers
for ST in 4 20; do
  for MK in flux qwen-image sd3 qwen-image-edit; do
    run_edge $MK onednn $ST
    run_edge $MK baseline $ST
    run_diff $MK $ST
  done
done
echo "=== 矩阵完成 $(date +%T) ==="
echo "汇总: $SUMMARY"
