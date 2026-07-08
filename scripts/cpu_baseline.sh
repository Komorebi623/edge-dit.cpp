#!/usr/bin/env bash
# CPU baseline 测量脚本 —— edge-dit.cpp
#
# 目的:在固定口径下测出 CPU 后端的性能基准,作为后续所有优化的对照分母。
# 口径(一经固定,后续优化重测不得更改):
#   模型     : SD3-medium (diffusers 目录)
#   分辨率   : 512x512
#   步数     : 8
#   seed     : 42 (固定,保证每次采样负载一致)
#   --no-t5  : 跳过 T5 (省 ~9GB,不影响 DiT/VAE 计算特征)
#   cache    : 默认 DISABLED (每步完整计算,计时干净)
#
# 记录两个指标:
#   主指标: DiT 采样每步耗时 = "sampling completed, taking Xs" / steps
#   副指标: 端到端 wall (从进程启动到出图)
#
# 抗噪:跑 N 轮取中位数;跑前检查 load average;固定线程数。
set -u

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# ---- 可调参数(通过环境变量覆盖,但默认即标准口径)----
BIN="${BIN:-./build-cpu/bin/ed-cli}"
MODEL="${MODEL:-/mnt/cfs/9n-das-admin/llm_models/stable-diffusion-3-medium-diffusers}"
W="${W:-512}"
H="${H:-512}"
STEPS="${STEPS:-8}"
SEED="${SEED:-42}"
THREADS="${THREADS:-0}"          # 0 = 用 hardware_concurrency (192)
ROUNDS="${ROUNDS:-3}"            # 重复轮数,取中位数
PROMPT="${PROMPT:-a cat}"
OUTDIR="${OUTDIR:-.wty_cpu_baseline}"
TAG="${TAG:-baseline}"           # 本组实验的标签(如 baseline / llamafile)

mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/${TAG}_summary.txt"
: > "$SUMMARY"

echo "===== CPU baseline 测量 =====" | tee -a "$SUMMARY"
echo "标签      : $TAG"              | tee -a "$SUMMARY"
echo "二进制    : $BIN"              | tee -a "$SUMMARY"
echo "模型      : $MODEL"            | tee -a "$SUMMARY"
echo "配置      : ${W}x${H} steps=$STEPS seed=$SEED threads=$THREADS rounds=$ROUNDS" | tee -a "$SUMMARY"
echo "日期      : (由运行时 date 记录)" | tee -a "$SUMMARY"
date '+%F %T'                        | tee -a "$SUMMARY"

# ---- 环境体检:负载过高就警告 ----
echo "---- 环境 ----" | tee -a "$SUMMARY"
echo "CPU       : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)" | tee -a "$SUMMARY"
echo "逻辑核    : $(nproc)"          | tee -a "$SUMMARY"
LOAD1=$(cut -d' ' -f1 /proc/loadavg)
echo "load(1m)  : $LOAD1"            | tee -a "$SUMMARY"
awk -v l="$LOAD1" -v n="$(nproc)" 'BEGIN{ if (l > n*0.3) print "  [警告] 负载偏高 ("l"),数据可能受干扰,建议等空闲再测"; else print "  [OK] 负载低,适合测量" }' | tee -a "$SUMMARY"

if [ ! -x "$BIN" ]; then echo "[错误] 找不到可执行 $BIN"; exit 1; fi
if [ ! -d "$MODEL" ]; then echo "[错误] 找不到模型目录 $MODEL"; exit 1; fi

# ---- N 轮测量 ----
declare -a STEP_TIMES
declare -a WALL_TIMES
for r in $(seq 1 "$ROUNDS"); do
  echo "---- Round $r/$ROUNDS ----" | tee -a "$SUMMARY"
  LOG="$OUTDIR/${TAG}_run${r}.log"
  T0=$(date +%s.%N)
  "$BIN" --backend cpu --model "$MODEL" --no-t5 \
    -p "$PROMPT" -W "$W" -H "$H" --steps "$STEPS" -s "$SEED" -t "$THREADS" \
    -o "$OUTDIR/${TAG}_run${r}.png" > "$LOG" 2>&1
  RC=$?
  T1=$(date +%s.%N)
  if [ $RC -ne 0 ]; then echo "  [错误] run $r 退出码 $RC,见 $LOG"; tail -5 "$LOG"; continue; fi

  WALL=$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.2f", b-a}')
  # 从日志抓 DiT 采样总时间
  SAMP=$(grep -oE "sampling completed, taking [0-9.]+s" "$LOG" | grep -oE "[0-9.]+" | head -1)
  if [ -n "$SAMP" ]; then
    PERSTEP=$(awk -v s="$SAMP" -v n="$STEPS" 'BEGIN{printf "%.3f", s/n}')
    STEP_TIMES+=("$PERSTEP")
    echo "  DiT 采样   : ${SAMP}s  (每步 ${PERSTEP}s)" | tee -a "$SUMMARY"
  else
    echo "  [警告] 未抓到 sampling 计时" | tee -a "$SUMMARY"
  fi
  WALL_TIMES+=("$WALL")
  echo "  端到端 wall: ${WALL}s" | tee -a "$SUMMARY"
done

# ---- 中位数 ----
median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{ if(NR%2){print a[(NR+1)/2]} else {printf "%.3f", (a[NR/2]+a[NR/2+1])/2} }'; }
echo "===== 汇总 (中位数 of $ROUNDS) =====" | tee -a "$SUMMARY"
if [ ${#STEP_TIMES[@]} -gt 0 ]; then
  echo "DiT 每步 (主指标): $(median "${STEP_TIMES[@]}") s" | tee -a "$SUMMARY"
fi
if [ ${#WALL_TIMES[@]} -gt 0 ]; then
  echo "端到端 wall (副) : $(median "${WALL_TIMES[@]}") s" | tee -a "$SUMMARY"
fi
echo "详细日志在: $OUTDIR/${TAG}_run*.log" | tee -a "$SUMMARY"
