#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

###############################################################################
# 直接改这里
###############################################################################

# 如果你的 build-cuda/bin/ld-cli 已经支持 --cache，就用这个。
# 如果你手动重链了 ld-cli-cache，就改成 ./build-cuda/bin/ld-cli-cache。
CLI="./build-cuda/bin/ld-cli"

BACKEND="cuda"
MODEL="/mnt/cfs/9n-das-admin/llm_models/flux-dev/"
PROMPT="a cinematic photo of a glass teapot on a wooden table, soft morning light"

WIDTH=1024
HEIGHT=1024
STEPS=50
SEED=0
GUIDANCE=3.5

OUT_DIR="cache_test_flux"
PREFIX="flux"

# 不需要时保持为空字符串。
THREADS=""
CFG_SCALE=""
FLOW_SHIFT=""

# 只跑想看的 case。可选：
#   baseline easycache ucache dbcache taylorseer cache-dit
CASES=(
  baseline
  easycache
  ucache
  dbcache
  taylorseer
  cache-dit
)

# 所有非 baseline cache case 都会带上的通用参数。
CACHE_COMMON_ARGS=(
  # --cache-start 0.15
  # --cache-end 0.95
)

# 每种 cache 单独的参数。直接取消注释或改数值即可。
EASYCACHE_ARGS=(
  --cache-threshold 0.20
  --cache-start 0.15
  --cache-end 0.95
)

UCACHE_ARGS=(
  --cache-threshold 1.00
  --cache-error-decay 1.00
  --cache-relative-threshold
  --cache-no-reset-error
  --cache-start 0.15
  --cache-end 0.95
)

DBCACHE_ARGS=(
  --cache-warmup-steps 5
  --cache-residual-threshold 0.08
  --cache-max-accumulated-residual-diff -1
  --cache-fn-blocks 8
  --cache-bn-blocks 0
  --cache-max-cached-steps -1
  --cache-max-continuous-cached-steps -1
  --cache-scm-mask 1,0,0,1
  --cache-static-scm
)

TAYLORSEER_ARGS=(
  --cache-warmup-steps 5
  --cache-taylor-order 1
  --cache-taylor-skip 3
  --cache-start 0.15
  --cache-end 0.95
)

CACHE_DIT_ARGS=(
  --cache-warmup-steps 5
  --cache-residual-threshold 0.08
  --cache-max-accumulated-residual-diff -1
  --cache-fn-blocks 8
  --cache-bn-blocks 0
  --cache-taylor-order 1
  --cache-taylor-skip 1
)

# 追加到每个 ld-cli 调用的额外参数。不需要就留空。
EXTRA_CLI_ARGS=(
  # --some-extra-flag
)

# true: 只打印命令，不跑模型。
# false: 正常跑 benchmark。
DRY_RUN=false

###############################################################################
# 下面一般不用改
###############################################################################

cmd=(
  python3 scripts/benchmark_flux_cache.py
  --cli "${CLI}"
  --backend "${BACKEND}"
  --model "${MODEL}"
  --prompt "${PROMPT}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --steps "${STEPS}"
  --seed "${SEED}"
  --guidance "${GUIDANCE}"
  --out-dir "${OUT_DIR}"
  --prefix "${PREFIX}"
)

if [[ -n "${THREADS}" ]]; then
  cmd+=(--threads "${THREADS}")
fi
if [[ -n "${CFG_SCALE}" ]]; then
  cmd+=(--cfg-scale "${CFG_SCALE}")
fi
if [[ -n "${FLOW_SHIFT}" ]]; then
  cmd+=(--flow-shift "${FLOW_SHIFT}")
fi
if [[ "${DRY_RUN}" == "true" ]]; then
  cmd+=(--dry-run)
fi

join_args() {
  local -n arr_ref="$1"
  local joined=""
  local arg
  for arg in "${arr_ref[@]}"; do
    if [[ -z "${joined}" ]]; then
      joined="${arg}"
    else
      joined+=" ${arg}"
    fi
  done
  printf '%s' "${joined}"
}

cache_common="$(join_args CACHE_COMMON_ARGS)"
easycache_args="$(join_args EASYCACHE_ARGS)"
ucache_args="$(join_args UCACHE_ARGS)"
dbcache_args="$(join_args DBCACHE_ARGS)"
taylorseer_args="$(join_args TAYLORSEER_ARGS)"
cache_dit_args="$(join_args CACHE_DIT_ARGS)"

case_spec() {
  local case_name="$1"
  case "${case_name}" in
    baseline)
      printf '%s' "baseline"
      ;;
    easycache)
      printf '%s' "easycache ${cache_common} ${easycache_args}"
      ;;
    ucache)
      printf '%s' "ucache ${cache_common} ${ucache_args}"
      ;;
    dbcache)
      printf '%s' "dbcache ${cache_common} ${dbcache_args}"
      ;;
    taylorseer)
      printf '%s' "taylorseer ${cache_common} ${taylorseer_args}"
      ;;
    cache-dit|cachedit|cache_dit)
      printf '%s' "cache-dit ${cache_common} ${cache_dit_args}"
      ;;
    *)
      echo "unknown case: ${case_name}" >&2
      return 1
      ;;
  esac
}

for case_name in "${CASES[@]}"; do
  cmd+=(--case "$(case_spec "${case_name}")")
done

for arg in "${EXTRA_CLI_ARGS[@]}"; do
  cmd+=(--extra-cli-arg "${arg}")
done

echo "Running benchmark command:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
