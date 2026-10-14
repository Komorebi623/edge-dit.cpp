# H3/CUDA 四方法复现（RTX 4090）

## 固定契约
- 五份 `rtx4090_four_*.json` 是复现计划，不是新增测试结果；本次不运行 GPU、重编译或下载。
- 单卡；T2VA：864×480、124 帧、24 fps、20 步、seed=42、CFG=1、res_multistep/simple；音频 32000 Hz 双声道。
- 保持原 precision lock 的完整混合 Q4（不是全张量 Q4）：FL2VA、Qwen、video VAE F16、audio VAE F32，共四模型；T2VA 不要求 Ref2VA。
- DiT/text encoder/VAE 全卸载，`--max-vram 22`（GiB 图规划预算，非显存硬上限）、CPU 6 线程；两组固定 `ED_CUDNN_SDPA_SYNC_F32_SELF_ATTN=1`。
- 每组独立 CLI 预热 1 次、正式 3 次；baseline 八个新总开关全 0，无 tuning；candidate 仅对应总开关为 1，不叠加融合。

| 文件后缀 | candidate 总开关 | 固定变体/参数 |
|---|---|---|
| baseline | 全 0 | 默认仅 baseline；build-06 全关 candidate 只作显式控制 |
| adaln | `ED_H3_ADALN_CACHE` | layer-major AdaLN cache；需核对同步计划下的数值 |
| cache | `ED_H3_CACHE_DIT` | `ED_H3_CACHE_THRESHOLD=0.12`：已测近似设置，不是推荐画质阈值或保证 |
| sage | `ED_H3_SAGE_ATTN` | `ED_H3_SAGE_PV_FP8=0`；INT8 QK、Q32 per-warp/K64、F16 PV、FP32 累积、smooth-K；不是 SM89 auto FP8 |
| subblock | `ED_H3_SPARSE_ATTN` | **native-indexed，非 gather**；keep=0.5、`ED_H3_SUBBLOCK_INDEXED=1` 显式固定当前默认 |

## 路径、安全及来源
- 路径均以 `{repo_root}` 为参数；默认取脚本仓库，可用 `--repo-root/--model-root/--asset-root` 覆写；`--profile host/container` 均可解析。
- baseline 沿用已测原构建 `{repo_root}/_scratch/rtx4090-20261013/build-baseline/baseline/bin/ed-cli`，源码固定 `97cc585d9dbd9ac5e30b78ef2e2c4a0d8bfbe9f2`。
- candidate 为 `{repo_root}/_scratch/h3-offload-20261014/four-methods-build-06/candidate/bin/ed-cli`；对应 `four-methods-source-06`（同一父目录）。`source_snapshot` 仅来源说明，不代替构建指纹。
- `--binary` 覆写当前阶段二进制；workflow 默认覆写 candidate，`--phase baseline-only` 时覆写 baseline；更换构建需重新检查来源/依赖，不能冒用旧数字。
- 全部零 UUID 是占位，不是设备预订；全部默认 `phase=baseline-only`，**不等于自动 dry-run**。以下始终显式 `--dry-run`；只有取得独占调度、制作本地 manifest 副本并填入真实 UUID/index 后才可去掉它。
- 空闲门禁不是独占锁；忙卡/遥测异常必须拒绝，不抢占、不停止他人进程。输出目录必须新建；不覆盖报告。

## 一键 CPU → 计划 → 正式比较
以下命令供后续操作者使用；CPU 一键回归会构建 CPU 测试，本次不执行。
```bash
ROOT="$(pwd)" # 在仓库根目录执行
B="$ROOT/benchmark/h3_cuda"; W="$ROOT/scripts/h3_cuda_workflow.py"
MODELS="${H3_MODEL_ROOT:-$ROOT/models/minimax-h3}"; OUT="$(mktemp -d)"
H3_BUILD_JOBS=6 bash "$ROOT/scripts/test_h3_cpu.sh"
python3 -B "$W" baseline --manifest "$B/rtx4090_four_baseline.json" --repo-root "$ROOT" --model-root "$MODELS" --dry-run --output-dir "$OUT/baseline-plan"
for method in adaln cache sage subblock; do
  python3 -B "$W" workflow --manifest "$B/rtx4090_four_$method.json" --repo-root "$ROOT" --model-root "$MODELS" --phase all --dry-run --output-dir "$OUT/$method-plan"
done
```
- 正式执行：`B` 改指向获准的本地运行副本，使用新 `OUT`；移除 `--dry-run`。workflow `--phase all` 自动新建同条件 baseline → candidate → compare，勿同时启动多份。
- 若共享一次已测 baseline：先用 `baseline` 命令生成严格锁库的真实 `report.json`，再按下列阶段执行（候选示例仍为计划；compare 只能使用已实际完成的候选报告）。
```bash
# REAL_BASELINE、REAL_CANDIDATE 必须指向已验证的真实 report.json，不能用上述 dry-run 报告。
python3 -B "$W" candidate --manifest "$B/rtx4090_four_cache.json" --repo-root "$ROOT" --model-root "$MODELS" --baseline "$REAL_BASELINE" --dry-run --output-dir "$OUT/cache-candidate-plan"
python3 -B "$W" compare --baseline "$REAL_BASELINE" --candidate "$REAL_CANDIDATE" --output-dir "$OUT/cache-compare"
```

## 证据与比较边界
- `lock_runtime_libraries=true`：实际运行记录 CLI、构建记录、解析出的共享库/本地 GGML/声明 cuDNN 路径和整文件 SHA256，以及 loader 环境；结束复核未变。拒绝 preload/audit；baseline 必须同样有严格库锁。
- 两组允许各自方法库不同，但必须审查差异和实际解析路径；旧未锁库 baseline 不能复用。tuning parser 门禁不是 kernel 命中证明：SubBlock 核对 indexed 路由，Sage 要有 `[H3SA] CUDA_DISPATCH`、`pv_fp8=0`、`tail_zero_fill=1` 及成功完成，不接受旧 gather/FP8 冒名。
- cold CLI 指每次新进程的 wall time，含启动、模型加载、进程内校验及封装，不含外部预校验/媒体质量验证；不清 OS 页缓存，**不是磁盘冷缓存或 load-once 稳态吞吐**。另报 DiT、预填耗时、3 次均值/离散度；采样显存峰值不是硬峰值。
- 正式运行重核四模型完整 SHA256/存储类型；compare 检查视频/音频规格、SSIM/PSNR、音频误差。有限值/数值相近不证明语义、动作、参考身份或音画同步；Cache/SubBlock/Sage 近似结果须单列质量审查，不能仅按速度采纳。
- 禁止跨 GPU、精度、模式、输入顺序/hash、prompt、采样、卸载/预算/CPU 线程、同步计划、loader 环境或 warmup/repeats 复用 baseline；变更构建/库需新指纹与重新审查，报告不可改写。

## mixed 单独建立 manifest
- 从本次模板另存本地 mixed 配置，参考 `{repo_root}/_scratch/h3-offload-20261014/mixed-sync-baseline-manifest.json` 的 `assets/modes`：使用本机 generated 素材 `{repo_root}/_scratch/rtx4090-20261013/reference-assets/{first.png,reference.mp4,paired.wav}`；保留 `<Picture 1>/<Video 1>/<Audio 1>` 及配对/额外音频顺序。
- 添加 precision lock 中完整版 Ref2VA 的 bytes/types/header/full SHA256，切为 `ref_mixed/dit=ref2va`；移除不使用的 FL2VA 或两组一致保留。重新锁定素材内容/hash及来源，统一保持本页目标几何、1+3、严格库锁并建立独立 mixed baseline；不要照搬旧 mixed 的 0+1。
- 本机生成素材是复现 fixture，不是外部优质案例或质量证明；较小 `modal-smoke-*` 仅查模态接线，**不能当作 864×480/124f/20steps 的 target performance**。本次不生成素材、不运行 mixed。
