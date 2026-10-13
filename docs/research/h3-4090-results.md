# H3 / RTX 4090：实测对照与接续点

记录覆盖 2026-10-13 晚至 10-14 凌晨（Asia/Shanghai）。接续 `71cc80d`，代码检查点 `d81041f`；**八种方法没有全部完成，未测项不填估算收益。** 机器可读数据及证据hash：`benchmark/h3_cuda/rtx4090-results.json`。

## 1. 环境、模型、控制变量

- 独立 worktree：原仓库内 `_scratch/h3-cuda-continuation`，分支 `codex/minimax-h3-cuda-optimizations`；只推个人 origin。保留原 main、用户报告、upstream及其进程/文件。
- 服务器8×RTX4090，**所有推理串行、只用GPU0**，UUID `GPU-95eb7a54-3471-4542-5a4d-894df955a273`；driver580.95.05。每次launch检查全部计算/图形进程，无可用卡就拒绝，不停止其他进程。
- CUDA12.2.140 / GCC12.4.0 / cuDNN9.24.0 / SM89；固定ggml `f270346d`、oneDNN `5e922403`、frontend `07b368a2`。baseline/candidate全新构建，共享配置hash一致；只有候选编译包含Sage能力，运行时关闭。候选全关控制组用于隔离此差异。
- CPU2×Xeon Gold6330 / RAM499GiB，构建与CLI CPU线程均限制6；性能复测期间没有并行编译。系统盘初始66GiB可用、下载后约46GiB；没有删除文件或系统安装。
- **五模型整文件SHA256全部匹配原锁**：完整FL2VA、完整Ref2VA、Qwen混合Q4，FP16 video VAE、FP32 audio VAE。仅补下载缺失的18,779,848,448-byte Ref2VA并整文件读回校验；不替用本机pruned、不再量化。
- 任务统一：**864×480 /124帧 /24fps /20步 /seed42 /CFG1 /res_multistep+simple**；`--type preserve`，DiT/TE/VAE全部offload，`--max-vram 22`，CPU6线程。22GiB是规划预算，不是硬峰值cap。原生log确认真实CUDA参数传输和79个DiT图段。
- 原baseline源 `97cc585`，保留已有norm/RoPE/modulation/SwiGLU优化。所有新增开关默认0；每次只开一个研究方法。不跨GPU、精度、模式、素材或placement计算加速比。

## 2. 主对照：T2VA（预热1次＋测量3次）

每次独立CLI进程；**不是load-once稳态**。CLI时间包含加载/生成/编码，不含外层SHA256和媒体复验。±为样本标准差。

| 方案 | CLI E2E，秒 | DiT，秒 | 单卡显存采样峰值 | 对照质量 | 结论 |
|---|---:|---:|---:|---|---|
| 原始baseline，新开关全0 | **254.54 ± 0.17** | **211.554 ± 0.060** | **16,035MiB** | 基准 | 1.000× |
| 同条件，仅 `ED_H3_FUSED_OPS=1` | **299.62 ± 0.52** | **256.380 ± 0.059** | **16,351MiB** | 3组SSIM=1；124/124帧相同；音频RMSE=0；MP4全hash相同 | **0.850×，慢17.71%；不采用** |

- 全部测量通过124帧/864×480/24fps、32kHz立体声和decoded finite检查；每声道165,888 PCM样本。结构/数值通过不等于音频语义、音画同步或人物一致性验收。
- 显存为只读**整卡采样峰值**，非严格硬峰值/进程专属值。baseline3269次、fusion3694次；实际最大间隔0.509s/0.543s，遥测错误0。周期配置0.25s，加查询开销。
- 同候选binary全关control（独立1次）：CLI255.36s、DiT211.57s，与原baseline相近，输出与fusion逐字节一致。说明当前开关确有速度回退；不把旧优化重复计功。
- 第一轮冷启动容量值保留：baseline256.37s、fusion299.45s；最初baseline视频与后来结果SSIM0.99351/PSNR48.06dB。后续**原baseline自身**及全部候选复测都收敛到同一MP4 hash，故初次微差不能归因于fusion或候选源码，原因仍待独立冷启动分析。
- T2VA使用原创固定山涧prompt（全文在JSON契约）；抽帧可见连续水流场景。没有把联网未验证的素材宣称为权威画质样例。

## 3. 混合模态：独立 Ref2VA baseline

图片＋带内嵌音频视频＋额外音频；同几何/步数/精度、研究开关全0。仅完成一次容量验证，**不与T2VA求加速比**。

| 次数 | CLI E2E | conditioning | DiT | decode | 显存采样峰值 | 输出 |
|---|---:|---:|---:|---:|---:|---|
| 预热0 / 测1 | **627.54s** | 23.85s | **547.71s** | 10.37s | **18,751MiB** | 124帧、32k立体声、finite有效 |

素材从本轮锁定T2VA输出提取，父视频hash、转换、图片/视频/PCM完整hash及输入顺序均记录。**这是自生成的受控参考，不是独立真人素材质量证明。** 显存1992次采样，最大间隔0.569s、错误0。首/末/首尾帧、各独立reference模式的完整生产矩阵尚未完成。

## 4. 八种方法现状

| 独立总开关 | 本轮真实进展 / 剩余限制 | 生产4090收益 |
|---|---|---|
| `ED_H3_FUSED_OPS` | primitives CUDA52/52；完整T2VA和重复对照通过，但速度回退 | **慢17.71%，保持关闭** |
| `ED_H3_ADALN_CACHE` | 新增逐投影预填、仅可达CPU参数传输、独立host缓存、失败原子回滚；真实CUDA tiny104对照/784图段通过 | **生产卸载仍拒绝，未测** |
| `ED_H3_CACHE_DIT` | 沿用既有实现；跨卸载段front/RF/delta生命周期及命中后跳过传输未接通 | 未测，卸载保护保留 |
| `ED_H3_SAGE_ATTN` | SM89编译通过；single-segment H3SA生命周期/完整模型数值待验证 | 未测，卸载保护保留 |
| `ED_H3_SPARSE_ATTN` | SubBlock主任务有界执行未完成，原展开容量保护保留 | 未测 |
| `ED_H3_TURBO` | 需真实匹配adapter和独立模型/步数baseline，不用原Q4少跑步替代 | 未测 |
| `ED_H3_VDN` | exact solve/short-conv/anchor/完整checkpoint及CUDA部署未完成 | 生产拒绝 |
| `ED_H3_VSA` | trained gate/native tile+compression部署未完成 | 生产拒绝 |

**AdaLN CUDA测试的准确范围：** 合成F32真实tiny H3，TimeEmbedder/curve、T2VA/mixed+keyframe、两CFG角色、3sigma、repeat/reset、预填读回后失败和consumer切段异常恢复；104对照、784图段、1768次图审计，禁止CPU fallback。它使用测试专用串行遍历并调用真实planner/bind/segment/execute，**未验证生产dispatch、异步预取、显存预算、完整Q4或124帧模型**。不能仅删除guard放行；没有释放投影权重或宣称显存节省。

## 5. 回归、证据和复现

- CPU CTest **12/12**，workflow/media **53/53（无跳过）**，来源18/18、LoRA5/5、build29/29、preflight56/56、cleanup29/29、4090配置14/14、显存7/7、GPU guard5/5、source稳定性2/2。
- CUDA baseline/candidate编译均通过；candidate production hash `cc6f87a2…` 与提交`d81041f`一致。构建前后检查源码漂移，避免并行编辑导致二进制证据失真。
- 新增4090 T2VA/mixed manifest、baseline-only执行路径、显式CPU线程/卸载预算契约、UUID与图形进程门禁、只读显存采样、opt-in CUDA测试。所有输出拒绝覆盖。
- 原始证据在研究worktree `_scratch/rtx4090-20261013/`：`model-verification.json`、build/logs、`t2va-baseline-repeat/`、`t2va-fusion-repeat/`、`t2va-repeat-comparison/`、`ref-mixed-capacity/`。临时工具、模型、媒体不提交；紧凑结果和原始报告hash进入Git。
- 有限串行队列`controlled-series-status.json`已 **complete**；没有持续后台任务、没有多卡推理、没有停止他人进程。

```bash
# 同机从worktree；GPU运行仍须通过空闲门禁，使用全新输出目录。
source _scratch/rtx4090-20261013/environment.sh
bash scripts/test_h3_cpu.sh
python3 scripts/h3_cuda_workflow.py workflow \
  --manifest _scratch/rtx4090-20261013/t2va-manifest.json \
  --phase all --warmup 1 --repeats 3 --dry-run \
  --output-dir _scratch/new-t2va-plan
# 只编译AdaLN CUDA helper，不执行GPU：
bash scripts/build_h3_adaln_offload_test.sh \
  _scratch/rtx4090-20261013/build-candidate/candidate /tmp/h3-adaln-test-new
```

## 6. 下一步（不重做调研）

1. **融合拆因诊断**：总开关强制RoPE tail非连续view，可能绕过原concat连续/vec4；CUSTOM与原MUL+ADD融合重合、可能影响buffer复用。段数和权重搬运相同；分别控制tail、block CUSTOM、FinalLayer并用已有graph-cut profiler定位，诊断数据不混入正式性能。
2. **AdaLN生产卸载接线**：在已通过tiny CUDA基础上，仅限定单卡、不可变CPU权重、串行/关闭graph-reuse的路径；真实预算/空参数段fallback/失败恢复须验收，先完整baseline对照后放行。Cache-DiT/Sage按交接顺序继续。
3. 扩展完整模态矩阵及独立优质参考、音频听评/音画同步；VDN/VSA/Turbo权重与部署另立baseline。
