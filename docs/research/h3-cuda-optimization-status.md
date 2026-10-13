# MiniMax-H3 CUDA 优化：环境、基线与逐项验证

> **4090接续更新（2026-10-13）**：已完成固定SM89构建、5/5整文件hash、真实CUDA tiny52/52，以及T2VA/Ref2VA mixed卸载baseline。单次fusion为回退而非加速；详细实测/范围见 `h3-4090-results.md`。下文H200“未推理”是历史记录，不代表本机状态。

## 环境与基线

- 开发分支：`codex/minimax-h3-cuda-optimizations`；原始源码：`97cc585`。
- 服务器：8×H200。**本轮未运行 GPU 推理，未停止任何 GPU／他人进程。**
- 构建复用已有开发容器：GCC 13.3、CUDA 12.8、SM90、cuDNN；不改旧构建目录。
- 主实验：完整 FL2VA/Ref2VA 的现有 Q4 GGUF、Qwen 现有 Q4 GGUF、FP16 video VAE、FP32 audio VAE。GGUF 实际为混合张量类型，详见 `h3-precision-lock.json`；五个组件的**整文件 SHA256 已锁定**。
- 固定 Ref2VA mixed：512×512、124帧、24fps、20步、seed42、CFG1、RES multistep/simple。另覆盖首帧、末帧、首尾、图、视频、视频配对音频、混合；固定输入顺序与 hash。
- 原始性能：**待 GPU 空闲后测量，暂无本项目加速数据**。CPU synthetic/媒体测试不能作真实性能或画质基线。

## 每项总开关（默认全部 0）

| 开关 | 思路及代码状态 | 已做验证／限制 | 性能提升 |
|---|---|---|---|
| `ED_H3_FUSED_OPS` | H3 现有调制/残差门控、SwiGLU、QKV/RoPE views 总控；不设置全局后端开关 | CUDA-only 入口，旧 flags 保留；GPU舍入与质量待测 | 未测 |
| `ED_H3_CACHE_DIT` | 真实 Fn probe，对每个模态 span 的 RF 作 RelL1 guard；命中复用 middle delta，执行真实 suffix | CPU actual H3: full/probe/miss/hit block数4/1/3/1；CFG、identity、窗口/连续次数/reset；拒绝 offload/并行/graph-reuse/可变adapter | 未测 |
| `ED_H3_ADALN_CACHE` | 请求内全 sigma 预填真实 TimeEmbedder/curve+AdaLN 输出；各 CFG 分支独立，hit 图不执行投影GEMM | actual tiny H3、curve/no-SiLU、完整 schedule、角色、hash、预算、stale/shape/finite；host表不释放权重，不宣称11GiB显存收益 | 未测 |
| `ED_H3_SPARSE_ATTN` | per-head Q64/K64、16-token subcell LSE router+TopK/gather；保护 text/audio/reference KV 与 query | 实际 ggml CPU图、尾padding、同图K变化；只F32、batch1，首尾步/层保dense；不是上游融合Triton速度复现 | 未测 |
| `ED_H3_SAGE_ATTN` | 独立H3SA tag复用既有 INT8-QK/F16-PV CUDA kernel；旧SAGQ global gate不变 | CPU scaling/layout/tag差分；首末层/token-refiner dense；不与SubBlock同开；CUDA候选已编译链接，真实GPU质量待测 | 未测 |
| `ED_H3_TURBO` | CPU-only LoRA offline merger+全选中DiT shard hash验证+公开Euler/simple配方 | 小矩阵merge、A/B pairing、resumeidentity、finite/FP16溢出；必须独立匹配权重与4/8步配方，不用基线Q4“少跑步”冒充 | 独立实验未测 |
| `ED_H3_VDN` | CPU exact `vdn_solve` oracle，以及native ggml SanaScaled/frame stats/双向窗外scan/localsoftmax/schema | **生产入口未完成、请求明确拒绝**。发布权重需要精确solve、shortconv、anchor与gate/adapter加载；不能用Sana替换发布vdn_solve | 未测 |
| `ED_H3_VSA` | 已实现CPU 3D tile/untile、多region TopK、fine+rawlinear-gate×compression、权重guard | **非生产CUDA实现**；base无gate/非FastH3不能启用。不删除token/FFN，gate不是sigmoid | 未测 |

缓存参数：`ED_H3_CACHE_FRONT_BLOCKS`/`BACK_BLOCKS`、`THRESHOLD`、`WARMUP_STEPS`、`MAX_CONSECUTIVE`、`START_PERCENT`/`END_PERCENT`。
AdaLN：`ED_H3_ADALN_MAX_MIB`，默认每 CFG 分支 host/staging 上限4096MiB，非GPU显存上限。
SubBlock：`ED_H3_SPARSE_KEEP_RATIO`，默认0.5；块大小固定64，首两步和末步保守。

## 一键开发与待GPU验证

在已有CUDA开发环境（或同一容器）执行：

```bash
bash scripts/test_h3_cpu.sh                 # 纯CPU构建+CTest+媒体/merge/source tests
bash scripts/build_h3_cuda.sh baseline      # 固定源码，旧构建保留
bash scripts/build_h3_cuda.sh candidate     # 候选编译，所有方法仍默认关闭
python3 scripts/h3_cuda_workflow.py workflow --manifest benchmark/h3_cuda/cache_dit.json --dry-run --output-dir /tmp/h3-plan-cache
```

- manifest 示例和逐方法计划在 `benchmark/h3_cuda/`；生成素材：`fixtures`；整文件身份：`lock`；基线→候选→比较：`workflow`；已有结果：`compare`。
- `--dry-run` 不查询GPU、不生成视频、不写性能值；真实执行有显存/利用率/GPU UUID安全门禁，忙则失败保存日志，**不会杀进程或抢占**。
- 一次只开一个优化。CPU fixture 是工程边界测试，画质应另用合法公开素材、人像/动作/说话/音乐/音画同步与参考身份盲评。
- SSIM/PSNR和音频RMSE只检查数值，不能替代prompt/动作/音画同步正确性。
- 不同硬件、量化、pruned、LoRA、solver或placement必须另建baseline。无权重/无支持后端 fail closed，不报假加速。

## 来源与日志

- 固定9仓库167份源码：`_scratch/h3-reference-src/`；离线复用和限制：`scripts/fetch_h3_references.py`。
- 来源表：`minimax-h3-optimization-sources.md`；下载日志：`h3-source-fetch-log.md`。
- 测试明细：`tests/h3-*-progress.md`；构建日志：`_scratch/h3-logs/`；CPU媒体日志：`_scratch/h3-local-logs/`。
- 本轮所有GPU质量/性能值保持未测。公开模型权重元数据网络核验仍有失败；不会自动下载数十GiB未知权重占满磁盘。

## 本轮软件验证汇总

| 检查 | 结果 |
|---|---|
| 固定原始CUDA版／候选CUDA版（含Sage） | 编译和链接通过，未运行GPU推理 |
| 纯CPU全工程＋CTest | 12/12通过 |
| 输入锁／媒体／工作流测试 | 48/48通过 |
| 固定源码下载安全测试 | 18/18通过 |
| CPU微型LoRA合并／溢出拒绝 | 5/5通过 |
| actual tiny H3完整图AdaLN对照 | 24/24逐位一致；投影GEMM 5→0 |
| 完整H3生产权重各模态质量／性能 | 未执行 |

以上是软件正确性与构建检查，不提供真实性能提升或“GPU已验证”结论。

多卡 TP/Ulysses 是独立通信/分片工程；当前 H3 研究入口拒绝并行 graph cuts，未将其他模型的多卡路径宣称可直接用于 H3。RunningHub 私有蒸馏权重未公开，无法复现其 headline。torch.compile 是 Python 图编译，不作为 C++ 下的伪开关。

## 2026-10-13 新服务器接续

接续 `31ac721`，不是从头研究。五个共享权重完整 SHA256 全部重新核验一致；8×H200 均被占用且利用率 99–100%，未运行 GPU 推理。CPU 12/12 CTest、49/49 工作流（含媒体）、18/18 来源、5/5 LoRA、29/29 构建配置、56/56 安全门禁通过。

修复未推送日志解析器、GPU报告循环引用、空hash锁定错误payload、异常遥测拒绝及逐次输入变化检查；构建脚本参数化并固定依赖／toolchain、全新输出且不覆盖。CUDA13 本机双版构建进行中，旧机器的编译通过结论不当作新机结果。新合成素材仅为独立 wiring dataset，真实质量和性能仍未测。续做顺序与容量边界见 `h3-server-continuation-plan.md`。

生产审查补充：SubBlock 在主任务尺寸的逐head/Q64展开超过图容量，已加建图前保守拒绝；当前不能进入生产 SubBlock 性能实验，后续需要有界批量化／原生CUDA实现。AdaLN/SubBlock/Sage的动态reuse/adapter/process-group已补compute前复查，CPU负例通过。新机baseline CUDA13编译链接已通过，candidate仍在构建；无GPU推理。

18:55 本机双CUDA版（含候选Sage及本轮生产修复）编译链接通过，公共配置一致、candidate源文件指纹与当前源码匹配；4个逐开关manifest的build gates通过。CPU全回归12/49/18/5/29/56/29全部通过。生产GPU仍未执行；SubBlock主任务仍拒绝，VDN/VSA仍未完成。

## 最终交接：4090路线

用户选择在Mac侧Codex连接另一服务器。接续入口`h3-coordination-handoff.md`、行动计划`h3-4090-continuation.md`。目标864×480/124帧/20步，完整混合Q4精度锁不变，GPU/几何/offload改变另baseline。4090短56帧历史实测不能当目标实测；5–10min仅计划估计。当前Cache/AdaLN/Sage拒绝DiT offload，先基准＋融合，再补卸载兼容；不把关闭拒绝当支持。GPU质量／性能仍未执行，VDN/VSA仍不完整。
