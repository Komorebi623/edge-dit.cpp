# MiniMax-H3：跨服务器接续交接

> **2026-10-14 LoRA实现里程碑**：IPv4/TLS合法下载已解决，原Q4＋BF16独立增量接入；Larry4步106.12s、LightFL4步110.85s vs原254.33s，Larry首末帧smoke通过。默认关闭、H3/CUDA单方法、不重跑baseline。Ref4已下载/转换但8卡被无关训练占用，mixed未启动；VSA/三链接身份仍独立待办。看`h3-lora-results.md`与`benchmark/h3_cuda/lora-results.json`。

> **2026-10-14 LoRA尝试最新状态**：ModelScope模型卡/文件SHA元数据已可读，收敛Larry4步与LightX2V4步候选；CDN实际返回校园网认证HTML（auth6.tsinghua.edu.cn），尚无真实adapter可加载。用户需认证服务器网络或提供本地LoRA；不绕过认证。无新模型baseline/生产代码改动/LoRA GPU实验。见`lora-trial-plan.json`与LoRA审计第8节。

> **2026-10-14 LoRA接续研究**：最新可访问GitHub已核验；优先原Q4＋高精度动态LoRA，候选Larry8步/LightX2V、FastVideo dense4步。HF证书失败，未确认最新adapter文件、未下载或推理；不得标记Q4 LoRA已支持。路线与缺口见`h3-lora-q4-deployment-audit-20261014.md`、`benchmark/h3_cuda/lora-q4-audit-20261014.json`。

> **2026-10-14追加keep25实验已完成**：T2VA205.72s、mixed538.70s，各1次；比keep50再省4.49%/2.28%，输出有额外稀疏差异。精度/二进制/输入不变，不改默认参数。见`h3-subblock-keep25.md`与`validation.json.subblock_keep25`。无剩余GPU测试。

> **2026-10-14 SubBlock修复完成**：同精度/输入/keep0.5，T2VA单次254.33→215.38s（↓15.3%）；mixed复用同配置baseline，单次627.91→551.30s（↓12.2%）。两个产物均与旧SubBlock逐字节相同，仍有相对dense的稀疏误差。详见`h3-subblock-fix.md`与`validation.json.subblock_repair`。按用户要求不再默认重复3次；先T2VA一次，再mixed一次。

> **2026-10-14 15:47接续**：AdaLN性能退化已修复，mixed已测571.53/565.67s vs baseline627.99s，各自与一个baseline产物逐字节一致（baseline重复本身有差异）；用户要求停止重复验收，三次正式矩阵未完成，勿误标完成。详见`h3-adaln-fix.md`及`validation.json.adaln_repair`。后续SubBlock优先，验证改为T2VA单次通过后mixed单次。

> **2026-10-14最新状态**：AdaLN / Cache-DiT / Sage(F16PV) / SubBlock(indexed) 已完成4090单卡、原精度、864×480/124帧/20步的T2VA三次对照，另有完整mixed单次与40项全模式功能回归。受控serial offload已接通；异步/并行/adapter组合仍拒绝。当前数据、收益与限制以 `h3-four-method-results.md`、`h3-four-method-reproduce.md` 和 `validation.json.four_method_validation` 为准；下文保留先前交接/计划历史，不能把旧“未测/只resident”当最新状态。


更新：2026-10-13（Asia/Shanghai）。当前选择：在 Mac 的 Codex 中连接另一台服务器，优先空闲 RTX 4090。**本文件是恢复进度入口；不需要重新从头研究，也不依赖旧聊天迁移。**

> **2026-10-13 RTX4090 已接续**：本机独立worktree `_scratch/h3-cuda-continuation`，生产源码提交 `d81041f`。最新实测和下一步看 `h3-4090-results.md` 与 `validation.json.rtx4090_continuation`；不要重新hash下载前已验证记录替代未来新机整文件核验。原始baseline/精度锁不变，AdaLN卸载仍未生产放行。

## 1. 分支、baseline 与代码状态

- 个人远端：`git@github.com:Komorebi623/edge-dit.cpp.git`；研究分支：`codex/minimax-h3-cuda-optimizations`。
- 本轮从 `31ac721acadd000b0e712d09d17553661e29748c` 接续；新的交接点为包含本文件及本轮修复的提交，接手时记录实际 `HEAD`，以用户给出的完整 hash 核对。
- 原始 baseline 固定 `97cc585d9dbd9ac5e30b78ef2e2c4a0d8bfbe9f2`。它已经包含原项目 CUDA norm/RoPE/modulation 路径和默认 SwiGLU 融合。全 `ED_H3_*` 关闭仅关闭新增研究方法，**不撤掉用户既有优化；不重复计功**。
- 固定子模块：ggml `f270346d688c7bec7c56bae2ae73343aba6817ad`；oneDNN `5e9224036021433d2577548ed0539fe9a53256bc`；cuDNN frontend `07b368a2696f5fd0fc2207d4635c51cea4c0bf71`。
- 不更新依赖、不合并 main、不强推；保留 `origin` 为个人仓库。已有目录先检查未提交内容，不覆盖或强制重置。只有一个集成者；旧服务器交接后不并行修改研究分支。

## 2. 用户要求

八种独立默认关闭开关：融合、Cache-DiT、AdaLN、SubBlock、Sage、Turbo、VDN、VSA。新优化仅 H3、仅 CUDA 生效，不改变其他模型／后端。允许多个智能体在不重叠文件范围内并行。

**禁止停止／杀死任何他人的进程，禁止删除用户文件，不抢占 GPU。** 空闲显存和瞬时利用率不是资源预留。GPU 不可用时继续 CPU 开发；需要用户提供的地址／权限才询问。先 baseline 再逐开关，固定精度／输入／solver／步数／placement；不能把模型换成 pruned／distilled／LoRA 或改量化后沿用旧 baseline。

## 3. 已完成与真实缺口

| 开关 | 当前真实状态 |
|---|---|
| `ED_H3_FUSED_OPS` | H3 分段调制／门控残差、视图和 SwiGLU 路径已接入；部分与原已有融合重合；GPU收益未测 |
| `ED_H3_CACHE_DIT` | 当前真实前段 probe、逐模态 RF guard、同锚 middle delta＋真实后段；CFG分离／请求失效；生产GPU未测 |
| `ED_H3_ADALN_CACHE` | 全固定sigma真实投影预填、分支及模型／输入身份检查；tiny CPU逐位一致；**未释放投影权重**，不宣称显存节省 |
| `ED_H3_SPARSE_ATTN` | per-head Q64/K64 TopK/gather＋全部Q更新；保护非目标video模态；**主任务图展开超容量，现建图前明确拒绝**，需要有界／批量化实现 |
| `ED_H3_SAGE_ATTN` | H3SA专用tag、INT8 QK＋F16 PV；首末层／refiner保dense，SubBlock不能叠加；CUDA编译通过，GPU未测 |
| `ED_H3_TURBO` | CPU离线LoRA合并／完整hash／recipe入口；真实匹配adapter未验证，需独立模型baseline |
| `ED_H3_VDN` | 数学参考及部分原生ggml组件；exact solve、short-conv、anchor、完整权重和CUDA部署未完成；生产入口拒绝 |
| `ED_H3_VSA` | tile routing、fine＋learned gated compression的CPU参考和gate检查；真实权重／原生CUDA／生产集成未完成；生产入口拒绝 |

TP/Ulysses尚未迁移H3；torch.compile不是C++算法开关。RunningHub私有权重未公开，不能承诺复现其headline。已读两份原PPT和固定官方关键源码；9仓库167份精选源码有hash锁，不代表读过所有仓库每一行。VSA不是“删除90%的Q token”，不能按PPT简述实现。

本轮修复：clean clone缺失CPU日志解析器；GPU报告循环引用；null hash锁定错误payload；异常遥测与逐launch输入变化门禁；SubBlock容量拒绝；compute/prefill前重验reuse/offload/adapter/process-group；请求成功、取消、早返回、异常清理，并在VAE decode前释放请求缓存，不释放模型权重。

## 4. 本轮验证与服务器环境

旧H200服务器：8×H200，CUDA13.0.88／驱动580.105.08／GCC12.3.1；固定CUDA baseline及candidate（含Sage及生产修复）编译链接通过，公共配置及源文件指纹核对一致。**没有生产GPU推理，无质量／性能／加速比。**

CPU全工程＋一键回归：12/12 CTest、49/49 workflow/media、18/18 sourcefetch、5/5 tinyLoRA、29/29 build config、56/56 preflight、29/29 request cleanup，全部通过，无跳过／预期失败。tiny H3 24/24逐位一致；fusion 56次CPU实际图对照误差0。生命周期测试有结构＋编译tiny场景，不是完整pipeline/GPU证明。

2026-10-13 19:21：每卡free约3939–3941MiB，全部被VLLM worker占用，即使util为0也不可用；没有停止服务。

## 5. 精度与输入

先读 `docs/research/h3-precision-lock.json`。五个模型本轮在共享存储重新读取**整文件SHA256，5/5匹配**；新机必须重新核验，不能沿用旧机“已验证”的结论。

原共享根：`/mnt/cfs/9n-das-admin/llm_models/MiniMax-H3-GGUF`。完整FL2VA/Ref2VA混合Q4_K/Q4_0/BF16/F32；Qwen混合Q4_K/Q6_K/BF16；video VAE FP16，audio VAE FP32。使用preserve，不自动再量化；单任务组件文件39.8708GiB，**不是运行峰值显存**。

旧主任务：Ref2VA mixed，512×512、124帧、24fps、20步、seed42、CFG1、res_multistep/simple。迁4090的新目标：**864×480、124帧、20步**，同seed／CFG／solver和权重；几何＋GPU＋placement均改变，必须新baseline。

覆盖mixed、paired audio、ref video/image、首尾／首帧／末帧、T2VA，以及CFG／重复请求／输入顺序标签变更／错误恢复。旧exact fixture未迁移；本轮生成独立合成wiring dataset，不冒充旧素材续跑；生产语义质量另需合法真实参考媒体。

## 6. 4090接续的首要任务

详细方案：`docs/research/h3-4090-continuation.md`。

**当前Cache-DiT/AdaLN/SubBlock/Sage要求DiT resident，拒绝DiT卸载，不能在4090直接打开做实验。** 原项目有按组件／图段卸载及固定VAE分块，24GiB短任务已跑通；先恢复默认关闭的offloaded baseline，再实现研究优化与卸载的兼容，不能删掉保护条件冒充支持。

4090的864×480/124帧/20步Q4任务先按5–10分钟/冷进程、保守12分钟规划；这是56帧历史记录外推，**目标任务未测、显存是否足够也未测**。混合参考输入仅粗略预留10–15分钟；不要写成保证。

## 7. 新机执行顺序

1. 只读环境／GPU／磁盘／依赖检查，核对HEAD与全部固定依赖；找五个模型并整文件hash。不假设有同名容器或旧本地路径。
2. `bash scripts/test_h3_cpu.sh`；媒体测试需ffmpeg/ffprobe，LoRA需CPU torch/safetensors。隔离依赖，不覆盖系统。合理限制CPU线程。
3. `scripts/build_h3_cuda.sh`已参数化：4090显式`H3_CUDA_ARCHITECTURES=89`，H200=90；显式本机CUDA/cuDNN/frontend路径及编译器，两版一致。先`all --dry-run`，再`all --validate`，最后全新目录构建；不下载、不覆盖、不跑GPU。**旧脚本SM90/旧路径提醒已过时。**
4. 复用`h3_cuda_workflow.py`的fixtures→lock→baseline→candidate→compare。新路径/新几何/新placement单独manifest；现有manifest不是4090即用配方。显式offload不代表CPU计算；不盲用auto-fit（会改量化）。输出全新，不覆盖结果。
5. 合法独占空闲GPU后测目标baseline，记录首次加载/prefill、TE/DiT/VAE、CLI冷进程E2E及真实峰值。预算不是万能硬cap，CPU/cuDNN工作区和最小graph segment要实测。
6. 优先增量融合→卸载兼容工程→AdaLN／Cache-DiT／Sage逐开关，未经单项验收不叠加；SubBlock另做有界实现；VDN/VSA/Turbo按独立模型路线推进。CPU可继续参考/错误回归，不能替代CUDA性能或数值验收。

## 8. 不随Git迁移的内容与日志

Git包含源码、tests、简洁docs、精度锁、参考source lock和示例manifest；**不含模型、旧构建/二进制、媒体、`_scratch`原始日志、附件、聊天、认证文件或私钥**。

可另行安全复制约1MiB本轮素材`_scratch/h3-server-20261013/fixtures/`及`locked-manifest.json`（路径通过asset-root重映射）；否则新机fixtures重新生成并锁定，重新baseline。源快照可用`fetch_h3_references.py --fetch`按锁重取，不执行第三方源码、不下载未知权重。

本轮证据旧机路径：`_scratch/h3-server-20261013/`；最新软件状态在`benchmark/h3_cuda/validation.json`的`new_server_validation`，旧顶层结果为历史。CPU可行性probe在`_scratch/h3-cpu-feasibility-20261013/`，8核synthetic算子外推不当生产时间。

日志只记环境、baseline、算法、数据差异与故障；不得把模型/素材哈希通过或编译通过写成GPU画质／性能通过。新Codex读本文件即可接续，不必尝试注入旧私密聊天。
