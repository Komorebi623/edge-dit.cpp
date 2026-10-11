# MiniMax-H3 优化来源核对与固定源码快照

核对日期：2026-10-11。输入为 `slides-1.txt`、`slides-2.txt`。已恢复研究/fetch 工具工作并完成固定源码快照；未运行性能测试、未下载模型、未修改推理实现。公开权重元数据及部分历史性能出处仍待复核，本文不是完整复现报告。

## 1. 官方来源与固定提交

以下提交通过 GitHub API 获取并用于源码下载；许可证为仓库代码许可证，不自动覆盖模型及所有第三方文件。

| 项目 | 官方来源与固定 commit | 实现关键路径（仓库内） | 许可证、权重限制与迁移结论 |
|---|---|---|---|
| SGLang Diffusion | [sgl-project/sglang](https://github.com/sgl-project/sglang/tree/53295c697255a5e8ddb263f98ab7e047e2700ad2)，`53295c697255a5e8ddb263f98ab7e047e2700ad2` | `python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py`；`runtime/models/dits/minimax_h3_adaln_cache.py`；`runtime/layers/attention/backends/subblock_sparse/router.py`；`runtime/cache/cache_dit_integration.py`；手册 `docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx` | 代码 Apache-2.0。融合算子、精确时间表缓存可借鉴；SubBlock 和跨步缓存属于无训练但有近似的推理优化，不是专用蒸馏模型。 |
| RunningHub H3 Lightning | [RH-RunningHub/MiniMax-H3-MultiGPU-Lightning](https://github.com/RH-RunningHub/MiniMax-H3-MultiGPU-Lightning/tree/e8f860bb71bc7793e2614e471bbb7e375a8acde7)，`e8f860bb71bc7793e2614e471bbb7e375a8acde7` | `README.md`、`sglang/RH-PIN.md`；内置 SGLang 固定 `1f6ce4b0686232fce5feb0d0d841df7d9bcabc02` | 代码 Apache-2.0。RH 自训加速权重未公开；公开 LoRA 可跑流程，不能保证复现 RH headline。TP2×Ulysses4 是八卡方案，不是单卡算子优化。 |
| FastVideo / FastH3 | [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo/tree/d5253fd71055927f12aa6c4aa679b93a00238851)，`d5253fd71055927f12aa6c4aa679b93a00238851` | `fastvideo/models/dits/minimax_h3.py`；`fastvideo/attention/backends/video_sparse_attn_h3.py`；`fastvideo/pipelines/basic/minimax_h3/vsa_guard.py`；`docs/inference/fasth3-distilled.md` | 代码 Apache-2.0。官方公布 FastH3 V1 4-forward 和 V2 8-forward 权重/适配器；需匹配学生权重、VSA 门控、采样梯度和音视频 shift。不是把原始 H3 的 steps 改小即可。 |
| VideoDeltaNet-H3 | [OpenVDN/vdn-minimax-h3](https://github.com/OpenVDN/vdn-minimax-h3/tree/e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b)，`e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b` | `src/models/hybrid_attention.py`；`src/models/linear_attention/delta_rule.py`；`src/models/softmax_attention/window.py`；`src/inference/utils/lora.py`；`src/training/` | 代码 Apache-2.0；模型继承 MiniMax-H3 Community License。新增线性分支、门控和 LoRA 来自训练，不能用原始 H3 权重无训练替换 Attention。 |
| vLLM-Omni | [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni/tree/213a1c835ed595d6a51d4f918e3920f4159dd7ed)，`213a1c835ed595d6a51d4f918e3920f4159dd7ed` | `vllm_omni/diffusion/models/minimax_h3/adaln_cache.py`、`minimax_h3_transformer.py`；`tools/minimax_h3/build_adaln_cache.py` | 代码 Apache-2.0。主线包含精确投影缓存/sidecar 及可选权重 offload；附件具体显存数据来自尚未合并的 PR，不能混称主线实测。 |
| Cache-DiT | [vipshop/cache-dit](https://github.com/vipshop/cache-dit/tree/a7898aaf1ba2c2aabdd94a9526fb5fb82ddc900e)，`a7898aaf1ba2c2aabdd94a9526fb5fb82ddc900e` | `src/cache_dit/caching/cache_blocks/pattern_3_4_5.py`；`src/cache_dit/caching/cache_contexts/cache_context.py` | 代码 Apache-2.0。残差复用可重新实现为 C++/CUDA 请求级缓存；必须维护分支、连续命中上限及调度状态。 |

SageAttention 官方仓库为 [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention/tree/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5)，已固定 `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`；RunningHub README 固定依赖 SageAttention 2.2.0。本轮已下载 `LICENSE`、`sageattention/core.py`、`quant.py` 及 SM80/89/90 原生 CUDA 源码。仓库代码为 Apache-2.0，部分头文件包含 FlashInfer、vLLM、NVIDIA 等第三方归属信息；移植需保留对应文件声明。固定 dispatcher 有 SM80、SM86、SM89、SM90、SM120、SM121 分支，没有通用覆盖全部 GPU 的保证；当前精选快照不含 SageAttention3/2++ 独立目录。

## 2. 纠正附件的关键误读

### VSA：稀疏的是连接，不是删除 90% Token/FFN

- `_pool_tiles()` 对 Q/K 分块池化，计算块间相关性；`_build_block_mask()` 对**每个查询块**沿 K 块维度选择 Top-K。随后 block-sparse Attention 产生所有查询行的输出，`untile()` 还原完整序列。不是“前 10% 高分 Token 算 Attention，其余仅残差直通”。
- `to_gate_compress` 控制**池化压缩 Attention 分支的输出合并**，不是 Token 重要性分类器。它从 `hidden_size` 映射到 `num_heads × head_dim`；固定 H3 配置为 5376→7168，不能描述为“数万参数、可忽略开销”。
- 主 Block 的 `self.ff(...)` 仍处理完整 hidden states；Q/K/V 投影也未因块连接稀疏而只处理 10% Token。附件“单步主体算力减少 90%”缺乏依据，90% 是部分视频→视频块连接的目标稀疏率，不是整网 FLOPs/Token 数下降。
- 官方 FastH3 V1 发布说明使用 0.9 稀疏率、64-token tiles；V2 文档记录 0.8、64-token tiles。不能把 90% 写成所有模型的固定值。文本/音频等受保护行保持稠密，实际保留率还受分块和配置影响。发布说明：[FastH3 Preview](https://haoailab.com/blogs/fasth3-preview/)（动态页面，未固定版本）。
- 官方 `vsa_guard.py` 拒绝所有 compression gate 都为零的 H3 VSA 模型；应加载携带训练门控的学生模型/适配器。用户无需再训练≠算法不需要训练，也不意味着基础 H3 可安全直接切换。

### VDN：混合注意力需要训练，LoRA 融合不等于无代价

- `HybridAttention` 共用原 QKV，但增加局部 softmax、双向线性 Delta 分支、门控和独立线性输出投影。线性分支是 forward/reverse 扫描，不能解释成仅“晚 Token 读早 Token”的单向自回归记忆。
- 发布训练阶段为 A1 新分支逐层对齐、A2 分支对齐、B 的 QKV/O LoRA 与新分支训练，以及 DMD2 的 8-step turbo LoRA。仅安装内核不能获得这些训练后的功能。
- 两个 LoRA 是 checkpoint 的适配器，不等于“局部/全局各一个”。`merge_lora_state()` 可把 BA 更新合入线性权重，消除独立 LoRA matmul；新增线性 Attention、门控、状态读写仍有算力、参数和显存成本，“不增加任何推理延迟”不成立。
- 视频局部窗口与 Delta 分支可降低长视频计算增长，但 H3 保留文本、音频、锚点等稠密交互，不能无条件称整个联合模型严格 O(N)。
- [SGLang PR #37903](https://github.com/sgl-project/sglang/pull/37903) 于 **2026-09-12** 合并，merge commit `ff1ce11348673eea9b51165bd6b2da7238c958b4`。这证明接入了**专用 VDN 模型**，不证明原 H3 可无训练更换后端。初始 PR 是 T2VA；当前固定手册支持 T2VA/FL2VA，拒绝原生 Ref2VA。VDN 官方的 Ref2VA-like 是借用 FL2VA 权重，不应当作完整 Ref2VA 等价物。
- 多 GPU 非算法的硬性前提：官方提供单卡入口，SGLang 也有 1/2/4/8 卡路径；论文/宣传的八 B200 结果不可外推到单消费卡。分支并行 5+3 与 SGLang Ulysses8 是不同实现。

### 缓存、融合与性能归因

- Cache-DiT 命中时用**当前 Fn 输出 + 最近完整计算的 Mn 残差**，然后计算 Bn（如 Bn>0）；附件公式只适用于 Bn=0 或省略尾段的简图。阈值、warmup、连续命中上限均影响质量/命中率；4–8 NFE 学生可复用机会较少，必须实测。
- SubBlock 属于训练自由块路由；所有 Q 行仍更新。当前 H3 实现保护非视频查询，且稠密 warmup 应使用与基线一致的后端。[PR #34578](https://github.com/sgl-project/sglang/pull/34578) 提醒稠密 kernel 不一致本身会造成输出差异；不能把它全归因于稀疏。
- 融合 RMSNorm/AdaLN/SwiGLU/QK Norm/RoPE 不需新权重，但舍入顺序可能改变，不能仅因“数学等价”就承诺字节相同。H3 的 128 head_dim 中 RoPE 只旋转 96 维，迁移需保留这类架构细节。
- RunningHub README 明确报告八 RTX 6000D、RH 自训权重的 50→4-step 对照，并说明 12.2×为蒸馏+SageAttention2+Cache-DiT+compile 的组合结果。不是单一 Attention 或纯工程优化收益；BF16 backbone 也不等于全部计算无损。其 `1f6ce4b0` 快照更新没有另行测出的加速收益。
- 附件 SGLang 1.85–6.24×及 217.31→34.80s 的完整历史实验出处尚未匹配；不将附件数字写为本项目实测，后续需固定旧实现、硬件、任务、缓存/stride、基线及 SSIM 定义。

### AdaLN：可无训练迁移，但要区分主线与未合并实验

- AdaLN 的 timestep/modality 输出可按精确采样时间表缓存；不能跨不同权重、LoRA、模型分区、音视频 flow shift 或时间表盲用。当前 vLLM-Omni sidecar 包含调度、权重/适配器身份和数学环境校验；CPU 保留权重以便新调度/失效回退。
- [vLLM-Omni PR #5783](https://github.com/vllm-project/vllm-omni/pull/5783) 核对时仍 **open、未合并**，实验 head `635215a96d3eaf6b5bac66020feae51b34daa5c9`。附件“11.26 GiB/GPU”是该 PR 的 **2×H20、TP2、50 sigma 点/49 DiT forwards** 特定实验：每卡移除 12.144 GiB 投影权重、增加 0.885 GiB 表，净 11.259 GiB；不等于任意 GPU/调度都节省相同显存。
- PR 对特定 eager 输入报告输出文件哈希相同，不能扩大为所有硬件、量化、compile、seed 都相同。权重释放降低稳态显存，不保证首次建表峰值下降；这是容量优化，不应宣称额外巨大延迟收益。

## 3. 公开权重与许可证边界

- 官方代码/文档列出 [FastH3 V1](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)、[FastH3 V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2)、[VDN-H3](https://huggingface.co/OpenVDN/vdn-minimax-h3)。本次 Hugging Face 直连及升级后重试仍失败；**没有独立核验其当前 model-card、gated 状态、权重文件列表、权重 revision 或哈希**。代码/官方发布说明中的公开性声明不能替代实际权重核验。
- RunningHub README 提供公开下载固定版本：base H3 `42ed227ee7df40d41602854ae760620d6eb651fe`；larryvrh Turbo LoRA `43a74557ac3f6539db8e0f2a959d03feb7a81480`。这些是官方 README 记录，非本次独立检查所得；RH 私有权重仍不可复现。
- 已读 [VDN 仓库附带的 H3 协议](https://github.com/OpenVDN/vdn-minimax-h3/blob/e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b/licenses/MiniMax-H3-Community-License-Agreement.txt)：协议明确排除美国、欧盟、英国、韩国等地域，并规定商业年收入超过 2000 万美元须另行事先书面授权；另有再分发、NOTICE 和使用限制。**Apache-2.0 代码≠H3 衍生权重可全球无条件使用**。此处是文件条款摘要，部署前需复核实际权重附带版本并进行许可审查。

## 4. 对 C++/CUDA 的迁移建议

| 分类 | 建议 |
|---|---|
| 优先：无训练、尽量保真 | 原生算子融合、精确 AdaLN 时间表缓存与请求级失效管理；保留 BF16/FP32 边界、96维 RoPE 和稠密一致性参考。compile 仅是 Python 工程路线，不是向 C++ 移植算法的必要条件。 |
| 次优：无训练但近似 | Cache-DiT 残差复用、SubBlock、SageAttention 类低精度 Attention；做可关闭开关，分别检查视频、音频和同步，不预先承诺收益。 |
| 独立模型路线 | FastH3/VSA、Turbo/DMD/PDD、VDN；加载学生/分支/LoRA及匹配调度，而非仅复制后端。先核验公开权重与许可证，再评估转换成本。 |
| 暂不当作单卡优化 | TP、Ulysses、分支并行；通信拓扑与总显存另立研究项。单卡运行能否完成取决于权重容量/offload，与算法是否可单卡不是一回事。 |

## 5. 可复用源码 fetch workflow（已完成）

- 工具：`/export/home/wangtianyang.21/code1/edge-dit.cpp/scripts/fetch_h3_references.py`；固定来源锁：`/export/home/wangtianyang.21/code1/edge-dit.cpp/benchmark/h3_cuda/reference-lock.json`。
- 默认 dry-run **不联网、不写文件**；显式 `--fetch` 后才下载。采用官方 raw GitHub **selected-files snapshot**，不是全量 clone/archive；保留仓库内目录结构，明确不能据此构建完整上游工程。
- 锁记录官方仓库、40位 commit、源文件大小、Git blob SHA1、代码许可证路径；下载后逐文件校验 Git blob/长度并记录 SHA256 回执。默认磁盘预算 64 MiB，单文件 2 MiB；计入既有文件及保守暂存预算，另做写入时检查。
- 固定输出根 `/export/home/wangtianyang.21/code1/edge-dit.cpp/_scratch/h3-reference-src`。拒绝外部路径、符号链接、路径穿越、非白名单仓库、浮动分支、二进制/权重/LFS pointer。匹配既有内容则校验复用；不匹配即失败，绝不覆盖用户文件。并发运行有排他锁；部分下载失败记录错误，可重试续传。
- 不执行参考源码，不运行 Git hooks，不取子模块，不安装依赖，不下载 weights。本轮 9 个官方仓库 **167 份源码、2,387,085 bytes** 已全部校验；9 个 `SNAPSHOT-*.json` 和一次完整回执已落盘。再次 dry-run 显示 **0 待下载、167 份匹配**。
- 离线验证：内置 `--self-test`，覆盖默认零网络/零写入、提交/路径限制、权重与 LFS 拒绝、预算、哈希、符号链接、原子不覆盖、断点复用、部分失败回执及并发锁；不引入 `requests`。

```bash
# 默认预览；可从任意目录使用绝对工具路径
python3 -B /export/home/wangtianyang.21/code1/edge-dit.cpp/scripts/fetch_h3_references.py
# 显式下载（网络受环境审批控制）
python3 -B /export/home/wangtianyang.21/code1/edge-dit.cpp/scripts/fetch_h3_references.py --fetch --max-mib 64 --max-file-mib 2
# 单项及离线测试
python3 -B /export/home/wangtianyang.21/code1/edge-dit.cpp/scripts/fetch_h3_references.py --source lightx2v --source lightx2v-turbo --dry-run
python3 -B /export/home/wangtianyang.21/code1/edge-dit.cpp/scripts/fetch_h3_references.py --self-test
```

新增固定来源：

| 来源 | 固定 commit | 原始实现关键路径 | 代码许可与 scope |
|---|---|---|---|
| [ModelTC/LightX2V](https://github.com/ModelTC/LightX2V/tree/4fe984c07b1610338e97f8c7a268dceef8fd9966) | `4fe984c07b1610338e97f8c7a268dceef8fd9966` | `lightx2v/models/networks/minimax_h3/lora.py`；`lightx2v/models/schedulers/minimax_h3/scheduler.py`；`configs/minimax_h3/dmd/minimax_h3_bf16_4step.json` | Apache-2.0；提供 PEFT 合并/TP 分片和独立音视频调度，不能把 DMD 学生当作原 H3 无训练改步数。 |
| [ModelTC/Minimax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo/tree/02e26d591f7a04d5d1a074c9566d5dd4f22f6225) | `02e26d591f7a04d5d1a074c9566d5dd4f22f6225` | `README.md`；`inference_minimax_h3.py`；`minimax_h3_ref2va_pipeline.py`；`DIFFUSERS_SETUP_AND_INFERENCE.md` | Apache-2.0 代码；权重另发于 HF，未独立取到其 revision/模型卡。已固定官方说明，不把其权重链接视作权重内容验证。 |

所有上游路径可在本地 `<源码根>/<source-name>/<固定commit>/<原仓库路径>` 浏览。详细绝对路径、回执及下载过程见 `/export/home/wangtianyang.21/code1/edge-dit.cpp/docs/research/h3-source-fetch-log.md`。

## 6. 具体迁移依赖及算法 scope

| 技术 | 必要依赖/状态 | C++/CUDA 接入边界 |
|---|---|---|
| 精确 AdaLN/融合算子 | 精确 timestep/modality、原投影权重、模型分区与 adapter 身份；BF16/FP32 舍入、RoPE 尾部保留 | 不需专用训练；vLLM-Omni `exact_projection_cache.py` 的失效/sidecar 契约可借鉴，不能只照抄 time→table。FastVideo `minimax_h3_fusions/exact.py` 显式保留 BF16 中间舍入并用非融合 mul/add，与只最终舍入的融合不是同一一致性目标。 |
| Cache-DiT | Fn/Mn/Bn 切分；warmup、相对残差阈值/累计阈值、连续缓存上限；CFG/请求状态 | Python wrapper 不直接链接到 ggml；在 block 层重新实现请求级缓冲/重置。`cache_config.py` 的库默认与 SGLang integration 默认不同，不能照搬一组常数。少步 LoRA 不保证有足够可缓存步骤。 |
| SubBlock | 分块/模态路由、全部 QKV、Top-K 索引、稠密 warmup；Triton router，CUDA/CuTe 稀疏后端 | 固定 SGLang 主线稀疏路径调用自带 FlashAttention CuTe；SM120 Sage 路径另用 FlashInfer CuTe。原 CUDA 源码及运行时布局需一起适配，不是复制 `router.py` 就可调用。路由分数/索引临时张量与 head 分块也影响单卡显存。 |
| SageAttention | Q/K smoothing+量化、V scale/均值修正、scale/granularity、累加模式；NHD/HND、head_dim、非因果、连续性 | 低精度 Attention 无训练，但输出近似。固定 `core.py`：SM80 使用 INT8-QK/FP16-PV CUDA，SM86 走 Triton，SM89/90 等使用对应 FP8-PV 内核；H3 128维 head 是接口适配起点，不等于所有长 packed AV 序列自动支持。native kernels 有 PyTorch/pybind 包装，ggml 接入需自写张量/stream/scale桥接。 |
| FastH3/VSA | 学生 backbone/LoRA+训练 compression gate；稀疏率、tile、精确 DMD 梯度及两套 scheduler shifts | 训练后的模型路线。块路由和池化压缩分支同时存在，不能只取 sparse mask；完整 FFN仍执行。V1/V2/PDD 的采样契约不通用。Triton/sm100a kernel 与分布式 Attention 不是 C++ 可直接引用的统一 ABI。 |
| VDN | 原始未旋转 Q/K 与局部旋转 Q/K；局部窗口+anchor、短卷积、frame statistics、双向 scan、边界 bridge、gate/norm；训练新增权重 | `branch.py`/`scan.py` 的逐帧状态依赖需整体移植。发布 `vdn_solve` 在 FP32 使用 Cholesky + triangular solve + GEMM，C++需 cuSOLVER/cuBLAS 或等价实现；状态有 d×d成本且双向，不能替换成随意顺序的累加。仅 LoRA融合不会消除新分支。 |
| TP/Ulysses/分支并行 | GPU组、collectives、head/sequence分片布局、拓扑 | 属于分布式执行项；不是单卡内核优化前提。RunningHub 8卡组合与 VDN 5+3/纯Ulysses 不混记。 |

本轮只核对代码许可证文件与第三方声明，不提供法律结论；权重许可、地域/商业条件应按具体模型卡和协议版本另行确认。

## 7. LightX2V Turbo 公开 LoRA conversion 接入路线（不修改实现）

既有本地 `feat/minimax-h3-turbo` 提交 **`a2cc96b9ebbb94706903094aed2ced2e98796654`** 可读但不在当前开发 HEAD 的祖先链上；没有 cherry-pick 或执行其脚本。可借鉴文件与职责：

| 历史文件（`git show a2cc96b:<path>`） | 借鉴点/不能直接照搬处 |
|---|---|
| `scripts/merge_minimax_h3_lora.py` | `load_adapter()`/`validate_adapter()`：PEFT A/B 配对、rank、alpha、基线key和矩阵shape校验；`merge_weight()`：FP32 `W + strength*(alpha/rank)*BA` 后回原dtype；原脚本可CPU/GPU、resume只核对tensor keys，**不能称为权重哈希验证**。主线程正恢复 CPU-only validated recipe与shard hash，此 worker不重复实现。 |
| `src/core/runtime/model_loader.cpp` | 独立 Diffusers transformer index 的加载参考；主线程已做 H3-only 路径识别，本 worker不改 loader。 |
| `src/dit_models/pipelines/minimax_h3_pipeline.cpp` / `.hpp` | Turbo默认NFE/Euler/simple、显式参数优先、视频/音频独立shift及Euler音频修正；与当前融合分支合并需保留基础H3 RES轨迹。历史 `768p` 路径tag启发式不能替代完整 recipe。 |
| `docs/minimax-h3-turbo.md` | matching FL2VA/Ref2VA baseline、离线合并后GGUF转换和音视频质量检查。历史性能数字没有在当前分支复测，不移作现状结论。 |

建议待主线程 CPU-only 转换工具就绪后采用以下验收路线（本轮未下载/合并权重）：

1. 固定**匹配 base 分区、adapter revision与文件SHA256**；选择 Diffusers PEFT 权重，不把 ComfyUI 键名/融合 QKV/SwiGLU 排列直接传入旧合并器。官方 native `lora.py` 支持额外前缀/键格式及TP分片，历史脚本仅严格接受 `.lora_A.default.weight`/`.lora_B.default.weight`，应以实际权重header验证。
2. CPU-only dry-run先检查每对A/B、2D shape、rank、有限scale/alpha、base索引与分片身份；明确拒绝或显式转换未知keys/格式，不能静默漏合并。alpha优先级记录在 recipe，不默认让所有adapter共享alpha=8；官方768p4步config显式alpha=128，旧544p4步说明采用8。
3. FP32逐层计算更新后回base dtype、原子写出；resume应验证recipe输入身份及输出分片hash，不仅tensor-key集合。冻结 base+adapter+scale+alpha+输出shard SHA256，并保留足够磁盘空间；本源码fetch的64MiB预算**不适用于模型合并**。
4. 把合并后 `diffusion_pytorch_model.safetensors.index.json` 输入H3-only loader；完整基线encoder/VAEs仍必需。GGUF为后续显式转换步骤，量化需独立质量对照，不在fetch工具内执行。
5. 采样使用正确NFE与**独立音视频shift**。固定官方Turbo README已包含：544p4步/8步为12/3，768p4步/**768p8步**为6/3，Ref2VA4步为12/3；新增768p8步不能遗漏。官方定义 q_i=(N-i)/N 后分别shift，N次forward加terminal-zero；不能混用FastH3 V2的非均匀DMD rungs。
6. Ref2VA官方入口默认 `reference-resize-mode=match`（随目标面积、保持比例、不无条件放大），不是原Diffusers固定2048短边；须记录resize策略。固定768p LightX2V config为`training_euler`、CFG关闭、`feature_caching=NoCaching`；先建立这类清晰基线再加近似缓存/量化。
7. 转换结构单测与采样回归分别做；视频、语音/音乐非静音与clipping、音视频时长/同步、基础H3未开启Turbo的回归都需保留。**未在本worker运行模型测试，不能声明已验证Turbo输出或性能**。

## 8. 本轮完成与后续边界

固定源码fetch及日志已完成。按用户要求不再扩张HF调研；本轮对 `huggingface.co` 的已批准元数据请求连接失败，公开权重revision/模型卡依然未独立核验。SGLang附件历史benchmark出处、目标硬件性能/数值验证留后续，不能由源码静态检查替代。

先前临时资料：`/tmp/edge-dit-h3-research-20261011/`。本轮实际源码、固定锁、回执位于仓库指定scratch和四个授权交付文件；不改其他worker实现，不下载GB级模型，不安装依赖、不执行上游源码。完成后停止本worker并让出并发slot。
