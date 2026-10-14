# H3 AdaLN：SGLang 公开收益与移植差距（2026-10-14，只读审计）

## 1. 结论与证据边界
- **有 AdaLN 相关公开性能数据，但不能说“纯缓存必定提速 X%”。** 已合并在线重建实验减少 6.4% 耗时，同时改变 DiT 权重驻留；两级缓存实验只在重复时间表请求显著受益；另有显存下降但时延几乎不变的未合并实验。
- 固定上游 `53295c697255a5e8ddb263f98ab7e047e2700ad2`：GitHub 提交时间 2026-10-11 17:25:43（北京时间）；手册及 AdaLN Python 文件完整 SHA256 均重新匹配本地 verified 清单。本项目审读基准 `79b8aa0722dd79f42032f340b03854a83819d7cf`；完成时主代理已并行修改生产文件，下文差距/行号描述该基准，不覆盖其在途修复。
- 本轮 web 检索无可用正文；随后经批准只读访问官方 GitHub API 的 PR 正文与状态，没有下载仓库、模型或运行实验。以下是**上游作者报告**，不是本机复现。
- 排除 Cache-DiT 的 1.4×、RMSNorm/AdaLN 融合、pruned/蒸馏模型及 vLLM-Omni 显存数字；它们不证明本项收益。

## 2. 公开单项/关联数据（必须保留比较口径）
| 来源及日期 | 配置 | 公开数据 | 能证明什么 / 不能证明什么 |
|---|---|---|---|
| [S1] 在线重建，2026-08-14 合并 | 8×RTX PRO 5000 SM120；TP1/Ulysses8；T2VA 1344×768、124帧、50步；3次，噪声底1.72% | E2E **135.74→127.03 s，↓6.4%（约1.069×）**；整卡峰值70.34→66.90 GiB | AdaLN使原 layerwise r35 可改为 DiT全驻留；不是固定驻留布局的纯投影缓存A/B。作者profile归因于H2D与Ulysses通信争用降低，不能照搬到单卡4090。 |
| [S1] 首次/复用 | 同上；逐层读取原投影权重，保持每个plan的GEMM形状 | 重建一轮 **2.4 s**；同时间表记忆命中无需重建 | 首次重建不能忽略；正文未明确127.03 s是否计入首次重建，也未完整定义加载/编码封装边界。 |
| [S2] 两级缓存，2026-09-03 合并 | 1×H200；在线AdaLN两组均开；T2VA、50步、seed1101；shift12/10交替，同server各4请求，排除warmup | 首见94.83→94.68 s、94.76→94.76 s；重复 **94.77→88.42 s、94.73→88.34 s（约↓6.7%）** | 对比host tier=0/8GB；收益是免去再次读24.2GiB权重及重建，不是相对完全关闭AdaLN。分辨率/帧数未在该正文写明。 |
| [S2] 多卡重复时间表 | 4×H200；Ulysses4/TP1；T2VA、50步 | 首见31.95/31.88 s，重复25.16/25.13 s | 首见vs重复，不另包装为同条件“纯AdaLN单项加速”；作者报告host开关两组及重复输出逐字节一致。 |
| [S3] sidecar＋INT8兼容，2026-10-12创建，审计时open未合并，**晚于锁定快照** | 1×RTX PRO 6000 SM120；FL2VA/T2VA、4s、16:9、768p、50步、seed7、shifts12/3、exact attention；同为在线ConvRot INT8，其51个AdaLN仍BF16 | resident→sidecar：generation **369.8→369.4 s**；denoise345.3→350.3 s；峰值58.0→33.8 **GB（原文单位）**；MP4一致 | 主要是约24.2GB显存收益，不能称可靠时延提升；sidecar离线构建不在generation内。不要拿另一行serialized GPTQ导出的382.3 s作配对基线。 |
- [S0] 最初sidecar PR（2026-08-07创建、closed但未合并）正文仅说明数值/峰值验证待做，没有单项时延表；后续实现见[S1]，不能把closed当merged。
- 固定手册[U1] L358–430只描述实验路径、BF16表、24.2GiB投影权重、首次数秒开销和GPU/host缓存；**没有同配置AdaLN开/关耗时表**，不能仅从手册声称某个加速倍数。

## 3. 固定源码差异与修复方向（只给主代理证据，不改生产）
| 项目 | SGLang固定实现 | 本项目已证实差距与边界 |
|---|---|---|
| 预填组织 | [U2] L615–624每个缺失plan仅算一次embed；L746–790逐层读权重并遍历这些embed；L561–568要求保留原GEMM形状 | [L1] L1779–1844、[L2] L735–790已是layer-major，**不是仍逐step反复搬每层权重**；但每层重新build time_embedding，重复temb可消除。不可把全部sigma拼成大GEMM改变数值。 |
| 表的位置/访问 | [U2] L512–545固定地址BF16 GPU slab；L834–845一次gather取所有层；L198–219 host miss才批量异步回填GPU，带stream fence | [L2] L408–423先同步D2H保存F32；L468–487每次构图分配host副本、memcpy并绑定输入，随后执行器负责上传。可考虑预算内常驻F32表、跨段别名/常量复用，不能直接改成BF16破坏原精度锁。 |
| 生命周期 | [U2] L575–613跨请求plan LRU命中直接返回；host按schedule组LRU，未命中再重建 | [L1] L1645–1665清理每请求缓存；当前CLI每次新进程。先修单请求退化，再另列常驻进程首请求/同时间表/交替时间表数据，不能用热server对冷CLI冒充A/B。 |
| 身份校验 | [U2] L746–759打开权重并建立索引；该模块未见每请求完整文件hash；[U1] L424–430通过权重更新/adapter拒绝与失效约束身份 | **纠正“每请求hash”表述**：[L3] L729–746是在load阶段完整hash一次；独立CLI测试每请求重新load才每次付费。不能说同一已加载runner每次生成都hash。不要删除完整SHA保护，应考虑加载校验复用/合并读取，保留真实冷启动成本。 |
| 权重与显存 | [U1] L406–415移除GPU上的原AdaLN权重，只保留plan表；[S1]因而把其任务改成DiT全驻留 | 我们仍保留CPU源权重，Q4混合模型/4090容量及其他组件不同，无法直接承诺“节省24.2GiB”或全驻留；表与执行workspace必须纳入预算。 |
| 图规划 | 上游固定GPU slab契合可中断CUDA graph；[U2] L512–516要求地址稳定 | [L1] L1862–1884在投影替换后调用通用resolve_graph_cut_plan / compute_with_graph_cuts；不是CacheOffload的scoped合并路径。图段、输入复制/扫描、分配、同步都需profile；**源码差异不能证明65s退化具体由哪一项引起**。 |

## 4. 我们的已有数字与主代理待补证据
- 原对照[ R ]：单4090、864×480/124帧/20步、锁定混合Q4、全组件卸载、22GiB预算；T2VA预热1＋3次测量，mixed仅1次。
- T2VA E2E **254.15→250.04 s**，DiT **211.160→186.231 s**；mixed E2E **627.91→714.72 s**，DiT **547.981→613.742 s**；两者视频/音频均逐字节一致。不是数值失败证据，也绝非“性能已修好”。
- 实际日志[ T ] L17/L146：完整hash **21.321 s**、预填 **2.263 s**；[ M ] L17/L325：**21.351 s / 2.226 s**。预填已计入DiT，不能重复扣减。
- 因此重复temb及额外hash是明确优化点，但**不能解释mixed在DiT内多65.761 s**：整轮预填仅2.226 s，hash又在DiT之外；主因要靠实际图段/复制/分配profile定位。
- 主代理建议验证顺序：①按阶段/段统计混合任务，区分扫描、复制、分配、计算与权重传输；②保持F32/原小GEMM，复用temb与稳定缓存输入，减少图段重绑/重复上传；③独立解决冷load哈希开销；④再做跨请求缓存。
- 验收仍需相同输入/seed/精度/卸载预算、仅AdaLN开关A/B，T2VA＋mixed重复统计及数值一致；上游资料不足以预设应达到的4090收益。此文无新运行、无修复/性能验收声明。

## 5. 原始来源与固定行号
- [S0] [sidecar 原始 PR](https://github.com/sgl-project/sglang/pull/33991) （head `d2c11858316b97f2c1a3f5f8725bfe1e4eeece11`）。
- [S1] [online rebuild PR](https://github.com/sgl-project/sglang/pull/34650) （head `f8d942d4a6a815d5a1a2961773bf79a97f72c666`，在线重建与通信profile）。
- [S2] [tiered cache PR](https://github.com/sgl-project/sglang/pull/37266) （head `2365c14c96722e390413a773095037183f6d45eb`，host-tier A/B）。
- [S3] [量化兼容 PR](https://github.com/sgl-project/sglang/pull/43745) （head `76d62531d0d58854ae6632662908cc2e91bc7e5e`，未合并sidecar兼容实验）。
- [U1] [固定手册 L358–430](https://github.com/sgl-project/sglang/blob/53295c697255a5e8ddb263f98ab7e047e2700ad2/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx#L358-L430)
- [U2] [固定 AdaLN 代码（相关行另见表）](https://github.com/sgl-project/sglang/blob/53295c697255a5e8ddb263f98ab7e047e2700ad2/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3_adaln_cache.py#L512-L845)
- 提交日期核验：[固定提交 GitHub API](https://api.github.com/repos/sgl-project/sglang/commits/53295c697255a5e8ddb263f98ab7e047e2700ad2)；PR正文/状态读取 `/repos/sgl-project/sglang/pulls/{33991,34650,37266,43745}`。
- [L1] `/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/src/dit_models/models/minimax_h3_full.hpp`，以上行号对应79b8aa0。
- [L2] `/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/src/dit_models/models/minimax_h3_cuda_adaln_cache.hpp`。
- [L3] `/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/src/dit_models/pipelines/minimax_h3_pipeline.cpp`。
- [ R ] `/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/docs/research/h3-four-method-results.md` L9–44。
- [ T ] `/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-offload-20261014/sync-adaln/t2va-001.stderr.log`。
- [ M ] `/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-offload-20261014/mixed-sync-adaln/ref_mixed-000.stderr.log`。

后续状态：修复与阶段实测见`h3-adaln-fix.md`；用户停止重复验收后的完成范围见`benchmark/h3_cuda/adaln-fix-results.json`。
