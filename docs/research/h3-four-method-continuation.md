# H3四项无训练方法：2026-10-14续做

> 最新已完成的T2VA三次实测与mixed单次对照见 `h3-four-method-results.md`；下文保留开发过程与曾遇到的问题，不代表最终未修复状态。

目标：10月14日晚完成AdaLN / Cache-DiT / Sage / SubBlock的同条件生产性能对照，15日早可查。Turbo/VDN/VSA暂停开发。**主四方法已完成约定配置的完整对照，见最新结果文档；本文件是过程中各阶段记录，不把tiny/编译通过当性能完成。**

## 已核对的上游与实质差异

固定reference-lock保持。raw HTTPS证书失败后，通过GitHub SSH稀疏取固定commit；112文件逐一按原锁length/blob hash核验。证据：`_scratch/h3-reference-ssh/*-verified.json`，没有运行第三方源码或下载新模型。

- **AdaLN**：SGLang `minimax_h3_adaln_cache.py::_project_plans_from_checkpoint`按layer outer / request plans inner，一次权重驻留服务所有sigma。旧segmented helper是plan outer，会重复H2D；本轮改layer-major小GEMM图合并，保留原每plan形状及原混合权重精度，而非扩大batch改变舍入。每步导入独立host副本，projection权重在denoiser图中不可达才能省传输。原host权重不释放。vLLM-Omni `exact_projection_cache.py`仅投影结果cache并不自动跳过block offload hooks，不能混称其本身省掉所有传输。
- **Cache-DiT**：核对`pattern_3_4_5.py`的Fn residual判定、Bn residual同锚复用和真实后段；迁移必须跨graph-cut保存独立RF/front/delta，命中真实跳过中段参数传输。开发中。
- **Sage**：固定官方SageAttention实现需区分QK量化粒度、smooth-K、PV类型和SM89路线；不得把所有Sage变体与已有INT8-QK/F16-PV一概而论。内核及分段生命周期开发中。
- **SubBlock**：SGLang Q64/K64，每侧4个16-token subcell，16对scaled dot score做LSE后per-head TopK；不是单均值router或删除Q。旧graph逐head逐Q展开不可用，本轮移植有界CUDA routing + batch/gather attention；保护所有非目标video输入，所有Q仍更新。开发中。

## 同条件实验

继承5/5整文件hash、原始97cc585现有优化baseline和4090单卡配置。864×480、124帧、24fps、20步、seed42、CFG1、res_multistep/simple、preserve、DiT/TE/VAE offload、22GiB规划、CPU6线程。先T2VA逐项、再mixed分别baseline。全关控制组与新源码一致，候选一次只开一个方法；先1次正确性/容量，再1预热+3测量。近似方法报告数值偏差/视频音频，而不是只给耗时。

## 不能跳过的门禁

- 固定依赖，不改ggml gitlink。所有新增生产路线H3/CUDA限定且默认关闭。
- 逐次检查同GPU UUID与所有计算/图形进程；无空闲卡继续CPU，禁止杀进程。
- 不只删除offload guard：限定不可变CPU参数、显式预算、单卡串行，graph reuse/async/adapter/parallel未经验证继续拒绝。
- AdaLN完整production configure/schedule/compute测试须覆盖两CFG角色、多sigma、repeat/reset、真实graph-plan/消费。空weights segment不允许回落到全模型H2D。
- SubBlock需原生有界容量证明，不增加MAX_GRAPH_SIZE硬顶。
- CUDA构建冻结snapshot，其他agent同时修改不改变被测二进制；源码hash标识实际snapshot。

当前工作/原始日志：`_scratch/h3-offload-20261014/`。原上一轮fusion回退数据保留，不算上游方法复现。具体拆因仍待逐路径profiler，不预先下“我们的融合一定更好”的结论。

## 实测阶段记录（进行中，非最终表）

- AdaLN layer-major生产tiny CUDA48/48逐位相同；完整Q4 T2VA首次：CLI331.70s / DiT187.268s / prefill2.331s / host cache1,532,397,305bytes / 显存采样15,173MiB。全关同binary控制CLI256.40s / DiT213.373s。方法额外整文件身份hash导致冷启动成本，尚不能声称端到端加速。
- AdaLN完整输出相对原始首次baseline SSIM0.977907 / PSNR36.69dB / audio RMSE0.00815，有真实差异；正在用原锁定投影权重probe核对，不以tiny F32证明完整Q4 exact。
- SubBlock CUDA：1663检查/214输出对照通过；完整目标T2VA正在执行，耗时待完整结束。不是官方FlashInfer原生indexed kernel吞吐复现，有界gather策略的复制开销明确记录。
- Cache-DiT CUDA helper29输出对照；生产入口24对照/16次命中通过。首次Q4触发显存预算拒绝（未生成成功、无性能值）；查明通用cache buffer计数已包含pool，重复charge已修，释放仅unused pool并预留planner margin，修复版snapshot在构建。
- Sage初次测试发现batch=1 offset view满足宽松contiguous却不满足scale stride断言，测试/adapter已严格化。继续测试在L63 near-uniform输入发现非有限值；**没有隐藏dense回退或放宽阈值**，正在检查H3专用tail zero-fill。完整模型Sage性能尚未启动。
- 本轮CPU总回归13/13 CTest、53/53 workflow/media及既有门禁通过。上述CUDA失败/拒绝日志均保留，不计通过。

### 更新：定位不能以小测代替

- SubBlock完整T2VA已完成：CLI424.503s / DiT381.484s / 采样峰值16,045MiB；相对首次baseline SSIM0.92612 / audio RMSE0.004996。当前gather策略比baseline慢，尚未正式重复测量，不冒充官方indexed稀疏吞吐。
- Cache预算第二次失败详细值：第一remainder的segment1 compute估计162,043,646,336bytes，而参数仅340,118,016bytes；**不只是pool计数问题，疑似attention退为dense**。继续诊断，不放宽22GiB预算硬跑，无成功性能值。
- AdaLN真实原锁权重probe：535参数完整加载，51投影×3sigma×T2VA/mixed共306项全部bit-exact，原权重字节未变；layer-major对独立原生投影图等价。但完整denoiser差异尚待定位，不能把306项外推成整模型exact。
- Sage H3专用tail zero-fill修复后，原126项数值测试通过、72实际H3SA dispatch；旧SAGQ四种kernel指令序列未变。完整Q4首轮运行进行中。
- 机器可读中间数据：`benchmark/h3_cuda/four-methods-progress.json`。状态明确in_progress，四项尚未全部交付。

- Sage完整T2VA首轮已完成：CLI282.743s / DiT239.591s / 采样峰值16,749MiB，SSIM0.97432 / audio RMSE0.007288。此处是Q32-per-warp INT8QK/F16PV/FP32accum+smoothK变体，首末层dense、所有20步；不是SM89自动选择FP8PV变体。相较已有baseline目前仍慢；还需完整重复与质量审查。

### Cache图规划进一步定位

CPU全尺寸图诊断发现`dense_square_nodes=0`，此前“dense attention fallback”只是假设，**现已排除**。失败来源是通用segment graph构造递归收集前序边界节点的全部祖先，内存估计把历史激活作为叶子累计到百GiB。H3专用边界截断诊断可把remainder最大真实段估计降到约8.64GB；修复/回归正在进行，未删除预算门禁。

### 新基线控制项与后续实测

- Cache scoped边界修复后完整目标成功：CLI228.443s / DiT185.311s / 峰值14,153MiB，但20步命中0。**不能把省时计作Cache-DiT跨步复用收益**；需RF实测与同执行器threshold0控制，图卸载改进另列。
- AdaLN目标几何实际Q4完整DiT诊断：默认cuDNN异步plan仅首次冷forward有差异，后5组exact；将现有 `ED_CUDNN_SDPA_SYNC_F32_SELF_ATTN=1` 固定给两组后6/6全部exact，profile明确sync build。确认冷启动算法选择是比较变量，不改变模型精度。
- workflow新增受限共同`runtime_env`仅允许此同步plan键，进入比较contract，旧数据不可混用。新同步baseline正在1预热+3测量。该设置不改项目生产默认行为。
- full-file identity保留，H3/CUDA AdaLN可用现有libcrypto3的EVP流式SHA256（无mandatory依赖、不可用则portable）；实际18.8GB hash21.84s且完整digest与锁一致。消除纯C++校验带来的不必要冷进程成本，不能省略完整hash。
