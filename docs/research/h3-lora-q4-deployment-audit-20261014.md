# H3 LoRA：最新公开路线与4090/Q4迁移审计

核验日期：2026-10-14（Asia/Shanghai），接续`58c4764`。本轮为**现成加速LoRA的迁移研究，不是训练或已完成部署**。未运行GPU、未下载模型/adapter大文件、未改精度锁、依赖、生产代码或默认开关。

## 1. 结论与“最新”的边界

**ComfyUI不是必要依赖。原Q4底模＋独立高精度LoRA输出分支，是优先工程路线；现在不能把文件放进去就宣称可用。**

证书校验正常的GitHub API已重新读取当前主线，固定到下表。Hugging Face模型页/API及官方别名HTTPS因self-signed证书失败；web工具无可用正文。未关闭TLS校验、未改系统证书。因此**不能断言Larry HF当前最新就是v4，也不能给出未经核验的adapter SHA256、大小或完整tensor清单**。下述Larry v4是当前SGLang/RH明确推荐、固定文件的可追溯候选，不是HF latest声明。

| 官方项目 | 本轮实际查询的主线HEAD | 提交UTC日期 | 结论 |
|---|---|---|---|
| RH-RunningHub/MiniMax-H3-MultiGPU-Lightning | `e8f860bb71bc7793e2614e471bbb7e375a8acde7` | 2026-09-26 | 仍使用SGLang；RH自有权重未公开 |
| RH-RunningHub/ComfyUI-RH-MiniMax-H3 | `f58a52fc6e4ec932e61ec6388c0b1858d3b35174` | 2026-08-28 | 独立Comfy节点，配套INT8 ConvRot，不是现有GGUF Q4格式 |
| hao-ai-lab/FastVideo | `d5253fd71055927f12aa6c4aa679b93a00238851` | 2026-10-09 | dense/VSA LoRA、V2、PDD须分别处理；本轮主线与原锁相同 |
| ModelTC/Minimax-H3-Turbo | `02e26d591f7a04d5d1a074c9566d5dd4f22f6225` | 2026-08-27 | 同时提供Diffusers与Comfy格式、推理脚本；不是仅Comfy |
| ModelTC/LightX2V | `8e86555809efd20138d3a4e1f937ee306cdf0ee0` | 2026-10-13 | 当前native LoRA合并明确拒绝quantized DiT/lazy-load |
| sgl-project/sglang | `c82e8bd5fe9cad787c2cd3ccf067a1ba5b95e0c4` | 2026-10-14 | 最新H3手册推荐Larry8步，明确GGUF＋LoRA不支持；auto merge有小增量保护 |

源码/元数据放`_scratch/h3-lora-audit-20261014/`，约7MiB；只读官方文件保存SHA256，不改已有研究source lock。原API响应、失败说明与选读文件见`benchmark/h3_cuda/lora-q4-audit-20261014.json`。

## 2. 三条路线不可混为一谈

### Larry / LightX2V：优先普通底模＋加速LoRA

- SGLang当前手册推荐`larryvrh/MiniMax-H3-Turbo-Lora/minimax_h3_turbo_v4_step600_ema.safetensors`，8次denoiser evaluation，strength1.0，FL2VA分区用于T2VA/FL2VA。RH README固定HF revision `43a74557ac3f6539db8e0f2a959d03feb7a81480`。**这是上游引用，尚未独立读取HF adapter header**。
- LightX2V完整公开代码同时支持Diffusers推理和Comfy工作流：544p FL4步/8步配方为视频/音频shift12/3；768p FL4步/8步为6/3。Ref2VA另有单独文件，不能把FL adapter用于同shape的Ref底模。
- SGLang对`minimax_h3_fl2v_turbo_4step_v0.1.safetensors`明确记载rank128但缺alpha，须显式alpha8；有效LoRA比例是strength×8/128。**不能假定所有LightX2V版本或Larry都用这个alpha。**
- RH较新README还引用Ref2VA 8步768p文件；ModelTC独立Turbo仓库较早规格表未列此项，不能据较早README断言它不存在。下载候选前仍须HF当前文件/模型卡核验。
- 优先工程目标：Larry的已知8步文件（HF核验后），随后选择明确匹配FL/Ref的LightX2V。维持864×480、124帧、seed42；不把官方训练分辨率换成我们的目标后称为原配方逐位复现。

### RunningHub：公开实现可借鉴，私有效果不能承诺

- 当前Lightning README仍明确RH in-house权重未公开。其12.2×为8×RTX6000D、BF16、50→4步，加Sage/Cache/compile的联合结果，且不含冷加载；不能当单4090 Q4的LoRA单项收益。
- 可在非Comfy的SGLang中加载社区LoRA，公开部署示例引用Larry/LightX2V。Comfy节点项目则使用完整INT8 ConvRot包（README约95GiB、建议110GiB空闲），不是本项目Q4 GGUF部署指令。

### FastVideo：dense-LoRA是可行候选，VSA/PDD是独立模型工程

- `FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA`中的`dense-datafree/adapter_model.safetensors`有官方`--no-vsa` Python样例，**无需Comfy、无需VSA**。这是与Larry/LightX2V并列的优先候选。
- 但FastH3 adapter不一定只有A/B：还可包含`.diff/.diff_b/.diff_param`精确dense增量和`.set_weight/.set_param`替换参数。我们旧PEFT-only合并器会拒绝这些；不能忽略未知key后宣称加载成功。
- 同仓库`vsa-datafree`等含`to_gate_compress`，必须运行匹配VSA。它是训练后的压缩注意力分支，**不是我们的SubBlock；不能用SubBlock开关替代**。
- V2为8-forward、80% VSA、shift10/3、非均匀rungs `[999,874,749,624,500,375,250,125]`。其9个sigma点含terminal0，并非9次forward。tile64有Triton路径，但不代表我们的C++/Q4 VSA已实现。
- OmniRef PDD-8改变video/audio输出头（32个细网格头融合成8次forward）；官方该配方tile128依赖sm100a/103a且无Triton fallback，不能直接用于4090(SM89)。
- FastVideo README 2026-10-06宣布4090消费卡路线，但同时有Trim/pruned、NVFP4等变体。**“FastH3支持4090”不等于所有adapter/backend都支持原始完整GGUF Q4。**

## 3. Q4如何用LoRA：两种不同计算契约

以列向量记法表示：

**A，首选：保留Q4底模＋高精度低秩输出分支**

`y = W_Q4(x) + strength*(alpha/rank)*B(Ax)`。

- 原GGUF字节和已有CUDA量化矩阵乘不变；A/B保留发布精度（BF16/F16或文件实际类型），选择受控累积精度。LoRA不需要也不应默认量化到Q4。
- 不构造整个高精度`W+BA`，减少附加显存；按层/图段上传adapter并管理其生命周期。dense delta/替换参数另行处理，不能当低秩矩阵硬套。
- 这是Q4近似底模上的新模型，不保证与官方BF16底模＋LoRA输出相同。需要独立adapter身份和少步采样契约；保留原baseline，不改原精度锁。
- 当前SGLang手册提示Larry部分更新低于BF16权重ULP，直接BF16 merge会抹掉大量更新（其作者称典型层77–97%）；这是官方报告，不是本机测量。Q4重新量化也存在吞掉小增量的风险，不能仅凭文件merge成功判定效果。

**B，备选：高精度合并后重新量化为新的Q4模型**

`W'_Q4 = Q4(W_high_precision + strength*(alpha/rank)*BA)`。

- 正式高精度底模合并→按选定量化策略转换→保存新文件、新hash/manifest/模型baseline。不能覆盖原GGUF，且重新量化有新的误差。
- `Q4(dequant(W_Q4)+BA)`只是从已损失信息的底模再量化，**不等价**于从原BF16合并，也不等价于A；不作为未经验证的默认捷径。
- A与B是不同质量/显存/速度权衡，不承诺A能零额外开销或B更优。LoRA少步带来的端到端收益要实测，20→8步不等于总耗时×0.4。

## 4. 本地具体差距与容量

只读GGUF header已与原33979-byte header结构对应：535张量，F32×13、BF16×264、Q4_0×50、Q4_K×208；不是全模型每个张量都是4bit。原DiT文件18,779,848,448字节（约17.49GiB）。同tensor元素若全按BF16存约61.70GiB（只估tensor payload，不是实际上游下载量）。

实际torch形状：QKV `[21504,5376]`，每个Q/K/V输出7168；fc1 `[28672,5376]`，两半各14336。FastVideo官方转换器明确说明：native与Diffusers的QKV需拆/合，SwiGLU两半顺序相反。我们base loader已有普通权重半交换，**不能证明adapter因子映射已正确**。分离Q/K/V的A因子不同，不能随意拼B并假设共享A；须保留三个增量或严格构造等价因子。

当前已有：
- `scripts/merge_minimax_h3_lora.py`：CPU、Diffusers全精度shard、严格`.lora_A.default.weight/.lora_B.default.weight`，FP32合并/回原dtype，hash保护；**不接收GGUF Q4，也不覆盖FastVideo dense/set载荷**。
- `MultiLoraAdapter`通用组件已有`base_linear + out_diff`机制，可借鉴，不必重新写低秩数学；但当前H3 pipeline没有把请求LoRA列表接到该组件，API有字段不等于功能已接通。
- H3 `ED_H3_TURBO`入口只验证离线合并safetensors及其recipe；没有动态Q4＋LoRA的身份契约，GGUF新artifact也不能只改文件名冒充已认证shard。
- H3现有Cache/AdaLN/SubBlock/Sage serial-offload保护拒绝mutable `weight_adapter`；adapter参数计入分段预算、上传、缓存失效与异常释放尚需工程化。**不得只删guard**。

空闲磁盘约37GiB、主机RAM约499GiB。磁盘不足以按常规完整落盘61.7GiB BF16底模再保留merged和Q4结果；未下载。动态路线只新增adapter/小元数据，HF恢复核验大小后若空间充足可按需下载；若必须下载完整BF16/INT8包或扩容，先请用户批准，禁止删除用户文件。现阶段不凭未知adapter大小承诺一定放得下。

## 5. 实施顺序与单次验收

1. 恢复可校验的HF访问或取得用户已有adapter文件；固定repo revision、完整SHA256、header、base分区、rank/alpha/strength/NFE/双shift。全key清单逐个映射，未知载荷/不匹配形状直接拒绝。
2. 先接**普通dense、无VSA**的Q4＋高精度LoRA输出分支；复用通用矩阵组件，H3/CUDA独立opt-in、默认关闭。不同时叠加Cache/SubBlock/Sage/AdaLN。
3. CPU/tiny验证`scale=0`原路径、QKV/SwiGLU因子转换、Q4原字节不变、adapter跨图段生命周期、取消/异常、同形状不同模型身份。真实LoRA固定单层对照不能由合成A/B替代。
4. 先8步Larry T2VA单次与原20步完整baseline比较，明确这是模型/步数一起变动；记录CLI、DiT、实际forward次数、峰值、输出视频及音频，**不拿dense输出SSIM当少步学生的唯一质量门**。
5. 匹配FL2VA关键帧单次，再用独立Ref2VA adapter跑mixed单次。FastVideo dense4步另立case；VSA/PDD/Trim留独立分支与模型baseline，不夹带进本轮。

**尚未完成真实adapter下载/header/fullhash、Q4动态LoRA接入或GPU效果/速度验证。本轮交付是已核对源码的选择与迁移计划，不能标“LoRA已可用”。**

## 6. 官方来源（均为固定版本的只读来源）

- [SGLang最新H3手册](https://github.com/sgl-project/sglang/blob/c82e8bd5fe9cad787c2cd3ccf067a1ba5b95e0c4/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx)：GGUF限制、Larry/Light配方、merge更新舍入警告。
- [RunningHub Lightning](https://github.com/RH-RunningHub/MiniMax-H3-MultiGPU-Lightning/blob/e8f860bb71bc7793e2614e471bbb7e375a8acde7/README.md)：私有权重限制、公开LoRA固定下载、测试口径。
- [RH Comfy节点](https://github.com/RH-RunningHub/ComfyUI-RH-MiniMax-H3/blob/f58a52fc6e4ec932e61ec6388c0b1858d3b35174/README.md)：INT8包大小与实际功能。
- [FastVideo LoRA入口](https://github.com/hao-ai-lab/FastVideo/blob/d5253fd71055927f12aa6c4aa679b93a00238851/examples/inference/basic/basic_fasth3_lora_preview.py)、同目录`run_fasth3_lora_preview_dense_datafree.sh`、`fastvideo/models/loader/lora_patch.py`：dense/VSA分流、完整载荷。
- [FastVideo Comfy转换器](https://github.com/hao-ai-lab/FastVideo/blob/d5253fd71055927f12aa6c4aa679b93a00238851/scripts/checkpoint_conversion/convert_minimax_h3_comfy_lora.py)：QKV与SwiGLU实质转换，不是重命名。
- [FastH3 V2/PDD配方](https://github.com/hao-ai-lab/FastVideo/blob/d5253fd71055927f12aa6c4aa679b93a00238851/docs/inference/fasth3-distilled.md)：梯度、门控、头数与硬件限制。
- [LightX2V Turbo规格](https://github.com/ModelTC/Minimax-H3-Turbo/blob/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/README.md)、[当前合并实现](https://github.com/ModelTC/LightX2V/blob/8e86555809efd20138d3a4e1f937ee306cdf0ee0/lightx2v/models/networks/minimax_h3/lora.py)：分区、NFE/双shift、原精度限制。

## 7. 后续精度/配对追问（2026-10-14）

进一步读完FastVideo `scripts/benchmarks/minimax_h3_4090/README.md`与`bench_pod.py`、`minimax_h3_sparse_int8.py`、`nvfp4_config.py`（仍固定d5253fd）：

- **有SM89专用VSA tile64路径**，可选择BF16 QK/BF16 PV或INT8 QK/BF16 PV；保留原tile路由、尾块和学习到的压缩分支。不能笼统说FastVideo VSA只能Blackwell。
- 该4090实测文档明确使用**private `FastVideo/FastH3-Pruned-8Step-FP8-ckpt300`**，修订`f2ef54f9ff2091762ab8689b6514dcab5bc1d383`，DiT为FP8，非完整Q4；记录其当时私有状态，不由代码仓库推断HF当前可下载性。脚本L93说明NVFP4选项为其他GPU保留，4090 DiT使用FP8。
- 该4090记录的text encoder是**NVFP4存储、逐linear反量化到BF16 GEMM**；这不证明4090以原生NVFP4运行整个DiT。当前NVFP4 quant config最小capability=100，不能直接把其Blackwell主计算路径给SM89。
- 官方RTX PRO6000文档明确引用公开FastH3 V1/V2 **NVFP4成品学生模型**，例如`FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4`。这是已训练/量化学生，不等于通用Q4底模＋任意LoRA的公开配对。
- Larry/LightX2V：在已核验官方源码/说明中，**未确认针对本项目Q4_K/Q4_0混合GGUF的作者验证配对**；SGLang当前GGUF+LoRA明确不支持，LightX2V合并器拒绝quantized DiT。HF API重试仍证书失败；不能把“未确认”写成“全球不存在”。

精度解释：LoRA训练精度、adapter存储精度、base量化格式、运行累积精度是不同维度。LoRA不天然只能用于某个dtype，但须匹配模型架构/具体训练base/张量布局；base量化误差、合并舍入、动态执行和重新量化会改变有效模型。`Q4_K/Q4_0`、NVFP4、NF4同为4bit也不互换，GGUF只是容器，不是统一4bit语义。

未量化的理想实数运算中，`(W+sBA)x = Wx+sB(Ax)`。量化后比较的是`Q(W+sBA)x`与`Q(W)x+sB(Ax)`，二者通常不等价。优先动态分支是避免把小LoRA增量再次压进低bit权重，但不消除原Q4误差，也不构成已验证质量保证。

若用户优先VSA：以VSA训练学生/携带完整gate与dense增量的adapter为模型基准，完整实现稀疏细粒度分支＋粗粒度压缩分支＋gate合并，严格遵守NFE/rungs/shifts。首选tile64 SM89可实现路径，不能以SubBlock顶替；之后再独立比较保留原Q4＋adapter与学生重新量化的质量/性能，不自动改原精度锁。无需重训才能移植已有模型，但不能忽略训练新增参数。

补充源码证据（官方固定GitHub版本）：
- [4090专项记录](https://github.com/hao-ai-lab/FastVideo/blob/d5253fd71055927f12aa6c4aa679b93a00238851/scripts/benchmarks/minimax_h3_4090/README.md)
- [4090实验入口](https://github.com/hao-ai-lab/FastVideo/blob/d5253fd71055927f12aa6c4aa679b93a00238851/scripts/benchmarks/minimax_h3_4090/bench_pod.py)
- [SM89 VSA实现](https://github.com/hao-ai-lab/FastVideo/blob/d5253fd71055927f12aa6c4aa679b93a00238851/fastvideo/attention/backends/minimax_h3_sparse_int8.py)
- [NVFP4计算能力门槛](https://github.com/hao-ai-lab/FastVideo/blob/d5253fd71055927f12aa6c4aa679b93a00238851/fastvideo/layers/quantization/nvfp4_config.py)
