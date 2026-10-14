# H3 SubBlock 有界 CUDA：接入与验证记录（2026-10-14）

> 后续状态（2026-10-14）：已完成SM89 indexed片内流水与有界K/V转换优化，完整T2VA/mixed均提速且与旧indexed产物逐字节相同；见`docs/research/h3-subblock-fix.md`。下文保留最初gather及后续indexed接入历史。

## 实现状态

已实现：两个 H3 专用 CUSTOM 节点（router / attention），三个 metadata leaves。
Q64/K64，4×4 个16-token子块均值点积的 LSE；每个 head、每个视频 Q 块独立 exact TopK。
视频查询访问选中视频 K 块＋所有受保护 K；所有非视频查询访问完整真实 K/V。
ragged cell/block 不贡献虚假 exp(0)；gather 的 padding 用 -INF mask，输出散回全部真实查询。
不增大图 cap、不做 mean-query-only 或全局TopK、不删除非视频输出、不降低模型精度。

执行策略是 **有界批量 gather + 项目已有 CUDA FlashAttention**，不是移植官方 FlashInfer 原生稀疏内核。
它避免按每个 head/Q 块扩展整个 ggml 图，但会重复搬运入选 K/V；是否获得速度收益只能由正式 GPU 性能对照判定。
无全局状态、D2H路由读取或私有 cudaMalloc；路由与 attention 使用同一 CUDA stream、ggml pool。

## 固定来源核对

只读来源：`_scratch/h3-reference-ssh/sglang`，HEAD `53295c697255a5e8ddb263f98ab7e047e2700ad2`。
主代理的 `sglang-verified.json` 与本 worker 的 `git hash-object` 均核对：

- `subblock_sparse/router.py`：`cedc1e655a68b675f39c4c03d8189fef386cc3ad`
- `subblock_sparse/kernels.py`：`8de3d732f3e1ae2b9c5c24df50228f83cdaf821c`

另已读 `subblock_sparse_attn.py` 的 heterogeneous-query 分支。
本实现独立写 CUDA/C++，未复制 Triton/CuTe 源码。

| 细节 | 官方固定实现 | 本次适配／理由 |
|---|---|---|
| 估计器 | Q64/K64，Q/K各4个子块，pairwise LSE | 相同，不改为 mean-Q 或 row-normalized mass |
| 池化数值 | FP32 mean后 Q乘scale·log2e，写 BF16；BF16 GEMM→FP32 score | **保留项目已有 F32 mean/score**与自然指数，避免偷偷改变原比较契约；非官方逐位复现 |
| 选择 | threshold interpolation，16/24/32轮，边界可能近似，输出顺序未定义 | shared-memory exact bitonic TopK、相同分数低ID优先；更严格但不等于官方相同kernel |
| 预算 | K64路径默认向8取整（FlashInfer收费粒度） | 保留 `ceil(keep_ratio*video_blocks)` 或显式topk；本消费内核无需8倍数，正式报告必须列实际topk |
| 非视频保护 | query-block mask可让未选查询dense；key保护不是此router本身提供 | 保留本项目语义：text/audio/reference/query均dense query且始终保留为K/V |
| attention | 架构专属原生 sparse BF16（另有SM90 sageFP8模式） | stock ggml F16 FlashAttention，K/V按baseline缩放1/128，输出×128；不引入Sage量化 |
| 目标设备 | 固定上游按SM90/100/120分派 | 本地SM89 4090，supported同时检查NVIDIA与stock Flash能力 |

## 主代理必须做的最小接入（不改固定 ggml）

根 `CMakeLists.txt` **`edgedit` target创建后** include：

```cmake
include("${CMAKE_CURRENT_SOURCE_DIR}/src/core/backend/ggml/cuda/h3_subblock.cmake")
```

构建 `-DED_ENABLE_CUDA_H3_SUBBLOCK=ON`。helper默认OFF；运行仍需已有、默认关闭的 `ED_H3_SPARSE_ATTN`。
helper只给ggml-cuda和edgedit加PRIVATE宏，不强制已有CPU参考tests进入CUDA路径。

共享CUSTOM dispatcher无动态registry，helper采用 **build-local副本注入**：

- 固定源 `third_party/ggml/src/ggml-cuda/ggml-cuda.cu` 完整SHA256
  `1284940278a57c529b88ba4a3b4f49a82cf431fea63848bfb74d9e7f1f6ee265`。
- 先核对hash，再核对include／Modulation compute／Modulation supports三个唯一锚点。
- 输出到 `<build>/h3-subblock-generated/ggml-cuda.cu`，加三个H3-tag hooks；替换ggml-cuda target中**恰好一个**原源文件。
- hash不符、锚点不唯一或其他helper已替换源文件均fail closed。原源加入`CMAKE_CONFIGURE_DEPENDS`，更改后重新核验。
- 默认OFF不读取/生成/替换；原gitlink和依赖tree不变。生成目录有完整原/新SHA256 provenance。
- 新include path包含原CUDA源码目录，使用原target编译参数。已使用实际candidate原有NVCC参数编译生成副本和新kernel，未运行GPU。

不修改 `full.hpp`/pipeline；首末层/步dense策略由原代码保留。
native adapter要求CUDA registry、F32 Q/K/V、flash启用；缺dispatcher直接失败。
图切分后的route leaf允许op=NONE，以shape校验，不依赖原生产节点指针；内部scratch使用同一stream/pool。

## 边界

- N≤65536，H≤64，视频块≤1024。额外按真实字节数拒绝。
- router：`H*B*4*128*4*2 + H*B*B*4` bytes，上限192MiB。
- routing输出：`H*B*topk*4` bytes，由ggml图allocator管理。
- attention显式scratch ≤128MiB，批大小1..64自适应；**不包含stock Flash自身工作区、图激活或模型权重**，不是整个max-vram保证。
- 9472 token/H56/B148：router 38,854,144 bytes；显式gather 128,679,936 bytes，batch21。
- 长序列/大head导致实际scratch越界时失败，不改cap绕过。

## 本 worker 验证

已通过：
- NVCC12.2/SM89 standalone CUDA TU编译（无GPU运行）。
- GCC12 shape validator编译；production header含native宏语法检查。
- CPU-only新测试：946项检查、214次输出对照通过，含batch边界与反简化路由fixtures。
- 构建helper配置回归5/5：OFF不生成、ON恰好三hook、源码hash失配拒绝、target源缺失/重复拒绝。
- 原CPU数学参考及原CPU ggml展开图回归通过：reference max_abs=1.2666e-7，dense=5.96046e-8，sparse=1.11759e-7，flash-baseline=5.68859e-4。
- 真实图形状：常规输入2个节点；非连续view输入5个节点，生产9472/H56图也不随块数展开。

新增测试不是CUDA已通过的声明。GPU矩阵待主代理通过既有wrapper串行安排。
CPU emulation独立重写lane-reduction/streaming-LSE/gather/attention，对照原CPU方法；CUDA模式读取真实路由和全部输出。
预设CUDA容差 abs4e-4 + rel4e-3（CPU abs2e-6+rel2e-6），不自动放宽。
覆盖：1/15/16/17/63/64/65/127/128/129/257-token边界、多个不连续target视频段、四类保护token、两个不同输入重放、非连续有偏移views、tie、全选与分头路由。
每次输出预填NaN、路由预填-1；任何缺写/非有限/错误设备/CPU fallback均失败。

## 构建/运行测试

用匹配candidate的库编译（不是旧版本candidate）：

```bash
ROOT=/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation
BUILD=/absolute/path/to/matching/candidate
c++ -std=c++17 -O2 -ffp-contract=off -DGGML_MAX_NAME=128 \
  -I"$ROOT/third_party/ggml/include" -I"$ROOT/src" \
  "$ROOT/tests/h3_subblock_test.cpp" \
  -L"$BUILD/bin" -Wl,-rpath,"$BUILD/bin" \
  -lggml-cuda -lggml-base -o "$BUILD/bin/h3-subblock-test"
"$BUILD/bin/h3-subblock-test" --cpu
```

本地旧库没有shape validator符号时，仅CPU构建需附加 `h3_subblock_shape.cpp`；正式candidate已包含它，不重复编译。

GPU只通过 `scripts/test_h3_gpu_parity.sh --run --device FULL_GPU_UUID --binary .../h3-subblock-test`；程序校验显式授权、完整唯一UUID、匹配且30秒内preflight，默认不运行。
完整Q4单卡卸载模型、实际864×480/124帧/20步质量与性能不由tiny单测替代，仍须主代理执行。

## 性能风险：staging 估算，不是原生 sparse 内核吞吐复现

以下是**每个 attention layer、每次 forward**的确定性 staging 写入量，由 `h3_subblock::traffic()` 按实际launcher计算；GiB=2^30。
H56、总N15000、keep≈0.5，protected为连续span的例子：

| protected tokens | video blocks/topk | batch | sparse + dense batches | K/V staging writes | mask writes | Q staging writes |
|---|---|---|---|---|---|---|
| 0 | 235/118 | 13 | 1013 + 0 | 50,884,771,840 B（47.39 GiB） | 12,721,192,960 B（11.85 GiB） | 431,226,880 B（0.40 GiB） |
| 500 | 227/114 | 13 | 978 + 35 | 54,268,526,592 B（50.54 GiB） | 13,567,131,648 B（12.64 GiB） | 431,226,880 B（0.40 GiB） |

这些只是显式写入：源F32 K/V读取、Flash对stage的再读取、输出/散回另计。不是HBM实测（缓存影响流量），也不据此伪造耗时。
因此虽然resident scratch≤128MiB，**累计搬运是数十GiB/层/forward**，几十层、多次forward可能很慢。13这个batch来自按dense最坏N预留空间，而不是只按sparse slots分配；保护span更多时dense批次还会增加。
正式benchmark应标为 `SubBlock F32-router + bounded gather + ggml FlashAttention`，不得标为SGLang/FlashInfer吞吐复现。

后续直接indexed kernel的必要点（当前未实现，不作为已完成能力）：
1. 保留已验证的router输出 `[head,qblock,topk]`，消除K/V gather与全mask tensor；K/V只做一次连续F16缩放prep或在indexed loader中处理。
2. 原生Q64 CTA读取本Q块route row，将每个selected K64映射回原K/V地址；末尾tail与受保护K span必须正确mask，不能用全局mask/全局TopK替代。
3. 一个online-softmax状态跨所有selected视频块及protected块，正确累积max/sum/PV；非视频Q的dense分支沿用stockflash或专门batch，不能少算输出。
4. SM89内核可专门化现有ggml MMA Flash load/loop（需独立H3实例、固定源码生成与许可审查），或移植已固定、支持SM89的indexed块实现；不要改共享模板影响其他模型。
5. 保留同样F32 Q、F16缩放K/V、输出补偿与重放/尾部/分段测试，先数值再完整Q4卸载性能。该路径才消除当前主性能风险。

## v2：直接 indexed MMA attention（本轮新增，待主代理 GPU 验收）

主代理提供的完整 Q4 对照：旧 gather **424.50s / DiT381.48s**，baseline约 **256s / DiT213s**，SSIM **0.926**。本 worker 没有重跑该实验；该结果属于旧 gather，**不属于下面新内核**。

### 执行与总开关

- `ED_H3_SPARSE_ATTN` 仍是唯一方法总开关、默认0。关闭它不会进入任何 SubBlock tag。
- 新 `H3IA (0x48334941)` tag 与旧 gather `H3RA` tag独立；无 gather/dense-mask 静默回退。
- 当前主代理已通过两路径共433次数值对照；`ED_H3_SPARSE_ATTN=1`默认选择indexed，`ED_H3_SUBBLOCK_INDEXED=0`仅作gather诊断。总开关默认仍为0，不会影响默认模型路径。
- **验收通过后应由主代理把 indexed 设为总开关=1的默认实现，gather仅保留显式诊断选择**。临时INDEXED变量不是要求用户额外开启的最终接口。
- 环境选择在构图时固化进tag；benchmark使用新进程，不在persistent/cached runner中途切换环境变量。
- 新实现使用已有hash-checked build-local dispatcher hooks，不修改固定ggml文件、full.hpp或pipeline；现有首末层/步dense策略原样保留。

### 直接索引核心

新文件 `h3_subblock_indexed.cuh`：每CTA=Q64、一个head、4warps；每warp计算16行。
Q只加载一次进寄存器。每轮依据本head/Qblock的route ID读取一个K64，K/V直接由原输入读到共享内存；之后依次读所有protected keys。
受保护Q由另一kernel完整遍历真实K，仍产生全部输出。视频Q从未改成全N dense mask。

- 不分配global K/V stage或全N mask，无D2H route读取。
- 每层至多 **2次attention launch**（视频Q sparse、非视频Q dense）；router原有kernel不变。
- 每CTA仅 **34,816 bytes shared memory**，K共享区先暂存Q、Q进寄存器后被K重用。
- QK和PV均使用原生 `mma.sync.m16n8k16.row.col.f32.f16.f16.f32`；PV概率为F16，累积为FP32。
- 行max/denominator/输出的在线softmax状态跨所有选中视频K64和protected块持续更新，不在模态边界reset。
- 支持span-local尾部K64、最后不足16的子块、非连续F32输入视图、多个target span以及packed protected keys；不强迫topk成为8倍数。
- 输入容量/布局门禁仍保留。CUDA capability要求SM80+并有匹配的已编译MMA架构；不支持时明确拒绝。

**流量不是零**：不同Q块仍需读取其选中K/V（块稀疏attention的本来工作），本次消除的是旧路径每Q产生的global staging写入/再读取及全mask物化，不宣称K/V每层只读取一次。未做异步双缓冲或跨CTA预取；剩余读带宽、occupancy及router成本需要完整模型实测。

### 精度差异明确说明

本地固定ggml MMA路径先把Q转half，再与half(scale)相乘，结果half；本实现保持该顺序：
`Q'=half(half(Q)*half(128/sqrt(128)))`，`K'=half(K/128)`，`V'=half(V/128)`。
QK与PV均FP32累积，softmax max/sum和输出重缩放FP32，P进入PV前转F16，最终输出乘128补偿。

- 原CPU数学参考使用F32 Q/K/V且没有上述half舍入，**不应期待bitexact**。
- 官方固定SGLang基于BF16池化/架构专属稀疏内核；本地router仍维持既有F32池化/精确TopK，以不改变当前对照预算与路由契约。
- 本online softmax以真实行max归一化；stock ggml另有max offset策略，运算排列/rounding也不完全相同。
- 单测沿用预先固定CUDA容差 `abs4e-4 + rel4e-3`，没有为新内核放宽。完整视频/音频质量仍需主代理验证，不把小张量结果替代全Q4结果。

### 借鉴来源与许可证

无新增下载、未执行上游代码。使用现有固定vendor文件：

| 文件 | 用途 | 许可证／SHA256 |
|---|---|---|
| `sage/mma.cuh` | 直接include现有ldmatrix和FP16 MMA wrapper，不修改它 | Apache-2.0；文件保留FlashInfer 2023及SageAttention 2024原声明；`841a0b028e25b48fd83ad042d7bf817f84bfd5e5630d19d08510b0b797914231` |
| `sage/attn_utils.cuh` | 核对m16n16寄存器行列映射、四线程row reduction、FP32 online rescale/PV模式；未include整套Sage量化attention | Apache-2.0；`d74a33cc4e186ad545f8005e113c375eadf9c40639bc562f9a44e995dd3e72f0` |
| 固定ggml `fattn-mma-f16.cuh` | 核对baseline Qhalf×scalehalf、K/V prep及float输出语义 | MIT；`7b72e041a8304d06b167587e573ac08f9fda7708a4a3be4908a8b00d62130a26` |

新wrapper没有启用Sage的INT8、FP8或额外模型量化，也未复制完整官方Sage/FlashInfer kernel冒充其吞吐复现。

### 本轮验证边界

- SM89 NVCC standalone编译：两个indexed specialization均198 registers、34,816 bytes shared、0 spill、0 stack（候选完整flags结果以最终交付为准）。
- CPU **68,051项检查、433次输出对照通过**：包括共享给CUDA的Q/K/V ldmatrix地址映射、P寄存器重组、全部输出地址唯一覆盖、half-prep在线softmax oracle、反序全选块与大幅移动max、两种implementation tag的既有尾部/换输入/多模态保护矩阵。
- GPU测试现在对gather和indexed两个**显式不同tag**均进行原始CPU参考对照；主代理需重建matching candidate并通过既有single-GPU wrapper执行。
- CPU oracle不能证明硬件MMA寄存器映射或真实CUDA同步正确；**本worker没有运行GPU，也没有新indexed完整模型性能数据**。
