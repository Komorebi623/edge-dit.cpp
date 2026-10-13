# H3 SubBlock 有界 CUDA：接入与验证记录（2026-10-14）

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
