# H3 AdaLN：对照上游后的性能修复

日期：2026-10-14（Asia/Shanghai）。从`79b8aa0`接续，仅修H3/CUDA AdaLN路径，默认开关仍关闭；模型、ggml/oneDNN/frontend版本与精度锁不变。**修复已实现；用户于2026-10-14 15:47要求停止重复验收，已只终止自有测试，保留所有产物。下表profile不是三次正式性能表。**

## 上游证据，不混淆收益口径

详见`h3-adaln-upstream-gap.md`，公开PR为作者报告、非本机复现：
- SGLang在线AdaLN PR34650：8×RTX PRO5000、1344×768/124帧/50步，135.74→127.03s（耗时↓6.4%）。AdaLN释放GPU权重后也改变了DiT驻留方式，不能当纯缓存固定placement收益。
- 两级cache PR37266：单H200两边都已开启AdaLN，重复时间表请求94.77→88.42s（↓6.7%），首见请求基本不变；这不是相对关闭AdaLN的对照。
- PR43745单RTX PRO6000 sidecar+INT8实验369.8→369.4s，主要收益是显存；该PR未合并、晚于锁定源码，未迁移其权重/精度。

## 差距与实际修复

1. **主要退化：host缓存替换投影后，通用图规划过度切分和重复分配。** 原mixed层数不变，但同样91个图段，权重传输减少却分配阶段增大。AdaLN现复用H3已有的boundary-aware测量/相邻合并，显式保持预算、单卡、不可变CPU权重、禁止async/reuse/adapter；不是开启Cache-DiT复用、也不跳block。完整DiT仍50层/20步。
2. **共享time embedding。** 上游每个plan一次temb后跨层复用；旧移植按layer虽只传一次投影权重，却重复执行temb。现先一次请求级temb pass，再原形状逐层小GEMM；原F32输出、混合Q4/BF16权重不改。临时存储计入缓存预算，reset/失败/异常回滚后不遗留。
3. **完整缓存重入。** 全部plan已完整时直接返回存储命中，不额外分配temb、不因剩余预算不足销毁有效cache。
4. 上游跨请求LRU、GPU常驻表与我们的host请求级策略仍不同。本轮不为了复现上游数字改变冷CLI口径、不换BF16缓存，不删除模型加载时的整文件SHA256。

## 完整mixed profile（均同一套instrumentation）

任务：单RTX4090、864×480/124f/24fps/20step/seed42/CFG1、res_multistep/simple、preserve、DiT/TE/VAE offload、22GiB预算、CPU6、共同同步cuDNN计划。固定原mixed图/视频/音频内容及顺序。

| 阶段 | 原baseline | 旧AdaLN | 修复首版 |
|---|---:|---:|---:|
| CLI E2E | 628.74s | 714.30s | **567.30s** |
| DiT调用 | 548.76s | 612.60s | **465.24s** |
| 图段计算合计 | 391.44s | 387.21s | 386.54s |
| 图/缓冲分配阶段 | 80.67s | **173.57s** | **34.33s** |
| 权重H2D阶段 | 56.00s | 22.42s | 19.72s |
| 输入copy阶段 | 1.45s | 2.32s | 1.72s |
| DiT图段/步 | 91 | 91 | **10** |

新旧“分配阶段”包含执行器计时所覆盖的分配/等待，不夸大成每次cudaMalloc已逐个归因。修复版视频124/124帧完全相同、音频RMSE0；采样峰值18,061MiB，仍为采样峰值而非硬cap。profile和正式perf分开。

## 已做验证

- 原生CPU TimeEmbedder/curve、GEMM形状、共享temb、预算/异常/重复CFG测试；完整cache重入存储不变。
- 生产CUDA tiny48组（2请求×CFG×3sigma×两架构/模态组合）通过；修复版原锁Q4实际51投影×3sigma×两布局共306组全部逐位一致。
- CPU总回归14/14 CTest、53/53 workflow/media及附加套件通过，无skip；不把任何tiny测试当作生产速度证明。

证据根 `_scratch/h3-adaln-fix-20261014/`：`baseline-profile/`、`old-adaln-profile/`、`fix01-profile/`、`fix01-profile-comparison/`、`logs/`、`fix-source-01/02`及构建provenance。正式T2VA与mixed均安排新baseline/修复版1次预热＋3次测量，动态库完整hash锁定，性能阶段无并行编译。原始失败/旧值不覆盖。

## 用户停止时的正式单次结果

原mixed baseline三次均值627.993s；修复版已完成的两次测量为571.529s、565.666s（DiT464.764s、464.638s），均比baseline快；两次各自与一个baseline产物逐字节相同（candidate1=baseline1，candidate2=baseline3）；baseline自身重复结果并非全相同，因此不能声称按序配对的三次比较全部逐位一致。第三次在加载阶段按用户要求终止，T2VA/最终8模式未启动，**不声称完整三次验收通过**。后续改为T2VA单次先验，成功后mixed单次。详见`benchmark/h3_cuda/adaln-fix-results.json`。
