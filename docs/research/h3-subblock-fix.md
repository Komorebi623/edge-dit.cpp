# H3/CUDA SubBlock：SM89 性能退化修复

2026-10-14，接续`86ab99a`。用户要求停止重复模型验收，后续**T2VA单次先验，确认有效后mixed单次**；不默认做预热＋三次完整生成。CPU/tiny覆盖是短软件测试，不当生产速度证据。

## 环境与原始性能

单RTX4090（SM89），固定CUDA12.2/GCC12/cuDNN9.24及ggml/oneDNN/frontend；不更换依赖或模型。864×480、124帧/24fps、20步、seed42、CFG1、res_multistep/simple、preserve、CPU6、DiT/TE/VAE卸载、22GiB规划预算、共同同步cuDNN计划。原始baseline`97cc585`保留项目已有优化。

上一轮：T2VA baseline254.15s vs SubBlock327.46s；mixed627.91s vs1018.73s。均是实测退化，不因“做了稀疏”而自动成立加速。更低10%保留率仍慢且质量退化，不能靠继续丢弃更多块掩盖内核问题。

## 上游实现与差距

固定SGLang `53295c697255a5e8ddb263f98ab7e047e2700ad2`，只读源码：
- `subblock_sparse_attn.py` L69–85：作者记录37.7k-token T2VA，丢弃75%/80%/85%块约1.14×/1.18×/1.21×；激进时间门控会改变画面。**这是源码中的作者报告，不是本机4090复现，也没有足够信息把它当完整固定配置性能表。**
- 同文件L314–342及手册L1387–1415：原生BF16后端按SM90/100/120分派，SM89不支持。不能直接照搬吞吐数字或宣称我们已使用上游内核。
- `subblock_sparse/kernels.py`：Tensor Core BF16路由评分＋寄存器LSE；专用稀疏attention。我们保持既有F32池化/路由和exact TopK，不偷偷改BF16路由、Sage量化、保留率或门控。

## 修复

1. **K/V转换摊销与局部性**：旧内核每个Q块重复读取F32 K/V并转换。现在同一head批次只做一次原有`F32*(1/128)→F16`，供所有Q块复用；最多8头，K/V暂存总量≤128MiB，尾head/尾token处理保留，ggml同stream pool所有权不变。
2. **真正的片内搬运流水**：旧循环`load→同步→QK→softmax→PV→同步`串行。现在K(next)搬运与当前softmax/PV重叠，V(next)搬运与下一次QK重叠；仍只使用原34KiB shared memory，不通过翻倍shared牺牲驻留。每个tile复用前有跨warp barrier；末轮wait-all。这里是CUDA内核内部`cp.async`，**不是启用异步权重卸载**，原async/offload组合限制不变。
3. **卸载图执行**：SubBlock单方法调用H3已有boundary-aware切分/合并，额外为K/V暂存留预算，减少不必要的分段与分配；仍完整执行50层/20步，**没有Cache-DiT跨步/跨block复用**。这是独立于稀疏attention本身的执行器收益。

默认`ED_H3_SPARSE_ATTN=0`不变；仅该H3/CUDA路径启用修改。Sage/其他模型/其他backend不切换执行器。keep=0.5、exact TopK及原稠密门控保持，未更改权重、sigma或输入。

## 单层定位（不能当整模型收益）

N=15000、H=56、D=128、235个视频块、TopK118；无protected token，合成同一QKV；分配/上传/读取校验不计时。每个图1次预热＋1次测量，非重复生成。

| 实现/阶段 | 毫秒 |
|---|---:|
| 旧indexed attention | 81.44 |
| 只改有界K/V暂存 | 71.83 |
| 加入片内流水 | **27.91** |
| 最终生产库：router + attention | **5.77 + 28.02 = 33.79** |
| 同输入生产cuDNN dense，含KV准备/输出缩放 | **50.51** |
| stock ggml dense诊断（不是生产baseline） | 103.43 |

最终库的cuDNN路径通过实际DSO调用计数确认成功，无stock fallback；稀疏含路由比该单层生产dense少约33.1%时间。tiny433输出比较通过，分片/尾块/跨head批次/受保护查询覆盖保留。原生内核只读已选K块，所有Q均更新。实际GPU计数器权限不足，保留失败日志、不修改系统设置；不虚构SM/DRAM计数器归因。

机器证据：`benchmark/h3_cuda/subblock-fix-kernel-profile.json`。完整源码snapshot指纹`9e38c09eff4531bf17e5079e2b20de535b71728717866787a5c06f3175df8985`，CPU14/14 CTest＋53/53 workflow/media及附加套件通过。完整权重T2VA与mixed单次验证均已完成，见下表。

## 验收与边界

证据根`_scratch/h3-subblock-fix-20261014/`，固定build/source/manifest/provenance。性能只启用SubBlock，不叠AdaLN/Cache/Sage。最终需分别列总CLI、DiT、峰值与baseline及旧SubBlock输出差异；近似方法不承诺无损，不把降低attention耗时直接当整模型倍率。mixed要在T2VA通过后启动。`--max-vram`是规划预算，不是外部CUDA workspace硬上限。

## 快速复现工具

沿用已有workflow，不新建模型运行框架。`scripts/build_h3_subblock_probe.py --build <固定candidate构建目录> --output <新路径>`只编译短单层诊断，不探测GPU；之后通过已有`test_h3_gpu_parity.sh --run --device <单卡UUID> --binary <绝对路径>`运行。probe显式区分生产cuDNN与stock ggml，检查实际dispatch，不拿较慢的stock路径冒充baseline。正式生成用`h3_cuda_workflow.py`，manifest固定warmup=0/repeats=1；CPU回归仍为`test_h3_cpu.sh`。

## 完整目标单次结果（已完成）

| 任务 | baseline CLI / DiT | 旧SubBlock CLI / DiT | 修复版 CLI / DiT | 相对baseline耗时下降 | 相对dense SSIM / audio RMSE |
|---|---:|---:|---:|---:|---:|
| T2VA | 254.33 / 211.297s | 327.46 / 284.533s | **215.38 / 172.403s** | **15.3%** | 0.917351 / 0.007621 |
| mixed | 627.91 / 547.981s | 1018.73 / 938.710s | **551.30 / 469.145s** | **12.2%** | 0.977898 / 0.004690 |

本轮仅新增3次完整生成：T2VA baseline与修复版各1次，之后mixed修复版1次；mixed复用已完成的同配置、完整动态库hash锁定baseline，两组比较均通过workflow受控契约检查。旧T2VA数字是上一轮3次均值，旧mixed为单次。没有再做3次重复，不声称统计稳定性。

**T2VA、mixed的修复版MP4均与对应旧SubBlock产物逐字节相同**，因此这次工程提速没有增加原有稀疏误差。相对dense依旧是有损近似，音频relative-L2分别0.4693/0.3041，不能称为无损或仅靠SSIM宣称语义/音画同步验收通过。8种完整模式没有再次全部重跑。

T2VA分段79→5，mixed91→10；该规划收益与原生稀疏内核收益共同计入上述CLI数字，没有把总提升全算成稀疏计算减少。采样显存峰值：T2VA baseline16035、修复15505MiB；mixed baseline18751、修复18529MiB。T2VA按报告时间窗口划分采样，均不是精确进程峰值。

完整机器证据与报告hash：`benchmark/h3_cuda/subblock-fix-results.json`。新T2VA证据`_scratch/h3-subblock-fix-20261014/t2va-once/`；mixed证据`mixed-once-02/`。第一次mixed准备因误用T2VA-only方法目录而在推理前拒绝，失败记录保留；修正为已有mixed目录后仅执行了一次模型生成。所有GPU任务已结束，无待执行队列。
