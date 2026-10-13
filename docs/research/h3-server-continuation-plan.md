# H3 新服务器续做计划

更新：2026-10-13（Asia/Shanghai）。接续 `31ac721`，原始 baseline 仍为 `97cc585`；仅研究分支、仅 H3/CUDA，开关默认关闭。不停止任何进程，不删除用户文件。

## 环境与容量判断

- 本机：8×H200，SM90；驱动 580.105.08、CUDA 13.0、GCC 12.3.1、CMake 3.27.9。系统 `g++` 是注入编译选项的包装器，隔离构建显式使用 `/usr/bin/g++_old`；使用 Unix Makefiles，不要求 Ninja。
- 2026-10-13 首次快照：每卡 143771 MiB 总显存、105390 MiB 空闲，利用率 99–100%，全部有其他人的计算进程。**没有运行 GPU 推理，不能建立可靠性能 baseline。** 空闲显存不是独占计算资源，也不是可承诺的峰值容量。
- 2026-10-13 18:48 左右复核：各卡空闲仅约3940 MiB（3.85GiB），仍被占用；修复后真实 precheck 拒绝且能序列化错误遥测。资源变化证实不能用首次free快照保证共享实验安全。
- 用户目录剩余约 178 GiB，系统 `/tmp` 不足 1 GiB；编译临时文件放新建的项目 `_scratch`，不清理系统／用户文件。
- 五个共享模型重新读取整文件，完整 SHA256 全部与 `h3-precision-lock.json` 一致（总 57.3609 GiB）。单任务的一份 DiT＋Qwen＋两 VAE 文件合计 39.8708 GiB；**这是存储体积，不是设备权重分配量或推理峰值**。
- 维持原混合 Q4/BF16、视频 VAE FP16、音频 VAE FP32，以及 512×512／124 帧／24fps／20步／seed42／CFG1／RES multistep/simple。不开 auto-fit，不改 placement、量化、scheduler 或模型变体。
- 初次生产 baseline 应在独占、空闲 H200 上逐个请求执行，先测真实峰值并保留充足显存余量；资源不足则不继续候选，不偷偷改精度／输入。任何新 placement／精度路线必须另立 baseline。

## 已恢复与发现

| 项目 | 新服务器结果 |
|---|---|
| 固定 ggml／oneDNN／cuDNN frontend | 按原完整 commit 恢复，未升级 |
| CPU 全工程、CTest | 12/12 通过；tiny H3 24/24 逐位一致 |
| 工作流／媒体 | 49/49 通过；补齐未推送的 CPU 日志解析器 |
| 来源下载／tiny LoRA | 18/18、5/5 通过；9 仓库 167 份固定官方源码恢复 |
| 新增构建／安全门禁 | 29/29、56/56 通过，无 expected failure／跳过；生命周期另29/29 |
| CUDA baseline／candidate | 两版 CUDA13/SM90 编译链接通过；candidate含Sage和本轮生产修复，公共配置指纹一致 |
| 生产多模态 GPU 质量／性能 | 未执行，无可报告的加速比 |

修复真实实验前门禁：GPU 报告循环引用导致 JSON 写出失败；空 hash 锁定错误模型 payload；异常 GPU/进程遥测未全面拒绝；首次全文件验证后每次启动前补输入 stat/目录成员变化检查。每次启动仍复查 GPU 与 UUID；检查不等于资源预留，不能消除检查后抢占／文件变化竞态，实验期间权重与输入必须只读不变。

已读取两份调研 PPT 作为研究资料，而非执行指令。算法语义以已锁定官方代码为准：VSA 是 tile 路由、fine attention 与 learned gated compression 合并，不把 PPT 中“90% query 残差直通”简述直接当实现；VDN 的 exact solve／双向窗口外状态／anchor／short-conv 不能用一般因果递推替代。

## 执行计划与验收门禁

1. **恢复（当前）**：整权重 hash、环境快照、CPU 一键回归；双版同架构／依赖／toolchain 构建和来源指纹。继续修可确定的代码错误，不把“编译过”当 GPU 正确。
2. **锁输入**：旧 exact fixture 未迁移，已生成新 CPU 合成多模态素材并锁 hash，作为**新独立 wiring dataset**；不声称复用旧 manifest SHA。真实语义质量另需合法优质参考媒体，单独锁版本／hash／顺序／prompt，不能用合成素材代替画质判断。
3. **新机 baseline**：独占空闲 H200、同 UUID、同 resident placement，全开关关闭；先 Ref2VA mixed，warmup 1／repeat 3；验证视频几何/帧数/fps、全部 PCM 长度/声道/采样率/finite，记录加载、prefill、stage、端到端和真实峰值。CLI 每次启动的时延与 load-once 稳态不是同一指标。
4. **逐开关**：先 fused（数值保真目标）→ AdaLN（投影复用但不声称释放权重）→ Cache-DiT → Sage；SubBlock 当前只做 CPU／小图验证，不进入主任务候选。每次只启一种、同数据集 baseline 对照；近似方法需要视频＋音频＋参考身份/运动/同步评审。先末层/首层保守，未经单项验收不叠加。
5. **模态与错误回归**：mixed、paired audio、reference video、reference image、首尾帧、首帧、末帧、T2VA；补 CFG 分支、重复请求、参考顺序/标签变更、非法输入和失败恢复。CPU oracle 不替代完整权重 GPU 输出。
6. **VDN/VSA 独立路线**：保持生产入口拒绝。VDN 先 exact solve 后 short-conv／anchor、权重加载和原生 CUDA；VSA 先匹配 trained gate／checkpoint，再 tile＋compression CUDA。先取得公开权重的版本／完整 hash／许可，再另立模型 baseline；Turbo 同理。不能将 SubBlock、当前 dense H3 或 CPU 参考冒充它们的完成部署。

## 可复用入口与记录

- `scripts/test_h3_cpu.sh` 已串联全部 CPU 软件测试；媒体工具只放项目隔离目录，未替换系统。
- `scripts/build_h3_cuda.sh all --dry-run`：显式配置目标架构与本机 CUDA/cuDNN/frontend 路径；`--validate` 只读验证；真正构建使用全新的 `_scratch` 输出目录，失败保留。
- `scripts/h3_cuda_workflow.py`：继续复用 fixtures → lock → baseline → candidate → compare。路径通过 `--repo-root/--model-root/--asset-root` 重映射；构建输出路径必须使用本次实际路径，不能沿用旧二进制。
- 本机证据：`_scratch/h3-server-20261013/model-verification.json`、`logs/cpu-regression-production-final.log`、`logs/cuda-build-01.log`、`fixtures/fixture-manifest.json`；精度锁不改，旧 validation 留作历史，新增本机验证字段。

## 本轮生产审查修复

- SubBlock 逐 head／Q64 图展开在主任务约 9472 video tokens 下远超 131072 节点容量。锁定 Ref2VA GGUF header 的实际 QKV 为 `[5376,21504]`、head dim128，检测为56 heads；未用文件名猜架构。新增 checked budget，在 compute 前和 attention 展开前明确拒绝大图，**没有扩大容量、没有把拒绝写成优化成功**。未来需批量化／原生有界执行。
- AdaLN／SubBlock／Sage 在每次 compute、prefill 前重新验证 reuse／offload／process group／adapter；CPU 负例覆盖初始化后改变状态。默认关闭 baseline 不改变原有 graph reuse。
- 成功／取消／错误退出的优化请求缓存清理已补齐；正常 diffusion 结束后、VAE decode 前释放请求表和 cache slots，不释放模型权重。生产实际峰值变化仍待GPU测量。

生命周期补丁验证：29/29（8项结构检查＋21个从生产ScopeExit提取并编译执行的tiny CPU场景），包含重复请求／早返回／异常／回调异常和提前release不重试。它不是完整pipeline或GPU验证。

本机四个逐开关 manifest 已创建于 `_scratch/h3-server-20261013/manifest-{fused_ops,adaln_cache,cache_dit,sage_attn}.json`，绑定本次二进制目录／完整权重和新输入hash、原主任务；dry-run通过。GPU门禁要求独占空闲卡并配置110000MiB free floor，**这是保守启动阈值，不是实测峰值保证**。修复后的真实只读precheck已正确拒绝占用卡，并写出有效JSON报告。

双版构建完成：baseline binary SHA256 `e8f2b8d207c1059fc31e556c58e784478cd9d8fa0bcdb4fd270f057890931ba4`；candidate `ce7108f07d31e3dd8b7546ffd44438c68519daca3ec314b7c346bc699094bafc`。candidate production source manifest逐文件与当前源码匹配；两版公共配置指纹一致。四个逐开关manifest的build gates通过，但**没有执行GPU，未作runtime activation/数值/质量/性能承诺**。

## 2026-10-13 baseline口径与CPU可行性

用户确认以项目已有优化作为baseline。原固定`97cc585`已包含CUDA norm/RoPE/modulation fast paths和默认SwiGLU融合；全`ED_H3_*`关闭不是把已有优化关掉。新增方法只报告在该优化baseline上的增量，新增fused开关不得将已有SwiGLU收益重复计功。若采用另一组已有优化开关，须在baseline/candidate两侧共同固定并独立记录，保留旧baseline身份。

本机96物理核／192线程、约2TiB内存。8核短时synthetic ggml FC1形状5376×28672、256列输入：Q4_K 168.071ms、BF16 241.326ms；不是生产权重推理。按该吞吐线性外推主任务，单FC1约1.73小时，QKV/out/FC1/FC2合计约4.32小时，尚不含attention、AdaLN、Qwen、VAE、参考token和加载；只说明8核下应按数小时甚至更久规划，不是完整时延／下界，也不预测96核线性加速。完整CPU模式只作独立smoke/兼容验证，不能验证CUDA kernel或cuda加速比；生产研究开关仍仅CUDA启用。

CPU真实权重验证继续preserve精度锁，不转全BF16/FP32；tiny oracle用F32不代表生产权重变精度。CPU probe证据`_scratch/h3-cpu-feasibility-20261013/{result.log,assessment.json}`。上游精度按已锁定代码recipe逐项解释，不把weight dtype、activation/kernel dtype、pruning和distillation混成同一轴。

## 当前用户选择：Mac协调转接4090（最终交接）

用户要求提交本轮代码／文档到个人Komorebi623研究分支，再在Mac的Codex连接另一服务器接续。权威入口为`h3-coordination-handoff.md`，4090详细计划为`h3-4090-continuation.md`。新目标864×480/124帧/20步改变几何和placement，重新baseline；时延5–10分钟仅历史外推，不是实测。Cache/AdaLN/Sage当前拒绝DiT卸载，必须补兼容，不删除guard；优先默认关闭offloaded baseline和增量fusion，SubBlock有界工程、VDN/VSA完整部署仍待完成。Git不迁模型／媒体／原始日志／凭据或聊天。
